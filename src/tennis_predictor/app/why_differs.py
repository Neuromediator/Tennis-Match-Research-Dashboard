"""Deterministic "why model differs from market" structural checks.

When the trained LightGBM probability disagrees with the market by
> 10pp, the Prediction page surfaces a short panel explaining the
gap. Phase 6.2 explicitly does NOT use the LLM for this — Phase 6
narrative bias is a known failure mode (CLAUDE.md hard rule #4). The
three checks here are dated, structural, and trivially auditable.

Each check returns a `WhyDifferReason` or None; the caller assembles
the non-None results in order of strength. The order matches the
phase notes:

1. **Activity asymmetry** — one player has played a lot recently while
   the other has been absent. Elo cannot decay an inactive player.
2. **Stale surface Elo** — `elo_state.last_updated_date` is older than
   180 days for either player on this surface.
3. **Career-length asymmetry** — one player has > 200 career matches
   and the other has < 30. Captures returning veteran vs new pro.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import duckdb

ActivityKind = Literal[
    "activity_gap",
    "stale_elo",
    "career_asymmetry",
    "surface_elo_agrees_with_model",
    "recent_form_gap",
    "unexplained",
]

# Thresholds — these are the numbers in the Phase 6.2 design doc and
# are kept here, not in CLAUDE.md, because tuning them is a UI tweak
# not a contract change. Bump after observing 10-match acceptance data.
ACTIVITY_WINDOW_DAYS: int = 90
ACTIVITY_ACTIVE_THRESHOLD: int = 10
ACTIVITY_DORMANT_THRESHOLD: int = 3
STALE_ELO_DAYS: int = 180
CAREER_VETERAN_THRESHOLD: int = 200
CAREER_NEW_PRO_THRESHOLD: int = 30
# Phase 6.2 follow-up: minimum match count for the recent-form check.
# Below this we don't trust the win-rate enough to attribute the gap.
RECENT_FORM_MIN_MATCHES: int = 5
RECENT_FORM_WINDOW_DAYS: int = 60
# The two model-vs-rank-driven-market checks fire only when the model
# and surface-Elo agree against the market by at least this margin —
# tighter than the 10pp page-level trigger so the panel doesn't claim
# "Elo agrees" on a tie.
ELO_AGREEMENT_PP_THRESHOLD: float = 5.0


@dataclass(frozen=True)
class WhyDifferReason:
    kind: ActivityKind
    headline: str  # one-line user-facing label
    detail: str  # explanatory sentence below the headline


def _matches_in_window(
    conn: duckdb.DuckDBPyConnection,
    player_id: str,
    as_of_date: date,
    window_days: int,
) -> int:
    """Count completed matches for `player_id` in `(as_of - window, as_of)`.
    Reads from the `matches` table directly — this is a UI helper, not a
    feature path, so it intentionally bypasses the feature-replay
    machinery."""
    cutoff = as_of_date - timedelta(days=window_days)
    row = conn.execute(
        """
        SELECT COUNT(*) FROM matches
        WHERE (winner_player_id = ? OR loser_player_id = ?)
          AND match_status = 'completed'
          AND tourney_date BETWEEN ? AND ?
        """,
        [player_id, player_id, cutoff, as_of_date],
    ).fetchone()
    return int(row[0]) if row else 0


def _total_career_matches(conn: duckdb.DuckDBPyConnection, player_id: str) -> int:
    """Total completed matches for `player_id` across all surfaces."""
    row = conn.execute(
        """
        SELECT COUNT(*) FROM matches
        WHERE (winner_player_id = ? OR loser_player_id = ?)
          AND match_status = 'completed'
        """,
        [player_id, player_id],
    ).fetchone()
    return int(row[0]) if row else 0


def _elo_last_updated(conn: duckdb.DuckDBPyConnection, player_id: str, surface: str) -> date | None:
    """Return the `last_updated_date` from `elo_state` for the (player,
    surface) pair, or None if no row exists (player never played on
    this surface in the trained period)."""
    row = conn.execute(
        "SELECT last_updated_date FROM elo_state WHERE player_id = ? AND surface = ?",
        [player_id, surface],
    ).fetchone()
    return row[0] if row else None


def check_activity_gap(
    conn: duckdb.DuckDBPyConnection,
    player_a_id: str,
    player_b_id: str,
    player_a_name: str,
    player_b_name: str,
    as_of_date: date,
) -> WhyDifferReason | None:
    """Flag when one player has been heavily active and the other
    dormant in the last `ACTIVITY_WINDOW_DAYS`."""
    a_matches = _matches_in_window(conn, player_a_id, as_of_date, ACTIVITY_WINDOW_DAYS)
    b_matches = _matches_in_window(conn, player_b_id, as_of_date, ACTIVITY_WINDOW_DAYS)
    active, dormant_name, active_matches, dormant_matches = None, None, 0, 0
    if a_matches >= ACTIVITY_ACTIVE_THRESHOLD and b_matches < ACTIVITY_DORMANT_THRESHOLD:
        active, dormant_name = player_a_name, player_b_name
        active_matches, dormant_matches = a_matches, b_matches
    elif b_matches >= ACTIVITY_ACTIVE_THRESHOLD and a_matches < ACTIVITY_DORMANT_THRESHOLD:
        active, dormant_name = player_b_name, player_a_name
        active_matches, dormant_matches = b_matches, a_matches
    if active is None:
        return None
    return WhyDifferReason(
        kind="activity_gap",
        headline="Recent activity asymmetry — Elo cannot decay an inactive player.",
        detail=(
            f"{active} has played {active_matches} matches in the last "
            f"{ACTIVITY_WINDOW_DAYS} days; {dormant_name} has played {dormant_matches}. "
            "The model's surface Elo for the dormant player reflects their last "
            "active form, not present-day fitness."
        ),
    )


def check_stale_elo(
    conn: duckdb.DuckDBPyConnection,
    player_a_id: str,
    player_b_id: str,
    player_a_name: str,
    player_b_name: str,
    surface: str,
    as_of_date: date,
) -> WhyDifferReason | None:
    """Flag when either player's surface-Elo snapshot is older than
    `STALE_ELO_DAYS` days. Phase 6.2 example: Djokovic's clay Elo last
    updated 2025-05-26 on a 2026-05-26 fixture date."""
    a_date = _elo_last_updated(conn, player_a_id, surface)
    b_date = _elo_last_updated(conn, player_b_id, surface)
    cutoff = as_of_date - timedelta(days=STALE_ELO_DAYS)
    stale_lines: list[str] = []
    if a_date is not None and a_date < cutoff:
        stale_lines.append(f"{player_a_name}: {a_date.isoformat()}")
    if b_date is not None and b_date < cutoff:
        stale_lines.append(f"{player_b_name}: {b_date.isoformat()}")
    if not stale_lines:
        return None
    return WhyDifferReason(
        kind="stale_elo",
        headline=f"Stale surface Elo (> {STALE_ELO_DAYS} days since last update on {surface}).",
        detail=(
            "Last surface-Elo update — "
            + "; ".join(stale_lines)
            + ". The model anchors its prediction to this frozen rating."
        ),
    )


def check_career_asymmetry(
    conn: duckdb.DuckDBPyConnection,
    player_a_id: str,
    player_b_id: str,
    player_a_name: str,
    player_b_name: str,
) -> WhyDifferReason | None:
    """Flag a returning-veteran-vs-new-pro mismatch: one player has
    > 200 career matches and the other has < 30."""
    a_total = _total_career_matches(conn, player_a_id)
    b_total = _total_career_matches(conn, player_b_id)
    veteran, newcomer = None, None
    veteran_total, newcomer_total = 0, 0
    if a_total >= CAREER_VETERAN_THRESHOLD and b_total < CAREER_NEW_PRO_THRESHOLD:
        veteran, newcomer = player_a_name, player_b_name
        veteran_total, newcomer_total = a_total, b_total
    elif b_total >= CAREER_VETERAN_THRESHOLD and a_total < CAREER_NEW_PRO_THRESHOLD:
        veteran, newcomer = player_b_name, player_a_name
        veteran_total, newcomer_total = b_total, a_total
    if veteran is None:
        return None
    return WhyDifferReason(
        kind="career_asymmetry",
        headline="Returning-veteran vs new-pro asymmetry.",
        detail=(
            f"{veteran}: {veteran_total} career main-draw matches; "
            f"{newcomer}: {newcomer_total}. Career Elo can lag dramatically "
            "on a veteran returning from a long absence."
        ),
    )


def _recent_win_rate(
    conn: duckdb.DuckDBPyConnection,
    player_id: str,
    as_of_date: date,
    surface: str | None,
    window_days: int,
) -> tuple[float | None, int]:
    """Win rate of `player_id` in `(as_of - window, as_of)`. Returns
    `(rate, matches_played)` or `(None, 0)` when the sample is empty.
    When `surface` is provided the rate is restricted to that surface
    — useful because clay form is a poor predictor of grass form."""
    cutoff = as_of_date - timedelta(days=window_days)
    if surface:
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN winner_player_id = ? THEN 1 ELSE 0 END) AS wins,
                COUNT(*) AS total
            FROM matches
            WHERE (winner_player_id = ? OR loser_player_id = ?)
              AND match_status = 'completed'
              AND surface = ?
              AND tourney_date BETWEEN ? AND ?
            """,
            [player_id, player_id, player_id, surface, cutoff, as_of_date],
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN winner_player_id = ? THEN 1 ELSE 0 END) AS wins,
                COUNT(*) AS total
            FROM matches
            WHERE (winner_player_id = ? OR loser_player_id = ?)
              AND match_status = 'completed'
              AND tourney_date BETWEEN ? AND ?
            """,
            [player_id, player_id, player_id, cutoff, as_of_date],
        ).fetchone()
    if row is None or row[1] is None or int(row[1]) == 0:
        return None, 0
    wins, total = int(row[0] or 0), int(row[1])
    return wins / total, total


def check_surface_elo_agrees_with_model(
    *,
    player_a_name: str,
    player_b_name: str,
    model_prob_a: float,
    market_prob_a: float | None,
    surface_elo_prob_a: float | None,
) -> WhyDifferReason | None:
    """Flag when **both** our trained model and the surface-Elo baseline
    sit on the same side against the market by a meaningful margin.

    This is the structural answer to "the model is right and the market
    is the one wrong" — when two independent signals disagree with the
    market the gap is unlikely to be a model artifact (it's still not a
    *correct* call by definition, but it's not the inverted-favourite
    bug Phase 6.1 close-out flagged either). Covers the Baez-Burruchaga
    Roland Garros 2026 case where Baez is rank-favoured but clay Elo
    + LightGBM both lean to Burruchaga."""
    if market_prob_a is None or surface_elo_prob_a is None:
        return None
    model_diff = (model_prob_a - market_prob_a) * 100
    elo_diff = (surface_elo_prob_a - market_prob_a) * 100
    # Both signals must lean the same way AND each by at least the
    # agreement threshold relative to the market.
    threshold = ELO_AGREEMENT_PP_THRESHOLD
    same_side = (model_diff > 0 and elo_diff > 0) or (model_diff < 0 and elo_diff < 0)
    both_meaningful = abs(model_diff) >= threshold and abs(elo_diff) >= threshold
    if not (same_side and both_meaningful):
        return None
    favoured_by_signals = player_a_name if model_diff > 0 else player_b_name
    favoured_by_market = player_a_name if market_prob_a >= 0.5 else player_b_name
    if favoured_by_signals == favoured_by_market:
        # Same favourite, just stronger preference — not a "differs" reason.
        return None
    return WhyDifferReason(
        kind="surface_elo_agrees_with_model",
        headline="Model + surface Elo lean opposite to the market.",
        detail=(
            f"Surface-Elo gives {favoured_by_signals} a {abs(elo_diff):.1f}pp edge "
            f"over the market line, and the trained model agrees by {abs(model_diff):.1f}pp. "
            "Two independent signals against the rank-driven market — typically "
            "a surface-specialist mismatch the bookmaker hasn't priced in. "
            "Inspect H2H and recent form below."
        ),
    )


def check_recent_form_gap(
    conn: duckdb.DuckDBPyConnection,
    player_a_id: str,
    player_b_id: str,
    player_a_name: str,
    player_b_name: str,
    surface: str,
    as_of_date: date,
) -> WhyDifferReason | None:
    """Flag when one player has a markedly better recent win rate on
    this surface than the other. Catches form swings the trained
    surface-Elo update at K=32 hasn't fully reflected yet."""
    a_rate, a_n = _recent_win_rate(conn, player_a_id, as_of_date, surface, RECENT_FORM_WINDOW_DAYS)
    b_rate, b_n = _recent_win_rate(conn, player_b_id, as_of_date, surface, RECENT_FORM_WINDOW_DAYS)
    if a_rate is None or b_rate is None:
        return None
    if a_n < RECENT_FORM_MIN_MATCHES or b_n < RECENT_FORM_MIN_MATCHES:
        return None
    gap = abs(a_rate - b_rate)
    if gap < 0.30:  # < 30 percentage points is normal variance
        return None
    hot_name, hot_rate, hot_n = (
        (player_a_name, a_rate, a_n) if a_rate > b_rate else (player_b_name, b_rate, b_n)
    )
    cold_name, cold_rate, cold_n = (
        (player_b_name, b_rate, b_n) if a_rate > b_rate else (player_a_name, a_rate, a_n)
    )
    return WhyDifferReason(
        kind="recent_form_gap",
        headline=f"Recent form gap on {surface}.",
        detail=(
            f"{hot_name}: {hot_rate:.0%} ({hot_n} matches) on {surface} in the "
            f"last {RECENT_FORM_WINDOW_DAYS} days; "
            f"{cold_name}: {cold_rate:.0%} ({cold_n} matches). "
            "Surface-Elo updates at K=32 — short, sharp form swings take 5-10 "
            "wins to fully register in the rating."
        ),
    )


def compute_reasons(
    conn: duckdb.DuckDBPyConnection,
    *,
    player_a_id: str,
    player_b_id: str,
    player_a_name: str,
    player_b_name: str,
    surface: str,
    as_of_date: date,
    model_prob_a: float | None = None,
    market_prob_a: float | None = None,
    surface_elo_prob_a: float | None = None,
) -> list[WhyDifferReason]:
    """Run every structural check; return non-None results in priority
    order. Order matters — the panel surfaces them top-to-bottom.

    Priority logic:
    1. Activity gap — most common; addresses the inactive-player Elo
       freeze pattern.
    2. Stale Elo — fallback when activity gap doesn't fire but the
       snapshot is genuinely old.
    3. Surface-Elo + model agree against market — the
       "rank-vs-surface-specialist" case (Phase 6.2 follow-up).
    4. Recent form gap — short-window momentum the trained model
       can't catch up to fast enough.
    5. Career asymmetry — rare returning-veteran case.
    6. Generic fallback — when none of the structural checks fire
       but the gap is large enough that the page-level trigger asked
       for an explanation, point the user at the detail blocks
       (H2H, recent form, news) and own that we can't pinpoint a
       single cause. Honest > silent."""
    out: list[WhyDifferReason] = []
    for reason in (
        check_activity_gap(
            conn, player_a_id, player_b_id, player_a_name, player_b_name, as_of_date
        ),
        check_stale_elo(
            conn, player_a_id, player_b_id, player_a_name, player_b_name, surface, as_of_date
        ),
        check_surface_elo_agrees_with_model(
            player_a_name=player_a_name,
            player_b_name=player_b_name,
            model_prob_a=model_prob_a if model_prob_a is not None else 0.0,
            market_prob_a=market_prob_a,
            surface_elo_prob_a=surface_elo_prob_a,
        )
        if model_prob_a is not None
        else None,
        check_recent_form_gap(
            conn, player_a_id, player_b_id, player_a_name, player_b_name, surface, as_of_date
        ),
        check_career_asymmetry(conn, player_a_id, player_b_id, player_a_name, player_b_name),
    ):
        if reason is not None:
            out.append(reason)
    if not out:
        # Nothing structural triggered, but the page-level trigger
        # only invokes us when |model - market| > 10pp. Don't go
        # silent on the user — surface a honest "couldn't pinpoint
        # but here's what to look at" message.
        out.append(
            WhyDifferReason(
                kind="unexplained",
                headline="Model differs from market — no obvious structural cause.",
                detail=(
                    "None of activity gap / stale surface Elo / recent form gap / "
                    "career asymmetry fires. Likely a style-matchup or H2H pattern "
                    "the model and market price differently. Inspect H2H, recent "
                    "form, and news blocks below before trusting either side."
                ),
            )
        )
    return out


__all__ = [
    "ACTIVITY_WINDOW_DAYS",
    "ELO_AGREEMENT_PP_THRESHOLD",
    "RECENT_FORM_WINDOW_DAYS",
    "STALE_ELO_DAYS",
    "WhyDifferReason",
    "check_activity_gap",
    "check_career_asymmetry",
    "check_recent_form_gap",
    "check_stale_elo",
    "check_surface_elo_agrees_with_model",
    "compute_reasons",
]
