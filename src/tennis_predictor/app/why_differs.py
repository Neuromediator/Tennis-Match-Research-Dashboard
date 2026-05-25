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

ActivityKind = Literal["activity_gap", "stale_elo", "career_asymmetry"]

# Thresholds — these are the numbers in the Phase 6.2 design doc and
# are kept here, not in CLAUDE.md, because tuning them is a UI tweak
# not a contract change. Bump after observing 10-match acceptance data.
ACTIVITY_WINDOW_DAYS: int = 90
ACTIVITY_ACTIVE_THRESHOLD: int = 10
ACTIVITY_DORMANT_THRESHOLD: int = 3
STALE_ELO_DAYS: int = 180
CAREER_VETERAN_THRESHOLD: int = 200
CAREER_NEW_PRO_THRESHOLD: int = 30


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


def compute_reasons(
    conn: duckdb.DuckDBPyConnection,
    *,
    player_a_id: str,
    player_b_id: str,
    player_a_name: str,
    player_b_name: str,
    surface: str,
    as_of_date: date,
) -> list[WhyDifferReason]:
    """Run all three checks; return non-None results in priority order
    (activity → stale Elo → career)."""
    out: list[WhyDifferReason] = []
    for reason in (
        check_activity_gap(
            conn, player_a_id, player_b_id, player_a_name, player_b_name, as_of_date
        ),
        check_stale_elo(
            conn, player_a_id, player_b_id, player_a_name, player_b_name, surface, as_of_date
        ),
        check_career_asymmetry(conn, player_a_id, player_b_id, player_a_name, player_b_name),
    ):
        if reason is not None:
            out.append(reason)
    return out


__all__ = [
    "ACTIVITY_WINDOW_DAYS",
    "STALE_ELO_DAYS",
    "WhyDifferReason",
    "check_activity_gap",
    "check_career_asymmetry",
    "check_stale_elo",
    "compute_reasons",
]
