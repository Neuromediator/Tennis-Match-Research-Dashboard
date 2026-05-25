"""DuckDB-backed LLM tools.

Each function:

1. Takes a DuckDB connection + the tool's Pydantic input model.
2. Resolves human-friendly player names to canonical_player_id via
   `player_aliases` (failing with `PlayerResolutionError` if a name is
   unknown or ambiguous).
3. Reads from `matches` / `rankings` with `match_date < as_of_date` to keep
   the tools honest about historical state (point-in-time correctness is a
   training-data invariant; the agent should respect it too so the
   narrative matches what the model is conditioned on).
4. Returns the matching Pydantic output model.

The agent loop in `llm/agent.py` is responsible for translating any
`PlayerResolutionError` raised here into a tool-result error block, so the
LLM sees the failure and can mention it in `caveats`. Other exceptions
bubble up (failure-mode 3 in CLAUDE.md: DB tool exceptions are programming
bugs and must not be silently masked).
"""

from __future__ import annotations

import duckdb

from tennis_predictor.data.reconcile import AliasIndex
from tennis_predictor.llm.tools.schemas import (
    GetHeadToHeadInput,
    GetPlayerRankingInput,
    GetPlayerStatsInput,
    GetRecentFormInput,
    GetSurfaceEloInput,
    H2HSummary,
    HeadToHeadMatch,
    HeadToHeadResult,
    PlayerResolutionError,
    PlayerStats,
    RankingSnapshot,
    RecentFormSummary,
    RecentMatch,
    SurfaceEloSummary,
    Tour,
)

# Confidence floor for taking an alias-index match as canonical. Below this
# the tool refuses to guess — better to surface the unresolved name in
# `caveats` than to silently attribute someone else's career to a player.
_MIN_RESOLUTION_CONFIDENCE: float = 0.85

# `match_tier` values that count as "tour-level" for the agent's context.
# Mirrors the training population so career numbers and recent form line
# up with what the model is calibrated against.
_TOUR_LEVEL_TIERS: tuple[str, ...] = ("main", "qual_chall", "qual_itf")

# Phase 2 documented Sackmann ingestion lag of 1-7 days for active tour
# players. Anything past this threshold is reason to suspect missing
# matches and to prefer web search for "current form" claims.
_STALE_FORM_DAYS: int = 7


def _resolve(conn: duckdb.DuckDBPyConnection, tour: Tour, name: str) -> tuple[str, str]:
    """Resolve `name` to (canonical_player_id, canonical_full_name).

    Strategy:
    1. Build a tour-scoped AliasIndex. Exact normalized match short-circuits
       and is cheap.
    2. Below `_MIN_RESOLUTION_CONFIDENCE`, raise `PlayerResolutionError`.
    3. Look up the canonical `full_name` from `players` so the agent can use
       the official spelling in narrative ("Djokovic, N." → "Novak Djokovic").
    """
    index = AliasIndex(conn, tour)
    result = index.lookup(name)
    if result.canonical_player_id is None or result.confidence < _MIN_RESOLUTION_CONFIDENCE:
        raise PlayerResolutionError(
            f"could not resolve player name {name!r} on {tour} tour "
            f"(best candidate {result.candidate_name!r}, confidence {result.confidence:.2f})"
        )
    row = conn.execute(
        "SELECT full_name FROM players WHERE player_id = ?",
        [result.canonical_player_id],
    ).fetchone()
    canonical_name = row[0] if row and row[0] else name
    return result.canonical_player_id, canonical_name


# ---------------------------------------------------------------------------
# get_player_stats
# ---------------------------------------------------------------------------


_PLAYER_STATS_SQL = """
    SELECT
        surface,
        SUM(CASE WHEN winner_player_id = ? THEN 1 ELSE 0 END) AS wins,
        SUM(CASE WHEN loser_player_id  = ? THEN 1 ELSE 0 END) AS losses
    FROM matches
    WHERE (winner_player_id = ? OR loser_player_id = ?)
      AND match_status = 'completed'
      AND match_tier IN ('main', 'qual_chall', 'qual_itf')
      AND tourney_date < ?
    GROUP BY surface
"""


def get_player_stats(
    conn: duckdb.DuckDBPyConnection,
    payload: GetPlayerStatsInput,
) -> PlayerStats:
    """Career totals + per-surface breakdown, as of `payload.as_of_date`."""
    canonical_id, canonical_name = _resolve(conn, payload.tour, payload.player_name)

    rows = conn.execute(
        _PLAYER_STATS_SQL,
        [canonical_id, canonical_id, canonical_id, canonical_id, payload.as_of_date],
    ).fetchall()

    surface_matches: dict[str, int] = {}
    surface_win_pct: dict[str, float] = {}
    total_wins = 0
    total_losses = 0
    for surface, wins, losses in rows:
        wins = int(wins or 0)
        losses = int(losses or 0)
        n = wins + losses
        if surface and n > 0:
            surface_matches[surface] = n
            surface_win_pct[surface] = wins / n
        total_wins += wins
        total_losses += losses

    career_matches = total_wins + total_losses
    career_win_pct = total_wins / career_matches if career_matches > 0 else None

    return PlayerStats(
        canonical_player_id=canonical_id,
        player_name=canonical_name,
        tour=payload.tour,
        as_of_date=payload.as_of_date,
        career_matches=career_matches,
        career_wins=total_wins,
        career_losses=total_losses,
        career_win_pct=career_win_pct,
        surface_matches=surface_matches,
        surface_win_pct=surface_win_pct,
    )


# ---------------------------------------------------------------------------
# get_head_to_head
# ---------------------------------------------------------------------------


_H2H_SQL = """
    SELECT
        tourney_date, surface, tourney_name, tourney_level, round, score,
        winner_player_id, loser_player_id
    FROM matches
    WHERE ((winner_player_id = ? AND loser_player_id = ?)
        OR (winner_player_id = ? AND loser_player_id = ?))
      AND match_status = 'completed'
      AND match_tier IN ('main', 'qual_chall', 'qual_itf')
      AND tourney_date < ?
    ORDER BY tourney_date ASC, match_num ASC
"""


def get_head_to_head(
    conn: duckdb.DuckDBPyConnection,
    payload: GetHeadToHeadInput,
) -> HeadToHeadResult:
    """Legacy aggregate H2H — kept for callers that still want the v1
    `HeadToHeadResult` shape (notably the Phase 5 tests).

    The Phase 6.1 LLM agent uses `get_head_to_head_v2` below, which
    returns the richer `H2HSummary` with per-surface breakdown,
    completion-status, and a matchstat-first / Sackmann-fallback data
    source.
    """
    a_id, a_name = _resolve(conn, payload.tour, payload.player_a_name)
    b_id, b_name = _resolve(conn, payload.tour, payload.player_b_name)
    if a_id == b_id:
        raise PlayerResolutionError(
            f"player_a_name and player_b_name resolve to the same player "
            f"({a_id!r}) — refusing to build self-H2H"
        )

    rows = conn.execute(
        _H2H_SQL,
        [a_id, b_id, b_id, a_id, payload.as_of_date],
    ).fetchall()

    matches: list[HeadToHeadMatch] = []
    a_wins = 0
    b_wins = 0
    for r in rows:
        (
            tourney_date,
            surface,
            tourney_name,
            tourney_level,
            round_name,
            score,
            winner_id,
            _loser_id,
        ) = r
        if winner_id == a_id:
            winner_name = a_name
            a_wins += 1
        else:
            winner_name = b_name
            b_wins += 1
        matches.append(
            HeadToHeadMatch(
                match_date=tourney_date,
                surface=surface,
                tournament_name=tourney_name,
                tournament_level=tourney_level,
                round_name=round_name,
                winner_name=winner_name,
                score=score,
            )
        )

    return HeadToHeadResult(
        player_a_name=a_name,
        player_b_name=b_name,
        tour=payload.tour,
        player_a_wins=a_wins,
        player_b_wins=b_wins,
        matches=matches,
    )


def get_head_to_head_v2(
    conn: duckdb.DuckDBPyConnection,
    payload: GetHeadToHeadInput,
) -> H2HSummary:
    """Phase 6.1 H2H tool returning matchstat-sourced detail rows with
    odds + completion status, falling back to Sackmann when matchstat
    quota is exhausted. See `data.recent_form_live.fetch_h2h_summary`
    for the underlying decision tree.

    Forward reference to `H2HSummary` keeps the import out of the
    legacy code path so existing tests don't accidentally touch the
    new schema.
    """
    # Local import to avoid the `data` → `llm` → `data` cycle that
    # would happen if we imported at module top.
    from tennis_predictor.data.recent_form_live import fetch_h2h_summary

    a_id, a_name = _resolve(conn, payload.tour, payload.player_a_name)
    b_id, b_name = _resolve(conn, payload.tour, payload.player_b_name)
    if a_id == b_id:
        raise PlayerResolutionError(
            f"player_a_name and player_b_name resolve to the same player "
            f"({a_id!r}) — refusing to build self-H2H"
        )

    return fetch_h2h_summary(
        conn,
        payload.tour,
        a_id,
        b_id,
        a_name,
        b_name,
        payload.as_of_date,
    )


# ---------------------------------------------------------------------------
# get_recent_form
# ---------------------------------------------------------------------------


_RECENT_FORM_SQL = """
    SELECT
        tourney_date, surface, tourney_name, tourney_level, round, score,
        winner_player_id, loser_player_id
    FROM matches
    WHERE (winner_player_id = ? OR loser_player_id = ?)
      AND match_status = 'completed'
      AND match_tier IN ('main', 'qual_chall', 'qual_itf')
      AND tourney_date < ?
    ORDER BY tourney_date DESC, match_num DESC
    LIMIT ?
"""


def get_recent_form(
    conn: duckdb.DuckDBPyConnection,
    payload: GetRecentFormInput,
) -> RecentFormSummary:
    """Most-recent N matches, newest-first, with W/L from the queried player's view."""
    canonical_id, canonical_name = _resolve(conn, payload.tour, payload.player_name)
    rows = conn.execute(
        _RECENT_FORM_SQL,
        [canonical_id, canonical_id, payload.as_of_date, payload.n_matches],
    ).fetchall()

    last_matches: list[RecentMatch] = []
    wins = 0
    losses = 0
    for r in rows:
        (
            tourney_date,
            surface,
            tourney_name,
            tourney_level,
            round_name,
            score,
            winner_id,
            loser_id,
        ) = r
        result: str
        opponent_id: str
        if winner_id == canonical_id:
            result = "W"
            opponent_id = loser_id
            wins += 1
        else:
            result = "L"
            opponent_id = winner_id
            losses += 1
        opponent_row = conn.execute(
            "SELECT full_name FROM players WHERE player_id = ?",
            [opponent_id],
        ).fetchone()
        opponent_name = opponent_row[0] if opponent_row and opponent_row[0] else opponent_id
        last_matches.append(
            RecentMatch(
                match_date=tourney_date,
                opponent_name=opponent_name,
                result=result,  # type: ignore[arg-type]
                surface=surface,
                tournament_name=tourney_name,
                tournament_level=tourney_level,
                round_name=round_name,
                score=score,
            )
        )

    n_returned = len(last_matches)
    win_pct = wins / n_returned if n_returned > 0 else None

    # Detect cold-data lag: the newest match returned vs the requested
    # `as_of_date`. If the gap exceeds the documented Sackmann ingestion
    # window, surface a warning so the agent prefers web search over DB
    # for any "current form" claim.
    latest_match_date = last_matches[0].match_date if last_matches else None
    data_freshness_warning: str | None = None
    if latest_match_date is not None:
        gap_days = (payload.as_of_date - latest_match_date).days
        if gap_days > _STALE_FORM_DAYS:
            data_freshness_warning = (
                f"Newest match in this record is {latest_match_date.isoformat()} "
                f"({gap_days} days before as_of_date {payload.as_of_date.isoformat()}). "
                "Sackmann cold-data ingestion lags 1-7 days for active tour players; "
                "any matches the player has played within that window are NOT in this "
                "list. Treat this record as historical, not 'current form'. Prefer "
                "web_search results for live status."
            )

    return RecentFormSummary(
        canonical_player_id=canonical_id,
        player_name=canonical_name,
        tour=payload.tour,
        as_of_date=payload.as_of_date,
        n_requested=payload.n_matches,
        n_returned=n_returned,
        wins=wins,
        losses=losses,
        win_pct=win_pct,
        last_matches=last_matches,
        latest_match_date=latest_match_date,
        data_freshness_warning=data_freshness_warning,
    )


# ---------------------------------------------------------------------------
# get_player_ranking
# ---------------------------------------------------------------------------


_RANKING_SQL = """
    SELECT ranking_date, rank, points
    FROM rankings
    WHERE player_id = ? AND ranking_date <= ?
    ORDER BY ranking_date DESC
    LIMIT 1
"""


def get_player_ranking(
    conn: duckdb.DuckDBPyConnection,
    payload: GetPlayerRankingInput,
) -> RankingSnapshot:
    """Most recent ranking snapshot on or before `as_of_date`."""
    canonical_id, canonical_name = _resolve(conn, payload.tour, payload.player_name)
    row = conn.execute(_RANKING_SQL, [canonical_id, payload.as_of_date]).fetchone()
    if row is None:
        return RankingSnapshot(
            canonical_player_id=canonical_id,
            player_name=canonical_name,
            tour=payload.tour,
            as_of_date=payload.as_of_date,
            rank=None,
            points=None,
            snapshot_date=None,
        )
    snapshot_date, rank, points = row
    return RankingSnapshot(
        canonical_player_id=canonical_id,
        player_name=canonical_name,
        tour=payload.tour,
        as_of_date=payload.as_of_date,
        rank=int(rank) if rank is not None else None,
        points=int(points) if points is not None else None,
        snapshot_date=snapshot_date,
    )


# ---------------------------------------------------------------------------
# get_surface_elo (Phase 6.1) — single round-trip for both players' surface
# Elo + diff + baseline win probability. Replaces the "issue two
# get_player_stats calls and infer something" pattern from Phase 5.
# ---------------------------------------------------------------------------


def _elo_logistic(diff: float) -> float:
    """Closed-form Elo expected-score: P(higher wins) = 1/(1+10^(-diff/400)).
    Inlined rather than imported to avoid a circular dependency on
    `features.elo` for what is a one-line formula."""
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def get_surface_elo(
    conn: duckdb.DuckDBPyConnection,
    payload: GetSurfaceEloInput,
) -> SurfaceEloSummary:
    """Read `elo_state` for both players on `payload.surface`.

    Players with no row on this surface get the default rating of 1500
    (the same prior `EloState` uses for first appearance). `as_of_date`
    is taken into account only for context — the persisted snapshot
    reflects all training data so the rating is always the most
    up-to-date one available."""
    a_id, a_name = _resolve(conn, payload.tour, payload.player_a_name)
    b_id, b_name = _resolve(conn, payload.tour, payload.player_b_name)
    if a_id == b_id:
        raise PlayerResolutionError(
            f"player_a_name and player_b_name resolve to the same player "
            f"({a_id!r}) — refusing to score a self-match Elo"
        )

    rows = conn.execute(
        "SELECT player_id, rating, last_updated_date FROM elo_state "
        "WHERE player_id IN (?, ?) AND surface = ?",
        [a_id, b_id, payload.surface],
    ).fetchall()
    elo_by_id: dict[str, float] = {}
    snapshot_dates: list = []
    for player_id, rating, last_updated in rows:
        elo_by_id[player_id] = float(rating)
        if last_updated is not None:
            snapshot_dates.append(last_updated)

    elo_a = elo_by_id.get(a_id, 1500.0)
    elo_b = elo_by_id.get(b_id, 1500.0)
    diff = elo_a - elo_b
    # Newest of the two snapshot dates — that's the most-recent rating
    # data we have on either player for this surface.
    snapshot_date = max(snapshot_dates) if snapshot_dates else None

    return SurfaceEloSummary(
        player_a_name=a_name,
        player_b_name=b_name,
        player_a_id=a_id,
        player_b_id=b_id,
        tour=payload.tour,
        surface=payload.surface,
        as_of_date=payload.as_of_date,
        player_a_elo=elo_a,
        player_b_elo=elo_b,
        diff_a_minus_b=diff,
        baseline_prob_a=_elo_logistic(diff),
        elo_state_snapshot_date=snapshot_date,
    )


__all__ = [
    "get_head_to_head",
    "get_head_to_head_v2",
    "get_player_ranking",
    "get_player_stats",
    "get_recent_form",
    "get_surface_elo",
]
