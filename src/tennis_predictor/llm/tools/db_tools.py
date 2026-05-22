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
    HeadToHeadMatch,
    HeadToHeadResult,
    PlayerResolutionError,
    PlayerStats,
    RankingSnapshot,
    RecentFormSummary,
    RecentMatch,
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
    """Aggregate H2H record + per-meeting detail rows."""
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


__all__ = [
    "get_head_to_head",
    "get_player_ranking",
    "get_player_stats",
    "get_recent_form",
]
