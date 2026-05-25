"""View-layer helpers for fresh "8 last matches" + detailed H2H lists.

Phase 6.1 introduces two surfaces that need data fresher than Sackmann's
1-7 day lag:

- Per-player last-N completed matches (rendered as the two-column recent
  form panel on the Prediction page).
- Per-pair detailed H2H history (with odds + completion status).

Both prefer matchstat (live, on-demand, 24h cached) and fall back to
Sackmann (cold layer) when quota is exhausted or the player has no
known matchstat external ID. The fallback path is **not** an error —
it's a documented graceful-degradation surface visible to the user via
the `data_source` field on every payload.

Why this module isn't merged into `db_tools.py`:

- `db_tools` are LLM-callable, so their I/O has to match the Pydantic
  schemas the JSON-schema layer exposes. These helpers are view-layer
  only — the LLM never sees them — so their I/O is internal-only and
  shaped for what Streamlit needs.
- The view layer renders these without the agent in the loop; coupling
  them to LLM tooling would force a redundant agent invocation for
  every page render.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

import duckdb

from tennis_predictor.data.matchstat import (
    RichMatch,
    RichMatchesPage,
    TourCode,
    parse_completion_status,
    winner_index,
)
from tennis_predictor.data.matchstat_live import (
    MatchstatBudgetExceeded,
    fetch_h2h,
    fetch_player_past_matches,
    quota_status,
)
from tennis_predictor.llm.tools.schemas import (
    H2HMatchDetail,
    H2HSummary,
    RecentFormPayload,
    RecentMatchDetail,
    Tour,
)

# matchstat surface label normalisation. The API returns "I.hard" for
# indoor hard (matching Sackmann's "IHard" after our existing pipeline
# normalises it). Mapping kept tiny so a future court value is easy to
# spot — anything not here is passed through unchanged.
_MATCHSTAT_SURFACE_MAP = {
    "I.hard": "IHard",
    "i.hard": "IHard",
    "I.Hard": "IHard",
}


def _normalise_surface(name: str | None) -> str | None:
    if name is None:
        return None
    return _MATCHSTAT_SURFACE_MAP.get(name, name)


def _tour_to_matchstat_code(tour: Tour) -> TourCode:
    """Convert the LLM-facing `"ATP"|"WTA"` to the matchstat path code
    `"atp"|"wta"`. Narrow `str.lower()` (LiteralString) into the
    expected Literal so pyright is happy."""
    if tour == "ATP":
        return "atp"
    return "wta"


def _parse_odds(raw: str | None) -> float | None:
    """matchstat returns odds as strings. Bad values silently become None
    rather than raising — odds are a display nicety, not load-bearing."""
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 1.0:
        return None
    return value


def _matchstat_id_from_canonical(
    conn: duckdb.DuckDBPyConnection,
    tour: Tour,
    canonical_player_id: str,
) -> int | None:
    """Find the matchstat external player ID for a canonical player_id
    by querying `scheduled_matches` rows that already carry both sides
    of the mapping.

    Returns None when the player has never appeared in a scheduled
    fixture we've ingested. The caller then falls back to Sackmann.

    `scheduled_matches.tour` is stored in uppercase ("ATP"/"WTA") by
    `refresh_hot.py`, so we query in that case directly.
    """
    row = conn.execute(
        """
        SELECT external_id FROM (
            SELECT player1_external_id AS external_id
            FROM scheduled_matches
            WHERE tour = ? AND player1_canonical_id = ?
            UNION
            SELECT player2_external_id AS external_id
            FROM scheduled_matches
            WHERE tour = ? AND player2_canonical_id = ?
        )
        LIMIT 1
        """,
        [tour, canonical_player_id, tour, canonical_player_id],
    ).fetchone()
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Recent form
# ---------------------------------------------------------------------------


def _rich_match_to_recent(
    match: RichMatch,
    *,
    queried_external_id: int,
    opponent_name_fallback: str,
) -> RecentMatchDetail:
    """Map one matchstat RichMatch row to the view-layer detail type
    from the perspective of the queried player.

    Winner side comes from `winner_index` which falls back to the
    score string when matchstat omits `matchWinner` (Phase 6.2 fix —
    `matchWinner: null` is common on older matches)."""
    winner_side = winner_index(match.match_winner, match.result)
    if match.player1_id == queried_external_id:
        opponent_name = match.player2.name
        odds_self = _parse_odds(match.odd1)
        odds_opp = _parse_odds(match.odd2)
        result: Literal["W", "L"] = "W" if winner_side == 1 else "L"
    else:
        opponent_name = match.player1.name
        odds_self = _parse_odds(match.odd2)
        odds_opp = _parse_odds(match.odd1)
        result = "W" if winner_side == 2 else "L"

    return RecentMatchDetail(
        match_date=(match.date.date() if match.date else date.min),
        opponent_name=opponent_name or opponent_name_fallback,
        result=result,
        surface=_normalise_surface(
            match.tournament.court.name if match.tournament and match.tournament.court else None
        ),
        tournament_name=match.tournament.name if match.tournament else None,
        round_name=match.round.name if match.round else None,
        score=match.result,
        odds_self=odds_self,
        odds_opponent=odds_opp,
        completion_status=parse_completion_status(match.result),
    )


def _sackmann_recent_form(
    conn: duckdb.DuckDBPyConnection,
    canonical_player_id: str,
    as_of_date: date,
    n: int,
) -> list[RecentMatchDetail]:
    """Cold-layer fallback for the recent-form panel. Same query shape
    as `db_tools.get_recent_form` but materialises into the richer
    detail type (with no odds — Sackmann doesn't carry pre-match odds
    in the `matches` table)."""
    rows = conn.execute(
        """
        SELECT m.tourney_date, m.surface, m.tourney_name, m.round, m.score,
               m.winner_player_id, m.loser_player_id, m.match_status
        FROM matches m
        WHERE (m.winner_player_id = ? OR m.loser_player_id = ?)
          AND m.match_status = 'completed'
          AND m.match_tier IN ('main', 'qual_chall', 'qual_itf')
          AND m.tourney_date < ?
        ORDER BY m.tourney_date DESC, m.match_num DESC
        LIMIT ?
        """,
        [canonical_player_id, canonical_player_id, as_of_date, n],
    ).fetchall()
    out: list[RecentMatchDetail] = []
    for r in rows:
        tdate, surface, tname, rnd, score, winner_id, loser_id, _ = r
        result: Literal["W", "L"] = "W" if winner_id == canonical_player_id else "L"
        opp_id = loser_id if result == "W" else winner_id
        opp_row = conn.execute(
            "SELECT full_name FROM players WHERE player_id = ?",
            [opp_id],
        ).fetchone()
        opponent_name = opp_row[0] if opp_row and opp_row[0] else opp_id
        out.append(
            RecentMatchDetail(
                match_date=tdate,
                opponent_name=opponent_name,
                result=result,
                surface=surface,
                tournament_name=tname,
                round_name=rnd,
                score=score,
                odds_self=None,
                odds_opponent=None,
                completion_status=parse_completion_status(score),
            )
        )
    return out


def fetch_recent_n_matches(
    conn: duckdb.DuckDBPyConnection,
    tour: Tour,
    canonical_player_id: str,
    player_name: str,
    as_of_date: date,
    *,
    n: int = 8,
    now: datetime | None = None,
) -> RecentFormPayload:
    """View-layer "last N matches" lookup with matchstat → Sackmann
    fallback.

    Decision tree:
    1. If we have a matchstat external ID AND quota is not exhausted,
       fetch via matchstat (24h cached) and return matchstat rows.
    2. Otherwise (no external ID known, OR quota exhausted, OR matchstat
       errored): query Sackmann `matches` directly.
    """
    moment = now or datetime.now(UTC)
    external_id = _matchstat_id_from_canonical(conn, tour, canonical_player_id)
    used, cap = quota_status(conn, now=moment)

    if external_id is not None:
        try:
            page: RichMatchesPage = fetch_player_past_matches(
                conn,
                _tour_to_matchstat_code(tour),
                external_id,
                now=moment,
                page_size=max(n, 10),
            )
            rows = [
                _rich_match_to_recent(
                    m,
                    queried_external_id=external_id,
                    opponent_name_fallback="opponent",
                )
                for m in page.data[:n]
            ]
            # Only return matchstat-sourced when it had data. An empty
            # payload for a known-active player is a sign matchstat's
            # backend doesn't have the player populated (same pattern
            # as the H2H empty-payload case). Falling through to
            # Sackmann surfaces a real list instead of a misleading
            # "no completed matches" caption.
            if rows:
                used, cap = quota_status(conn, now=moment)
                return RecentFormPayload(
                    player_id=canonical_player_id,
                    player_name=player_name,
                    tour=tour,
                    as_of_date=as_of_date,
                    matches=rows,
                    data_source="matchstat",
                    fetched_at=moment.replace(tzinfo=None),
                    matchstat_quota_used=used,
                    matchstat_quota_cap=cap,
                )
        except MatchstatBudgetExceeded:
            # fall through to Sackmann
            pass

    fallback_rows = _sackmann_recent_form(conn, canonical_player_id, as_of_date, n)
    return RecentFormPayload(
        player_id=canonical_player_id,
        player_name=player_name,
        tour=tour,
        as_of_date=as_of_date,
        matches=fallback_rows,
        data_source="sackmann",
        fetched_at=moment.replace(tzinfo=None),
        matchstat_quota_used=used,
        matchstat_quota_cap=cap,
    )


# ---------------------------------------------------------------------------
# H2H
# ---------------------------------------------------------------------------


def _rich_h2h_match_to_detail(
    match: RichMatch,
    *,
    player_a_external_id: int,
    player_a_canonical_id: str,
    player_b_canonical_id: str,
    player_a_name: str,
    player_b_name: str,
) -> H2HMatchDetail:
    """Build one H2HMatchDetail row from a matchstat RichMatch + the
    pre-known a/b mapping. Winner side comes from `winner_index` (the
    Phase 6.2 fallback resolver) which uses matchstat's `matchWinner`
    when set and falls back to parsing the score string. Necessary
    because matchstat's H2H endpoint frequently returns
    `matchWinner: null` for older matches (observed on the 2016
    Barcelona Open Q3 Khachanov-Trungelliti row, among others)."""
    winner_side = winner_index(match.match_winner, match.result)
    if match.player1_id == player_a_external_id:
        winner_a = winner_side == 1
        odds_winner_raw = match.odd1 if winner_a else match.odd2
        odds_loser_raw = match.odd2 if winner_a else match.odd1
    else:
        winner_a = winner_side == 2
        odds_winner_raw = match.odd2 if winner_a else match.odd1
        odds_loser_raw = match.odd1 if winner_a else match.odd2

    winner_canonical_id = player_a_canonical_id if winner_a else player_b_canonical_id
    winner_name = player_a_name if winner_a else player_b_name

    return H2HMatchDetail(
        match_date=(match.date.date() if match.date else date.min),
        tournament_name=match.tournament.name if match.tournament else None,
        round_name=match.round.name if match.round else None,
        surface=_normalise_surface(
            match.tournament.court.name if match.tournament and match.tournament.court else None
        ),
        winner_player_id=winner_canonical_id,
        winner_name=winner_name,
        score=match.result,
        odds_winner=_parse_odds(odds_winner_raw),
        odds_loser=_parse_odds(odds_loser_raw),
        completion_status=parse_completion_status(match.result),
    )


def _summarise_by_surface(
    matches: list[H2HMatchDetail],
    player_a_canonical_id: str,
) -> dict[str, tuple[int, int]]:
    """Return {'Clay': (a_wins, b_wins), ...} skipping rows with no
    surface info or no winner identification."""
    counts: dict[str, list[int]] = {}
    for m in matches:
        if not m.surface or not m.winner_player_id:
            continue
        bucket = counts.setdefault(m.surface, [0, 0])
        if m.winner_player_id == player_a_canonical_id:
            bucket[0] += 1
        else:
            bucket[1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def _sackmann_h2h(
    conn: duckdb.DuckDBPyConnection,
    player_a_id: str,
    player_b_id: str,
    player_a_name: str,
    player_b_name: str,
    as_of_date: date,
) -> list[H2HMatchDetail]:
    """Cold-layer H2H fallback: query `matches` for every meeting up to
    `as_of_date`. Newest first, no odds (Sackmann doesn't carry them on
    the `matches` table)."""
    rows = conn.execute(
        """
        SELECT tourney_date, surface, tourney_name, round, score,
               winner_player_id, loser_player_id
        FROM matches
        WHERE ((winner_player_id = ? AND loser_player_id = ?)
            OR (winner_player_id = ? AND loser_player_id = ?))
          AND match_status = 'completed'
          AND match_tier IN ('main', 'qual_chall', 'qual_itf')
          AND tourney_date < ?
        ORDER BY tourney_date DESC, match_num DESC
        """,
        [player_a_id, player_b_id, player_b_id, player_a_id, as_of_date],
    ).fetchall()
    out: list[H2HMatchDetail] = []
    for tdate, surface, tname, rnd, score, winner_id, _ in rows:
        winner_a = winner_id == player_a_id
        out.append(
            H2HMatchDetail(
                match_date=tdate,
                tournament_name=tname,
                round_name=rnd,
                surface=surface,
                winner_player_id=player_a_id if winner_a else player_b_id,
                winner_name=player_a_name if winner_a else player_b_name,
                score=score,
                odds_winner=None,
                odds_loser=None,
                completion_status=parse_completion_status(score),
            )
        )
    return out


def fetch_h2h_summary(
    conn: duckdb.DuckDBPyConnection,
    tour: Tour,
    player_a_canonical_id: str,
    player_b_canonical_id: str,
    player_a_name: str,
    player_b_name: str,
    as_of_date: date,
    *,
    now: datetime | None = None,
) -> H2HSummary:
    """H2H lookup with the same matchstat-first / Sackmann-fallback
    decision tree as `fetch_recent_n_matches`. Returns the full
    `H2HSummary` view-layer & LLM-tool callers consume."""
    moment = now or datetime.now(UTC)
    a_external = _matchstat_id_from_canonical(conn, tour, player_a_canonical_id)
    b_external = _matchstat_id_from_canonical(conn, tour, player_b_canonical_id)

    used_matchstat = False
    matches: list[H2HMatchDetail] = []

    if a_external is not None and b_external is not None:
        try:
            page = fetch_h2h(
                conn,
                _tour_to_matchstat_code(tour),
                a_external,
                b_external,
                now=moment,
            )
            # Defensive filter (Phase 6.2): matchstat's `/h2h/matches/`
            # endpoint should only return completed history, but we drop
            # any row whose `result_type` is set to anything other than
            # "completed" (e.g. an upcoming-fixture leak) so a future
            # schema drift can't re-introduce the Svitolina-Bondar
            # "score unknown" bug. Rows with `result_type` absent are
            # kept — matchstat omits the field on older completed rows.
            h2h_rows = [m for m in page.data if m.result_type in (None, "completed")]
            matches = [
                _rich_h2h_match_to_detail(
                    m,
                    player_a_external_id=a_external,
                    player_a_canonical_id=player_a_canonical_id,
                    player_b_canonical_id=player_b_canonical_id,
                    player_a_name=player_a_name,
                    player_b_name=player_b_name,
                )
                for m in h2h_rows
            ]
            # Only flag the row as matchstat-sourced when matchstat
            # actually had something to return. An empty payload from
            # the live API is suspicious — Sinner-Djokovic on tour
            # have >5 meetings, but matchstat's H2H endpoint occasionally
            # returns `data: []` for legitimate famous matchups (their
            # backend isn't populated for every pair). Falling through
            # to Sackmann avoids advertising "never met" with false
            # confidence; the cold layer has every recorded meeting.
            used_matchstat = len(matches) > 0
        except MatchstatBudgetExceeded:
            pass

    if not used_matchstat:
        matches = _sackmann_h2h(
            conn,
            player_a_canonical_id,
            player_b_canonical_id,
            player_a_name,
            player_b_name,
            as_of_date,
        )

    a_wins = sum(1 for m in matches if m.winner_player_id == player_a_canonical_id)
    b_wins = sum(1 for m in matches if m.winner_player_id == player_b_canonical_id)
    by_surface = _summarise_by_surface(matches, player_a_canonical_id)

    return H2HSummary(
        player_a_name=player_a_name,
        player_b_name=player_b_name,
        player_a_id=player_a_canonical_id,
        player_b_id=player_b_canonical_id,
        tour=tour,
        player_a_wins=a_wins,
        player_b_wins=b_wins,
        by_surface=by_surface,
        matches=matches,
        data_source="matchstat" if used_matchstat else "sackmann",
        fetched_at=moment.replace(tzinfo=None),
    )


__all__ = [
    "fetch_h2h_summary",
    "fetch_recent_n_matches",
]
