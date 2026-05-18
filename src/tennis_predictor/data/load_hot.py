"""Transform matchstat Pydantic models into DuckDB row inserts (Phase 2).

The boundary between the typed API responses (`matchstat.py`) and the
schema (`schema.py`). Intentionally thin:

- Pure SQL writes; no HTTP, no resolver logic.
- Player canonical resolution is the caller's responsibility (passed as
  a `PlayerResolver` callable). This lets the orchestrator wire either a
  full `AliasIndex` with manual-review writeback, or a stub for tests.
- All inserts are idempotent via `ON CONFLICT DO NOTHING` on the natural
  unique key, so daily refresh is safe to re-run.

Each function returns `LoadCounts` so the orchestrator can sum them and
write the totals into `ingestion_runs`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import duckdb

from tennis_predictor.data.matchstat import Fixture, Match, RankingEntry

SOURCE_MATCHSTAT = "matchstat"
ODDS_SOURCE_MATCHSTAT = "matchstat"

PlayerResolver = Callable[[str, str], str | None]
"""(raw_player_name, tour) -> canonical_player_id or None.

Returning None means "leave canonical_id NULL" — acceptable for a
scheduled_matches row, but blocks a `matches` row from being inserted
(since matches.winner_player_id is NOT NULL)."""


@dataclass(frozen=True)
class LoadCounts:
    added: int = 0
    skipped: int = 0
    failed: int = 0

    def __add__(self, other: LoadCounts) -> LoadCounts:
        return LoadCounts(
            added=self.added + other.added,
            skipped=self.skipped + other.skipped,
            failed=self.failed + other.failed,
        )


def _normalize_overround(odds_w: float, odds_l: float) -> tuple[float, float]:
    """Decimal odds → implied probabilities, normalized to sum to 1.

    Duplicated from `load_market._normalize_overround` to avoid a Phase-1
    file edit; both are the same 4-line formula. If a third caller appears
    these should be merged into a shared `odds.py`.
    """
    p_w_raw = 1.0 / odds_w
    p_l_raw = 1.0 / odds_l
    total = p_w_raw + p_l_raw
    return p_w_raw / total, p_l_raw / total


# matchstat `court.name` -> our canonical `surface` field (Sackmann convention).
# Observed in probe: "Clay", "I.hard". Likely also "Hard", "Grass", "Carpet".
SURFACE_MAP: dict[str, str] = {
    "Clay": "Clay",
    "Hard": "Hard",
    "I.hard": "Hard",  # indoor hard collapses to "Hard" for the predictor
    "Grass": "Grass",
    "Carpet": "Carpet",
}


def _map_surface(raw: str | None) -> str | None:
    if raw is None:
        return None
    return SURFACE_MAP.get(raw, raw)  # pass through unknowns rather than dropping


def _parse_odd(raw: str | None) -> float | None:
    """matchstat returns odds as strings (e.g. "1.38"). Some are None."""
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 1.0 else None  # decimal odds must be > 1


# ---------------------------------------------------------------------------
# scheduled_matches


def insert_scheduled_matches(
    conn: duckdb.DuckDBPyConnection,
    fixtures: Iterable[Fixture],
    *,
    tour: str,
    resolve_player: PlayerResolver,
    tournament_tier_by_id: dict[int, str | None] | None = None,
    now: datetime | None = None,
) -> LoadCounts:
    """Insert upcoming fixtures into `scheduled_matches`.

    `tournament_tier_by_id` is the orchestrator's calendar cache: matchstat's
    fixture payload doesn't carry the human-readable tier ("ATP 250" etc.)
    even with `include=tournament.rank` — that field lives on the calendar
    payload. The caller passes it through so we can populate
    `tournament_tier` here.
    """
    now = now or datetime.now(UTC)
    tier_lookup = tournament_tier_by_id or {}
    counts = LoadCounts()
    for fx in fixtures:
        scheduled_match_id = f"{SOURCE_MATCHSTAT}::{fx.id}"
        surface = _map_surface(
            fx.tournament.court.name if fx.tournament and fx.tournament.court else None
        )
        tournament_name = fx.tournament.name if fx.tournament else None
        tournament_country_acr = fx.tournament.country_acr if fx.tournament else None
        tournament_tier = tier_lookup.get(fx.tournament_id)
        round_name = fx.round.name if fx.round else None

        p1_canonical = resolve_player(fx.player1.name, tour)
        p2_canonical = resolve_player(fx.player2.name, tour)

        rowcount_before = _scheduled_count(conn)
        conn.execute(
            """
            INSERT INTO scheduled_matches (
                scheduled_match_id, source, fixture_external_id,
                tour, tournament_external_id, tournament_name, tournament_tier,
                tournament_country_acr, surface, round_external_id, round_name,
                player1_external_id, player2_external_id,
                player1_canonical_id, player2_canonical_id,
                player1_name, player2_name,
                player1_country_acr, player2_country_acr,
                player1_seed, player2_seed,
                scheduled_start_utc, ingested_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            ) ON CONFLICT (scheduled_match_id) DO NOTHING
            """,
            [
                scheduled_match_id,
                SOURCE_MATCHSTAT,
                str(fx.id),
                tour,
                str(fx.tournament_id),
                tournament_name,
                tournament_tier,
                tournament_country_acr,
                surface,
                str(fx.round_id) if fx.round_id is not None else None,
                round_name,
                str(fx.player1_id),
                str(fx.player2_id),
                p1_canonical,
                p2_canonical,
                fx.player1.name,
                fx.player2.name,
                fx.player1.country_acr,
                fx.player2.country_acr,
                fx.seed1,
                fx.seed2,
                fx.date,
                now,
            ],
        )
        if _scheduled_count(conn) > rowcount_before:
            counts += LoadCounts(added=1)
        else:
            counts += LoadCounts(skipped=1)
    return counts


def _scheduled_count(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM scheduled_matches").fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# matches (completed)


def insert_completed_matches(
    conn: duckdb.DuckDBPyConnection,
    matches: Iterable[Match],
    *,
    tour: str,
    tournament_name: str | None,
    tournament_tier: str | None,
    surface: str | None,
    tourney_date: date,
    resolve_player: PlayerResolver,
    match_tier: str = "main",
) -> LoadCounts:
    """Insert completed matches into `matches`.

    Refuses to insert a row if either player can't be resolved to a canonical
    ID (matches.winner_player_id / loser_player_id are NOT NULL). Counts those
    rows as `failed` — the orchestrator decides whether to write them to the
    aliases-review CSV for later resolution.

    `match_tier` is the predictor's coarse main/qualifying distinction — pass
    "main" for the `singles` array from tournament/results, "qualifying" for
    the `qualifying` array.
    """
    counts = LoadCounts()
    for m in matches:
        if m.match_winner is None:
            counts += LoadCounts(skipped=1)
            continue

        p1_canonical = resolve_player(m.player1.name, tour)
        p2_canonical = resolve_player(m.player2.name, tour)
        if p1_canonical is None or p2_canonical is None:
            counts += LoadCounts(failed=1)
            continue

        if m.match_winner == m.player1_id:
            winner_canonical, loser_canonical = p1_canonical, p2_canonical
        elif m.match_winner == m.player2_id:
            winner_canonical, loser_canonical = p2_canonical, p1_canonical
        else:
            counts += LoadCounts(failed=1)
            continue

        match_id = f"{SOURCE_MATCHSTAT}::{m.id}"
        match_status = "completed" if m.result else "unknown"
        match_num = _safe_int(m.id) or 0

        rowcount_before = _matches_count(conn)
        conn.execute(
            """
            INSERT INTO matches (
                match_id, source, match_external_id, tour, match_tier,
                tourney_id, tourney_name, tourney_level, tourney_date, surface,
                match_num, round, best_of, score, match_status,
                winner_player_id, loser_player_id
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            ) ON CONFLICT (match_id) DO NOTHING
            """,
            [
                match_id,
                SOURCE_MATCHSTAT,
                m.id,
                tour,
                match_tier,
                str(m.tournament_id),
                tournament_name,
                tournament_tier,
                tourney_date,
                surface,
                match_num,
                str(m.round_id) if m.round_id is not None else None,
                m.best_of,
                m.result,
                match_status,
                winner_canonical,
                loser_canonical,
            ],
        )
        if _matches_count(conn) > rowcount_before:
            counts += LoadCounts(added=1)
        else:
            counts += LoadCounts(skipped=1)
    return counts


def _matches_count(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM matches").fetchone()
    return int(row[0]) if row else 0


def _safe_int(s: str) -> int | None:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# market_implied_probabilities (pre-match odds bonus from /tournament/results)


def insert_market_odds_from_matches(
    conn: duckdb.DuckDBPyConnection,
    matches: Iterable[Match],
) -> LoadCounts:
    """Write pre-match odds from `Match.odd1`/`odd2` into market_implied_probabilities.

    Skips matches with missing odds, missing winner, or invalid odds (≤ 1.0).
    Uses the same overround-normalization as the cold tennis-data.co.uk
    loader. The companion `matches` row is expected to already exist (we
    use the same composite `match_id`); FK is not enforced by DuckDB but
    by convention the orchestrator calls `insert_completed_matches` first.
    """
    counts = LoadCounts()
    for m in matches:
        odd1 = _parse_odd(m.odd1)
        odd2 = _parse_odd(m.odd2)
        if odd1 is None or odd2 is None or m.match_winner is None:
            counts += LoadCounts(skipped=1)
            continue

        if m.match_winner == m.player1_id:
            odds_w, odds_l = odd1, odd2
        elif m.match_winner == m.player2_id:
            odds_w, odds_l = odd2, odd1
        else:
            counts += LoadCounts(failed=1)
            continue

        p_w, p_l = _normalize_overround(odds_w, odds_l)
        match_id = f"{SOURCE_MATCHSTAT}::{m.id}"

        rowcount_before = _market_count(conn)
        conn.execute(
            """
            INSERT INTO market_implied_probabilities (
                match_id, odds_source,
                odds_winner_close, odds_loser_close,
                p_winner_close, p_loser_close
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (match_id, odds_source) DO NOTHING
            """,
            [match_id, ODDS_SOURCE_MATCHSTAT, odds_w, odds_l, p_w, p_l],
        )
        if _market_count(conn) > rowcount_before:
            counts += LoadCounts(added=1)
        else:
            counts += LoadCounts(skipped=1)
    return counts


def _market_count(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM market_implied_probabilities").fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# rankings overlay (inter-week refresh from hot source)


def upsert_ranking_overlay(
    conn: duckdb.DuckDBPyConnection,
    entries: Iterable[RankingEntry],
    *,
    tour: str,
    resolve_player: PlayerResolver,
    as_of_date: date | None = None,
) -> LoadCounts:
    """Write current-day rankings as an overlay on top of weekly Sackmann snapshots.

    Same (ranking_date, player_id) row may already exist from an earlier run
    on the same day; we re-INSERT under ON CONFLICT DO NOTHING so first
    write wins. (If matchstat changes the number intra-day we keep the
    earlier value, which is fine — daily granularity is the contract.)
    """
    counts = LoadCounts()
    for entry in entries:
        ranking_date = (
            (entry.date or datetime.now(UTC)).date() if as_of_date is None else as_of_date
        )
        canonical = resolve_player(entry.player.name, tour)
        if canonical is None:
            counts += LoadCounts(failed=1)
            continue

        points = entry.point if entry.point is not None else entry.player.points

        rowcount_before = _rankings_count(conn)
        conn.execute(
            """
            INSERT INTO rankings (ranking_date, player_id, rank, points)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (ranking_date, player_id) DO NOTHING
            """,
            [ranking_date, canonical, entry.position, points],
        )
        if _rankings_count(conn) > rowcount_before:
            counts += LoadCounts(added=1)
        else:
            counts += LoadCounts(skipped=1)
    return counts


def _rankings_count(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM rankings").fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# scheduled_matches -> matches promotion


def promote_completed_fixtures(conn: duckdb.DuckDBPyConnection) -> int:
    """Remove scheduled_matches rows that already have a corresponding matches row.

    Linkage uses the composite key documented in the data-ingestion skill:
    `(tournament_external_id, player1_external_id, player2_external_id, round_external_id)`.
    A fixture matches if such a row exists in `matches` for the SAME source
    (i.e., both came from matchstat — not cross-source). Player ordering can
    flip between fixture and result depending on who's the winner, so we
    match player IDs as an unordered set.

    Returns the number of fixtures removed.
    """
    rowcount_before = _scheduled_count(conn)
    conn.execute(
        """
        DELETE FROM scheduled_matches sm
        WHERE EXISTS (
            SELECT 1
            FROM matches m
            WHERE m.source = sm.source
              AND m.tourney_id = sm.tournament_external_id
              AND COALESCE(m.round, '') = COALESCE(sm.round_external_id, '')
              AND (
                  (m.winner_player_id = sm.player1_canonical_id
                   AND m.loser_player_id = sm.player2_canonical_id)
               OR (m.winner_player_id = sm.player2_canonical_id
                   AND m.loser_player_id = sm.player1_canonical_id)
              )
        )
        """
    )
    return rowcount_before - _scheduled_count(conn)
