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

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import duckdb

from tennis_predictor.data.matchstat import (
    Fixture,
    Match,
    RankingEntry,
    fixture_on_court_time,
)

SOURCE_MATCHSTAT = "matchstat"
ODDS_SOURCE_MATCHSTAT = "matchstat"

# matchstat returns ISO-8601 with a `Z` suffix that is empirically NOT
# real UTC: user-confirmed 2026-05-24 that fixtures appearing as e.g.
# `2026-05-25T12:00:00.000Z` are actually 09:00 real UTC (a +3 hour
# shift). The most plausible cause is matchstat's backend storing times
# in Moscow time (UTC+3) and mislabeling them as `Z`. To recover real
# UTC we re-interpret the labelled `Z` as `MATCHSTAT_SOURCE_TZ` and
# convert.
#
# Phase 6.1 default change: `Europe/Moscow` (was `UTC` in Phase 6).
# The +3h shift is empirically consistent across non-Moscow tournaments
# (Roland Garros, etc.), so we default to undoing it for everyone.
# Override via env var if matchstat fixes their labelling. Read inside
# the helper on each call so monkeypatch.setenv in tests (and a manual
# export at runtime) takes effect without a module reload.
_MATCHSTAT_SOURCE_TZ_DEFAULT: str = "Europe/Moscow"


def _matchstat_source_tz() -> ZoneInfo:
    name = os.environ.get("MATCHSTAT_SOURCE_TZ", _MATCHSTAT_SOURCE_TZ_DEFAULT)
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        # Bad env value -> silently fall back to UTC rather than crash
        # the refresh script. The fix will be obvious in the displayed
        # times still being wrong; better than a refresh that errors.
        return ZoneInfo("UTC")


def _to_naive_utc(dt: datetime | None) -> datetime | None:
    """Normalize a matchstat-supplied datetime to naive real-UTC.

    Two distinct corrections happen here:

    1. **DuckDB tz-aware → naive shift bug:** `TIMESTAMP` is naive. Passing
       a tz-aware datetime triggers a silent conversion to the host's
       local time before storage (Estonia EEST = UTC+3 → +3h shift).
       Solved by stripping tzinfo after explicit conversion.
    2. **matchstat `Z`-but-not-UTC bug:** if `MATCHSTAT_SOURCE_TZ` is set
       to something other than UTC, we reinterpret the incoming naive
       wall-clock as being in that source TZ and convert to real UTC.
       Use `MATCHSTAT_SOURCE_TZ=Europe/Moscow` to undo the +3 shift.
    """
    if dt is None:
        return None
    source_tz = _matchstat_source_tz()
    # Strip the (misleading) tzinfo to get the raw wall-clock matchstat
    # actually sent, then re-tag with the source TZ we believe it to be.
    wall = dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
    localised = wall.replace(tzinfo=source_tz)
    return localised.astimezone(UTC).replace(tzinfo=None)


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
        # Drop doubles teams — matchstat's `filter=PlayerGroup:singles` query
        # param doesn't actually exclude them on the fixtures endpoint; they
        # leak through with composite "Player1/Player2" names. Skipped here
        # so the resolver isn't bothered with them and the review buffer
        # stays focused on legitimate singles ambiguity.
        if "/" in fx.player1.name or "/" in fx.player2.name:
            counts += LoadCounts(skipped=1)
            continue
        scheduled_match_id = f"{SOURCE_MATCHSTAT}::{fx.id}"
        surface = _map_surface(
            fx.tournament.court.name if fx.tournament and fx.tournament.court else None
        )
        tournament_name = fx.tournament.name if fx.tournament else None
        tournament_country_acr = fx.tournament.country_acr if fx.tournament else None
        # Phase 6.2: prefer the calendar's precise tier ("ATP 250" etc.)
        # when available, but fall back to the fixture's `tournament.rank.name`
        # ("Grand Slam", "Main tour"). matchstat's calendar endpoint is
        # forward-only and silently drops currently-active tournaments
        # (Phase 2 known issue), which leaves Roland Garros and other
        # in-progress events with `tournament_tier=NULL` if we only
        # consult the calendar. The fallback gives the Slam-prune logic
        # the signal it needs without a per-tournament workaround.
        tournament_tier = tier_lookup.get(fx.tournament_id)
        if tournament_tier is None and fx.tournament and fx.tournament.rank:
            tournament_tier = fx.tournament.rank.name
        round_name = fx.round.name if fx.round else None

        p1_canonical = resolve_player(fx.player1.name, tour)
        p2_canonical = resolve_player(fx.player2.name, tour)

        rowcount_before = _scheduled_count(conn)
        # ON CONFLICT DO UPDATE refreshes the mutable fields so a re-ingest
        # corrects rows written before tz-normalization landed. We still
        # count an UPDATE as "skipped" (row already existed) so the
        # ingestion_runs row reflects new inserts only.
        # Phase 6.2: prefer `timeGame` over `date` when matchstat sets
        # it (carries an alternate on-court time when present).
        #
        # `time_confirmed` is kept in the schema for back-compat but
        # always stored as True after Phase 6.2 follow-up — matchstat's
        # `T12:00:00Z` is ambiguous (it represents both a genuine 11:00
        # CEST morning-wave start AND a day-level placeholder), and the
        # two cases are indistinguishable in the wire format. The user
        # decision was to always display matchstat's time as an
        # estimated start (Slam start times are inherently approximate
        # — rain delays, previous-match overruns — so showing the
        # estimate is more useful than hiding it behind "time TBD").
        on_court_time = fixture_on_court_time(fx.date, fx.time_game)
        time_confirmed = on_court_time is not None
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
                scheduled_start_utc, time_confirmed, ingested_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            ) ON CONFLICT (scheduled_match_id) DO UPDATE SET
                tournament_name        = excluded.tournament_name,
                tournament_tier        = excluded.tournament_tier,
                tournament_country_acr = excluded.tournament_country_acr,
                surface                = excluded.surface,
                round_external_id      = excluded.round_external_id,
                round_name             = excluded.round_name,
                -- Phase 6.2: matchstat re-uses fixture_external_id for
                -- different matchups over time (observed live: Popyrin
                -- -Svajda took over fx_id=1271 from Griekspoor-Arnaldi
                -- at Roland Garros 2026). Without refreshing player
                -- identity on conflict the row keeps the stale matchup
                -- and the new matchup is silently lost. Treat the
                -- latest payload as authoritative for every column.
                player1_external_id    = excluded.player1_external_id,
                player2_external_id    = excluded.player2_external_id,
                player1_canonical_id   = excluded.player1_canonical_id,
                player2_canonical_id   = excluded.player2_canonical_id,
                player1_name           = excluded.player1_name,
                player2_name           = excluded.player2_name,
                player1_country_acr    = excluded.player1_country_acr,
                player2_country_acr    = excluded.player2_country_acr,
                player1_seed           = excluded.player1_seed,
                player2_seed           = excluded.player2_seed,
                scheduled_start_utc    = excluded.scheduled_start_utc,
                time_confirmed         = excluded.time_confirmed,
                ingested_at            = excluded.ingested_at
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
                _to_naive_utc(on_court_time),
                time_confirmed,
                _to_naive_utc(now),
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
