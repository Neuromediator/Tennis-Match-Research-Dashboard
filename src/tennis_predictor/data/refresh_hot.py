"""Daily hot-data refresh orchestrator (Phase 2).

Composes the API client, the resolver, and the load layer into one
end-to-end run. Per the Phase 2 close-out decision (Path C, documented
in `docs/tutorials/phase_2.md` and `docs/phases.md`), the orchestrator
pulls only **fixtures** and **rankings** from matchstat; completed
matches are sourced from the cold Sackmann layer (weekly git submodule)
because matchstat's `calendar/{year}` is forward-only and silently
drops currently-active tournaments, which makes the calendar-driven
results path unreliable.

Steps:
1. Open an `ingestion_runs` row with `status='running'`.
2. For each tour:
   a. Fetch the year's calendar — used **only** to build a
      `{tournament_id: tier}` lookup so `scheduled_matches.tournament_tier`
      can be populated. tournament/results is NOT called.
   b. Fetch `fixtures/{today}` (and `{tomorrow}`, paginated), insert into
      `scheduled_matches`.
   c. Fetch current rankings, write overlay.
3. Run `promote_completed_fixtures()` — currently a near-no-op under
   Path C (Sackmann tourney_id won't match matchstat tournament_id), but
   kept so the path is wired if Path B ever ships.
4. Write the resolver's `review_buffer` to `aliases_review_matchstat.csv`.
5. Close the `ingestion_runs` row with `status='success' | 'partial' |
   'failed'`, totals, `requests_used`.

The `_refresh_tournament_results` helper, `insert_completed_matches`,
`insert_market_odds_from_matches`, and `MatchstatClient.tournament_results`
remain in the codebase: they're tested and harmless when not called.
A future Path B (discover seasonids from fixtures) would rewire them.

Functions in this module take an injected `client` so unit tests can pass
a fake; only `scripts/refresh_hot.py` wires the real `MatchstatClient`.
"""

from __future__ import annotations

import csv
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import duckdb

from tennis_predictor.data.load_hot import (
    SOURCE_MATCHSTAT,
    LoadCounts,
    _map_surface,
    insert_completed_matches,
    insert_market_odds_from_matches,
    insert_scheduled_matches,
    promote_completed_fixtures,
    upsert_ranking_overlay,
)
from tennis_predictor.data.matchstat import (
    TOUR_LEVEL_TIERS,
    CalendarTournament,
    FixturesPage,
    RankingEntry,
    TourCode,
    TournamentResults,
)
from tennis_predictor.data.matchstat_resolver import MatchstatResolver

ACTIVE_WINDOW_DAYS = 21
"""How long after a tournament's start date we keep polling its results.

Slams run 14 days; ATP/WTA 1000s run up to 12; everything else is ≤8.
21 days covers the Slam case plus a buffer for late-arriving results
without triggering re-fetches of long-finished events."""

REVIEW_CSV_FIELDNAMES = [
    "raw_name",
    "tour",
    "candidate_name",
    "candidate_canonical_id",
    "confidence",
    "runner_up_confidence",
    "verdict",
]
"""Columns of the matchstat review CSV.

- `raw_name`            : the matchstat name we tried to resolve
- `tour`                : 'ATP' or 'WTA'
- `candidate_name`      : best Sackmann alias the fuzzy matcher landed on
- `candidate_canonical_id`: that candidate's canonical player_id (so the
                          reviewer doesn't have to look it up). The reviewer's
                          job is verify, not produce IDs.
- `confidence`          : 0..1
- `runner_up_confidence`: 0..1
- `verdict`             : blank in the generated CSV. Reviewer writes 'y' to
                          confirm raw_name -> candidate_canonical_id;
                          anything else (blank/'n'/'no') means reject.
"""


class MatchstatClientProtocol(Protocol):
    """Subset of MatchstatClient used by the orchestrator — for fake-client tests."""

    requests_used: int

    def calendar(self, tour: TourCode, year: int) -> list[CalendarTournament]: ...
    def fixtures_for_date(
        self,
        tour: TourCode,
        match_date: date,
        *,
        singles_only: bool = ...,
        page_size: int = ...,
        page_no: int = ...,
    ) -> FixturesPage: ...
    def fixtures_for_tournament(
        self,
        tour: TourCode,
        tournament_id: int,
        *,
        singles_only: bool = ...,
        page_size: int = ...,
        page_no: int = ...,
    ) -> FixturesPage: ...
    def tournament_results(self, tour: TourCode, season_id: int) -> TournamentResults: ...
    def rankings(
        self,
        tour: TourCode,
        *,
        page_size: int = ...,
        page_no: int = ...,
    ) -> list[RankingEntry]: ...


@dataclass
class TourSummary:
    tour: str
    matches: LoadCounts = field(default_factory=LoadCounts)
    qualifying: LoadCounts = field(default_factory=LoadCounts)
    market_odds: LoadCounts = field(default_factory=LoadCounts)
    fixtures: LoadCounts = field(default_factory=LoadCounts)
    rankings: LoadCounts = field(default_factory=LoadCounts)


@dataclass
class RefreshSummary:
    """Aggregate outcome of one refresh run — returned for CLI display and tests."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    status: str  # 'success' | 'partial' | 'failed'
    per_tour: dict[str, TourSummary]
    requests_used: int
    promoted_fixtures: int
    review_candidates_written: int
    error_message: str | None = None

    @property
    def totals(self) -> LoadCounts:
        total = LoadCounts()
        for s in self.per_tour.values():
            total += s.matches + s.qualifying + s.market_odds + s.fixtures + s.rankings
        return total


def _is_recently_active(t: CalendarTournament, today: date) -> bool:
    if t.date is None:
        return False
    start = t.date.date()
    return start <= today <= start + timedelta(days=ACTIVE_WINDOW_DAYS)


def _open_run(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    tour: str | None,
    started_at: datetime,
    notes: str | None = None,
) -> str:
    run_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO ingestion_runs (run_id, source, tour, started_at, status, notes)
        VALUES (?, ?, ?, ?, 'running', ?)
        """,
        [run_id, source, tour, started_at, notes],
    )
    return run_id


def _close_run(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    finished_at: datetime,
    status: str,
    counts: LoadCounts,
    requests_used: int,
    error_message: str | None,
) -> None:
    conn.execute(
        """
        UPDATE ingestion_runs SET
            finished_at = ?, status = ?,
            rows_added = ?, rows_skipped = ?, rows_failed = ?,
            requests_used = ?, error_message = ?
        WHERE run_id = ?
        """,
        [
            finished_at,
            status,
            counts.added,
            counts.skipped,
            counts.failed,
            requests_used,
            error_message,
            run_id,
        ],
    )


def _write_review_csv(buffer_path: Path, candidates: list, append: bool = True) -> int:
    if not candidates:
        return 0
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (append and buffer_path.exists())
    mode = "a" if append else "w"
    with open(buffer_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for c in candidates:
            writer.writerow(
                {
                    "raw_name": c.raw_name,
                    "tour": c.tour,
                    "candidate_name": c.candidate_name,
                    "candidate_canonical_id": c.candidate_canonical_id,
                    "confidence": f"{c.confidence:.4f}",
                    "runner_up_confidence": f"{c.runner_up_confidence:.4f}",
                    "verdict": "",  # left blank for the reviewer
                }
            )
    return len(candidates)


def _refresh_tournament_results(
    conn: duckdb.DuckDBPyConnection,
    client: MatchstatClientProtocol,
    tour_code: TourCode,
    tour: str,
    active: list[CalendarTournament],
    resolver: MatchstatResolver,
    summary: TourSummary,
) -> None:
    for t in active:
        results = client.tournament_results(tour_code, t.id)
        if t.date is None:
            continue
        surface = _map_surface(t.court.name if t.court else None)
        common = {
            "tour": tour,
            "tournament_name": t.name,
            "tournament_tier": t.tier,
            "surface": surface,
            "tourney_date": t.date.date(),
            "resolve_player": resolver,
        }
        summary.matches += insert_completed_matches(
            conn, results.singles, **common, match_tier="main"
        )
        summary.qualifying += insert_completed_matches(
            conn, results.qualifying, **common, match_tier="qualifying"
        )
        summary.market_odds += insert_market_odds_from_matches(conn, results.singles)


def _refresh_fixtures_for_date(
    conn: duckdb.DuckDBPyConnection,
    client: MatchstatClientProtocol,
    tour_code: TourCode,
    tour: str,
    match_date: date,
    tier_by_tournament_id: dict[int, str | None],
    resolver: MatchstatResolver,
    summary: TourSummary,
    seen_fixture_ids: set[str],
    max_pages: int = 10,
) -> None:
    page_no = 1
    while page_no <= max_pages:
        page = client.fixtures_for_date(tour_code, match_date, page_no=page_no)
        for fx in page.data:
            seen_fixture_ids.add(f"{SOURCE_MATCHSTAT}::{fx.id}")
        summary.fixtures += insert_scheduled_matches(
            conn,
            page.data,
            tour=tour,
            resolve_player=resolver,
            tournament_tier_by_id=tier_by_tournament_id,
        )
        if not page.has_next_page:
            break
        page_no += 1


# Tour-level tournament-rank tags as matchstat returns them on the
# fixture payload's `tournament.rank.name`. Used by the discovery step
# to filter out Challengers, ITFs, etc. without needing the calendar
# (which is forward-only and silently drops active tournaments).
_TOUR_LEVEL_RANK_NAMES: frozenset[str] = frozenset({"Grand Slam", "Main tour"})


def _discover_active_tournament_ids(
    client: MatchstatClientProtocol,
    tour_code: TourCode,
    today: date,
    *,
    lookahead_days: int = 1,
) -> set[int]:
    """Return the set of tour-level tournament IDs that matchstat
    currently lists fixtures for.

    Walks `/fixtures/{today}` (and an optional 1-day lookahead) once
    per tour, collects distinct `tournament_id` values whose
    `tournament.rank.name` is in `_TOUR_LEVEL_RANK_NAMES`. Cheap
    (1-2 API calls per tour) and authoritative: matchstat lists a
    tournament under `/fixtures/{date}` if and only if there's at
    least one fixture scheduled for that date — which is exactly when
    we want to pull its full draw.

    Lookahead = 1 covers the case where today has no main-tour matches
    yet (e.g., before Day 1 of a Slam) but tomorrow does."""
    active: set[int] = set()
    for offset in range(lookahead_days + 1):
        d = today + timedelta(days=offset)
        page_no = 1
        while True:
            page = client.fixtures_for_date(tour_code, d, page_no=page_no)
            for fx in page.data:
                if fx.tournament is None:
                    continue
                rank_name = fx.tournament.rank.name if fx.tournament.rank else None
                if rank_name in _TOUR_LEVEL_RANK_NAMES:
                    active.add(fx.tournament_id)
            if not page.has_next_page:
                break
            page_no += 1
    return active


def _refresh_fixtures_for_tournament(
    conn: duckdb.DuckDBPyConnection,
    client: MatchstatClientProtocol,
    tour_code: TourCode,
    tour: str,
    tournament_id: int,
    tier_by_tournament_id: dict[int, str | None],
    resolver: MatchstatResolver,
    summary: TourSummary,
    seen_fixture_ids: set[str],
    max_pages: int = 5,
) -> None:
    """Pull every fixture for one tournament — all rounds, all dates —
    via `/fixtures/tournament/{id}`. One credit per call (page size
    200 covers a 128-player Slam draw plus all later rounds in a
    single page; pagination kept for safety)."""
    page_no = 1
    while page_no <= max_pages:
        page = client.fixtures_for_tournament(tour_code, tournament_id, page_no=page_no)
        for fx in page.data:
            seen_fixture_ids.add(f"{SOURCE_MATCHSTAT}::{fx.id}")
        summary.fixtures += insert_scheduled_matches(
            conn,
            page.data,
            tour=tour,
            resolve_player=resolver,
            tournament_tier_by_id=tier_by_tournament_id,
        )
        if not page.has_next_page:
            break
        page_no += 1


def _prune_stale_scheduled_matches(
    conn: duckdb.DuckDBPyConnection,
    *,
    tours: list[str],
    seen_fixture_ids: set[str],
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> int:
    """Delete `scheduled_matches` rows whose fixture ID was NOT returned by
    matchstat during this refresh.

    matchstat's `/fixtures/{date}` endpoint sometimes leaks completed
    matches into the future-fixtures payload — we observed this on the
    Roland Garros 2026 R1 (e.g., Khachanov-Gea, completed 2026-05-24,
    re-returned as an upcoming fixture for 2026-05-26 on a refresh the
    next morning). matchstat re-cleans the leak hours later; without
    this prune the leaked row lives forever in our DB and shows up as
    a phantom "upcoming match".

    The prune is intentionally scoped:
    - **Only the tours actually queried this run.** A `--tours ATP`
      refresh must not wipe WTA rows.
    - **Only rows whose `scheduled_start_utc` falls in the date window
      we actually queried,** with a ±1d buffer to cover Moscow→UTC
      shifts. Fixtures outside the window stay untouched — they're
      either far-future placeholders or already-old debris a future
      run will handle on its own pass.
    - **No-op if `seen_fixture_ids` is empty.** A run that fetched
      zero fixtures (matchstat outage, key rotation) must not delete
      every existing row.
    """
    if not seen_fixture_ids:
        return 0
    placeholders = ",".join("?" for _ in tours)
    stale_rows = conn.execute(
        f"""
        SELECT scheduled_match_id FROM scheduled_matches
        WHERE source = ?
          AND tour IN ({placeholders})
          AND scheduled_start_utc IS NOT NULL
          AND scheduled_start_utc >= ?
          AND scheduled_start_utc < ?
        """,
        [SOURCE_MATCHSTAT, *tours, window_start_utc, window_end_utc],
    ).fetchall()
    stale_ids = [r[0] for r in stale_rows if r[0] not in seen_fixture_ids]
    if not stale_ids:
        return 0
    id_placeholders = ",".join("?" for _ in stale_ids)
    conn.execute(
        f"DELETE FROM scheduled_matches WHERE scheduled_match_id IN ({id_placeholders})",
        stale_ids,
    )
    return len(stale_ids)


# matchstat round labels we've observed, mapped to a monotonic rank used
# by the contradiction-prune below. Unknown labels rank as 0 → skipped
# defensively (no row deleted via a comparison we can't make sense of).
_ROUND_RANK: dict[str, int] = {
    "Q1": -3,
    "Q2": -2,
    "Q3": -1,
    "First": 1,
    "1st Round": 1,
    "Second": 2,
    "2nd Round": 2,
    "Third": 3,
    "3rd Round": 3,
    "Fourth": 4,
    "4th Round": 4,
    "Round of 16": 4,
    "1/8": 4,
    "Quarter-final": 5,
    "Quarterfinal": 5,
    "1/4": 5,
    "QF": 5,
    "Semi-final": 6,
    "Semifinal": 6,
    "1/2": 6,
    "SF": 6,
    "Final": 7,
    "F": 7,
}


def _prune_contradicted_round_fixtures(
    conn: duckdb.DuckDBPyConnection,
    *,
    tours: list[str],
) -> int:
    """Delete `scheduled_matches` rows that contradict a later-round
    fixture at the same tournament.

    Heuristic: if player P appears in an upcoming Round-N fixture at
    tournament T, **and** P also appears in an upcoming Round-(N+k>0)
    fixture at the same tournament, then the Round-N fixture is stale.
    P can only advance to a later round once Round N is already won —
    so a R1 row co-existing with a R2 row for the same player is
    matchstat's `/fixtures/{date}` payload leaking a completed match.

    Examples we hit in Phase 6.2 manual testing (Roland Garros 2026):
    - Bonzi-Zverev R1 (May 26) co-existed with Machac-Zverev R2
      (May 27). Zverev's R1 had been played the day before; matchstat
      kept returning the R1 row in /fixtures/2026-05-26 intermittently.
    - Khachanov-Gea R1 was already caught by `seen_fixture_ids` prune
      because matchstat eventually stopped returning it, but this
      heuristic would also have caught it via the Khachanov-Trungelliti
      R2 fixture for May 27.

    Limitations:
    - Skipped for rows whose `round_name` is not in `_ROUND_RANK` (we
      don't want to delete a row whose round we can't classify).
    - Only deletes within the tours queried this run.
    """
    if not tours:
        return 0
    placeholders = ",".join("?" for _ in tours)
    rows = conn.execute(
        f"""
        SELECT s1.scheduled_match_id, s1.round_name,
               s1.player1_external_id, s1.player2_external_id,
               s1.tournament_external_id, s1.tour
        FROM scheduled_matches s1
        WHERE s1.source = ?
          AND s1.tour IN ({placeholders})
        """,
        [SOURCE_MATCHSTAT, *tours],
    ).fetchall()

    # Build per-(tour, tournament) index: player_id -> highest round_rank seen.
    by_tournament: dict[tuple[str, str], dict[str, int]] = {}
    rows_by_id: dict[str, tuple[str, int, str, str, str, str]] = {}
    for sm_id, round_name, p1, p2, t_id, tour in rows:
        rank = _ROUND_RANK.get(round_name or "", 0)
        if rank == 0:
            continue
        key = (tour, t_id)
        bucket = by_tournament.setdefault(key, {})
        for p in (p1, p2):
            if p is None:
                continue
            prior = bucket.get(p, 0)
            if rank > prior:
                bucket[p] = rank
        rows_by_id[sm_id] = (round_name, rank, p1, p2, t_id, tour)

    stale_ids: list[str] = []
    for sm_id, (_, rank, p1, p2, t_id, tour) in rows_by_id.items():
        bucket = by_tournament.get((tour, t_id), {})
        max_p1 = bucket.get(p1, 0) if p1 else 0
        max_p2 = bucket.get(p2, 0) if p2 else 0
        # Either player already in a strictly later round → this row is stale.
        if max_p1 > rank or max_p2 > rank:
            stale_ids.append(sm_id)

    if not stale_ids:
        return 0
    id_placeholders = ",".join("?" for _ in stale_ids)
    conn.execute(
        f"DELETE FROM scheduled_matches WHERE scheduled_match_id IN ({id_placeholders})",
        stale_ids,
    )
    return len(stale_ids)


def _prune_duplicate_matchups(
    conn: duckdb.DuckDBPyConnection,
    *,
    tours: list[str],
) -> int:
    """Delete `scheduled_matches` rows that duplicate another row's
    matchup. matchstat occasionally returns the same fixture twice
    under different `fixture_external_id`s — typically once for "today"
    and once for "tomorrow" — and our `INSERT ... ON CONFLICT` keys
    on `scheduled_match_id` so both rows survive. Observed live on
    Sinner-Tabur (1294 + 1295) and Baez-Burruchaga (1298 + 1303) at
    Roland Garros 2026.

    Dedupe key: `(tour, tournament_external_id, sorted player pair,
    round_name)`. Keeps the row with the **most recent `ingested_at`**
    — matchstat's latest publishing wins (Phase 6.2 finding: their
    `/fixtures/` payload mutates throughout the day, including the
    fixture's `id` and `date` for the same logical match). The older
    `fixture_external_id` will normally be swept by
    `_prune_stale_scheduled_matches` once matchstat stops returning it.
    """
    if not tours:
        return 0
    placeholders = ",".join("?" for _ in tours)
    rows = conn.execute(
        f"""
        SELECT scheduled_match_id, tour, tournament_external_id,
               player1_external_id, player2_external_id, round_name,
               ingested_at, fixture_external_id
        FROM scheduled_matches
        WHERE source = ?
          AND tour IN ({placeholders})
        """,
        [SOURCE_MATCHSTAT, *tours],
    ).fetchall()

    groups: dict[tuple, list[tuple]] = {}
    for r in rows:
        sm_id, tour, t_id, p1, p2, round_name, ingested_at, fx_id = r
        if not p1 or not p2 or not round_name:
            continue
        pair = tuple(sorted([p1, p2]))
        key = (tour, t_id, pair, round_name)
        groups.setdefault(key, []).append((sm_id, ingested_at, fx_id))

    stale_ids: list[str] = []
    for entries in groups.values():
        if len(entries) <= 1:
            continue

        # Newest `ingested_at` wins (latest matchstat publishing);
        # ties broken by larger numeric `fixture_external_id`
        # (later-assigned ID).
        def sort_key(t: tuple) -> tuple:
            _, ingested_at, fx_id = t
            try:
                fx_num = int(fx_id) if fx_id is not None else 0
            except (TypeError, ValueError):
                fx_num = 0
            return (ingested_at or datetime.min, fx_num)

        ordered = sorted(entries, key=sort_key, reverse=True)
        # ordered[0] = freshest = keep; remaining are stale duplicates.
        stale_ids.extend(sm_id for sm_id, _, _ in ordered[1:])

    if not stale_ids:
        return 0
    id_placeholders = ",".join("?" for _ in stale_ids)
    conn.execute(
        f"DELETE FROM scheduled_matches WHERE scheduled_match_id IN ({id_placeholders})",
        stale_ids,
    )
    return len(stale_ids)


def _prune_completed_slam_fixtures(
    conn: duckdb.DuckDBPyConnection,
    client: MatchstatClientProtocol,
    *,
    tours: list[str],
) -> int:
    """Cross-check active Grand Slam tournaments via `/tournament/results/`
    and delete `scheduled_matches` rows whose match already appears as
    completed there.

    matchstat's `/fixtures/{date}` endpoint sometimes lists already-played
    Slam R1 matches with future dates (observed: Duckworth-Diallo at
    Roland Garros 2026, played 2026-05-24, listed as upcoming for
    2026-05-26 with no R2 yet — so the round-contradiction prune can't
    fire). The results endpoint is the authoritative completed-match
    source for tour-level tournaments.

    Scope guard: only Slams (`tournament_tier = 'Grand Slam'`), because
    the matchstat free-tier budget can't afford a per-active-tournament
    sweep. Slam draws are big enough that the leak is most visible
    there and other tour-week tournaments cycle rounds daily so the
    round-contradiction prune handles them quickly.

    Adds ~1-2 API calls per refresh in a Slam week, 0 otherwise.
    """
    if not tours:
        return 0
    placeholders = ",".join("?" for _ in tours)
    rows = conn.execute(
        f"""
        SELECT DISTINCT tour, tournament_external_id
        FROM scheduled_matches
        WHERE source = ?
          AND tour IN ({placeholders})
          AND tournament_tier = 'Grand Slam'
          AND tournament_external_id IS NOT NULL
        """,
        [SOURCE_MATCHSTAT, *tours],
    ).fetchall()
    if not rows:
        return 0

    # Gather every (tour, tournament, ordered-pair) tuple that matchstat
    # confirms is completed, then map back to scheduled_match_ids in one
    # query each. Two phases keeps the DELETE deterministic and the
    # caller-visible count accurate.
    completed_keys: set[tuple[str, str, str, str]] = set()
    for tour, t_id_str in rows:
        try:
            season_id = int(t_id_str)
        except (TypeError, ValueError):
            continue
        tour_code: TourCode = "atp" if tour == "ATP" else "wta"
        try:
            results = client.tournament_results(tour_code, season_id)
        except Exception:
            # Best-effort cleanup; matchstat 5xx or auth issues must not
            # fail the rest of the refresh.
            continue
        for match in results.singles:
            p1, p2 = match.player1_id, match.player2_id
            if p1 is None or p2 is None:
                continue
            lo, hi = sorted([str(p1), str(p2)])
            completed_keys.add((tour, t_id_str, lo, hi))

    if not completed_keys:
        return 0

    stale_ids: list[str] = []
    for tour, t_id_str, lo, hi in completed_keys:
        for (sm_id,) in conn.execute(
            """
            SELECT scheduled_match_id FROM scheduled_matches
            WHERE source = ?
              AND tour = ?
              AND tournament_external_id = ?
              AND (
                (player1_external_id = ? AND player2_external_id = ?)
                OR (player1_external_id = ? AND player2_external_id = ?)
              )
            """,
            [SOURCE_MATCHSTAT, tour, t_id_str, lo, hi, hi, lo],
        ).fetchall():
            stale_ids.append(sm_id)

    if not stale_ids:
        return 0
    id_placeholders = ",".join("?" for _ in stale_ids)
    conn.execute(
        f"DELETE FROM scheduled_matches WHERE scheduled_match_id IN ({id_placeholders})",
        stale_ids,
    )
    return len(stale_ids)


def refresh_hot(
    conn: duckdb.DuckDBPyConnection,
    client: MatchstatClientProtocol,
    *,
    tours: list[str],
    today: date | None = None,
    review_csv_path: Path | None = None,
    fixture_lookahead_days: int = 3,
) -> RefreshSummary:
    """Run one daily refresh end-to-end. Returns a RefreshSummary regardless
    of partial failure — the orchestrator catches exceptions and records
    them in the ingestion_runs row so the UI's staleness signal stays
    accurate even after a crash.

    Phase 6.2 (per-tournament refactor): fixture refresh no longer
    iterates `/fixtures/{date}` per day. Instead it discovers active
    tour-level tournament IDs from a single `/fixtures/{today}` probe
    (plus one-day lookahead) and pulls each tournament's full draw via
    `/fixtures/tournament/{id}`. Pros:

    - Slam R1 fixtures whose day-of-play hasn't been published in the
      Order of Play yet still surface (Roland Garros announces day N's
      schedule the evening before; per-date refresh systematically
      missed those rows until the next refresh).
    - Fewer API credits per refresh on a typical week (~5-8 calls vs
      ~13 for the per-date loop).
    - R2 / R3 / ... fixtures appear automatically as matchstat
      publishes them; no need to widen `fixture_lookahead_days`.

    `fixture_lookahead_days` is kept as a parameter — only the prune
    window still uses it — but no longer drives the fixture-fetch
    loop. Default of 3 means we still prune stale rows up to 3 days
    ahead of `today`.
    """
    today = today or datetime.now(UTC).date()
    # DuckDB TIMESTAMP is naive; write naive UTC to avoid round-trip tz
    # shifts on read-back. RefreshSummary still uses tz-aware for callers.
    started_at_aware = datetime.now(UTC)
    started_at = started_at_aware.replace(tzinfo=None)
    run_id = _open_run(
        conn,
        source=SOURCE_MATCHSTAT,
        tour=None if len(tours) != 1 else tours[0],
        started_at=started_at,
        notes=f"today={today.isoformat()}, tours={','.join(tours)}",
    )

    resolver = MatchstatResolver(conn)
    per_tour: dict[str, TourSummary] = {}
    status = "success"
    error_message: str | None = None
    # Phase 6.2: track every fixture ID matchstat returned this run so
    # `_prune_stale_scheduled_matches` can delete rows that matchstat
    # has silently stopped returning (the leaked-completed-fixture bug).
    seen_fixture_ids: set[str] = set()

    try:
        for tour in tours:
            tour_code: TourCode = tour.lower()  # type: ignore[assignment]
            summary = TourSummary(tour=tour)
            per_tour[tour] = summary

            # Calendar is used only for tier metadata (Path C). It's
            # forward-only — currently-active tournaments may be absent,
            # in which case fixtures for those tournaments get
            # tournament_tier=NULL (the UI handles this).
            calendar = client.calendar(tour_code, today.year)
            tier_by_id: dict[int, str | None] = {
                t.id: t.tier for t in calendar if t.tier in TOUR_LEVEL_TIERS
            }

            # Path C: do NOT fetch tournament/results — completed matches
            # come from Sackmann (cold). The helper `_refresh_tournament_results`
            # is kept available if Path B (discover seasonids from fixtures)
            # is wired in later.

            # Phase 6.2 refactor: instead of iterating /fixtures/{date}
            # per day (which only returns matches whose Order of Play is
            # firmed up), discover which tour-level tournaments are
            # currently active and pull each one's full draw via
            # /fixtures/tournament/{id}. matchstat's per-tournament
            # payload is the authoritative source — every round it
            # knows about (R1 / R2 / ... / F) comes back in one call,
            # so Slam R1 fixtures whose day-of-play hasn't been
            # announced yet still appear (only their time is TBD).
            active_tournament_ids = _discover_active_tournament_ids(
                client, tour_code, today, lookahead_days=1
            )
            for tid in sorted(active_tournament_ids):
                _refresh_fixtures_for_tournament(
                    conn,
                    client,
                    tour_code,
                    tour,
                    tid,
                    tier_by_id,
                    resolver,
                    summary,
                    seen_fixture_ids,
                )

            ranking_entries = client.rankings(tour_code)
            summary.rankings += upsert_ranking_overlay(
                conn,
                ranking_entries,
                tour=tour,
                resolve_player=resolver,
                as_of_date=today,
            )
    except Exception as e:  # pylint: disable=broad-except
        status = "failed"
        error_message = f"{type(e).__name__}: {e}"

    # Phase 6.2: prune fixtures that matchstat used to return but no
    # longer does. Run only when this refresh actually saw something —
    # an empty seen-set on a totally-failed run must not wipe the table.
    # ±1d buffer around the queried window covers the Moscow→UTC offset
    # plus any small clock drift between us and matchstat.
    if status != "failed" and seen_fixture_ids:
        window_start = datetime.combine(today - timedelta(days=1), datetime.min.time())
        window_end = datetime.combine(
            today + timedelta(days=fixture_lookahead_days + 2), datetime.min.time()
        )
        _prune_stale_scheduled_matches(
            conn,
            tours=tours,
            seen_fixture_ids=seen_fixture_ids,
            window_start_utc=window_start,
            window_end_utc=window_end,
        )
        # Second prune pass: drop fixtures that contradict a later-round
        # fixture at the same tournament. Catches the case where matchstat
        # is still returning a stale R1 fixture even after publishing the
        # corresponding R2 — `seen_fixture_ids` alone can't help because
        # both rows came back in this same refresh.
        _prune_contradicted_round_fixtures(conn, tours=tours)
        # Third prune pass: dedupe the same matchup returned under two
        # different `fixture_external_id`s (Sinner-Tabur / Baez-Burruchaga
        # at Roland Garros 2026 — matchstat emitted both a today- and a
        # tomorrow-dated row for each).
        _prune_duplicate_matchups(conn, tours=tours)
        # Fourth prune pass: authoritative completed-match cross-check
        # against matchstat `/tournament/results/` for active Slam draws.
        # 1-2 API calls per refresh in a Slam week; catches the leak case
        # where matchstat puts a played R1 in /fixtures/ and the next-round
        # fixture hasn't been published yet (no contradiction to lean on).
        _prune_completed_slam_fixtures(conn, client, tours=tours)

    promoted = promote_completed_fixtures(conn)

    review_written = 0
    if review_csv_path is not None and resolver.review_buffer:
        review_written = _write_review_csv(review_csv_path, resolver.review_buffer)

    totals = LoadCounts()
    for s in per_tour.values():
        totals += s.matches + s.qualifying + s.market_odds + s.fixtures + s.rankings

    if status == "success" and totals.failed > 0:
        status = "partial"

    finished_at_aware = datetime.now(UTC)
    finished_at = finished_at_aware.replace(tzinfo=None)
    _close_run(
        conn,
        run_id=run_id,
        finished_at=finished_at,
        status=status,
        counts=totals,
        requests_used=client.requests_used,
        error_message=error_message,
    )

    return RefreshSummary(
        run_id=run_id,
        started_at=started_at_aware,
        finished_at=finished_at_aware,
        status=status,
        per_tour=per_tour,
        requests_used=client.requests_used,
        promoted_fixtures=promoted,
        review_candidates_written=review_written,
        error_message=error_message,
    )
