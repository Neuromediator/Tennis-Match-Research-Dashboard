"""Daily hot-data refresh orchestrator (Phase 2).

Composes the API client, the resolver, and the load layer into one
end-to-end run:

1. Open an `ingestion_runs` row with `status='running'`.
2. For each tour:
   a. Fetch the year's calendar; keep tour-level tournaments that are
      "recently active" (start_date <= today <= start_date + 21d).
   b. For each active tournament: fetch `tournament/results/{seasonid}`,
      insert singles + qualifying matches and pre-match odds.
   c. Fetch `fixtures/{today}` (and `{tomorrow}`, paginated to cover all
      currently-known draws), insert into `scheduled_matches`.
   d. Fetch current rankings, write overlay.
3. Promote any `scheduled_matches` row whose composite key now appears
   in `matches` (i.e., the fixture has completed).
4. Write the resolver's `review_buffer` to `aliases_review_matchstat.csv`.
5. Close the `ingestion_runs` row with `status='success' | 'partial' |
   'failed'`, totals, `requests_used`.

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

REVIEW_CSV_FIELDNAMES = ["raw_name", "tour", "candidate_name", "confidence", "runner_up_confidence"]


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
                    "confidence": f"{c.confidence:.4f}",
                    "runner_up_confidence": f"{c.runner_up_confidence:.4f}",
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
    max_pages: int = 10,
) -> None:
    page_no = 1
    while page_no <= max_pages:
        page = client.fixtures_for_date(tour_code, match_date, page_no=page_no)
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


def refresh_hot(
    conn: duckdb.DuckDBPyConnection,
    client: MatchstatClientProtocol,
    *,
    tours: list[str],
    today: date | None = None,
    review_csv_path: Path | None = None,
    fixture_lookahead_days: int = 1,
) -> RefreshSummary:
    """Run one daily refresh end-to-end. Returns a RefreshSummary regardless
    of partial failure — the orchestrator catches exceptions and records
    them in the ingestion_runs row so the UI's staleness signal stays
    accurate even after a crash.
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

    try:
        for tour in tours:
            tour_code: TourCode = tour.lower()  # type: ignore[assignment]
            summary = TourSummary(tour=tour)
            per_tour[tour] = summary

            calendar = client.calendar(tour_code, today.year)
            active = [
                t
                for t in calendar
                if (t.tier in TOUR_LEVEL_TIERS and _is_recently_active(t, today))
            ]
            tier_by_id = {t.id: t.tier for t in active}

            _refresh_tournament_results(conn, client, tour_code, tour, active, resolver, summary)

            for offset in range(fixture_lookahead_days + 1):
                d = today + timedelta(days=offset)
                _refresh_fixtures_for_date(
                    conn, client, tour_code, tour, d, tier_by_id, resolver, summary
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
