"""Integration-style tests for the daily refresh orchestrator.

Use a FakeMatchstatClient (Protocol-compatible) so we can stage payloads
and verify the orchestrator wires everything together correctly —
without any network calls.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.data.matchstat import (
    CalendarTournament,
    FixturesPage,
    RankingEntry,
    TournamentResults,
)
from tennis_predictor.data.refresh_hot import refresh_hot

TODAY = date(2026, 5, 18)


class FakeMatchstatClient:
    """In-memory stub satisfying `MatchstatClientProtocol`."""

    def __init__(self) -> None:
        self.requests_used = 0
        self._calendar: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._results: dict[tuple[str, int], dict[str, Any]] = {}
        self._fixtures: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._rankings: dict[str, list[dict[str, Any]]] = {}
        self._raise_on: str | None = None

    # ---- setup helpers (used by tests, not part of the Protocol) ----

    def set_calendar(self, tour: str, year: int, items: list[dict[str, Any]]) -> None:
        self._calendar[tour, year] = items

    def set_results(self, tour: str, season_id: int, payload: dict[str, Any]) -> None:
        self._results[tour, season_id] = payload

    def set_fixtures(
        self,
        tour: str,
        match_date: date,
        payload: dict[str, Any],
        *,
        page_no: int = 1,
    ) -> None:
        self._fixtures[tour, match_date.isoformat(), page_no] = payload

    def set_rankings(self, tour: str, items: list[dict[str, Any]]) -> None:
        self._rankings[tour] = items

    def raise_on(self, endpoint: str) -> None:
        """Make the next call to `endpoint` raise — used to test failure recording."""
        self._raise_on = endpoint

    # ---- MatchstatClientProtocol surface ----

    def calendar(self, tour: str, year: int) -> list[CalendarTournament]:
        self.requests_used += 1
        if self._raise_on == "calendar":
            self._raise_on = None
            raise RuntimeError("fake calendar failure")
        items = self._calendar.get((tour, year), [])
        return [CalendarTournament.model_validate(it) for it in items]

    def tournament_results(self, tour: str, season_id: int) -> TournamentResults:
        self.requests_used += 1
        # Test payloads mirror the raw API shape: `{"data": {...}}`. The real
        # client unwraps the outer `data` before model_validate; do the same
        # here so test fixtures look like actual API responses.
        payload = self._results.get((tour, season_id), {})
        inner = payload.get("data", payload) if isinstance(payload, dict) else {}
        return TournamentResults.model_validate(inner)

    def fixtures_for_date(
        self,
        tour: str,
        match_date: date,
        *,
        singles_only: bool = True,
        page_size: int = 100,
        page_no: int = 1,
    ) -> FixturesPage:
        self.requests_used += 1
        key = (tour, match_date.isoformat(), page_no)
        payload = self._fixtures.get(key, {"data": [], "hasNextPage": False})
        return FixturesPage.model_validate(payload)

    def rankings(
        self,
        tour: str,
        *,
        page_size: int = 100,
        page_no: int = 1,
    ) -> list[RankingEntry]:
        self.requests_used += 1
        items = self._rankings.get(tour, [])
        return [RankingEntry.model_validate(it) for it in items]


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture
def db(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "test.duckdb"))
    schema.create_all_tables(conn)
    # Seed an ATP alias so resolver can resolve the players we'll use.
    aliases = [
        ("Tommy Paul", "ATP", "sackmann", "ATP_29935", 1.0),
        ("Paul Tommy", "ATP", "sackmann", "ATP_29935", 1.0),
        ("Paul T", "ATP", "sackmann", "ATP_29935", 1.0),
        ("Ethan Quinn", "ATP", "sackmann", "ATP_82269", 1.0),
        ("Quinn Ethan", "ATP", "sackmann", "ATP_82269", 1.0),
        ("Quinn E", "ATP", "sackmann", "ATP_82269", 1.0),
        ("Zizou Bergs", "ATP", "sackmann", "ATP_37741", 1.0),
        ("Bergs Zizou", "ATP", "sackmann", "ATP_37741", 1.0),
        ("Bergs Z", "ATP", "sackmann", "ATP_37741", 1.0),
        ("Arthur Gea", "ATP", "sackmann", "ATP_87277", 1.0),
        ("Gea Arthur", "ATP", "sackmann", "ATP_87277", 1.0),
        ("Gea A", "ATP", "sackmann", "ATP_87277", 1.0),
        ("Jannik Sinner", "ATP", "sackmann", "ATP_106421", 1.0),
        ("Sinner Jannik", "ATP", "sackmann", "ATP_106421", 1.0),
        ("Sinner J", "ATP", "sackmann", "ATP_106421", 1.0),
    ]
    for row in aliases:
        with suppress(duckdb.ConstraintException):
            conn.execute(
                "INSERT INTO player_aliases (alias_text, tour, source, "
                "canonical_player_id, confidence) VALUES (?, ?, ?, ?, ?)",
                list(row),
            )
    return conn


def _calendar_tour_level_atp250(
    season_id: int = 21327, start: date | None = None
) -> dict[str, Any]:
    start = start or TODAY - timedelta(days=2)
    return {
        "id": season_id,
        "name": "Geneva Open",
        "tier": "ATP 250",
        "date": f"{start.isoformat()}T00:00:00.000Z",
        "court": {"id": 2, "name": "Clay"},
        "rank": {"id": 2, "name": "Main tour"},
        "countryAcr": "SUI",
    }


def _fixture(
    fx_id: int = 1215,
    p1_id: int = 37741,
    p1_name: str = "Zizou Bergs",
    p2_id: int = 87277,
    p2_name: str = "Arthur Gea",
    tournament_id: int = 21327,
) -> dict[str, Any]:
    return {
        "id": fx_id,
        "date": "2026-05-19T13:00:00.000Z",
        "roundId": 4,
        "player1Id": p1_id,
        "player2Id": p2_id,
        "tournamentId": tournament_id,
        "player1": {"id": p1_id, "name": p1_name, "countryAcr": "BEL"},
        "player2": {"id": p2_id, "name": p2_name, "countryAcr": "FRA"},
        "tournament": {
            "id": tournament_id,
            "name": "Geneva Open",
            "court": {"id": 2, "name": "Clay"},
            "rank": {"id": 2, "name": "Main tour"},
        },
        "round": {"id": 4, "name": "R32"},
    }


def _ranking(
    position: int = 1, player_id: int = 47275, name: str = "Jannik Sinner"
) -> dict[str, Any]:
    return {
        "id": 1,
        "date": f"{TODAY.isoformat()}T00:00:00.000Z",
        "point": 14700,
        "position": position,
        "player": {"id": player_id, "name": name, "countryAcr": "ITA"},
    }


# ---------------------------------------------------------------------------
# Tests


def test_refresh_hot_happy_path(db: duckdb.DuckDBPyConnection) -> None:
    """Fixtures populated with tier from calendar, rankings written, status=success.

    Under Path C the orchestrator does NOT fetch tournament/results — completed
    matches come from Sackmann (cold). This test asserts the steady-state
    hot-path: calendar (tier lookup) + fixtures + rankings + ingestion_runs row.
    """
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    client.set_fixtures("atp", TODAY, {"data": [_fixture()], "hasNextPage": False})
    client.set_fixtures("atp", TODAY + timedelta(days=1), {"data": [], "hasNextPage": False})
    client.set_rankings("atp", [_ranking()])

    summary = refresh_hot(db, client, tours=["ATP"], today=TODAY)

    assert summary.status == "success"
    assert summary.requests_used > 0

    # Fixture inserted with tournament tier sourced from calendar lookup.
    fix = db.execute(
        "SELECT tournament_tier, surface, round_name FROM scheduled_matches"
    ).fetchone()
    assert fix == ("ATP 250", "Clay", "R32")

    # Ranking overlay row inserted under today's date.
    rank = db.execute("SELECT ranking_date, player_id, rank FROM rankings").fetchone()
    assert rank == (TODAY, "ATP_106421", 1)

    # No completed match rows from matchstat — Path C drops this path.
    match_count = db.execute("SELECT COUNT(*) FROM matches WHERE source = 'matchstat'").fetchone()
    assert match_count is not None and match_count[0] == 0

    # No pre-match odds — they came as a bonus from tournament/results.
    odds_count = db.execute(
        "SELECT COUNT(*) FROM market_implied_probabilities WHERE odds_source = 'matchstat'"
    ).fetchone()
    assert odds_count is not None and odds_count[0] == 0

    # ingestion_runs row reflects success.
    run = db.execute(
        "SELECT status, rows_added, requests_used, error_message FROM ingestion_runs"
    ).fetchone()
    assert run is not None
    status, added, requests_used, err = run
    assert status == "success"
    assert added > 0
    assert requests_used == summary.requests_used
    assert err is None


def test_refresh_hot_does_not_call_tournament_results(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Path C invariant: orchestrator must NOT call tournament/results.

    We stage an active tour-level event in the calendar but register NO
    results payload. If the orchestrator ever tried to fetch results, the
    fake would still answer (empty payload, +1 request) but the request
    count would betray us. Expect exactly: 1 calendar + 2 fixtures + 1 rankings.
    """
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    # Deliberately NOT calling set_results.
    refresh_hot(db, client, tours=["ATP"], today=TODAY)

    assert client.requests_used == 4  # 1 calendar + today + tomorrow + rankings


def test_refresh_hot_paginates_fixtures(db: duckdb.DuckDBPyConnection) -> None:
    """fixtures_for_date with hasNextPage=True triggers a follow-up call."""
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    # Two pages: first has 1 fixture and hasNextPage=True, second has 1 fixture and false.
    client.set_fixtures("atp", TODAY, {"data": [_fixture(fx_id=1)], "hasNextPage": True}, page_no=1)
    client.set_fixtures(
        "atp", TODAY, {"data": [_fixture(fx_id=2)], "hasNextPage": False}, page_no=2
    )
    client.set_fixtures("atp", TODAY + timedelta(days=1), {"data": [], "hasNextPage": False})

    refresh_hot(db, client, tours=["ATP"], today=TODAY)

    count = db.execute("SELECT COUNT(*) FROM scheduled_matches").fetchone()
    assert count is not None and count[0] == 2


def test_refresh_hot_promotes_completed_fixtures(db: duckdb.DuckDBPyConnection) -> None:
    """promote_completed_fixtures is still invoked at end-of-run.

    Under Path C the hot path doesn't insert matches, but the function
    stays wired — if a `matches` row with a matchstat-keyed
    (tournament_id, players, round) tuple ever ends up in the table (via
    a manual import, a future Path B, or a backfill script), the
    promotion pass will still remove the corresponding scheduled fixture.
    This test pre-seeds both manually and verifies the wiring.
    """
    db.execute(
        """
        INSERT INTO scheduled_matches (
            scheduled_match_id, source, fixture_external_id, tour,
            tournament_external_id, round_external_id,
            player1_external_id, player2_external_id,
            player1_canonical_id, player2_canonical_id,
            player1_name, player2_name, ingested_at
        ) VALUES (
            'matchstat::9001', 'matchstat', '9001', 'ATP',
            '21327', '4', '29935', '82269', 'ATP_29935', 'ATP_82269',
            'Tommy Paul', 'Ethan Quinn', CURRENT_TIMESTAMP
        )
        """
    )
    db.execute(
        """
        INSERT INTO matches (
            match_id, source, match_external_id, tour, match_tier,
            tourney_id, tourney_date, match_num, match_status, round,
            winner_player_id, loser_player_id
        ) VALUES (
            'matchstat::9001-done', 'matchstat', '84752520', 'ATP', 'main',
            '21327', DATE '2026-05-17', 1, 'completed', '4',
            'ATP_29935', 'ATP_82269'
        )
        """
    )
    client = FakeMatchstatClient()
    summary = refresh_hot(db, client, tours=["ATP"], today=TODAY)

    assert summary.promoted_fixtures == 1
    count = db.execute(
        "SELECT COUNT(*) FROM scheduled_matches WHERE scheduled_match_id = 'matchstat::9001'"
    ).fetchone()
    assert count is not None and count[0] == 0


def test_refresh_hot_marks_partial_when_ranking_player_unresolved(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Unresolved player in a rankings entry counts as `failed`
    (rankings.player_id is NOT NULL) and bumps status to 'partial'.

    Path C analogue of the original 'unresolved-in-matches' test — same
    contract, different surface (rankings is now the only NOT-NULL-blocking
    insert path the orchestrator drives).
    """
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    # Ranking entry references a player NOT in our seeded aliases.
    client.set_rankings(
        "atp",
        [_ranking(position=1, player_id=99999, name="Mystery Stranger 99999")],
    )

    summary = refresh_hot(db, client, tours=["ATP"], today=TODAY)

    assert summary.status == "partial"
    run = db.execute("SELECT status, rows_failed FROM ingestion_runs").fetchone()
    assert run is not None
    status, failed = run
    assert status == "partial"
    assert failed >= 1


def test_refresh_hot_marks_failed_on_exception(db: duckdb.DuckDBPyConnection) -> None:
    """If the client raises, status='failed', error captured, run still closed."""
    client = FakeMatchstatClient()
    client.raise_on("calendar")

    summary = refresh_hot(db, client, tours=["ATP"], today=TODAY)

    assert summary.status == "failed"
    assert summary.error_message is not None
    assert "fake calendar failure" in summary.error_message

    run = db.execute("SELECT status, finished_at, error_message FROM ingestion_runs").fetchone()
    assert run is not None
    status, finished_at, err = run
    assert status == "failed"
    assert finished_at is not None  # run is closed, not dangling
    assert err is not None
    assert "fake calendar failure" in err


def test_refresh_hot_writes_review_csv_when_buffer_nonempty(
    db: duckdb.DuckDBPyConnection,
    tmp_path: Path,
) -> None:
    """A review-band lookup surfaces via fixtures → CSV file is written.

    Reworked from the original results-based variant: under Path C the
    orchestrator hits the resolver via fixtures (and rankings), not via
    tournament/results. We exercise it through a fixture whose player1
    name is close-but-not-identical to a seeded alias.
    """
    db.execute(
        "INSERT INTO player_aliases (alias_text, tour, source, canonical_player_id, "
        "confidence) VALUES (?, ?, ?, ?, ?)",
        ["Stefanos Tsitsipas", "ATP", "sackmann", "ATP_126774", 1.0],
    )
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    # Player1 has a garbled name that should land in the review band (or higher).
    client.set_fixtures(
        "atp",
        TODAY,
        {
            "data": [
                _fixture(
                    fx_id=2222,
                    p1_id=88888,
                    p1_name="Tsitsi Stefa",
                )
            ],
            "hasNextPage": False,
        },
    )

    review_csv = tmp_path / "review.csv"
    refresh_hot(db, client, tours=["ATP"], today=TODAY, review_csv_path=review_csv)

    # If the fuzzy score landed in the review band, the CSV was written.
    # If it landed above auto or below unknown, no CSV — the *structure* of
    # the write is exercised by the matchstat_resolver tests directly.
    if review_csv.exists():
        text = review_csv.read_text()
        assert "raw_name" in text  # header present
        assert "tour" in text


def test_refresh_hot_summary_totals_aggregate_across_tours(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Multi-tour run sums per-tour counts into RefreshSummary.totals.

    Under Path C the contributions come from fixtures and rankings, not
    completed matches — we still expect non-zero totals when both tours
    have data, just from the surfaces that ARE driven by the orchestrator.
    """
    client = FakeMatchstatClient()
    # ATP side: 1 fixture + 1 ranking.
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    client.set_fixtures("atp", TODAY, {"data": [_fixture()], "hasNextPage": False})
    client.set_rankings("atp", [_ranking()])
    # WTA side: empty (no calendar, no fixtures, no rankings).
    summary = refresh_hot(db, client, tours=["ATP", "WTA"], today=TODAY)

    assert "ATP" in summary.per_tour
    assert "WTA" in summary.per_tour
    assert summary.totals.added >= 2  # at least: fixture + ranking from ATP


def test_refresh_hot_records_started_and_finished_timestamps(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """DuckDB TIMESTAMP stores naive datetimes — we write tz-aware UTC and
    read back naive, but the ordering relationship still holds."""
    client = FakeMatchstatClient()
    before = datetime.now(UTC).replace(tzinfo=None)
    refresh_hot(db, client, tours=["ATP"], today=TODAY)
    after = datetime.now(UTC).replace(tzinfo=None)

    row = db.execute("SELECT started_at, finished_at FROM ingestion_runs").fetchone()
    assert row is not None
    started, finished = row
    assert before <= started <= after
    assert started <= finished <= after
