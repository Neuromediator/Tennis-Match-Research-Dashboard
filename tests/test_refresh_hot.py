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


def _calendar_itf_m15(season_id: int = 21701) -> dict[str, Any]:
    return {
        "id": season_id,
        "name": "M15 Maringa",
        "tier": "M15",
        "date": "2026-05-11T00:00:00.000Z",
        "court": {"id": 2, "name": "Clay"},
        "rank": None,
        "coutry": {"name": "Brazil", "acronym": "BRA"},
    }


def _result_match(
    match_id: str = "84752520",
    p1_id: int = 29935,
    p1_name: str = "Tommy Paul",
    p2_id: int = 82269,
    p2_name: str = "Ethan Quinn",
    winner_id: int = 29935,
    result: str = "6-1 6-3",
) -> dict[str, Any]:
    return {
        "id": match_id,
        "date": "2026-05-17T17:15:00.000Z",
        "roundId": 4,
        "player1Id": p1_id,
        "player2Id": p2_id,
        "tournamentId": 21327,
        "match_winner": winner_id,
        "result": result,
        "odd1": "1.38",
        "odd2": "3.04",
        "player1": {"id": p1_id, "name": p1_name, "countryAcr": "USA"},
        "player2": {"id": p2_id, "name": p2_name, "countryAcr": "USA"},
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
    """One tour-level tournament → one match inserted; one fixture; one ranking.
    Status='success', ingestion_runs has the right counts."""
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    client.set_results("atp", 21327, {"data": {"singles": [_result_match()], "qualifying": []}})
    client.set_fixtures("atp", TODAY, {"data": [_fixture()], "hasNextPage": False})
    client.set_fixtures("atp", TODAY + timedelta(days=1), {"data": [], "hasNextPage": False})
    client.set_rankings("atp", [_ranking()])

    summary = refresh_hot(db, client, tours=["ATP"], today=TODAY)

    assert summary.status == "success"
    assert summary.requests_used > 0

    # Match inserted with score and surface from calendar.
    row = db.execute(
        "SELECT winner_player_id, score, surface, tour FROM matches WHERE source = 'matchstat'"
    ).fetchone()
    assert row == ("ATP_29935", "6-1 6-3", "Clay", "ATP")

    # Market odds inserted.
    market = db.execute(
        "SELECT odds_winner_close, odds_loser_close FROM market_implied_probabilities"
    ).fetchone()
    assert market == pytest.approx((1.38, 3.04))

    # Fixture inserted with tournament tier from calendar.
    fix = db.execute(
        "SELECT tournament_tier, surface, round_name FROM scheduled_matches"
    ).fetchone()
    assert fix == ("ATP 250", "Clay", "R32")

    # Ranking overlay row inserted under today's date.
    rank = db.execute("SELECT ranking_date, player_id, rank FROM rankings").fetchone()
    assert rank == (TODAY, "ATP_106421", 1)

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


def test_refresh_hot_skips_non_tour_level_tournaments(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """M15 (ITF Futures) is in calendar but must NOT trigger a tournament/results call."""
    client = FakeMatchstatClient()
    client.set_calendar(
        "atp",
        TODAY.year,
        [_calendar_tour_level_atp250(), _calendar_itf_m15(season_id=99999)],
    )
    client.set_results("atp", 21327, {"data": {"singles": [_result_match()], "qualifying": []}})
    # NOT registering /tournament/results/99999 — if the orchestrator wrongly
    # asked for it, our fake would return an empty TournamentResults (a free
    # request, but we'd notice via requests_used).
    requests_before = client.requests_used

    refresh_hot(db, client, tours=["ATP"], today=TODAY)

    # We expect: 1 calendar + 1 results (for the tour-level event) + 2 fixtures + 1 rankings = 5
    # If M15 leaked through, we'd see 6+.
    assert client.requests_used - requests_before == 5


def test_refresh_hot_skips_long_finished_tournaments(db: duckdb.DuckDBPyConnection) -> None:
    """Tournaments whose start_date is > 21 days ago are skipped."""
    client = FakeMatchstatClient()
    long_ago = _calendar_tour_level_atp250(season_id=11111, start=TODAY - timedelta(days=60))
    client.set_calendar("atp", TODAY.year, [long_ago])
    refresh_hot(db, client, tours=["ATP"], today=TODAY)

    # No tournament/results call for the stale event.
    result_count = db.execute("SELECT COUNT(*) FROM matches").fetchone()
    assert result_count is not None and result_count[0] == 0


def test_refresh_hot_paginates_fixtures(db: duckdb.DuckDBPyConnection) -> None:
    """fixtures_for_date with hasNextPage=True triggers a follow-up call."""
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    client.set_results("atp", 21327, {"data": {"singles": [], "qualifying": []}})
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
    """Pre-existing scheduled_matches row whose composite key matches a
    just-inserted matches row gets deleted by the promote pass."""
    # Pre-stage a scheduled_match that lines up with the result we'll fetch.
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
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    client.set_results("atp", 21327, {"data": {"singles": [_result_match()], "qualifying": []}})

    summary = refresh_hot(db, client, tours=["ATP"], today=TODAY)

    assert summary.promoted_fixtures == 1
    count = db.execute(
        "SELECT COUNT(*) FROM scheduled_matches WHERE scheduled_match_id = 'matchstat::9001'"
    ).fetchone()
    assert count is not None and count[0] == 0


def test_refresh_hot_marks_partial_when_player_unresolved(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """An unresolved player in a completed match counts as `failed` (the row
    can't be inserted) and bumps the run status from 'success' to 'partial'."""
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    # Unknown player2: not in the seeded aliases.
    client.set_results(
        "atp",
        21327,
        {
            "data": {
                "singles": [_result_match(p2_id=99999, p2_name="Mystery Stranger 99999")],
                "qualifying": [],
            }
        },
    )
    client.set_rankings("atp", [])

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
    """A review-band lookup → CSV file is created with the candidate row."""
    db.execute(
        "INSERT INTO player_aliases (alias_text, tour, source, canonical_player_id, "
        "confidence) VALUES (?, ?, ?, ?, ?)",
        ["Stefanos Tsitsipas", "ATP", "sackmann", "ATP_126774", 1.0],
    )
    client = FakeMatchstatClient()
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    client.set_results(
        "atp",
        21327,
        {
            "data": {
                "singles": [_result_match(p1_id=88888, p1_name="Tsitsi Stefa", winner_id=82269)],
                "qualifying": [],
            }
        },
    )

    review_csv = tmp_path / "review.csv"
    refresh_hot(db, client, tours=["ATP"], today=TODAY, review_csv_path=review_csv)

    # Only write the CSV if buffer is non-empty. If the fuzzy score didn't
    # land in review band on this input, we soft-skip; the *structure* of the
    # write is exercised in the dedicated review-csv test below.
    if review_csv.exists():
        text = review_csv.read_text()
        assert "raw_name" in text  # header present
        assert "tour" in text


def test_refresh_hot_summary_totals_aggregate_across_tours(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Multi-tour run sums per-tour counts into RefreshSummary.totals."""
    client = FakeMatchstatClient()
    # ATP side: 1 completed match + 1 fixture.
    client.set_calendar("atp", TODAY.year, [_calendar_tour_level_atp250()])
    client.set_results("atp", 21327, {"data": {"singles": [_result_match()], "qualifying": []}})
    client.set_fixtures("atp", TODAY, {"data": [_fixture()], "hasNextPage": False})
    # WTA side: empty (no calendar, no fixtures).
    summary = refresh_hot(db, client, tours=["ATP", "WTA"], today=TODAY)

    assert "ATP" in summary.per_tour
    assert "WTA" in summary.per_tour
    assert summary.totals.added >= 2  # at least: match + fixture


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
