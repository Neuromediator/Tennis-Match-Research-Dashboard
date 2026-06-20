"""Tests for the catch-up-on-wake refresh in `app/scheduler.py`.

The catch-up exists because HF Spaces (and any scale-to-zero host) sleeps
when idle: the 21:00 cron cannot fire while the container is asleep, so the
first page load after a wake must trigger a background refresh if the hot
data is stale. These tests cover the decision logic — *whether* a background
job is scheduled — without starting a real APScheduler.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from tennis_predictor import config
from tennis_predictor.app import scheduler as sched_mod
from tennis_predictor.data import schema


def _make_db(tmp_path: Path, *, finished_at: datetime | None, with_upcoming: bool = True) -> Path:
    """Create a DuckDB with the full schema and, optionally, one successful
    matchstat ingestion_runs row finished at `finished_at` (naive).

    When `with_upcoming` is True, also insert one scheduled fixture dated
    tomorrow so the "no upcoming fixtures" catch-up trigger does not fire —
    isolating the ingestion-age path under test."""
    db_path = tmp_path / "processed" / "tennis.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    schema.create_all_tables(conn)
    if finished_at is not None:
        naive = finished_at.replace(tzinfo=None)
        conn.execute(
            "INSERT INTO ingestion_runs (run_id, source, tour, started_at, "
            "finished_at, status, rows_added, rows_skipped, rows_failed, "
            "requests_used, error_message, notes) "
            "VALUES ('r', 'matchstat', NULL, ?, ?, 'success', 0, 0, 0, 15, NULL, NULL)",
            [naive, naive],
        )
    if with_upcoming:
        tomorrow = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)
        conn.execute(
            "INSERT INTO scheduled_matches (scheduled_match_id, source, "
            "fixture_external_id, tour, tournament_external_id, tournament_name, "
            "surface, player1_external_id, player2_external_id, player1_name, "
            "player2_name, scheduled_start_utc, time_confirmed, ingested_at) "
            "VALUES ('m1', 'matchstat', 'f1', 'ATP', 't1', 'Terra Wortmann Open', "
            "'Grass', 'p1', 'p2', 'A B', 'C D', ?, FALSE, ?)",
            [tomorrow, tomorrow],
        )
    conn.close()
    return db_path


@pytest.fixture(autouse=True)
def _reset_catch_up_flag():
    """Each test starts with the once-per-process guard cleared."""
    sched_mod._catch_up_attempted = False
    yield
    sched_mod._catch_up_attempted = False


def test_noop_when_scheduler_is_none() -> None:
    # Gated-off host (ENABLE_SCHEDULER unset): get_scheduler() returns None.
    # Must not raise and must not flip the attempted flag.
    sched_mod.maybe_catch_up_refresh(None)
    assert sched_mod._catch_up_attempted is False


def test_schedules_refresh_when_data_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Last successful refresh far in the past → unambiguously stale.
    db_path = _make_db(tmp_path, finished_at=datetime(2020, 1, 1, 0, 0, tzinfo=UTC))
    monkeypatch.setattr(config, "DUCKDB_PATH", db_path)
    scheduler = MagicMock()

    sched_mod.maybe_catch_up_refresh(scheduler)

    scheduler.add_job.assert_called_once()
    assert scheduler.add_job.call_args.kwargs["id"] == "catch_up_refresh"


def test_schedules_refresh_when_fresh_but_no_upcoming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ingestion is recent (fresh by age) but the snapshot has no upcoming
    # fixtures — e.g. yesterday's fixtures have aged into the past. Home
    # would be empty, so catch-up must still fire.
    db_path = _make_db(tmp_path, finished_at=datetime.now(UTC), with_upcoming=False)
    monkeypatch.setattr(config, "DUCKDB_PATH", db_path)
    scheduler = MagicMock()

    sched_mod.maybe_catch_up_refresh(scheduler)

    scheduler.add_job.assert_called_once()
    assert scheduler.add_job.call_args.kwargs["id"] == "catch_up_refresh"


def test_no_refresh_when_data_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _make_db(tmp_path, finished_at=datetime.now(UTC))
    monkeypatch.setattr(config, "DUCKDB_PATH", db_path)
    scheduler = MagicMock()

    sched_mod.maybe_catch_up_refresh(scheduler)

    scheduler.add_job.assert_not_called()


def test_no_refresh_when_db_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fresh host before the one-shot bootstrap populates the volume.
    monkeypatch.setattr(config, "DUCKDB_PATH", tmp_path / "absent.duckdb")
    scheduler = MagicMock()

    sched_mod.maybe_catch_up_refresh(scheduler)

    scheduler.add_job.assert_not_called()


def test_runs_at_most_once_per_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _make_db(tmp_path, finished_at=datetime(2020, 1, 1, 0, 0, tzinfo=UTC))
    monkeypatch.setattr(config, "DUCKDB_PATH", db_path)
    scheduler = MagicMock()

    sched_mod.maybe_catch_up_refresh(scheduler)
    sched_mod.maybe_catch_up_refresh(scheduler)  # second visit, same process

    scheduler.add_job.assert_called_once()
