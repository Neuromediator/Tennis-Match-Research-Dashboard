"""Schema-creation tests.

Validates:
- All expected tables come into existence.
- create_all_tables is idempotent (safe to call repeatedly).
- A representative table accepts an INSERT with the documented column types.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import schema


@pytest.fixture
def fresh_db(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    yield conn
    conn.close()


def _table_names(conn: duckdb.DuckDBPyConnection) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return {row[0] for row in rows}


def test_create_all_tables_creates_every_expected_table(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    schema.create_all_tables(fresh_db)
    actual = _table_names(fresh_db)
    missing = schema.EXPECTED_TABLES - actual
    assert not missing, f"Expected tables not created: {missing}"


def test_create_all_tables_is_idempotent(fresh_db: duckdb.DuckDBPyConnection) -> None:
    schema.create_all_tables(fresh_db)
    schema.create_all_tables(fresh_db)
    schema.create_all_tables(fresh_db)
    assert _table_names(fresh_db) >= schema.EXPECTED_TABLES


def test_matches_table_accepts_well_formed_row(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    schema.create_all_tables(fresh_db)
    fresh_db.execute(
        """
        INSERT INTO matches (
            match_id, source, match_external_id, tour, match_tier,
            tourney_id, tourney_date, match_num, match_status,
            winner_player_id, loser_player_id
        ) VALUES (
            'sackmann::ATP-2023-9900-300', 'sackmann', 'ATP-2023-9900-300',
            'ATP', 'main',
            '2023-9900', DATE '2023-01-02', 300, 'completed',
            'ATP_126203', 'ATP_126610'
        )
        """
    )
    count = fresh_db.execute("SELECT COUNT(*) FROM matches").fetchone()
    assert count is not None and count[0] == 1


def test_llm_traces_autoincrements_trace_id(fresh_db: duckdb.DuckDBPyConnection) -> None:
    schema.create_all_tables(fresh_db)
    fresh_db.execute(
        "INSERT INTO llm_traces (ts, model) VALUES (CURRENT_TIMESTAMP, 'claude-sonnet-4-6')"
    )
    fresh_db.execute(
        "INSERT INTO llm_traces (ts, model) VALUES (CURRENT_TIMESTAMP, 'claude-sonnet-4-6')"
    )
    ids = [row[0] for row in fresh_db.execute("SELECT trace_id FROM llm_traces").fetchall()]
    assert len(ids) == 2
    assert ids[0] != ids[1]


def test_scheduled_matches_accepts_well_formed_row(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    schema.create_all_tables(fresh_db)
    fresh_db.execute(
        """
        INSERT INTO scheduled_matches (
            scheduled_match_id, source, fixture_external_id,
            tour, tournament_external_id, tournament_name, tournament_tier,
            surface, round_external_id, round_name,
            player1_external_id, player2_external_id,
            player1_name, player2_name,
            scheduled_start_utc, ingested_at
        ) VALUES (
            'matchstat::1215', 'matchstat', '1215',
            'ATP', '21327', 'Geneva Open', 'ATP 250',
            'Clay', '4', 'R32',
            '37741', '87277',
            'Zizou Bergs', 'Arthur Gea',
            TIMESTAMP '2026-05-19 13:00:00', CURRENT_TIMESTAMP
        )
        """
    )
    count = fresh_db.execute("SELECT COUNT(*) FROM scheduled_matches").fetchone()
    assert count is not None and count[0] == 1


def test_scheduled_matches_rejects_duplicate_external_id(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    """The UNIQUE (source, fixture_external_id) constraint is what makes daily
    refresh idempotent — re-running must not produce duplicates."""
    schema.create_all_tables(fresh_db)
    base_sql = """
        INSERT INTO scheduled_matches (
            scheduled_match_id, source, fixture_external_id,
            tour, tournament_external_id,
            player1_external_id, player2_external_id,
            player1_name, player2_name,
            ingested_at
        ) VALUES (?, 'matchstat', '1215', 'ATP', '21327',
                  '37741', '87277', 'Zizou Bergs', 'Arthur Gea', CURRENT_TIMESTAMP)
    """
    fresh_db.execute(base_sql, ["matchstat::1215"])
    with pytest.raises(duckdb.ConstraintException):
        fresh_db.execute(base_sql, ["matchstat::1215-dup"])


def test_ingestion_runs_accepts_well_formed_row(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    schema.create_all_tables(fresh_db)
    fresh_db.execute(
        """
        INSERT INTO ingestion_runs (
            run_id, source, tour, started_at, finished_at, status,
            rows_added, rows_skipped, rows_failed, requests_used
        ) VALUES (
            'run-abc-123', 'matchstat', 'ATP',
            TIMESTAMP '2026-05-18 06:00:00', TIMESTAMP '2026-05-18 06:00:12',
            'success', 42, 3, 0, 7
        )
        """
    )
    row = fresh_db.execute(
        "SELECT status, rows_added, requests_used FROM ingestion_runs WHERE run_id = 'run-abc-123'"
    ).fetchone()
    assert row == ("success", 42, 7)
