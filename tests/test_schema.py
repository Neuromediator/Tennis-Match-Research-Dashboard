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
