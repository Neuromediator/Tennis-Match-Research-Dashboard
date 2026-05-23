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


def test_training_features_has_phase4_1_layout(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    """Phase 4.1 v2 schema must include every FeatureVector field plus identifiers."""
    schema.create_all_tables(fresh_db)
    cols = {
        row[0]
        for row in fresh_db.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'training_features'"
        ).fetchall()
    }
    expected = {
        # identifiers + label
        "match_id",
        "tour",
        "match_date",
        "p1_player_id",
        "p2_player_id",
        "label_winner_is_p1",
        # Surface-Elo (3)
        "elo_p1_surface",
        "elo_p2_surface",
        "elo_diff_surface",
        # Recent form (4)
        "win_pct_last10_p1",
        "win_pct_last10_p2",
        "win_pct_last25_surface_p1",
        "win_pct_last25_surface_p2",
        # Serve/return (8)
        "first_serve_win_pct_p1",
        "first_serve_win_pct_p2",
        "second_serve_win_pct_p1",
        "second_serve_win_pct_p2",
        "bp_saved_pct_p1",
        "bp_saved_pct_p2",
        "bp_converted_pct_p1",
        "bp_converted_pct_p2",
        # H2H (3)
        "h2h_p1_wins",
        "h2h_p2_wins",
        "h2h_recency_days",
        # Fatigue (4)
        "fatigue_matches_7d_p1",
        "fatigue_matches_7d_p2",
        "fatigue_sets_14d_p1",
        "fatigue_sets_14d_p2",
        # Ranking (3)
        "rank_p1",
        "rank_p2",
        "rank_diff",
        # Tournament context (3)
        "tournament_level",
        "best_of",
        "surface",
        # Phase 4.1 — handedness (2)
        "hand_p1",
        "hand_p2",
        # Phase 4.1 — age (4)
        "age_p1",
        "age_p2",
        "age_vs_peak_p1",
        "age_vs_peak_p2",
        # Phase 4.1 — height (3)
        "height_p1",
        "height_p2",
        "height_diff_cm",
        # Phase 4.1 — recovery (2)
        "days_since_last_match_p1",
        "days_since_last_match_p2",
        # bookkeeping
        "schema_version",
    }
    missing = expected - cols
    assert not missing, f"training_features missing Phase 4.1 columns: {missing}"


def test_training_features_migration_drops_phase1_placeholder(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    """If a Phase 1 placeholder shape exists, create_all_tables must replace it."""
    fresh_db.execute(
        "CREATE TABLE training_features ("
        "match_id VARCHAR PRIMARY KEY, "
        "label_winner_is_p1 INTEGER NOT NULL, "
        "schema_version INTEGER NOT NULL DEFAULT 1)"
    )
    schema.create_all_tables(fresh_db)
    cols = {
        row[0]
        for row in fresh_db.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'training_features'"
        ).fetchall()
    }
    assert "tournament_level" in cols, "Migration did not run; placeholder still in place"


def test_training_features_migration_drops_v1_phase3_shape(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    """If the Phase 3 (v1) shape exists — has `tournament_level` but is missing
    the Phase 4.1 columns — create_all_tables must drop it so the v2 DDL
    can take its place. We always re-run `scripts/build_features.py` after
    a feature-set change."""
    fresh_db.execute(
        """
        CREATE TABLE training_features (
            match_id VARCHAR PRIMARY KEY,
            tour VARCHAR NOT NULL,
            match_date DATE NOT NULL,
            p1_player_id VARCHAR NOT NULL,
            p2_player_id VARCHAR NOT NULL,
            label_winner_is_p1 INTEGER NOT NULL,
            tournament_level VARCHAR NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    schema.create_all_tables(fresh_db)
    cols = {
        row[0]
        for row in fresh_db.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'training_features'"
        ).fetchall()
    }
    assert "days_since_last_match_p1" in cols, (
        "v1→v2 migration did not run; Phase 3 shape still present"
    )


def test_llm_traces_has_phase5_columns(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Phase 5 added `web_search_count` and `estimated_cost_usd`; Phase 5.1
    added `fetch_url_count`. All three must be present on a fresh build."""
    schema.create_all_tables(fresh_db)
    cols = {
        row[0]
        for row in fresh_db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'llm_traces'"
        ).fetchall()
    }
    assert {"web_search_count", "estimated_cost_usd", "fetch_url_count"} <= cols


def test_llm_traces_migration_adds_phase5_columns_to_legacy_table(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    """Pre-Phase-5 llm_traces shape must be migrated in place, preserving rows.
    Covers both the Phase 5 columns (web_search_count, estimated_cost_usd)
    and the Phase 5.1 column (fetch_url_count)."""
    fresh_db.execute(schema.LLM_TRACES_SEQUENCE_DDL)
    fresh_db.execute(
        """
        CREATE TABLE llm_traces (
            trace_id               BIGINT PRIMARY KEY DEFAULT nextval('seq_llm_traces'),
            ts                     TIMESTAMP NOT NULL,
            model                  VARCHAR NOT NULL,
            system_prompt_hash     VARCHAR,
            input_messages         JSON,
            tool_calls             JSON,
            output                 JSON,
            tokens_in              INTEGER,
            tokens_out             INTEGER,
            cache_read_tokens      INTEGER,
            cache_creation_tokens  INTEGER,
            latency_ms             INTEGER,
            error                  VARCHAR
        )
        """
    )
    fresh_db.execute("INSERT INTO llm_traces (ts, model) VALUES (CURRENT_TIMESTAMP, 'legacy-row')")

    schema.create_all_tables(fresh_db)

    cols = {
        row[0]
        for row in fresh_db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'llm_traces'"
        ).fetchall()
    }
    assert {"web_search_count", "estimated_cost_usd", "fetch_url_count"} <= cols

    row = fresh_db.execute(
        "SELECT model, web_search_count, estimated_cost_usd, fetch_url_count FROM llm_traces"
    ).fetchone()
    assert row == ("legacy-row", None, None, None)


def test_llm_traces_migration_adds_fetch_url_count_to_phase5_table(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    """Phase-5-shape table (has web_search_count + estimated_cost_usd but no
    fetch_url_count) must be migrated in place when create_all_tables runs."""
    fresh_db.execute(schema.LLM_TRACES_SEQUENCE_DDL)
    fresh_db.execute(
        """
        CREATE TABLE llm_traces (
            trace_id               BIGINT PRIMARY KEY DEFAULT nextval('seq_llm_traces'),
            ts                     TIMESTAMP NOT NULL,
            model                  VARCHAR NOT NULL,
            system_prompt_hash     VARCHAR,
            input_messages         JSON,
            tool_calls             JSON,
            output                 JSON,
            tokens_in              INTEGER,
            tokens_out             INTEGER,
            cache_read_tokens      INTEGER,
            cache_creation_tokens  INTEGER,
            latency_ms             INTEGER,
            error                  VARCHAR,
            web_search_count       INTEGER,
            estimated_cost_usd     DOUBLE
        )
        """
    )
    fresh_db.execute(
        "INSERT INTO llm_traces (ts, model, web_search_count, estimated_cost_usd) "
        "VALUES (CURRENT_TIMESTAMP, 'phase5-row', 2, 0.085)"
    )

    schema.create_all_tables(fresh_db)

    row = fresh_db.execute(
        "SELECT web_search_count, estimated_cost_usd, fetch_url_count FROM llm_traces"
    ).fetchone()
    assert row == (2, 0.085, None)


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
