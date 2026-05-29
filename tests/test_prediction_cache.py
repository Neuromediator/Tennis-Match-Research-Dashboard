"""Tests for `prediction_cache` — the 24h DuckDB cache for AgentResponse.

Covers the contract that matters for Phase 7 cost defense:
- Store + read roundtrip returns the same response (including news items).
- Missing key returns None.
- Stale entry (older than 24h) returns None.
- Re-storing same key upserts (no UNIQUE-violation crash).
- Corrupted JSON in storage returns None (treats schema drift as a miss).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from tennis_predictor.data.prediction_cache import CACHE_TTL, get_cached, store
from tennis_predictor.data.schema import create_all_tables
from tennis_predictor.llm.tools.submit import AgentResponse


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    create_all_tables(c)
    return c


def _sample_response(prob_a: float = 0.6) -> AgentResponse:
    return AgentResponse(
        model_probability_player_a=prob_a,
        model_probability_player_b=1.0 - prob_a,
        news_items=[],
        news_lookup_status="no_results",
        tools_used=["get_model_prediction", "get_head_to_head"],
    )


def test_roundtrip_returns_same_response(conn: duckdb.DuckDBPyConnection) -> None:
    sm_id = "sched_match_42"
    original = _sample_response(prob_a=0.73)
    store(conn, sm_id, original)
    got = get_cached(conn, sm_id)
    assert got == original


def test_missing_key_returns_none(conn: duckdb.DuckDBPyConnection) -> None:
    assert get_cached(conn, "no_such_match") is None


def test_stale_entry_returns_none(conn: duckdb.DuckDBPyConnection) -> None:
    sm_id = "stale_match"
    old_moment = datetime.now(UTC) - CACHE_TTL - timedelta(hours=1)
    store(conn, sm_id, _sample_response(), now=old_moment)
    assert get_cached(conn, sm_id) is None


def test_fresh_entry_just_inside_ttl_returns_response(conn: duckdb.DuckDBPyConnection) -> None:
    sm_id = "fresh_match"
    just_inside = datetime.now(UTC) - CACHE_TTL + timedelta(minutes=5)
    store(conn, sm_id, _sample_response(), now=just_inside)
    assert get_cached(conn, sm_id) is not None


def test_restoring_same_key_upserts(conn: duckdb.DuckDBPyConnection) -> None:
    sm_id = "upsert_match"
    store(conn, sm_id, _sample_response(prob_a=0.4))
    store(conn, sm_id, _sample_response(prob_a=0.9))
    got = get_cached(conn, sm_id)
    assert got is not None
    assert got.model_probability_player_a == 0.9
    # Sanity: only one row, not two.
    n_rows = conn.execute(
        "SELECT COUNT(*) FROM prediction_cache WHERE scheduled_match_id = ?", [sm_id]
    ).fetchone()
    assert n_rows is not None and n_rows[0] == 1


def test_corrupted_json_returns_none(conn: duckdb.DuckDBPyConnection) -> None:
    """Schema drift safety: if a stored row no longer parses, return None
    rather than blowing up the page. Next predict() will overwrite it."""
    sm_id = "corrupt_match"
    moment = datetime.now(UTC).replace(tzinfo=None)
    conn.execute(
        "INSERT INTO prediction_cache (scheduled_match_id, cached_at, agent_response_json) "
        "VALUES (?, ?, ?)",
        [sm_id, moment, "this is not valid JSON for AgentResponse"],
    )
    assert get_cached(conn, sm_id) is None
