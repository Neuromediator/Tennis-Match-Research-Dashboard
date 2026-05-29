"""Tests for `llm.budget` — daily LLM call cap.

Covers:
- `get_budget` reads DAILY_LLM_BUDGET env, falls back to default on
  missing / invalid values.
- `today_trace_count` correctly windows by UTC midnight, including
  edge cases at midnight boundary.
- `is_budget_exhausted` flips at the cap threshold.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from tennis_predictor.data.schema import create_all_tables
from tennis_predictor.llm.budget import (
    DAILY_LLM_BUDGET_DEFAULT,
    get_budget,
    is_budget_exhausted,
    today_trace_count,
)


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    create_all_tables(c)
    return c


def _insert_trace(conn: duckdb.DuckDBPyConnection, ts: datetime) -> None:
    """Minimal llm_traces insert — only `ts` matters for budget counting.
    Other columns are NULL-able per the schema; we omit them."""
    conn.execute(
        "INSERT INTO llm_traces (ts, model) VALUES (?, 'test')",
        [ts.replace(tzinfo=None)],
    )


# ---- get_budget ----------------------------------------------------------


def test_get_budget_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAILY_LLM_BUDGET", raising=False)
    assert get_budget() == DAILY_LLM_BUDGET_DEFAULT


def test_get_budget_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_LLM_BUDGET", "42")
    assert get_budget() == 42


def test_get_budget_invalid_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_LLM_BUDGET", "not-a-number")
    assert get_budget() == DAILY_LLM_BUDGET_DEFAULT


def test_get_budget_zero_or_negative_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_LLM_BUDGET", "0")
    assert get_budget() == DAILY_LLM_BUDGET_DEFAULT
    monkeypatch.setenv("DAILY_LLM_BUDGET", "-5")
    assert get_budget() == DAILY_LLM_BUDGET_DEFAULT


# ---- today_trace_count ---------------------------------------------------


def test_today_trace_count_empty_table(conn: duckdb.DuckDBPyConnection) -> None:
    assert today_trace_count(conn) == 0


def test_today_trace_count_includes_today(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    _insert_trace(conn, now)
    _insert_trace(conn, now - timedelta(hours=2))
    assert today_trace_count(conn, now=now) == 2


def test_today_trace_count_excludes_yesterday(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    yesterday_evening = datetime(2026, 5, 25, 22, 30, tzinfo=UTC)
    _insert_trace(conn, yesterday_evening)
    assert today_trace_count(conn, now=now) == 0


def test_today_trace_count_includes_midnight_exactly(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """A trace logged at exactly 00:00:00 UTC is counted as today."""
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    midnight = datetime(2026, 5, 26, 0, 0, 0, tzinfo=UTC)
    _insert_trace(conn, midnight)
    assert today_trace_count(conn, now=now) == 1


# ---- is_budget_exhausted -------------------------------------------------


def test_budget_not_exhausted_below_cap(
    conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DAILY_LLM_BUDGET", "5")
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    for _ in range(4):
        _insert_trace(conn, now)
    assert is_budget_exhausted(conn, now=now) is False


def test_budget_exhausted_at_cap(
    conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap is inclusive — at exactly N traces we're considered exhausted."""
    monkeypatch.setenv("DAILY_LLM_BUDGET", "5")
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    for _ in range(5):
        _insert_trace(conn, now)
    assert is_budget_exhausted(conn, now=now) is True


def test_budget_exhausted_above_cap(
    conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Soft-cap reality: an in-flight prediction can push us past N."""
    monkeypatch.setenv("DAILY_LLM_BUDGET", "5")
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    for _ in range(8):
        _insert_trace(conn, now)
    assert is_budget_exhausted(conn, now=now) is True
