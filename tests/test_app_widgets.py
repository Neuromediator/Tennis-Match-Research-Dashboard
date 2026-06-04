"""Unit tests for pure-Python helpers in `app.widgets`.

Streamlit-rendering functions (`cost_monitor_block`, `stale_data_banner`,
`prediction_card`, `freshness_indicator`) are not covered here — they
would require Streamlit's experimental `AppTest` API and Phase 6 chose
to defer that (see `docs/tutorials/phase_6_notes.md` Part 7). The SQL
aggregations and the staleness threshold are the real logic and are
both exercised here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.app.widgets import (
    STALE_THRESHOLD_HOURS,
    is_data_stale,
    is_quota_error,
    query_cost_summary,
    query_last_hot_refresh,
    query_last_hot_run_error,
    query_matchstat_usage_month,
)
from tennis_predictor.data import schema


@pytest.fixture
def conn(tmp_path: Path):
    db = duckdb.connect(str(tmp_path / "widgets.duckdb"))
    schema.create_all_tables(db)
    yield db
    db.close()


def _insert_trace(
    conn: duckdb.DuckDBPyConnection,
    *,
    ts: datetime,
    cost_usd: float = 0.10,
    tokens_in: int = 1000,
    tokens_out: int = 500,
    cache_read_tokens: int = 2000,
    cache_creation_tokens: int = 0,
    web_search_count: int = 1,
    fetch_url_count: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO llm_traces (
            ts, model, system_prompt_hash, input_messages, tool_calls, output,
            tokens_in, tokens_out, cache_read_tokens, cache_creation_tokens,
            latency_ms, error, web_search_count, estimated_cost_usd, fetch_url_count
        ) VALUES (?, 'claude-sonnet-4-6', 'h', NULL, NULL, NULL,
                  ?, ?, ?, ?, 1000, NULL, ?, ?, ?)
        """,
        [
            ts,
            tokens_in,
            tokens_out,
            cache_read_tokens,
            cache_creation_tokens,
            web_search_count,
            cost_usd,
            fetch_url_count,
        ],
    )


def _insert_run(
    conn: duckdb.DuckDBPyConnection,
    *,
    finished_at: datetime,
    source: str = "matchstat",
    status: str = "success",
) -> None:
    # DuckDB's `TIMESTAMP` is naive; pass naive datetimes the way production
    # ingestion code does (matchstat refresh writes `datetime.utcnow()`-ish
    # values without tzinfo).
    naive = finished_at.replace(tzinfo=None)
    conn.execute(
        """
        INSERT INTO ingestion_runs (
            run_id, source, tour, started_at, finished_at, status,
            rows_added, rows_skipped, rows_failed, requests_used,
            error_message, notes
        ) VALUES (?, ?, NULL, ?, ?, ?, 0, 0, 0, 0, NULL, NULL)
        """,
        [
            f"run-{naive.isoformat()}",
            source,
            naive - timedelta(minutes=1),
            naive,
            status,
        ],
    )


# ---------------------------------------------------------------------------
# Cost summary
# ---------------------------------------------------------------------------


def test_query_cost_summary_empty_table(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
    summary = query_cost_summary(conn, now=now)
    assert summary.today_usd == 0.0
    assert summary.today_calls == 0
    assert summary.month_usd == 0.0
    assert summary.cache_hit_rate_24h == 0.0


def test_query_cost_summary_buckets_today_and_month(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
    # Today (UTC): two calls at $0.10 each, 1 web search each.
    _insert_trace(conn, ts=datetime(2026, 5, 23, 10, 0, tzinfo=UTC), cost_usd=0.10)
    _insert_trace(conn, ts=datetime(2026, 5, 23, 11, 30, tzinfo=UTC), cost_usd=0.10)
    # Earlier this month, not today.
    _insert_trace(conn, ts=datetime(2026, 5, 10, 8, 0, tzinfo=UTC), cost_usd=0.05)
    # Previous month — outside both buckets.
    _insert_trace(conn, ts=datetime(2026, 4, 28, 8, 0, tzinfo=UTC), cost_usd=0.20)

    summary = query_cost_summary(conn, now=now)

    assert summary.today_usd == pytest.approx(0.20)
    assert summary.today_calls == 2
    assert summary.today_web_searches == 2
    assert summary.month_usd == pytest.approx(0.25)
    assert summary.month_calls == 3


def test_query_cost_summary_cache_hit_rate_uses_last_24h(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
    # Within 24h: heavy cache reads.
    _insert_trace(
        conn,
        ts=datetime(2026, 5, 23, 11, 0, tzinfo=UTC),
        tokens_in=1000,
        cache_read_tokens=9000,
        cache_creation_tokens=0,
    )
    # Outside the window — should not affect the rate.
    _insert_trace(
        conn,
        ts=datetime(2026, 5, 20, 0, 0, tzinfo=UTC),
        tokens_in=10_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )

    summary = query_cost_summary(conn, now=now)
    # 9000 / (1000 + 9000 + 0) = 0.9
    assert summary.cache_hit_rate_24h == pytest.approx(0.9)


def test_query_cost_summary_fetch_url_counter(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
    _insert_trace(
        conn,
        ts=datetime(2026, 5, 23, 11, 0, tzinfo=UTC),
        web_search_count=1,
        fetch_url_count=2,
    )
    summary = query_cost_summary(conn, now=now)
    assert summary.today_fetch_urls == 2


# ---------------------------------------------------------------------------
# Hot-refresh freshness
# ---------------------------------------------------------------------------


def test_query_last_hot_refresh_empty_returns_none(conn: duckdb.DuckDBPyConnection) -> None:
    assert query_last_hot_refresh(conn) is None


def test_query_last_hot_refresh_picks_latest_success(conn: duckdb.DuckDBPyConnection) -> None:
    _insert_run(conn, finished_at=datetime(2026, 5, 20, 3, 0, tzinfo=UTC))
    _insert_run(conn, finished_at=datetime(2026, 5, 22, 3, 0, tzinfo=UTC))
    _insert_run(
        conn,
        finished_at=datetime(2026, 5, 23, 4, 0, tzinfo=UTC),
        status="failed",
    )

    last = query_last_hot_refresh(conn)
    assert last is not None
    # tzinfo is attached if the DB column lost it during round-tripping.
    assert last.replace(tzinfo=UTC) == datetime(2026, 5, 22, 3, 0, tzinfo=UTC)


def test_query_last_hot_refresh_treats_partial_as_success(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """A run that marked itself `partial` still pulled fresh data — the
    staleness signal must NOT fire just because one sub-step errored."""
    _insert_run(conn, finished_at=datetime(2026, 5, 20, 3, 0, tzinfo=UTC))
    _insert_run(
        conn,
        finished_at=datetime(2026, 5, 23, 3, 0, tzinfo=UTC),
        status="partial",
    )

    last = query_last_hot_refresh(conn)
    assert last is not None
    assert last.replace(tzinfo=UTC) == datetime(2026, 5, 23, 3, 0, tzinfo=UTC)


def test_query_last_hot_refresh_ignores_other_sources(conn: duckdb.DuckDBPyConnection) -> None:
    _insert_run(
        conn,
        finished_at=datetime(2026, 5, 23, 4, 0, tzinfo=UTC),
        source="sackmann",
    )
    assert query_last_hot_refresh(conn, source="matchstat") is None


# ---------------------------------------------------------------------------
# Stale threshold
# ---------------------------------------------------------------------------


def test_is_data_stale_none_input_is_stale() -> None:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
    assert is_data_stale(None, now=now) is True


def test_is_data_stale_just_under_threshold_is_fresh() -> None:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
    last = now - timedelta(hours=STALE_THRESHOLD_HOURS - 0.5)
    assert is_data_stale(last, now=now) is False


def test_is_data_stale_just_over_threshold_is_stale() -> None:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
    last = now - timedelta(hours=STALE_THRESHOLD_HOURS + 0.5)
    assert is_data_stale(last, now=now) is True


def test_is_data_stale_naive_datetime_is_treated_as_utc() -> None:
    now = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
    last = datetime(2026, 5, 23, 11, 0)  # 1h ago, naive
    assert is_data_stale(last, now=now) is False


# ---------------------------------------------------------------------------
# Matchstat usage aggregator (Phase 6.2 follow-up)
# ---------------------------------------------------------------------------


def test_query_matchstat_usage_sums_ingestion_runs_and_quota(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """matchstat usage = SUM(ingestion_runs.requests_used for current
    month) + matchstat_quota.requests_used (per-prediction calls)."""
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)

    # Two May refresh runs at 13 and 15 credits.
    conn.execute(
        "INSERT INTO ingestion_runs (run_id, source, tour, started_at, status, "
        "rows_added, rows_skipped, rows_failed, requests_used, error_message, notes) "
        "VALUES ('r1', 'matchstat', NULL, ?, 'success', 0, 0, 0, 13, NULL, NULL)",
        [datetime(2026, 5, 24, 6, 0)],
    )
    conn.execute(
        "INSERT INTO ingestion_runs (run_id, source, tour, started_at, status, "
        "rows_added, rows_skipped, rows_failed, requests_used, error_message, notes) "
        "VALUES ('r2', 'matchstat', NULL, ?, 'success', 0, 0, 0, 15, NULL, NULL)",
        [datetime(2026, 5, 25, 6, 0)],
    )
    # A previous-month refresh — must NOT be counted.
    conn.execute(
        "INSERT INTO ingestion_runs (run_id, source, tour, started_at, status, "
        "rows_added, rows_skipped, rows_failed, requests_used, error_message, notes) "
        "VALUES ('r-old', 'matchstat', NULL, ?, 'success', 0, 0, 0, 99, NULL, NULL)",
        [datetime(2026, 4, 28, 6, 0)],
    )
    # A non-matchstat refresh in the same month — must NOT be counted.
    conn.execute(
        "INSERT INTO ingestion_runs (run_id, source, tour, started_at, status, "
        "rows_added, rows_skipped, rows_failed, requests_used, error_message, notes) "
        "VALUES ('r-odds', 'the_odds_api', NULL, ?, 'success', 0, 0, 0, 3, NULL, NULL)",
        [datetime(2026, 5, 25, 7, 0)],
    )
    # Per-prediction quota for the current month.
    conn.execute(
        "INSERT INTO matchstat_quota (month, requests_used, cap) VALUES ('2026-05', 7, 500)"
    )

    used, cap = query_matchstat_usage_month(conn, now=now)
    assert used == 13 + 15 + 7
    assert cap == 500


def test_query_matchstat_usage_empty_returns_zero(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    used, cap = query_matchstat_usage_month(conn, now=now)
    assert used == 0
    assert cap == 500


# ---------------------------------------------------------------------------
# Quota-exhaustion (429) detection
# ---------------------------------------------------------------------------


def _insert_run_with_error(
    conn: duckdb.DuckDBPyConnection,
    *,
    started_at: datetime,
    status: str,
    error_message: str | None,
    source: str = "matchstat",
) -> None:
    conn.execute(
        "INSERT INTO ingestion_runs (run_id, source, tour, started_at, status, "
        "rows_added, rows_skipped, rows_failed, requests_used, error_message, notes) "
        "VALUES (?, ?, NULL, ?, ?, 0, 0, 0, 1, ?, NULL)",
        [f"run-{started_at.isoformat()}", source, started_at, status, error_message],
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("MatchstatError: matchstat 429 at /atp/...: Too Many Requests", True),
        ("matchstat 429 at /wta/tournament/calendar/2026", True),
        ("Some Too Many Requests body", True),
        ("MatchstatError: matchstat 500 at /atp/...", False),
        ("ConnectionError: timed out", False),
        (None, False),
        ("", False),
    ],
)
def test_is_quota_error(message: str | None, expected: bool) -> None:
    assert is_quota_error(message) is expected


def test_query_last_hot_run_error_returns_message_when_latest_failed(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _insert_run_with_error(
        conn,
        started_at=datetime(2026, 6, 3, 21, 0),
        status="failed",
        error_message="MatchstatError: matchstat 429 at /atp/...",
    )
    err = query_last_hot_run_error(conn)
    assert err is not None
    assert is_quota_error(err) is True


def test_query_last_hot_run_error_none_when_latest_succeeded(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    # An older failure followed by a newer success → resolved, no error.
    _insert_run_with_error(
        conn,
        started_at=datetime(2026, 6, 3, 21, 0),
        status="failed",
        error_message="MatchstatError: matchstat 429 ...",
    )
    _insert_run_with_error(
        conn,
        started_at=datetime(2026, 6, 4, 21, 0),
        status="partial",
        error_message=None,
    )
    assert query_last_hot_run_error(conn) is None


def test_query_last_hot_run_error_ignores_other_sources(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _insert_run_with_error(
        conn,
        started_at=datetime(2026, 6, 4, 21, 0),
        status="failed",
        error_message="odds api 429",
        source="the_odds_api",
    )
    assert query_last_hot_run_error(conn, source="matchstat") is None
