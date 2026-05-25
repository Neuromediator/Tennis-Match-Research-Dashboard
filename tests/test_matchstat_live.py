"""Unit tests for the matchstat live-fetcher (Phase 6.1).

Covers the four code paths the caller cares about:
- Cache hit (fresh) — no API call, no quota increment.
- Cache miss + successful live fetch — payload returned, cache written,
  quota incremented.
- Cache miss + quota already exhausted — `MatchstatBudgetExceeded`
  raised BEFORE the API call.
- Cache miss + live fetch returns 429 — `MatchstatBudgetExceeded`
  raised with the current quota counter.

Plus invariants on the H2H canonicalisation (order-agnostic cache key)
and the quota row bootstrap.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import duckdb
import httpx
import pytest

from tennis_predictor.data.matchstat import BASE_URL, MatchstatClient
from tennis_predictor.data.matchstat_live import (
    CACHE_TTL,
    QUOTA_BUFFER,
    MatchstatBudgetExceeded,
    _canonical_h2h_pair,
    fetch_h2h,
    fetch_player_past_matches,
    quota_status,
)
from tennis_predictor.data.schema import create_all_tables


def _make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_all_tables(conn)
    return conn


def _make_client(handler: Any) -> MatchstatClient:
    transport = httpx.MockTransport(handler)
    inner = httpx.Client(base_url=BASE_URL, transport=transport, headers={})
    return MatchstatClient(api_key="test-key", client=inner)


def _rich_match(match_id: str, *, p1: int = 100, p2: int = 200) -> dict[str, Any]:
    return {
        "id": match_id,
        "date": "2026-05-20T12:00:00.000Z",
        "roundId": 1,
        "round": {"id": 1, "name": "R32"},
        "tournamentId": 999,
        "tournament": {
            "id": 999,
            "name": "Test Open",
            "court": {"id": 1, "name": "Clay"},
            "rank": {"id": 1, "name": "ATP 250"},
        },
        "player1Id": p1,
        "player2Id": p2,
        "player1": {"id": p1, "name": "A"},
        "player2": {"id": p2, "name": "B"},
        "matchWinner": 1,
        "result": "6-4 6-3",
        "bestOf": 3,
        "odd1": "1.5",
        "odd2": "2.5",
    }


# ---------------------------------------------------------------------------
# Fresh cache hit — no API call.
# ---------------------------------------------------------------------------


def test_player_past_matches_cache_hit_skips_api() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)

    def fail_if_called(_: httpx.Request) -> httpx.Response:
        raise AssertionError("API must not be called on cache hit")

    # Seed cache by doing one fetch through a "real" client.
    seed_calls = 0

    def seed_handler(_: httpx.Request) -> httpx.Response:
        nonlocal seed_calls
        seed_calls += 1
        return httpx.Response(200, json={"data": [_rich_match("1")], "hasNextPage": False})

    with _make_client(seed_handler) as client:
        first = fetch_player_past_matches(conn, "atp", 100, client=client, now=now)
    assert seed_calls == 1
    assert len(first.data) == 1

    # Immediately re-fetch — must hit cache.
    with _make_client(fail_if_called) as client:
        second = fetch_player_past_matches(
            conn, "atp", 100, client=client, now=now + timedelta(hours=1)
        )
    assert len(second.data) == 1
    assert second.data[0].id == "1"

    # Quota incremented exactly once (the seed call), not twice.
    used, _ = quota_status(conn, now=now)
    assert used == 1


def test_player_past_matches_cache_expires_after_ttl() -> None:
    """After CACHE_TTL the cache must be treated as stale and re-fetched."""
    conn = _make_conn()
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": [_rich_match(f"M{calls}")], "hasNextPage": False})

    with _make_client(handler) as client:
        fetch_player_past_matches(conn, "atp", 100, client=client, now=now)
    # Just past the TTL.
    with _make_client(handler) as client:
        fetch_player_past_matches(
            conn,
            "atp",
            100,
            client=client,
            now=now + CACHE_TTL + timedelta(minutes=1),
        )
    assert calls == 2


# ---------------------------------------------------------------------------
# Cache miss + successful live fetch.
# ---------------------------------------------------------------------------


def test_player_past_matches_writes_cache_and_increments_quota() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/player/past-matches/100" in request.url.path
        return httpx.Response(200, json={"data": [_rich_match("X")], "hasNextPage": False})

    with _make_client(handler) as client:
        page = fetch_player_past_matches(conn, "atp", 100, client=client, now=now)

    assert page.data[0].id == "X"
    cached = conn.execute(
        "SELECT fetched_at, payload FROM matchstat_player_recent_cache "
        "WHERE tour=? AND player_id=?",
        ["atp", 100],
    ).fetchone()
    assert cached is not None

    used, cap = quota_status(conn, now=now)
    assert used == 1
    assert cap == 500


# ---------------------------------------------------------------------------
# Quota exhaustion — pre-flight raise.
# ---------------------------------------------------------------------------


def test_quota_exhaustion_raises_before_api_call() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
    # Pre-fill the quota row at the buffered cap (500 - 20 = 480).
    conn.execute(
        "INSERT INTO matchstat_quota (month, requests_used, cap) VALUES (?, ?, ?)",
        ["2026-05", 500 - QUOTA_BUFFER, 500],
    )

    def fail_if_called(_: httpx.Request) -> httpx.Response:
        raise AssertionError("API must not be called when quota exhausted")

    with (
        _make_client(fail_if_called) as client,
        pytest.raises(MatchstatBudgetExceeded) as exc_info,
    ):
        fetch_player_past_matches(conn, "atp", 100, client=client, now=now)
    assert exc_info.value.requests_used == 480
    assert exc_info.value.cap == 500


# ---------------------------------------------------------------------------
# Live 429 -> BudgetExceeded.
# ---------------------------------------------------------------------------


def test_live_429_translates_to_budget_exceeded() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"message":"Too Many Requests"}')

    with _make_client(handler) as client, pytest.raises(MatchstatBudgetExceeded):
        fetch_player_past_matches(conn, "atp", 100, client=client, now=now)

    # Quota was NOT incremented (the call failed before increment).
    used, _ = quota_status(conn, now=now)
    assert used == 0


# ---------------------------------------------------------------------------
# H2H canonicalisation invariants.
# ---------------------------------------------------------------------------


def test_h2h_canonical_pair_is_order_independent() -> None:
    assert _canonical_h2h_pair(5, 9) == (5, 9)
    assert _canonical_h2h_pair(9, 5) == (5, 9)
    assert _canonical_h2h_pair(7, 7) == (7, 7)


def test_h2h_cache_serves_both_orientations_from_one_row() -> None:
    """Fetching A vs B then B vs A must produce exactly one API call —
    the second lookup is a cache hit even though arguments are swapped."""
    conn = _make_conn()
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"data": [_rich_match("H1", p1=100, p2=200)], "hasNextPage": False}
        )

    with _make_client(handler) as client:
        fetch_h2h(conn, "atp", 100, 200, client=client, now=now)
        fetch_h2h(conn, "atp", 200, 100, client=client, now=now + timedelta(hours=1))

    assert calls == 1
    used, _ = quota_status(conn, now=now)
    assert used == 1


# ---------------------------------------------------------------------------
# Quota bootstrap.
# ---------------------------------------------------------------------------


def test_quota_status_creates_row_on_first_call() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
    used, cap = quota_status(conn, now=now)
    assert used == 0
    assert cap == 500
    # Row must persist for inspection by Dashboard SQL.
    row = conn.execute("SELECT month, requests_used, cap FROM matchstat_quota").fetchone()
    assert row == ("2026-05", 0, 500)


def test_quota_uses_separate_buckets_per_month() -> None:
    conn = _make_conn()
    may = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
    june = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [_rich_match("X")], "hasNextPage": False})

    with _make_client(handler) as client:
        fetch_player_past_matches(conn, "atp", 100, client=client, now=may)
        fetch_player_past_matches(conn, "atp", 101, client=client, now=june)

    may_used, _ = quota_status(conn, now=may)
    june_used, _ = quota_status(conn, now=june)
    assert may_used == 1
    assert june_used == 1
