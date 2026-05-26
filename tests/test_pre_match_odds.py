"""Unit tests for the `pre_match_odds` persistence + matcher layer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from tennis_predictor.data.odds_api import AggregatedOdds, OddsApiQuotaExceeded
from tennis_predictor.data.pre_match_odds import (
    check_quota_or_raise,
    fixture_match_key,
    increment_quota,
    is_row_fresh,
    lookup_for_fixture,
    quota_status,
    upsert_aggregated,
)
from tennis_predictor.data.schema import create_all_tables


def _make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_all_tables(conn)
    return conn


def _make_row(
    *,
    tour: str = "ATP",
    a: str = "Jannik Sinner",
    b: str = "Novak Djokovic",
    sport_key: str = "tennis_atp_french_open",
    commence: datetime | None = None,
) -> AggregatedOdds:
    return AggregatedOdds(
        sport_key=sport_key,
        event_id="evt-1",
        tour=tour,
        player_a_name=a,
        player_b_name=b,
        commence_time_utc=commence or datetime(2026, 5, 26, 18, 0, tzinfo=UTC),
        median_odds_a=1.08,
        median_odds_b=10.5,
        best_odds_a=1.10,
        best_odds_b=11.0,
        median_implied_prob_a=0.92,
        median_implied_prob_b=0.08,
        books_count=3,
        pinnacle_odds_a=1.07,
        pinnacle_odds_b=11.0,
        pinnacle_implied_prob_a=0.92,
        pinnacle_implied_prob_b=0.08,
    )


def test_fixture_match_key_is_order_independent() -> None:
    when = datetime(2026, 5, 26, 18, 0, tzinfo=UTC)
    k1 = fixture_match_key("ATP", "Jannik Sinner", "Novak Djokovic", when)
    k2 = fixture_match_key("ATP", "Novak Djokovic", "Jannik Sinner", when)
    assert k1 == k2


def test_fixture_match_key_treats_hyphen_as_space() -> None:
    """Live observation: matchstat returns 'Felix Auger Aliassime' (no
    hyphen) while The Odds API returns 'Felix Auger-Aliassime' (hyphen).
    Similarly Pablo Carreno-Busta vs Pablo Carreno Busta. The key must
    collapse the hyphen so both providers land on the same row."""
    when = datetime(2026, 5, 26, 18, 0, tzinfo=UTC)
    matchstat_key = fixture_match_key("ATP", "Felix Auger Aliassime", "Daniel Altmaier", when)
    odds_key = fixture_match_key("ATP", "Felix Auger-Aliassime", "Daniel Altmaier", when)
    assert matchstat_key == odds_key
    # And the reverse case (matchstat hyphenated, odds not).
    ms_key2 = fixture_match_key("ATP", "Thanasi Kokkinakis", "Pablo Carreno-Busta", when)
    odds_key2 = fixture_match_key("ATP", "Thanasi Kokkinakis", "Pablo Carreno Busta", when)
    assert ms_key2 == odds_key2


def test_fixture_match_key_distinguishes_dates_and_tours() -> None:
    when = datetime(2026, 5, 26, 18, 0, tzinfo=UTC)
    later = datetime(2026, 5, 27, 18, 0, tzinfo=UTC)
    base = fixture_match_key("ATP", "A", "B", when)
    assert fixture_match_key("ATP", "A", "B", later) != base
    assert fixture_match_key("WTA", "A", "B", when) != base


def test_upsert_and_lookup_roundtrip() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    upsert_aggregated(conn, [_make_row()], now=now)
    row = lookup_for_fixture(
        conn,
        "ATP",
        "Jannik Sinner",
        "Novak Djokovic",
        datetime(2026, 5, 26, 18, 0, tzinfo=UTC),
    )
    assert row is not None
    assert row["median_odds_a"] == 1.08
    assert row["pinnacle_odds_a"] == 1.07
    assert row["books_count"] == 3
    assert row["source"] == "the_odds_api"


def test_lookup_is_order_independent() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    upsert_aggregated(conn, [_make_row()], now=now)
    # Query with players swapped — the keying is symmetric so it should still hit.
    row = lookup_for_fixture(
        conn,
        "ATP",
        "Novak Djokovic",
        "Jannik Sinner",
        datetime(2026, 5, 26, 18, 0, tzinfo=UTC),
    )
    assert row is not None


def test_upsert_updates_existing_row_on_conflict() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    upsert_aggregated(conn, [_make_row()], now=now)
    refreshed = _make_row().model_copy(update={"median_odds_a": 1.20, "books_count": 5})
    later = now + timedelta(hours=2)
    upsert_aggregated(conn, [refreshed], now=later)
    row = lookup_for_fixture(
        conn, "ATP", "Jannik Sinner", "Novak Djokovic", _make_row().commence_time_utc
    )
    assert row is not None
    assert row["median_odds_a"] == 1.20
    assert row["books_count"] == 5
    # Only one row total — confirm conflict path triggered.
    count = conn.execute("SELECT COUNT(*) FROM pre_match_odds").fetchone()
    assert count is not None
    assert count[0] == 1


def test_is_row_fresh_within_24h_else_stale() -> None:
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    row = {"fetched_at": now - timedelta(hours=4)}
    assert is_row_fresh(row, now=now) is True
    row_stale = {"fetched_at": now - timedelta(hours=25)}
    assert is_row_fresh(row_stale, now=now) is False
    assert is_row_fresh(None, now=now) is False


def test_quota_tracking_round_trip() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    used, cap = quota_status(conn, now=now)
    assert used == 0
    assert cap == 500
    increment_quota(conn, 3, now=now)
    increment_quota(conn, 2, now=now)
    used, _ = quota_status(conn, now=now)
    assert used == 5


def test_check_quota_or_raise_blocks_when_over_cap_buffer() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    # Buffer is 20 below cap 500 → trip at 480.
    increment_quota(conn, 480, now=now)
    with pytest.raises(OddsApiQuotaExceeded):
        check_quota_or_raise(conn, now=now)


def test_tavily_source_is_recorded_separately() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    upsert_aggregated(conn, [_make_row()], source="tavily", now=now)
    row = lookup_for_fixture(
        conn, "ATP", "Jannik Sinner", "Novak Djokovic", _make_row().commence_time_utc
    )
    assert row is not None
    assert row["source"] == "tavily"
