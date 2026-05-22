"""Unit tests for the Phase 4.1 LastMatchState state object.

Covers: cold start (None), single update, the 365-day cap, monotonic
behavior under chronological replay, defensive same-day query, and the
DB save/load round-trip.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.features.last_match import LastMatchState

P1 = "ATP_AAA"
P2 = "ATP_BBB"


@pytest.fixture
def conn(tmp_path: Path):
    db = duckdb.connect(str(tmp_path / "last_match.duckdb"))
    schema.create_all_tables(db)
    yield db
    db.close()


def test_days_since_returns_none_when_player_unseen() -> None:
    state = LastMatchState()
    assert state.days_since(P1, date(2024, 6, 1)) is None


def test_single_update_then_snapshot() -> None:
    state = LastMatchState()
    state.update(P1, date(2024, 6, 1))
    assert state.days_since(P1, date(2024, 6, 8)) == 7


def test_cap_at_365_days() -> None:
    state = LastMatchState()
    state.update(P1, date(2023, 1, 1))
    # 500 days later — must clamp to CAP_DAYS = 365.
    far_future = date(2023, 1, 1) + timedelta(days=500)
    assert state.days_since(P1, far_future) == LastMatchState.CAP_DAYS


def test_exactly_at_cap_returns_cap() -> None:
    """Boundary: exactly 365 days returns 365 (not capped down)."""
    state = LastMatchState()
    state.update(P1, date(2023, 1, 1))
    one_year_later = date(2023, 1, 1) + timedelta(days=365)
    assert state.days_since(P1, one_year_later) == 365


def test_update_advances_when_later_date_arrives() -> None:
    """Chronological replay — each subsequent match moves the recorded
    date forward."""
    state = LastMatchState()
    state.update(P1, date(2024, 1, 1))
    state.update(P1, date(2024, 3, 1))
    state.update(P1, date(2024, 5, 1))
    assert state.last_date(P1) == date(2024, 5, 1)
    assert state.days_since(P1, date(2024, 5, 8)) == 7


def test_out_of_order_update_keeps_max_date() -> None:
    """Defensive: if a caller hands updates out of order, we keep the
    maximum date — the chronological-replay contract isn't strictly
    enforceable from inside the state."""
    state = LastMatchState()
    state.update(P1, date(2024, 5, 1))
    state.update(P1, date(2024, 1, 1))  # earlier — must NOT clobber
    assert state.last_date(P1) == date(2024, 5, 1)


def test_same_day_query_returns_zero() -> None:
    """Snapshot-after-update on the same day clamps to 0 rather than going
    negative. Indicates a caller bug, but the state shouldn't crash."""
    state = LastMatchState()
    state.update(P1, date(2024, 6, 1))
    assert state.days_since(P1, date(2024, 6, 1)) == 0


def test_independent_players(conn: duckdb.DuckDBPyConnection) -> None:
    state = LastMatchState()
    state.update(P1, date(2024, 1, 1))
    state.update(P2, date(2024, 5, 1))
    assert state.days_since(P1, date(2024, 6, 1)) == 152
    assert state.days_since(P2, date(2024, 6, 1)) == 31


def test_round_trip_via_db(conn: duckdb.DuckDBPyConnection) -> None:
    original = LastMatchState()
    original.update(P1, date(2024, 1, 15))
    original.update(P2, date(2024, 4, 22))
    original.save_to_db(conn)

    reloaded = LastMatchState.from_db(conn)
    assert len(reloaded) == 2
    assert reloaded.last_date(P1) == date(2024, 1, 15)
    assert reloaded.last_date(P2) == date(2024, 4, 22)


def test_save_overwrites_previous_snapshot(conn: duckdb.DuckDBPyConnection) -> None:
    """save_to_db should be a full replace, not an append."""
    first = LastMatchState()
    first.update(P1, date(2024, 1, 1))
    first.save_to_db(conn)

    second = LastMatchState()
    second.update(P2, date(2024, 5, 1))
    second.save_to_db(conn)

    rows = conn.execute("SELECT player_id, last_match_date FROM last_match_state").fetchall()
    assert len(rows) == 1
    assert rows[0] == (P2, date(2024, 5, 1))


def test_save_empty_clears_table(conn: duckdb.DuckDBPyConnection) -> None:
    LastMatchState().save_to_db(conn)
    rows = conn.execute("SELECT COUNT(*) FROM last_match_state").fetchone()
    assert rows is not None and rows[0] == 0


def test_contains_membership() -> None:
    state = LastMatchState()
    state.update(P1, date(2024, 1, 1))
    assert P1 in state
    assert P2 not in state
