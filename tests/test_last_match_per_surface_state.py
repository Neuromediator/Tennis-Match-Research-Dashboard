"""Unit tests for the Phase 4.2 LastMatchPerSurfaceState state object.

Mirrors `test_last_match_state.py` plus the surface-dimension cases that
make the per-surface state object distinct from the global one:
independent tracking across (player, surface) keys, and the round-trip
through `last_match_per_surface_state` preserving the surface column.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.features.last_match_surface import LastMatchPerSurfaceState

P1 = "ATP_AAA"
P2 = "ATP_BBB"
HARD = "Hard"
CLAY = "Clay"
GRASS = "Grass"


@pytest.fixture
def conn(tmp_path: Path):
    db = duckdb.connect(str(tmp_path / "last_match_surface.duckdb"))
    schema.create_all_tables(db)
    yield db
    db.close()


def test_days_since_returns_none_when_pair_unseen() -> None:
    state = LastMatchPerSurfaceState()
    assert state.days_since(P1, HARD, date(2024, 6, 1)) is None


def test_single_update_then_snapshot() -> None:
    state = LastMatchPerSurfaceState()
    state.update(P1, HARD, date(2024, 6, 1))
    assert state.days_since(P1, HARD, date(2024, 6, 8)) == 7


def test_cap_at_365_days() -> None:
    state = LastMatchPerSurfaceState()
    state.update(P1, CLAY, date(2023, 1, 1))
    far_future = date(2023, 1, 1) + timedelta(days=500)
    assert state.days_since(P1, CLAY, far_future) == LastMatchPerSurfaceState.CAP_DAYS


def test_exactly_at_cap_returns_cap() -> None:
    state = LastMatchPerSurfaceState()
    state.update(P1, HARD, date(2023, 1, 1))
    one_year_later = date(2023, 1, 1) + timedelta(days=365)
    assert state.days_since(P1, HARD, one_year_later) == 365


def test_update_advances_when_later_date_arrives() -> None:
    """Chronological replay — each subsequent match on the same surface
    moves the recorded date forward."""
    state = LastMatchPerSurfaceState()
    state.update(P1, HARD, date(2024, 1, 1))
    state.update(P1, HARD, date(2024, 3, 1))
    state.update(P1, HARD, date(2024, 5, 1))
    assert state.last_date(P1, HARD) == date(2024, 5, 1)
    assert state.days_since(P1, HARD, date(2024, 5, 8)) == 7


def test_out_of_order_update_keeps_max_date() -> None:
    state = LastMatchPerSurfaceState()
    state.update(P1, HARD, date(2024, 5, 1))
    state.update(P1, HARD, date(2024, 1, 1))  # earlier — must NOT clobber
    assert state.last_date(P1, HARD) == date(2024, 5, 1)


def test_same_day_query_returns_zero() -> None:
    state = LastMatchPerSurfaceState()
    state.update(P1, HARD, date(2024, 6, 1))
    assert state.days_since(P1, HARD, date(2024, 6, 1)) == 0


def test_surfaces_tracked_independently_for_same_player() -> None:
    """The core Phase 4.2 invariant: a player's Hard gap and Clay gap are
    different cells in the dict. A recent Hard match must NOT reset the
    Clay gap."""
    state = LastMatchPerSurfaceState()
    state.update(P1, CLAY, date(2024, 1, 1))
    state.update(P1, HARD, date(2024, 5, 1))
    # Snapshot at 2024-06-01.
    assert state.days_since(P1, HARD, date(2024, 6, 1)) == 31
    # Clay still pinned to January — five months later, regardless of
    # the more-recent Hard activity.
    assert (
        state.days_since(P1, CLAY, date(2024, 6, 1)) == (date(2024, 6, 1) - date(2024, 1, 1)).days
    )
    # And Grass remains unseen.
    assert state.days_since(P1, GRASS, date(2024, 6, 1)) is None


def test_independent_players(conn: duckdb.DuckDBPyConnection) -> None:
    state = LastMatchPerSurfaceState()
    state.update(P1, HARD, date(2024, 1, 1))
    state.update(P2, HARD, date(2024, 5, 1))
    assert state.days_since(P1, HARD, date(2024, 6, 1)) == 152
    assert state.days_since(P2, HARD, date(2024, 6, 1)) == 31


def test_round_trip_via_db(conn: duckdb.DuckDBPyConnection) -> None:
    original = LastMatchPerSurfaceState()
    original.update(P1, HARD, date(2024, 1, 15))
    original.update(P1, CLAY, date(2024, 4, 22))
    original.update(P2, GRASS, date(2024, 6, 1))
    original.save_to_db(conn)

    reloaded = LastMatchPerSurfaceState.from_db(conn)
    assert len(reloaded) == 3
    assert reloaded.last_date(P1, HARD) == date(2024, 1, 15)
    assert reloaded.last_date(P1, CLAY) == date(2024, 4, 22)
    assert reloaded.last_date(P2, GRASS) == date(2024, 6, 1)
    # Surfaces that weren't saved come back as None.
    assert reloaded.last_date(P1, GRASS) is None


def test_save_overwrites_previous_snapshot(conn: duckdb.DuckDBPyConnection) -> None:
    """save_to_db should be a full replace, not an append."""
    first = LastMatchPerSurfaceState()
    first.update(P1, HARD, date(2024, 1, 1))
    first.save_to_db(conn)

    second = LastMatchPerSurfaceState()
    second.update(P2, CLAY, date(2024, 5, 1))
    second.save_to_db(conn)

    rows = conn.execute(
        "SELECT player_id, surface, last_match_date FROM last_match_per_surface_state"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0] == (P2, CLAY, date(2024, 5, 1))


def test_save_empty_clears_table(conn: duckdb.DuckDBPyConnection) -> None:
    LastMatchPerSurfaceState().save_to_db(conn)
    rows = conn.execute("SELECT COUNT(*) FROM last_match_per_surface_state").fetchone()
    assert rows is not None and rows[0] == 0


def test_contains_membership() -> None:
    state = LastMatchPerSurfaceState()
    state.update(P1, HARD, date(2024, 1, 1))
    assert (P1, HARD) in state
    assert (P1, CLAY) not in state
    assert (P2, HARD) not in state
