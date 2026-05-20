"""Tests for the surface-Elo state.

Covers:
- Default rating + zero-history behavior.
- Standard Elo update math (known input → known output).
- Zero-sum invariant per match.
- Snapshot-before-update protocol (the central anti-leakage rule).
- Multi-surface independence.
- DB round-trip via `from_db` / `save_to_db`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.features.elo import EloState, MatchOutcome


@pytest.fixture
def fresh_db(tmp_path: Path):
    db_path = tmp_path / "elo_test.duckdb"
    conn = duckdb.connect(str(db_path))
    schema.create_all_tables(conn)
    yield conn
    conn.close()


def test_unseen_pair_returns_default_rating() -> None:
    state = EloState()
    assert state.get("ATP_X", "Hard") == 1500.0
    assert state.matches_played("ATP_X", "Hard") == 0


def test_expected_score_symmetry() -> None:
    """E(a, b) + E(b, a) == 1 — fundamental Elo identity."""
    e_ab = EloState.expected_score(1700.0, 1500.0)
    e_ba = EloState.expected_score(1500.0, 1700.0)
    assert e_ab + e_ba == pytest.approx(1.0)


def test_known_update_value() -> None:
    """A 1500 vs 1500 match: E=0.5 for both, winner gains K*(1-0.5)=16."""
    state = EloState()
    state.update("W", "L", "Hard", date(2020, 1, 1))
    assert state.get("W", "Hard") == pytest.approx(1516.0)
    assert state.get("L", "Hard") == pytest.approx(1484.0)


def test_update_is_zero_sum() -> None:
    """Winner's gain equals loser's loss — Elo conservation."""
    state = EloState()
    state.update("A", "B", "Clay", date(2021, 6, 1))
    delta_a = state.get("A", "Clay") - 1500.0
    delta_b = 1500.0 - state.get("B", "Clay")
    assert delta_a == pytest.approx(delta_b)


def test_underdog_win_yields_larger_swing() -> None:
    """When the lower-rated player wins, their gain > what the favorite
    would have gained beating them."""
    state = EloState()
    # Seed unequal ratings via a few prior matches.
    for _ in range(5):
        state.update("FAV", "UND", "Hard", date(2020, 1, 1))
    r_fav_pre = state.get("FAV", "Hard")
    r_und_pre = state.get("UND", "Hard")
    assert r_fav_pre > r_und_pre  # sanity

    # Underdog wins → big swing.
    state.update("UND", "FAV", "Hard", date(2020, 6, 1))
    und_gain = state.get("UND", "Hard") - r_und_pre
    assert und_gain > 16.0  # > "default" K*0.5 because they were not favored


def test_matches_played_increments() -> None:
    state = EloState()
    state.update("A", "B", "Hard", date(2020, 1, 1))
    state.update("A", "B", "Hard", date(2020, 2, 1))
    state.update("A", "C", "Hard", date(2020, 3, 1))
    assert state.matches_played("A", "Hard") == 3
    assert state.matches_played("B", "Hard") == 2
    assert state.matches_played("C", "Hard") == 1


def test_surfaces_are_independent() -> None:
    """A win on Hard must not move the Clay rating, and vice-versa."""
    state = EloState()
    state.update("A", "B", "Hard", date(2020, 1, 1))
    assert state.get("A", "Clay") == 1500.0
    assert state.get("A", "IHard") == 1500.0
    assert state.get("A", "Grass") == 1500.0
    assert state.get("A", "Hard") > 1500.0


def test_snapshot_before_update_protocol() -> None:
    """The CRITICAL contract: when reading a snapshot BEFORE applying the
    match result, the rating must reflect pre-match state. This is what the
    orchestrator does, and what feature-leakage tests assert."""
    state = EloState()
    # Warm up so the rating is not the default.
    state.update("A", "B", "Hard", date(2020, 1, 1))
    pre = state.get("A", "Hard")

    # Snapshot first, then update with the next match.
    snapshot_value = state.get("A", "Hard")
    state.update("A", "B", "Hard", date(2020, 2, 1))
    post = state.get("A", "Hard")

    assert snapshot_value == pre
    assert post != pre  # state actually changed
    assert snapshot_value != post  # snapshot did not see the future update


def test_roll_forward_applies_in_order() -> None:
    """roll_forward must walk the list in caller's order, sequentially."""
    state_seq = EloState()
    state_seq.update("A", "B", "Hard", date(2020, 1, 1))
    state_seq.update("A", "C", "Hard", date(2020, 2, 1))

    state_batch = EloState()
    state_batch.roll_forward(
        [
            MatchOutcome("A", "B", "Hard", date(2020, 1, 1)),
            MatchOutcome("A", "C", "Hard", date(2020, 2, 1)),
        ]
    )

    assert state_seq.get("A", "Hard") == pytest.approx(state_batch.get("A", "Hard"))
    assert state_seq.get("B", "Hard") == pytest.approx(state_batch.get("B", "Hard"))
    assert state_seq.get("C", "Hard") == pytest.approx(state_batch.get("C", "Hard"))


def test_save_and_load_round_trip(fresh_db: duckdb.DuckDBPyConnection) -> None:
    state = EloState()
    state.update("ATP_1", "ATP_2", "Hard", date(2022, 3, 1))
    state.update("ATP_1", "ATP_3", "Clay", date(2022, 4, 1))
    state.update("ATP_2", "ATP_3", "IHard", date(2022, 11, 1))

    state.save_to_db(fresh_db)

    reloaded = EloState.from_db(fresh_db)

    # Every key from the original state survives the round-trip with the
    # same values — checked via the public surface.
    for surface in ("Hard", "Clay", "IHard"):
        for pid in ("ATP_1", "ATP_2", "ATP_3"):
            assert reloaded.get(pid, surface) == pytest.approx(state.get(pid, surface))
            assert reloaded.matches_played(pid, surface) == state.matches_played(pid, surface)
            assert reloaded.last_updated(pid, surface) == state.last_updated(pid, surface)

    assert len(reloaded) == len(state)


def test_save_replaces_previous_contents(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Saving must be an atomic full-snapshot replacement — not an append."""
    s1 = EloState()
    s1.update("A", "B", "Hard", date(2020, 1, 1))
    s1.save_to_db(fresh_db)

    s2 = EloState()
    s2.update("C", "D", "Clay", date(2021, 1, 1))
    s2.save_to_db(fresh_db)

    rows = fresh_db.execute("SELECT player_id FROM elo_state").fetchall()
    player_ids = {r[0] for r in rows}
    assert player_ids == {"C", "D"}  # 'A' and 'B' from s1 are gone


def test_empty_state_save_clears_table(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Saving an empty state must wipe the table (and not crash)."""
    s = EloState()
    s.update("A", "B", "Hard", date(2020, 1, 1))
    s.save_to_db(fresh_db)
    assert fresh_db.execute("SELECT count(*) FROM elo_state").fetchone() == (2,)

    EloState().save_to_db(fresh_db)
    assert fresh_db.execute("SELECT count(*) FROM elo_state").fetchone() == (0,)
