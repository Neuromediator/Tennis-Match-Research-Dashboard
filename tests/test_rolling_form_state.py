"""Tests for the rolling-form state.

Covers:
- Default-state (no history) → (None, None).
- Below-threshold windows → None (sparse data is noise).
- Window slicing: last 10 (any surface) and last 25 (surface-specific).
- Surface filtering: matches on other surfaces don't pollute the
  surface-specific window.
- Snapshot-before-update protocol.
- Both winner and loser are tracked (loser keeps `won=False`).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tennis_predictor.features.rolling_form import RollingFormState


def _play_matches(
    state: RollingFormState,
    player_id: str,
    surface: str,
    results: list[bool],
    start_date: date = date(2020, 1, 1),
) -> None:
    """Helper: append `n` matches where `results[i]=True` means player wins
    match `i`. Pairs against a throwaway opponent."""
    for i, won in enumerate(results):
        match_date = start_date + timedelta(days=i)
        if won:
            state.update(player_id, f"OPP_{i}", surface, match_date)
        else:
            state.update(f"OPP_{i}", player_id, surface, match_date)


def test_default_state_returns_none_for_both() -> None:
    state = RollingFormState()
    assert state.snapshot("ATP_X", "Hard") == (None, None)


def test_below_threshold_returns_none() -> None:
    """Fewer than MIN_MATCHES_FOR_RATE (3) matches → None for both rates."""
    state = RollingFormState()
    _play_matches(state, "P", "Hard", [True, True])
    any_rate, surf_rate = state.snapshot("P", "Hard")
    assert any_rate is None
    assert surf_rate is None


def test_at_threshold_returns_rate() -> None:
    state = RollingFormState()
    _play_matches(state, "P", "Hard", [True, True, True])
    any_rate, surf_rate = state.snapshot("P", "Hard")
    assert any_rate == pytest.approx(1.0)
    assert surf_rate == pytest.approx(1.0)


def test_last_10_any_window() -> None:
    """When there are more than 10 matches, only the last 10 count."""
    state = RollingFormState()
    # 5 losses, then 10 wins: last_10 = all wins → 1.0
    _play_matches(state, "P", "Hard", [False] * 5 + [True] * 10)
    any_rate, _ = state.snapshot("P", "Hard")
    assert any_rate == pytest.approx(1.0)


def test_last_25_surface_window() -> None:
    """Surface window takes the last 25 entries OF THAT SURFACE."""
    state = RollingFormState()
    # 30 matches on Clay: alternating W/L → last 25 give 12 or 13 wins
    pattern = [i % 2 == 0 for i in range(30)]
    _play_matches(state, "P", "Clay", pattern)
    _, surf_rate = state.snapshot("P", "Clay")
    assert surf_rate is not None
    # last 25 of alternating starting from index 5: pattern[5:30]
    expected = sum(pattern[5:30]) / 25
    assert surf_rate == pytest.approx(expected)


def test_surface_filter_excludes_other_surfaces() -> None:
    """Hard matches must not influence the Clay surface-window."""
    state = RollingFormState()
    _play_matches(state, "P", "Hard", [True] * 10, start_date=date(2020, 1, 1))
    _play_matches(state, "P", "Clay", [False] * 5, start_date=date(2020, 6, 1))

    any_rate, clay_rate = state.snapshot("P", "Clay")
    # Last 10 ANY = last 5 Clay (0%) + last 5 Hard (100%) = 50%
    assert any_rate == pytest.approx(0.5)
    # Clay only: 5 losses → 0%
    assert clay_rate == pytest.approx(0.0)


def test_surface_window_below_threshold_is_none_even_when_any_is_set() -> None:
    """A player can have a valid `any` rate but only 1 Clay match — the
    surface window stays None."""
    state = RollingFormState()
    _play_matches(state, "P", "Hard", [True] * 10, start_date=date(2020, 1, 1))
    _play_matches(state, "P", "Clay", [True], start_date=date(2020, 6, 1))

    any_rate, clay_rate = state.snapshot("P", "Clay")
    assert any_rate is not None
    assert clay_rate is None  # only 1 Clay match — below threshold


def test_loser_history_is_tracked() -> None:
    """The losing player's history must include the match with `won=False`."""
    state = RollingFormState()
    state.update("W", "L", "Hard", date(2020, 1, 1))
    state.update("W", "L", "Hard", date(2020, 1, 2))
    state.update("L", "W", "Hard", date(2020, 1, 3))  # L wins one

    # Wait — that flips W and L roles in match 3. Let's check L's history.
    # L was loser in matches 1, 2 and winner in match 3 → 1/3 win rate.
    any_rate, _ = state.snapshot("L", "Hard")
    assert any_rate == pytest.approx(1 / 3)


def test_snapshot_before_update_protocol() -> None:
    state = RollingFormState()
    _play_matches(state, "P", "Hard", [True, True, True])
    pre_any, pre_surf = state.snapshot("P", "Hard")
    assert pre_any == pytest.approx(1.0)
    assert pre_surf == pytest.approx(1.0)

    # Now P loses a 4th match. Snapshot BEFORE that update should still be 1.0.
    snap_any_pre, _ = state.snapshot("P", "Hard")
    state.update("OPP", "P", "Hard", date(2020, 1, 10))  # P loses
    snap_any_post, _ = state.snapshot("P", "Hard")

    assert snap_any_pre == pytest.approx(1.0)
    assert snap_any_post == pytest.approx(3 / 4)
    assert snap_any_pre != snap_any_post


def test_different_players_are_independent() -> None:
    state = RollingFormState()
    _play_matches(state, "A", "Hard", [True, True, True])
    _play_matches(state, "B", "Hard", [False, False, False])

    a_any, _ = state.snapshot("A", "Hard")
    b_any, _ = state.snapshot("B", "Hard")
    assert a_any == pytest.approx(1.0)
    assert b_any == pytest.approx(0.0)


def test_matches_played_counts_total() -> None:
    state = RollingFormState()
    _play_matches(state, "P", "Hard", [True] * 5)
    _play_matches(state, "P", "Clay", [False] * 3, start_date=date(2020, 6, 1))
    assert state.matches_played("P") == 8
    assert state.matches_played("OTHER") == 0


def test_window_at_exactly_25_surface_uses_all() -> None:
    state = RollingFormState()
    # Exactly 25 Clay matches, all wins → 1.0
    _play_matches(state, "P", "Clay", [True] * 25)
    _, surf_rate = state.snapshot("P", "Clay")
    assert surf_rate == pytest.approx(1.0)


def test_above_25_surface_takes_only_last_25() -> None:
    state = RollingFormState()
    # 30 matches: first 5 are wins, next 25 are losses → surface rate = 0.0
    _play_matches(state, "P", "Clay", [True] * 5 + [False] * 25)
    _, surf_rate = state.snapshot("P", "Clay")
    assert surf_rate == pytest.approx(0.0)
