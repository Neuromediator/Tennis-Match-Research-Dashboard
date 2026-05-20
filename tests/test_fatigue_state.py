"""Tests for fatigue state + set-count parser."""

from __future__ import annotations

from datetime import date

from tennis_predictor.features.fatigue import FatigueState, count_sets

# --- count_sets ---------------------------------------------------------------


def test_count_sets_two_set_match() -> None:
    assert count_sets("6-4 6-3") == 2


def test_count_sets_three_set_match() -> None:
    assert count_sets("6-4 3-6 6-2") == 3


def test_count_sets_with_tiebreaks() -> None:
    assert count_sets("7-6(5) 6-3") == 2
    assert count_sets("6-2 2-6 7-6(4)") == 3
    assert count_sets("7-6(8) 6-2") == 2


def test_count_sets_five_set_match() -> None:
    assert count_sets("6-4 4-6 6-3 4-6 7-5") == 5


def test_count_sets_retirement_counts_partial_set() -> None:
    """In '6-4 2-1 RET', the player played 1 complete set and started a 2nd —
    both should count toward fatigue (physical exertion happened)."""
    assert count_sets("6-4 2-1 RET") == 2
    assert count_sets("3-6 6-3 1-0 RET") == 3
    assert count_sets("6-1 RET") == 1


def test_count_sets_walkover_is_zero() -> None:
    """Walkover = no play happened."""
    assert count_sets("W/O") == 0


def test_count_sets_default_counts_partial() -> None:
    """Default with played games: e.g., '5-5 DEF' = 1 partial set."""
    assert count_sets("5-5 DEF") == 1
    assert count_sets("6-4 0-1 DEF") == 2


def test_count_sets_handles_null_and_empty() -> None:
    assert count_sets(None) == 0
    assert count_sets("") == 0


# --- FatigueState -------------------------------------------------------------


def test_default_state_returns_zeros() -> None:
    state = FatigueState()
    assert state.snapshot("X", date(2020, 1, 1)) == (0, 0)


def test_single_match_within_window() -> None:
    state = FatigueState()
    state.update("W", "L", sets_played=3, match_date=date(2020, 1, 1))
    m, s = state.snapshot("W", date(2020, 1, 5))
    assert m == 1
    assert s == 3


def test_match_at_exact_7_day_boundary_counts() -> None:
    """A match exactly 7 days ago is INCLUDED in matches_7d."""
    state = FatigueState()
    state.update("W", "L", sets_played=2, match_date=date(2020, 1, 1))
    m, s = state.snapshot("W", date(2020, 1, 8))  # delta = 7
    assert m == 1
    assert s == 2


def test_match_8_days_ago_excluded_from_matches_7d_but_in_sets_14d() -> None:
    """8 days ago drops from matches_7d but stays in sets_14d."""
    state = FatigueState()
    state.update("W", "L", sets_played=3, match_date=date(2020, 1, 1))
    m, s = state.snapshot("W", date(2020, 1, 9))  # delta = 8
    assert m == 0
    assert s == 3


def test_match_at_14_day_boundary_counts_in_sets() -> None:
    state = FatigueState()
    state.update("W", "L", sets_played=3, match_date=date(2020, 1, 1))
    _, s = state.snapshot("W", date(2020, 1, 15))  # delta = 14
    assert s == 3


def test_match_15_days_ago_excluded_entirely() -> None:
    state = FatigueState()
    state.update("W", "L", sets_played=3, match_date=date(2020, 1, 1))
    m, s = state.snapshot("W", date(2020, 1, 16))  # delta = 15
    assert m == 0
    assert s == 0


def test_both_winner_and_loser_pay_fatigue() -> None:
    """A 3-set match adds 3 sets to BOTH players' fatigue."""
    state = FatigueState()
    state.update("W", "L", sets_played=3, match_date=date(2020, 1, 1))
    assert state.snapshot("W", date(2020, 1, 2)) == (1, 3)
    assert state.snapshot("L", date(2020, 1, 2)) == (1, 3)


def test_multiple_matches_accumulate() -> None:
    """A tournament week: R32, R16, QF, SF over 5 days."""
    state = FatigueState()
    state.update("P", "OPP1", sets_played=2, match_date=date(2020, 1, 1))
    state.update("P", "OPP2", sets_played=3, match_date=date(2020, 1, 2))
    state.update("P", "OPP3", sets_played=2, match_date=date(2020, 1, 4))
    state.update("P", "OPP4", sets_played=3, match_date=date(2020, 1, 5))
    m, s = state.snapshot("P", date(2020, 1, 6))
    assert m == 4
    assert s == 2 + 3 + 2 + 3


def test_snapshot_before_update_protocol() -> None:
    state = FatigueState()
    state.update("P", "OPP", sets_played=2, match_date=date(2020, 1, 1))
    pre = state.snapshot("P", date(2020, 1, 5))
    state.update("P", "OPP2", sets_played=3, match_date=date(2020, 1, 5))
    post = state.snapshot("P", date(2020, 1, 5))
    assert pre == (1, 2)
    assert post == (2, 5)


def test_different_players_independent() -> None:
    state = FatigueState()
    state.update("A", "B", sets_played=2, match_date=date(2020, 1, 1))
    # C never played
    assert state.snapshot("A", date(2020, 1, 2)) == (1, 2)
    assert state.snapshot("C", date(2020, 1, 2)) == (0, 0)


def test_match_on_same_day_counts() -> None:
    """An earlier match on the same `as_of_date` (e.g., same-day doubles
    when a player also plays singles later) must count toward fatigue."""
    state = FatigueState()
    state.update("P", "OPP", sets_played=3, match_date=date(2020, 1, 1))
    m, s = state.snapshot("P", date(2020, 1, 1))  # delta = 0
    assert m == 1
    assert s == 3
