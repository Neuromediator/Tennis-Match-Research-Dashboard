"""Tests for serve/return rolling state."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tennis_predictor.features.serve_return import MatchStats, ServeReturnState


def _balanced_stats(
    w_first_won: int = 50,
    w_first_in: int = 70,
    l_first_won: int = 40,
    l_first_in: int = 70,
    *,
    w_second_won: int = 20,
    w_svpt: int = 100,
    l_second_won: int = 18,
    l_svpt: int = 100,
    w_bp_saved: int = 4,
    w_bp_faced: int = 6,
    l_bp_saved: int = 2,
    l_bp_faced: int = 6,
) -> MatchStats:
    """A reasonable default MatchStats for use across tests."""
    return MatchStats(
        w_first_in=w_first_in,
        w_first_won=w_first_won,
        w_second_won=w_second_won,
        w_svpt=w_svpt,
        w_bp_saved=w_bp_saved,
        w_bp_faced=w_bp_faced,
        l_first_in=l_first_in,
        l_first_won=l_first_won,
        l_second_won=l_second_won,
        l_svpt=l_svpt,
        l_bp_saved=l_bp_saved,
        l_bp_faced=l_bp_faced,
    )


def _seed(
    state: ServeReturnState,
    player_id: str,
    n: int,
    *,
    surface: str = "Hard",
    as_winner: bool = True,
    start: date = date(2020, 1, 1),
    stats_fn=None,
) -> None:
    """Helper: append n matches for `player_id` against throwaway opponents."""
    for i in range(n):
        s = stats_fn() if stats_fn else _balanced_stats()
        d = start + timedelta(days=i)
        if as_winner:
            state.update(player_id, f"OPP_{i}", surface, d, s)
        else:
            state.update(f"OPP_{i}", player_id, surface, d, s)


def test_default_state_returns_all_none() -> None:
    state = ServeReturnState()
    assert state.snapshot("X", "Hard") == (None, None, None, None)


def test_below_min_stat_matches_returns_none() -> None:
    state = ServeReturnState()
    _seed(state, "P", 4)  # below MIN_STAT_MATCHES=5
    assert state.snapshot("P", "Hard") == (None, None, None, None)


def test_null_stats_does_not_enter_window() -> None:
    """A match with stats=None must not count toward the stat-rich threshold."""
    state = ServeReturnState()
    for i in range(10):
        state.update("P", f"O_{i}", "Hard", date(2020, 1, 1) + timedelta(days=i), stats=None)
    assert state.stat_match_count("P", "Hard") == 0
    assert state.snapshot("P", "Hard") == (None, None, None, None)


def test_at_threshold_returns_rates() -> None:
    state = ServeReturnState()
    _seed(state, "P", 5)
    rates = state.snapshot("P", "Hard")
    for r in rates:
        assert r is not None


def test_first_serve_win_pct_aggregation() -> None:
    """5 identical matches each at (50/70) first-serve win rate -> 50/70 in window."""
    state = ServeReturnState()
    _seed(state, "P", 5)  # P is winner → uses w_first_*
    first_pct, _, _, _ = state.snapshot("P", "Hard")
    assert first_pct == pytest.approx(50 / 70)


def test_second_serve_win_pct_aggregation() -> None:
    """second_attempts = svpt - first_in = 100 - 70 = 30 per match.
    second_won = 20 per match. Rate = 20/30."""
    state = ServeReturnState()
    _seed(state, "P", 5)
    _, second_pct, _, _ = state.snapshot("P", "Hard")
    assert second_pct == pytest.approx(20 / 30)


def test_bp_saved_pct_aggregation() -> None:
    """w_bp_saved=4 of w_bp_faced=6 per match → 4/6."""
    state = ServeReturnState()
    _seed(state, "P", 5)
    _, _, bp_saved_pct, _ = state.snapshot("P", "Hard")
    assert bp_saved_pct == pytest.approx(4 / 6)


def test_bp_converted_pct_uses_opponent_bp_stats() -> None:
    """On the return side, P converts opp_bp_faced - opp_bp_saved per match.
    With loser's bp_saved=2, bp_faced=6 → P converts 4/6 (as winner here)."""
    state = ServeReturnState()
    _seed(state, "P", 5)  # P is winner; opp = loser; l_bp_faced=6, l_bp_saved=2
    _, _, _, bp_conv = state.snapshot("P", "Hard")
    assert bp_conv == pytest.approx(4 / 6)


def test_loser_records_loser_side_stats() -> None:
    """When P is the loser, we should see l_* stats applied to P, and
    opponent BP stats taken from w_*."""
    state = ServeReturnState()
    _seed(state, "P", 5, as_winner=False)
    first_pct, _, bp_saved_pct, bp_conv = state.snapshot("P", "Hard")
    # As loser: l_first_won/l_first_in = 40/70
    assert first_pct == pytest.approx(40 / 70)
    # As loser: l_bp_saved/l_bp_faced = 2/6
    assert bp_saved_pct == pytest.approx(2 / 6)
    # As loser, P's bp_converted comes from w_*: (6 - 4)/6 = 2/6
    assert bp_conv == pytest.approx(2 / 6)


def test_surface_filter_isolates_window() -> None:
    """Hard matches must not pollute the Clay-surface window."""
    state = ServeReturnState()
    _seed(state, "P", 5, surface="Hard")
    _seed(state, "P", 2, surface="Clay", start=date(2021, 1, 1))
    # Only 2 Clay matches → below threshold for Clay.
    assert state.snapshot("P", "Clay") == (None, None, None, None)
    # But Hard window has 5 → returns rates.
    assert all(r is not None for r in state.snapshot("P", "Hard"))


def test_window_caps_at_25_surface_matches() -> None:
    """When there are 30 matches on the same surface, only the LAST 25 are
    aggregated."""

    # Stage 1: 5 matches with first_won = 70 (every 1st serve won).
    # Stage 2: 25 matches with first_won = 0.
    # Window of 25 should see only stage 2 → first_serve_win_pct = 0.
    def hot_stats() -> MatchStats:
        return _balanced_stats(w_first_won=70)

    def cold_stats() -> MatchStats:
        return _balanced_stats(w_first_won=0)

    state = ServeReturnState()
    _seed(state, "P", 5, stats_fn=hot_stats)
    _seed(state, "P", 25, stats_fn=cold_stats, start=date(2020, 7, 1))

    first_pct, _, _, _ = state.snapshot("P", "Hard")
    assert first_pct == pytest.approx(0.0)


def test_snapshot_before_update_protocol() -> None:
    state = ServeReturnState()
    _seed(state, "P", 5)
    pre = state.snapshot("P", "Hard")

    # Now P plays a 6th match (as winner) with VERY different stats.
    state.update(
        "P",
        "OPP",
        "Hard",
        date(2020, 6, 1),
        _balanced_stats(w_first_won=10, w_first_in=20),
    )
    post = state.snapshot("P", "Hard")

    assert pre != post


def test_division_by_zero_yields_none() -> None:
    """If 5 stat-rich matches somehow all had first_in=0 (pathological), the
    rate must come back as None rather than crash."""
    state = ServeReturnState()

    def zero_first_stats() -> MatchStats:
        return MatchStats(
            w_first_in=0,
            w_first_won=0,
            w_second_won=10,
            w_svpt=20,
            w_bp_saved=0,
            w_bp_faced=0,
            l_first_in=10,
            l_first_won=5,
            l_second_won=2,
            l_svpt=20,
            l_bp_saved=0,
            l_bp_faced=2,
        )

    _seed(state, "P", 5, stats_fn=zero_first_stats)
    first_pct, second_pct, bp_saved_pct, _ = state.snapshot("P", "Hard")
    assert first_pct is None  # 0/0 → None
    # Second has attempts (svpt - first_in = 20 - 0 = 20) → real rate.
    assert second_pct == pytest.approx(10 / 20)
    # BP saved: 0/0 → None
    assert bp_saved_pct is None


def test_different_players_independent() -> None:
    state = ServeReturnState()
    _seed(state, "A", 5)
    # B never played
    assert all(r is None for r in state.snapshot("B", "Hard"))
    assert all(r is not None for r in state.snapshot("A", "Hard"))
