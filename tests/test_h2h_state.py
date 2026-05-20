"""Tests for the H2H state."""

from __future__ import annotations

from datetime import date

from tennis_predictor.features.h2h import H2HState


def test_never_met_returns_zero_zero_none() -> None:
    state = H2HState()
    assert state.snapshot("A", "B", date(2020, 1, 1)) == (0, 0, None)
    assert state.has_met("A", "B") is False


def test_single_match_recorded() -> None:
    state = H2HState()
    state.update("A", "B", date(2020, 1, 1))
    p1_wins, p2_wins, recency = state.snapshot("A", "B", date(2020, 1, 11))
    assert p1_wins == 1
    assert p2_wins == 0
    assert recency == 10
    assert state.has_met("A", "B") is True


def test_argument_order_mirrors_counts() -> None:
    """snapshot(A, B) and snapshot(B, A) must return mirrored counts —
    h2h is symmetric on the pair, asymmetric on the perspective."""
    state = H2HState()
    state.update("A", "B", date(2020, 1, 1))
    state.update("A", "B", date(2020, 2, 1))
    state.update("B", "A", date(2020, 3, 1))

    a_view = state.snapshot("A", "B", date(2020, 4, 1))
    b_view = state.snapshot("B", "A", date(2020, 4, 1))

    assert a_view == (2, 1, 31)  # A leads 2-1, last meeting 31 days ago
    assert b_view == (1, 2, 31)  # mirrored


def test_multiple_meetings_accumulate() -> None:
    state = H2HState()
    state.update("A", "B", date(2020, 1, 1))
    state.update("A", "B", date(2020, 2, 1))
    state.update("A", "B", date(2020, 3, 1))
    p1, p2, _ = state.snapshot("A", "B", date(2020, 4, 1))
    assert p1 == 3
    assert p2 == 0


def test_recency_days_computed_against_as_of_date() -> None:
    state = H2HState()
    state.update("A", "B", date(2020, 1, 1))
    _, _, recency = state.snapshot("A", "B", date(2020, 1, 8))
    assert recency == 7


def test_recency_uses_last_meeting_not_first() -> None:
    state = H2HState()
    state.update("A", "B", date(2019, 6, 1))
    state.update("A", "B", date(2020, 6, 1))
    _, _, recency = state.snapshot("A", "B", date(2020, 6, 30))
    assert recency == 29  # vs 2020-06-01, not 2019-06-01


def test_different_pairs_are_independent() -> None:
    state = H2HState()
    state.update("A", "B", date(2020, 1, 1))
    state.update("A", "C", date(2020, 1, 2))

    ab = state.snapshot("A", "B", date(2020, 2, 1))
    ac = state.snapshot("A", "C", date(2020, 2, 1))
    bc = state.snapshot("B", "C", date(2020, 2, 1))

    assert ab == (1, 0, 31)
    assert ac == (1, 0, 30)
    assert bc == (0, 0, None)


def test_snapshot_before_update_protocol() -> None:
    state = H2HState()
    state.update("A", "B", date(2020, 1, 1))

    pre = state.snapshot("A", "B", date(2020, 2, 1))
    state.update("B", "A", date(2020, 2, 1))
    post = state.snapshot("A", "B", date(2020, 2, 1))

    assert pre == (1, 0, 31)
    assert post == (1, 1, 0)
    assert pre != post


def test_key_canonicalization_does_not_double_count() -> None:
    """Whether the orchestrator calls update(A, B) or update(B, A), the
    pair must dedupe into the same canonical key."""
    state = H2HState()
    state.update("A", "B", date(2020, 1, 1))  # A wins
    state.update("B", "A", date(2020, 2, 1))  # B wins
    state.update("A", "B", date(2020, 3, 1))  # A wins again

    assert len(state) == 1  # one canonical pair, not three
    p1, p2, _ = state.snapshot("A", "B", date(2020, 4, 1))
    assert p1 == 2
    assert p2 == 1


def test_lex_ordering_of_ids_does_not_affect_counts() -> None:
    """The canonical key sorts player_ids lexicographically — counts must
    survive that internal reordering regardless of who's actually 'p1'."""
    state = H2HState()
    # "Z" > "A" — so canonical key is ("A", "Z"). Z wins.
    state.update("Z", "A", date(2020, 1, 1))
    a_view = state.snapshot("A", "Z", date(2020, 1, 2))
    z_view = state.snapshot("Z", "A", date(2020, 1, 2))
    assert a_view == (0, 1, 1)
    assert z_view == (1, 0, 1)
