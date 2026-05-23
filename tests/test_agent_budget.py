"""Unit tests for `AgentBudget` / `BudgetTracker`.

Covers each of the four hard limits from CLAUDE.md "Budget discipline":
iterations, total tokens, wall-clock, web searches. Both directions are
exercised: the `should_force_submit` predicate and the
`check_within_limits` overshoot raise.

`BudgetTracker.started_at` is overridden in the wall-clock test so we can
simulate elapsed seconds without sleeping in CI.
"""

from __future__ import annotations

import time

import pytest

from tennis_predictor.llm.agent import (
    AgentBudget,
    BudgetExceededError,
    BudgetTracker,
)
from tennis_predictor.llm.client import LLMResponse


def _response(
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cache_read: int = 0,
    cache_creation: int = 0,
    web_searches: int = 0,
) -> LLMResponse:
    return LLMResponse(
        stop_reason="end_turn",
        raw_content=[],
        text="",
        tool_uses=[],
        server_tool_uses=[],
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        web_search_count=web_searches,
        estimated_cost_usd=0.0,
        latency_ms=10,
    )


# ---------------------------------------------------------------------------


def test_register_iteration_accumulates_counters() -> None:
    """Budget is cost-weighted — counts tokens_in + tokens_out +
    cache_creation but NOT cache_read (the latter is billed at ~10%)."""
    tracker = BudgetTracker(AgentBudget())
    tracker.register_iteration(_response(tokens_in=100, tokens_out=50))
    tracker.register_iteration(
        _response(tokens_in=200, tokens_out=80, cache_read=500, cache_creation=300)
    )
    assert tracker.iterations_used == 2
    # cache_read=500 is intentionally excluded from the running total.
    assert tracker.tokens_used == 100 + 50 + 200 + 80 + 300


def test_should_force_submit_when_one_iteration_remaining() -> None:
    budget = AgentBudget(max_tool_iterations=2)
    tracker = BudgetTracker(budget)
    tracker.register_iteration(_response(tokens_in=10))
    assert tracker.should_force_submit() is True


def test_should_force_submit_when_token_buffer_hit() -> None:
    budget = AgentBudget(max_total_tokens=10_000)
    tracker = BudgetTracker(budget)
    # _FORCE_SUBMIT_TOKEN_BUFFER == 4000 — leave 3000 left to trigger.
    tracker.register_iteration(_response(tokens_in=7_000))
    assert tracker.should_force_submit() is True


def test_should_force_submit_when_web_searches_exhausted() -> None:
    """Phase 5.1: web_search count comes from agent-side reservation,
    not from `response.web_search_count` (that was a Phase 5 native-tool
    accounting path; the Phase 5.1 client tool reserves atomically)."""
    budget = AgentBudget(max_web_searches=2)
    tracker = BudgetTracker(budget)
    tracker.reserve_web_search()
    tracker.reserve_web_search()
    assert tracker.should_force_submit() is True


def test_should_force_submit_when_wall_clock_buffer_hit() -> None:
    budget = AgentBudget(max_wall_clock_seconds=20.0)
    tracker = BudgetTracker(budget)
    # _FORCE_SUBMIT_WALL_CLOCK_BUFFER == 15.0; consume 10s of wall clock
    # so only 10s remain, which is <= the buffer.
    tracker.started_at = time.monotonic() - 10.0
    assert tracker.should_force_submit() is True


def test_should_not_force_submit_with_full_budget() -> None:
    tracker = BudgetTracker(AgentBudget())
    assert tracker.should_force_submit() is False


def test_check_within_limits_raises_on_iteration_overflow() -> None:
    budget = AgentBudget(max_tool_iterations=1)
    tracker = BudgetTracker(budget)
    tracker.register_iteration(_response())
    with pytest.raises(BudgetExceededError, match="max_tool_iterations"):
        tracker.register_iteration(_response())


def test_check_within_limits_raises_on_token_overflow() -> None:
    budget = AgentBudget(max_total_tokens=1_000)
    tracker = BudgetTracker(budget)
    with pytest.raises(BudgetExceededError, match="max_total_tokens"):
        tracker.register_iteration(_response(tokens_in=2_000))


def test_check_within_limits_raises_on_web_search_overflow_via_manual_increment() -> None:
    """If the count somehow exceeds the cap (e.g. a future bug bypasses
    reserve), check_within_limits is the second wall and raises. Reach
    the overflow state manually since `reserve_web_search` would have
    refused at the 2nd call."""
    budget = AgentBudget(max_web_searches=1)
    tracker = BudgetTracker(budget)
    tracker.web_searches_used = 2  # simulate a leak past reserve
    with pytest.raises(BudgetExceededError, match="max_web_searches"):
        tracker.check_within_limits()


def test_check_within_limits_raises_on_wall_clock_overflow() -> None:
    budget = AgentBudget(max_wall_clock_seconds=1.0)
    tracker = BudgetTracker(budget)
    tracker.started_at = time.monotonic() - 5.0
    with pytest.raises(BudgetExceededError, match="max_wall_clock_seconds"):
        tracker.register_iteration(_response())


def test_remaining_helpers_decrement_from_full_budget() -> None:
    """`register_iteration` decrements iteration + token budgets;
    `reserve_web_search` is the path that decrements the web_search
    budget (Phase 5.1 separation)."""
    budget = AgentBudget(max_tool_iterations=6, max_total_tokens=30_000, max_web_searches=3)
    tracker = BudgetTracker(budget)
    tracker.register_iteration(_response(tokens_in=500, tokens_out=200))
    tracker.reserve_web_search()
    assert tracker.iterations_remaining() == 5
    assert tracker.tokens_remaining() == 30_000 - 700
    assert tracker.web_searches_remaining() == 2


# ---------------------------------------------------------------------------
# Phase 5.1 additions: register_tool_search / register_tool_fetch + pending
# cost accounting consumed by the next LLM call.
# ---------------------------------------------------------------------------


def test_reserve_web_search_increments_count_and_pending() -> None:
    tracker = BudgetTracker(AgentBudget())
    assert tracker.reserve_web_search() is True
    assert tracker.reserve_web_search() is True
    assert tracker.web_searches_used == 2
    assert tracker.pending_web_searches == 2
    tracker.register_tool_search(cost_usd=0.005)
    tracker.register_tool_search(cost_usd=0.005)
    assert tracker.pending_tool_cost_usd == pytest.approx(0.010)


def test_reserve_web_search_returns_false_when_exhausted() -> None:
    """Atomic reservation gate — prevents parallel dispatches from
    overshooting the cap."""
    tracker = BudgetTracker(AgentBudget(max_web_searches=2))
    assert tracker.reserve_web_search() is True
    assert tracker.reserve_web_search() is True
    assert tracker.reserve_web_search() is False
    assert tracker.web_searches_used == 2


def test_reserve_fetch_url_returns_false_when_exhausted() -> None:
    tracker = BudgetTracker(AgentBudget(max_fetch_urls=1))
    assert tracker.reserve_fetch_url() is True
    assert tracker.reserve_fetch_url() is False
    assert tracker.fetch_urls_used == 1


def test_refund_web_search_returns_slot() -> None:
    tracker = BudgetTracker(AgentBudget(max_web_searches=1))
    tracker.reserve_web_search()
    tracker.refund_web_search()
    assert tracker.web_searches_used == 0
    assert tracker.reserve_web_search() is True


def test_consume_pending_returns_and_resets() -> None:
    tracker = BudgetTracker(AgentBudget())
    tracker.reserve_web_search()
    tracker.register_tool_search(cost_usd=0.005)
    tracker.reserve_fetch_url()
    tracker.register_tool_fetch(cost_usd=0.005)
    cost, searches, fetches = tracker.consume_pending()
    assert cost == pytest.approx(0.010)
    assert searches == 1
    assert fetches == 1
    # Pending zeroed; cumulative usage counters untouched.
    assert tracker.pending_tool_cost_usd == 0.0
    assert tracker.pending_web_searches == 0
    assert tracker.pending_fetch_urls == 0
    assert tracker.web_searches_used == 1
    assert tracker.fetch_urls_used == 1


def test_fetch_urls_remaining_decrements() -> None:
    tracker = BudgetTracker(AgentBudget(max_fetch_urls=2))
    assert tracker.fetch_urls_remaining() == 2
    tracker.reserve_fetch_url()
    assert tracker.fetch_urls_remaining() == 1
