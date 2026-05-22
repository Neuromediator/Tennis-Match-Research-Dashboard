"""Unit tests for `llm/cost.py`.

The Anthropic pricing table is a moving target; these tests pin the
arithmetic, not the exact dollar value. If the rate card changes, update
the tests and the table together.
"""

from __future__ import annotations

import math

import pytest

from tennis_predictor.llm.cost import (
    ANTHROPIC_PRICING,
    WEB_SEARCH_USD_PER_CALL,
    cache_hit_rate,
    estimate_call_cost,
)


def test_estimate_call_cost_matches_rate_card_math() -> None:
    cost = estimate_call_cost(
        model="claude-sonnet-4-6",
        tokens_in=1_000_000,
        tokens_out=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        web_search_count=0,
    )
    assert math.isclose(cost, ANTHROPIC_PRICING["claude-sonnet-4-6"]["input"])


def test_estimate_call_cost_adds_web_search_line_items() -> None:
    cost = estimate_call_cost(
        model="claude-sonnet-4-6",
        tokens_in=0,
        tokens_out=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        web_search_count=3,
    )
    assert math.isclose(cost, 3 * WEB_SEARCH_USD_PER_CALL)


def test_estimate_call_cost_uses_cache_read_rate_not_input_rate() -> None:
    """Cache reads are heavily discounted — the math must use the cheaper rate."""
    pricing = ANTHROPIC_PRICING["claude-sonnet-4-6"]
    cache_read_cost = estimate_call_cost(
        model="claude-sonnet-4-6",
        tokens_in=0,
        tokens_out=0,
        cache_read_tokens=1_000_000,
        cache_creation_tokens=0,
        web_search_count=0,
    )
    assert math.isclose(cache_read_cost, pricing["cache_read"])
    assert pricing["cache_read"] < pricing["input"]


def test_estimate_call_cost_unknown_model_falls_back_to_sonnet() -> None:
    fallback = estimate_call_cost(
        model="claude-imaginary-9-9",
        tokens_in=1_000_000,
        tokens_out=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        web_search_count=0,
    )
    assert math.isclose(fallback, ANTHROPIC_PRICING["claude-sonnet-4-6"]["input"])


@pytest.mark.parametrize(
    "tokens_in,cache_read,cache_creation,expected",
    [
        (200, 1800, 0, 0.9),
        (1000, 0, 0, 0.0),
        (0, 0, 0, 0.0),
        (100, 100, 800, 0.1),
    ],
)
def test_cache_hit_rate(
    tokens_in: int, cache_read: int, cache_creation: int, expected: float
) -> None:
    assert math.isclose(
        cache_hit_rate(
            tokens_in=tokens_in,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        ),
        expected,
    )
