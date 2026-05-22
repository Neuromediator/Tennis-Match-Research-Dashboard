"""Per-call cost estimation for Anthropic + web_search.

The numbers below are USD per million input / output tokens for the
Anthropic models we may use. Cache reads are heavily discounted; cache
writes carry a small surcharge over uncached input. Web search is a flat
per-search line item.

Pricing snapshot taken 2026-05-22 from the Anthropic pricing page. The
table is intentionally a plain dict so updates are a single-line PR. If
Anthropic changes its pricing structure (e.g. introduces tiered output
pricing), bump the rate-table shape and update this docstring.

# Why this is approximate

Two reasons real bills can drift from `estimate_call_cost`:

1. Anthropic publishes prices in coarse rate cards; the actual invoice
   uses higher-resolution rates per call. The discrepancy is typically
   <1%, but on a $10 month that's a rounding-error column.

2. `web_search_count` counts `server_tool_use` blocks with name
   `web_search`. Anthropic occasionally bills per *query* including ones
   the model issued internally but didn't surface as a separate block.
   Again, small discrepancy in practice.

For the dashboard the estimate is "good enough"; the live monthly cap in
the Anthropic console is the actual ceiling.
"""

from __future__ import annotations

from typing import TypedDict

# USD per million tokens / per search.
USD_PER_MILLION: float = 1_000_000.0


class _ModelPricing(TypedDict):
    """Per-million-token prices for one model."""

    input: float
    output: float
    cache_read: float
    cache_write: float


# Sonnet 4.6 is the default; Opus 4.7 and Haiku 4.5 entries let us swap
# `ANTHROPIC_MODEL` without rewriting the cost code. Numbers are best
# guesses; verify against the Anthropic console before reading them as
# truth for finance purposes.
ANTHROPIC_PRICING: dict[str, _ModelPricing] = {
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-opus-4-7": {
        "input": 15.00,
        "output": 75.00,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
}

# Anthropic native web_search: ~$10 per 1000 searches as of 2026-05-22.
WEB_SEARCH_USD_PER_CALL: float = 0.010


def _resolve_pricing(model: str) -> _ModelPricing:
    """Return the rate card for `model`. Falls back to Sonnet 4.6 if the
    name is unknown — keeps the cost-tracking column populated rather
    than emitting `NaN` rows that the dashboard then has to special-case."""
    if model in ANTHROPIC_PRICING:
        return ANTHROPIC_PRICING[model]
    return ANTHROPIC_PRICING["claude-sonnet-4-6"]


def estimate_call_cost(
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    web_search_count: int,
) -> float:
    """Best-effort USD cost for one LLM API call.

    Token splits follow Anthropic's billing model:
    - `tokens_in` from `usage.input_tokens` excludes cached / cache-creation.
    - `cache_read_tokens` is billed at the cache_read rate.
    - `cache_creation_tokens` is billed at the cache_write rate
      (a one-time surcharge to write the prefix into the cache).
    - `tokens_out` is billed at the standard output rate.

    `web_search_count` is added as a flat per-search line item.
    """
    rates = _resolve_pricing(model)
    cost = 0.0
    cost += (tokens_in / USD_PER_MILLION) * rates["input"]
    cost += (tokens_out / USD_PER_MILLION) * rates["output"]
    cost += (cache_read_tokens / USD_PER_MILLION) * rates["cache_read"]
    cost += (cache_creation_tokens / USD_PER_MILLION) * rates["cache_write"]
    cost += web_search_count * WEB_SEARCH_USD_PER_CALL
    return round(cost, 6)


def cache_hit_rate(
    *,
    tokens_in: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> float:
    """Fraction of input tokens served from the prompt cache.

    Defined as ``cache_read / (tokens_in + cache_read + cache_creation)``,
    so the first call in a TTL window scores near-zero (cache miss) and
    follow-up calls score high. Used by `LLMClient` callers and by the
    CLI footer line to sanity-check that caching is actually working.

    Returns 0.0 when the denominator is zero (no tokens billed at all —
    only happens in mocked tests)."""
    denom = tokens_in + cache_read_tokens + cache_creation_tokens
    if denom <= 0:
        return 0.0
    return cache_read_tokens / denom


__all__ = [
    "ANTHROPIC_PRICING",
    "WEB_SEARCH_USD_PER_CALL",
    "cache_hit_rate",
    "estimate_call_cost",
]
