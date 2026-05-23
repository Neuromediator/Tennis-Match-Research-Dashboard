"""A/B compare Anthropic native `web_search` vs Tavily snippet-only.

The Phase 5 cost reality (~$0.10/predict, 3x the pre-implementation estimate)
is dominated by web_search payload — Anthropic's native search fetches full
page content and we pay ~$0.04/predict in `cache_creation` for content the
agent rarely needs past the headline.

This script empirically tests whether Tavily's snippet-only search finds the
same information for the kind of queries the agent actually runs:

  - Top player + popular news  — both providers should find the same big story.
  - Mid-tier + niche journalism — main value test; can Tavily surface a
    Kasatkina-style interview that's not on every front page?
  - Long-tail + local press    — discovery on under-indexed sources.
  - Twitter-granularity probe  — expected miss for BOTH; documenting the gap.

For each query the script prints, side by side, the top results from both
providers (title + URL + snippet preview), plus latency, token cost, and
whether the result sets overlap.

Cost of one run: ~$0.05 against Anthropic budget, 4 of your 1000 free Tavily
queries. Read the output, judge whether discovery quality is comparable on
the niche queries, then decide go / no-go on Phase 5.1.

Usage:
    uv run python scripts/compare_search_providers.py

Optional:
    --query "your query"   Add a custom query alongside the defaults.
    --json                 Dump structured JSON instead of pretty text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from anthropic import APIError, AsyncAnthropic

from tennis_predictor.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, TAVILY_API_KEY

logger = logging.getLogger("compare_search_providers")


# Default query set. Each tuple is (label, query string).
DEFAULT_QUERIES: list[tuple[str, str]] = [
    (
        "top-popular",
        "Carlos Alcaraz Roland Garros 2026 injury withdrawal news",
    ),
    (
        "mid-tier niche",
        "Daria Kasatkina absence WTA tour 2026 return interview",
    ),
    (
        "long-tail",
        "Tatjana Maria comeback 2026 tournament",
    ),
    (
        "twitter-granularity",
        "Djokovic missed training session today 2026",
    ),
]

# Same betting / pick-of-the-day blocklist Phase 5 uses on Anthropic native.
# Tavily gets the same list via `exclude_domains` for fair comparison.
BLOCKED_DOMAINS: tuple[str, ...] = (
    "draftkings.com",
    "fanduel.com",
    "betmgm.com",
    "pickwise.com",
    "actionnetwork.com",
)

MAX_RESULTS: int = 5
SNIPPET_PREVIEW_CHARS: int = 280


# Sonnet 4.6 rate card (cost.py mirrors these).
_USD_PER_M_INPUT: float = 3.00
_USD_PER_M_OUTPUT: float = 15.00
_USD_PER_M_CACHE_READ: float = 0.30
_USD_PER_M_CACHE_WRITE: float = 3.75
_USD_PER_WEB_SEARCH: float = 0.010
# Tavily basic on the paid plan; free tier is just $0 up to 1000/month.
_TAVILY_USD_PER_QUERY_PAID: float = 0.005


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    raw_length: int


@dataclass
class ProviderRun:
    provider: str
    query: str
    hits: list[SearchHit]
    latency_ms: int
    cost_usd: float
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    note: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


async def search_tavily(query: str) -> ProviderRun:
    """One Tavily basic search. Snippets only, no body fetch."""
    started = time.monotonic()
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": MAX_RESULTS,
                    "exclude_domains": list(BLOCKED_DOMAINS),
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        return ProviderRun(
            provider="Tavily basic",
            query=query,
            hits=[],
            latency_ms=latency_ms,
            cost_usd=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    latency_ms = int((time.monotonic() - started) * 1000)
    hits: list[SearchHit] = []
    for item in data.get("results") or []:
        content = item.get("content") or ""
        hits.append(
            SearchHit(
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=content[:SNIPPET_PREVIEW_CHARS],
                raw_length=len(content),
            )
        )
    return ProviderRun(
        provider="Tavily basic",
        query=query,
        hits=hits,
        latency_ms=latency_ms,
        cost_usd=_TAVILY_USD_PER_QUERY_PAID,
        cost_breakdown={"tavily_search": _TAVILY_USD_PER_QUERY_PAID},
        note="snippet-only; free tier hides cost — pricing shown for paid-plan reference",
    )


# ---------------------------------------------------------------------------
# Anthropic native web_search
# ---------------------------------------------------------------------------


_ANTHROPIC_SYSTEM = (
    "You are a search proxy. The user asks one query; you make one call to "
    "the `web_search` tool with the user's exact query and then immediately "
    "end your turn with no commentary. Do not summarise the results."
)


async def search_anthropic(query: str) -> ProviderRun:
    """One Anthropic call with native web_search. We force a single search,
    then parse the `web_search_tool_result` block(s) the API returns. The
    model's text output is suppressed (max_tokens=80, system prompt tells
    it to stay silent) — we only care about the raw search results."""
    if ANTHROPIC_API_KEY is None:
        return ProviderRun(
            provider="Anthropic native",
            query=query,
            hits=[],
            latency_ms=0,
            cost_usd=0.0,
            error="ANTHROPIC_API_KEY not set",
        )

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    started = time.monotonic()
    try:
        response = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=80,
            system=_ANTHROPIC_SYSTEM,
            messages=[{"role": "user", "content": query}],
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 1,
                    "blocked_domains": list(BLOCKED_DOMAINS),
                }
            ],
        )
    except APIError as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        return ProviderRun(
            provider="Anthropic native",
            query=query,
            hits=[],
            latency_ms=latency_ms,
            cost_usd=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    latency_ms = int((time.monotonic() - started) * 1000)
    hits: list[SearchHit] = []
    note: str | None = None
    for block in response.content:
        d = block.model_dump() if hasattr(block, "model_dump") else dict(block)
        if d.get("type") != "web_search_tool_result":
            continue
        inner = d.get("content")
        if isinstance(inner, dict) and inner.get("type") == "web_search_tool_result_error":
            note = f"web_search_tool_result_error: {inner.get('error_code')}"
            continue
        if isinstance(inner, list):
            for it in inner:
                if not isinstance(it, dict):
                    continue
                if it.get("type") != "web_search_result":
                    continue
                # Anthropic returns `title`, `url`, `page_age`, and
                # `encrypted_content` (opaque to us — the model uses it
                # internally but we cannot decode it here). We surface
                # title + URL; the snippet column shows whatever non-opaque
                # text fields are present.
                body_keys = ("snippet", "summary", "content", "text", "page_age")
                snippet_parts = [
                    str(it[k]) for k in body_keys if it.get(k) and isinstance(it[k], str)
                ]
                snippet = " | ".join(snippet_parts)
                hits.append(
                    SearchHit(
                        title=it.get("title") or "",
                        url=it.get("url") or "",
                        snippet=snippet[:SNIPPET_PREVIEW_CHARS],
                        raw_length=len(snippet),
                    )
                )

    # Cost breakdown — same math as llm/cost.py.
    usage = response.usage
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
    cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    breakdown = {
        "input": tokens_in / 1_000_000 * _USD_PER_M_INPUT,
        "output": tokens_out / 1_000_000 * _USD_PER_M_OUTPUT,
        "cache_creation": cache_creation / 1_000_000 * _USD_PER_M_CACHE_WRITE,
        "cache_read": cache_read / 1_000_000 * _USD_PER_M_CACHE_READ,
        "web_search": _USD_PER_WEB_SEARCH if hits or note else 0.0,
    }
    cost_usd = sum(breakdown.values())

    if not hits and note is None:
        note = "no web_search_tool_result block returned — model may have skipped the tool"

    return ProviderRun(
        provider="Anthropic native",
        query=query,
        hits=hits,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        cost_breakdown=breakdown,
        note=note,
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _short_url(url: str, width: int = 70) -> str:
    return url if len(url) <= width else url[: width - 3] + "..."


def print_provider_run(run: ProviderRun) -> None:
    print(f"  --- {run.provider} ---")
    if run.error:
        print(f"  ERROR: {run.error}")
        return
    print(f"  latency: {run.latency_ms} ms   cost: ${run.cost_usd:.4f}")
    if run.cost_breakdown:
        items = ", ".join(f"{k}=${v:.4f}" for k, v in run.cost_breakdown.items() if v > 0)
        if items:
            print(f"  cost breakdown: {items}")
    if run.note:
        print(f"  note: {run.note}")
    print(f"  hits ({len(run.hits)}):")
    if not run.hits:
        print("    (none)")
        return
    for i, h in enumerate(run.hits[:MAX_RESULTS], 1):
        print(f"    [{i}] {h.title or '(no title)'}")
        print(f"        url:     {_short_url(h.url)}")
        print(
            f"        snippet: {h.snippet[:200] if h.snippet else '(none returned by API)'}"
            + ("..." if len(h.snippet) > 200 else "")
        )
        if h.raw_length > 200:
            print(f"        raw length: {h.raw_length} chars")


def overlap_summary(a: ProviderRun, b: ProviderRun) -> str:
    a_urls = {_normalize_url(h.url) for h in a.hits}
    b_urls = {_normalize_url(h.url) for h in b.hits}
    both = a_urls & b_urls
    only_a = a_urls - b_urls
    only_b = b_urls - a_urls
    return (
        f"  URL overlap: {len(both)} shared, "
        f"{len(only_a)} unique to {a.provider.split()[0]}, "
        f"{len(only_b)} unique to {b.provider.split()[0]}"
    )


def _normalize_url(url: str) -> str:
    """Normalise URL for set comparison: strip scheme, trailing slash, query.
    Keeps host + path stable across http/https and tracking-params variants."""
    u = url.lower()
    for scheme in ("https://", "http://"):
        if u.startswith(scheme):
            u = u[len(scheme) :]
    if "?" in u:
        u = u.split("?", 1)[0]
    if "#" in u:
        u = u.split("#", 1)[0]
    return u.rstrip("/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(queries: list[tuple[str, str]], json_output: bool) -> int:
    if not TAVILY_API_KEY:
        print("ERROR: TAVILY_API_KEY is not set in .env", file=sys.stderr)
        return 1
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY is not set in .env", file=sys.stderr)
        return 1

    aggregated: list[dict[str, Any]] = []
    total_anthropic_cost = 0.0
    total_tavily_cost = 0.0

    for label, query in queries:
        if not json_output:
            print()
            print("=" * 80)
            print(f"Query [{label}]: {query!r}")
            print("=" * 80)

        tavily_run, anthropic_run = await asyncio.gather(
            search_tavily(query),
            search_anthropic(query),
        )
        total_tavily_cost += tavily_run.cost_usd
        total_anthropic_cost += anthropic_run.cost_usd

        if json_output:
            aggregated.append(
                {
                    "label": label,
                    "query": query,
                    "tavily": asdict(tavily_run),
                    "anthropic": asdict(anthropic_run),
                }
            )
        else:
            print_provider_run(tavily_run)
            print()
            print_provider_run(anthropic_run)
            print()
            print(overlap_summary(tavily_run, anthropic_run))

    if json_output:
        print(
            json.dumps(
                {
                    "queries": aggregated,
                    "total_anthropic_cost_usd": total_anthropic_cost,
                    "total_tavily_cost_usd": total_tavily_cost,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"Tavily total cost ({len(queries)} queries):    ${total_tavily_cost:.4f} "
        "(free on Tavily free tier)"
    )
    print(f"Anthropic total cost ({len(queries)} queries): ${total_anthropic_cost:.4f}")
    if total_tavily_cost > 0:
        ratio = total_anthropic_cost / total_tavily_cost
        print(f"Ratio: Anthropic is {ratio:.1f}x more expensive per search")
    print()
    print("Decision guide:")
    print("  - If URL overlap is high on the popular queries AND Tavily")
    print("    finds the same niche journalism on mid-tier queries → switch.")
    print("  - If Tavily misses obvious tennis sites on niche queries → stay.")
    print("  - Twitter-granularity gap is expected on both; not a deciding factor.")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="compare_search_providers")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Add a custom query (repeatable) alongside the default 4.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump structured JSON instead of pretty-printed text.",
    )
    args = parser.parse_args(argv)
    queries = list(DEFAULT_QUERIES) + [("custom", q) for q in args.query]
    return asyncio.run(run(queries, json_output=args.json))


if __name__ == "__main__":
    sys.exit(main())
