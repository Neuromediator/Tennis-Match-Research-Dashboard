"""Tavily snippet-only `web_search` tool — Phase 5.1.

Replaces Anthropic's native `web_search_20250305` server-side tool with our
own client-side wrapper around the Tavily Search API. Same name in the
tool catalogue, same role in the agent loop; what changes is the payload
shape we feed back to the model:

- Anthropic native: encrypted full-page content (~5-15k tokens/result)
- Tavily basic:     snippet (~150-300 chars/result) + title + URL + date

The Phase 5 A/B (`scripts/compare_search_providers.py`, 2026-05-23) showed
this swap is ~9x cheaper and ~3x faster with comparable discovery quality
on tennis queries; the rationale lives in
`docs/tutorials/phase_5_1_notes.md` Step 1-2.

Failure mode #1 from CLAUDE.md: Tavily 5xx/4xx → raise `TavilyError`. The
agent loop catches it and surfaces an `is_error=True` tool_result so the
LLM mentions the failure in `caveats` rather than fabricating around it.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from tennis_predictor.config import TAVILY_API_KEY
from tennis_predictor.llm.cost import TAVILY_BASIC_USD_PER_QUERY
from tennis_predictor.llm.tools.schemas import (
    TavilyError,
    WebSearchHit,
    WebSearchInput,
    WebSearchOutput,
)

logger = logging.getLogger(__name__)

# Same betting / pick-of-the-day blocklist Phase 5 used on Anthropic native
# (CLAUDE.md "Web search"). Applied client-side via Tavily's exclude_domains.
BLOCKED_DOMAINS: tuple[str, ...] = (
    "draftkings.com",
    "fanduel.com",
    "betmgm.com",
    "pickwise.com",
    "actionnetwork.com",
)

# Cap on tool invocations per agent call. Enforced by AgentBudget; mirrored
# here so callers can read the limit from a single import. CLAUDE.md sets
# this to 3 (typically 1 per player + optionally 1 for the tournament).
WEB_SEARCH_MAX_USES: int = 3

# Tavily API endpoint. Hard-coded rather than env-configurable because there
# is no staging endpoint; if Tavily ever moves the URL we want a single
# explicit change here, not silent env drift.
TAVILY_SEARCH_URL: str = "https://api.tavily.com/search"

# HTTP timeout (seconds). Tavily basic typically responds in 1-2s; 15s leaves
# room for tail-latency network hiccups without letting a stuck request eat
# the agent's 120s wall-clock budget.
_HTTP_TIMEOUT_SECONDS: float = 15.0


async def search_web(payload: WebSearchInput) -> WebSearchOutput:
    """Call Tavily basic search once. Raises `TavilyError` on any HTTP or
    parse failure — the agent loop turns that into a tool_result error.

    Caller is responsible for budget accounting (each call burns one slot
    of `AgentBudget.max_web_searches`).
    """
    if not TAVILY_API_KEY:
        raise TavilyError(
            "TAVILY_API_KEY is not set in .env; web_search cannot run. "
            "Sign up at app.tavily.com (free tier, 1000 req/month, no card)."
        )

    # Phase 6.2 (May 2026) tightening:
    # - `topic="news"`: Tavily news index biases results to dated news
    #   articles instead of general web pages. Without this we were
    #   getting stale tournament recaps with no `published_date` (e.g.,
    #   January 2026 Australian Open injury for a player in his May 26
    #   Roland Garros R1).
    # - `days=32`: Tavily-side recency filter, matches our agent's
    #   32-day window. Pre-filters at the source so the agent doesn't
    #   even see months-old items.
    body = {
        "query": payload.query,
        "search_depth": "basic",
        "topic": "news",
        "days": 32,
        "max_results": payload.max_results,
        "exclude_domains": list(BLOCKED_DOMAINS),
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=2),
            timeout=_HTTP_TIMEOUT_SECONDS,
        ) as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        raise TavilyError(
            f"Tavily search returned {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise TavilyError(f"Tavily search HTTP failure: {type(exc).__name__}: {exc}") from exc

    hits: list[WebSearchHit] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        hits.append(
            WebSearchHit(
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=item.get("content") or "",
                published_date=item.get("published_date"),
            )
        )

    return WebSearchOutput(
        query=payload.query,
        results=hits,
        cost_usd=TAVILY_BASIC_USD_PER_QUERY,
    )


# ---------------------------------------------------------------------------
# Tool definition the LLM receives.
# ---------------------------------------------------------------------------


WEB_SEARCH_TOOL_NAME: str = "web_search"

# Hand-written JSON schema rather than Pydantic-generated — same reasoning
# as `submit.py`: keeps the cacheable-prefix byte-stable across Pydantic
# minor versions and lets the description live next to the field.
WEB_SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query"],
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
            "description": (
                "Search query — a short tennis-context phrase. Examples: "
                "'Carlos Alcaraz injury Roland Garros 2026', 'Kasatkina "
                "absence return interview'. Include the player name, the "
                "year if topical, and one or two key topical words."
            ),
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": (
                "How many results to return (1-10). 5 is usually enough; "
                "go higher only if the first batch is mostly off-topic."
            ),
        },
    },
}


WEB_SEARCH_TOOL: dict[str, Any] = {
    "name": WEB_SEARCH_TOOL_NAME,
    "description": (
        "Search the web for recent tennis news. Returns up to 10 results "
        "with a ~200-character snippet, title, URL, and publication date. "
        "Snippets are usually enough to answer 'is there breaking news for "
        "this player?'. If a snippet truncates an important detail, you MAY "
        "call `fetch_url(url)` to retrieve the cleaned full article body — "
        "but use sparingly (<= 2 fetches per prediction). "
        "Prefer ESPN, BBC, tennis.com, tennis365.com and journalists on "
        "X/Twitter when surfaced. Avoid betting / pick-of-the-day sites "
        "(filtered for you by `exclude_domains`)."
    ),
    "input_schema": WEB_SEARCH_INPUT_SCHEMA,
}


__all__ = [
    "BLOCKED_DOMAINS",
    "TAVILY_SEARCH_URL",
    "WEB_SEARCH_INPUT_SCHEMA",
    "WEB_SEARCH_MAX_USES",
    "WEB_SEARCH_TOOL",
    "WEB_SEARCH_TOOL_NAME",
    "search_web",
]
