"""Tavily Extract `fetch_url` tool — Phase 5.1 escape-hatch.

`web_search` returns snippets (~200-300 chars). For the ~5% of cases where
a snippet truncates a key detail and the LLM needs the cleaned full body
of one specific article, this tool fetches and parses it via Tavily's
Extract endpoint.

Why Tavily Extract rather than DIY `httpx.get(...)` + readability:
- Tavily handles JS-rendered pages, paywall walls, cookie banners.
- Same vendor / same key as `web_search` — one less env to track.
- $0.005 per fetch, same as a search call.

Budget: `AgentBudget.max_fetch_urls = 2` per agent call. Beyond that the
loop forces `submit_analysis`. CLAUDE.md failure-mode #7 (new in Phase 5.1)
treats Tavily Extract failure / paywall as degraded mode — agent continues
with the snippet it already has.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from tennis_predictor.config import TAVILY_API_KEY
from tennis_predictor.llm.cost import TAVILY_EXTRACT_USD_PER_QUERY
from tennis_predictor.llm.tools.schemas import (
    FetchUrlInput,
    FetchUrlOutput,
    TavilyError,
)

logger = logging.getLogger(__name__)

TAVILY_EXTRACT_URL: str = "https://api.tavily.com/extract"

# Per-agent-call cap on fetch_url invocations. Enforced by AgentBudget;
# mirrored here so callers can read the limit from a single import.
FETCH_URL_MAX_USES: int = 2

# HTTP timeout (seconds). Extract is slower than search because Tavily
# actually fetches and parses the page; 20s leaves room for tail latency.
_HTTP_TIMEOUT_SECONDS: float = 20.0

# Cap on returned content size — we don't want a 50k-token Wikipedia article
# blowing up the next LLM turn. 6000 chars ~= 1500 tokens, enough to surface
# the substantive body of a typical news article.
_MAX_CONTENT_CHARS: int = 6000


async def fetch_url(payload: FetchUrlInput) -> FetchUrlOutput:
    """Call Tavily Extract once. Raises `TavilyError` on any HTTP or parse
    failure. A successful response with empty content (paywall, JS-only,
    Tavily couldn't extract) returns `FetchUrlOutput(extraction_success=False)`
    rather than raising — that's degraded mode #7, not a programming bug.
    """
    if not TAVILY_API_KEY:
        raise TavilyError("TAVILY_API_KEY is not set in .env; fetch_url cannot run.")

    body = {
        "urls": [payload.url],
        "extract_depth": "basic",
    }

    try:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=2),
            timeout=_HTTP_TIMEOUT_SECONDS,
        ) as client:
            response = await client.post(
                TAVILY_EXTRACT_URL,
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
            f"Tavily extract returned {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise TavilyError(f"Tavily extract HTTP failure: {type(exc).__name__}: {exc}") from exc

    # Tavily Extract returns:
    #   { "results": [{"url": "...", "raw_content": "..."}],
    #     "failed_results": [{"url": "...", "error": "..."}] }
    # On a successful URL we get a `results` entry; on paywalled/JS-only
    # pages it goes to `failed_results` — treated as extraction_success=False.
    results = data.get("results") or []
    failed = data.get("failed_results") or []

    if results:
        first = results[0] if isinstance(results[0], dict) else {}
        content = (first.get("raw_content") or "")[:_MAX_CONTENT_CHARS]
        return FetchUrlOutput(
            url=first.get("url") or payload.url,
            content=content,
            extraction_success=bool(content),
            cost_usd=TAVILY_EXTRACT_USD_PER_QUERY,
        )

    if failed:
        first = failed[0] if isinstance(failed[0], dict) else {}
        return FetchUrlOutput(
            url=first.get("url") or payload.url,
            content="",
            extraction_success=False,
            cost_usd=TAVILY_EXTRACT_USD_PER_QUERY,
        )

    # Empty response — treat as extraction failure (no exception, degraded).
    return FetchUrlOutput(
        url=payload.url,
        content="",
        extraction_success=False,
        cost_usd=TAVILY_EXTRACT_USD_PER_QUERY,
    )


# ---------------------------------------------------------------------------
# Tool definition the LLM receives.
# ---------------------------------------------------------------------------


FETCH_URL_TOOL_NAME: str = "fetch_url"

FETCH_URL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["url"],
    "properties": {
        "url": {
            "type": "string",
            "format": "uri",
            "minLength": 1,
            "maxLength": 2000,
            "description": (
                "Full URL of one article to fetch and clean. Use only "
                "URLs returned by a previous `web_search` call. "
                "Returns the article body (up to ~6000 chars) or an "
                "empty content + extraction_success=False on paywalls / "
                "JS-only sites."
            ),
        },
    },
}


FETCH_URL_TOOL: dict[str, Any] = {
    "name": FETCH_URL_TOOL_NAME,
    "description": (
        "Fetch the cleaned full body of one specific URL. Use ONLY when a "
        "`web_search` snippet truncates an important detail you need to "
        "interpret. Most predictions don't need this — call at most twice "
        "per match, never on a URL you haven't seen in a web_search result. "
        "If extraction_success is False, the page was paywalled or "
        "JS-rendered; mention that in caveats and move on."
    ),
    "input_schema": FETCH_URL_INPUT_SCHEMA,
}


__all__ = [
    "FETCH_URL_INPUT_SCHEMA",
    "FETCH_URL_MAX_USES",
    "FETCH_URL_TOOL",
    "FETCH_URL_TOOL_NAME",
    "TAVILY_EXTRACT_URL",
    "fetch_url",
]
