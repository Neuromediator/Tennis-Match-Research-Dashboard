"""Unit tests for the Tavily-backed `web_search` tool — Phase 5.1.

Mocks Tavily's HTTP API via `respx` so the tests are deterministic, free,
and don't touch the network. Covers happy path, 5xx error, empty results,
the blocked-domain payload check, and the JSON-schema literal that
Anthropic sees on the wire.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from tennis_predictor.llm.tools.schemas import TavilyError, WebSearchInput
from tennis_predictor.llm.tools.web_search import (
    BLOCKED_DOMAINS,
    TAVILY_SEARCH_URL,
    WEB_SEARCH_INPUT_SCHEMA,
    WEB_SEARCH_TOOL,
    WEB_SEARCH_TOOL_NAME,
    search_web,
)

# ---------------------------------------------------------------------------
# JSON-schema-shape tests — what the LLM sees in the tool catalog.
# ---------------------------------------------------------------------------


def test_web_search_tool_name_and_schema_shape() -> None:
    assert WEB_SEARCH_TOOL["name"] == WEB_SEARCH_TOOL_NAME == "web_search"
    assert WEB_SEARCH_TOOL["input_schema"] is WEB_SEARCH_INPUT_SCHEMA
    assert WEB_SEARCH_INPUT_SCHEMA["additionalProperties"] is False
    assert "query" in WEB_SEARCH_INPUT_SCHEMA["required"]
    assert "query" in WEB_SEARCH_INPUT_SCHEMA["properties"]
    assert "max_results" in WEB_SEARCH_INPUT_SCHEMA["properties"]


def test_web_search_schema_does_not_expose_blocked_domains() -> None:
    """Block list is applied server-side by us — LLM never sees / can override it."""
    assert "exclude_domains" not in WEB_SEARCH_INPUT_SCHEMA["properties"]
    assert "blocked_domains" not in WEB_SEARCH_INPUT_SCHEMA["properties"]


# ---------------------------------------------------------------------------
# search_web behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_search_web_happy_path_parses_tavily_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", "test-key")
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "query": "Alcaraz injury",
                "results": [
                    {
                        "title": "Alcaraz withdraws from Roland Garros",
                        "url": "https://example.com/article-1",
                        "content": "Carlos Alcaraz pulled out of Roland Garros with a wrist injury, his team said.",
                        "published_date": "2026-05-02",
                    },
                    {
                        "title": "BBC: Wimbledon update",
                        "url": "https://bbc.com/sport/tennis/article-2",
                        "content": "Two-time champion Carlos Alcaraz will miss Wimbledon...",
                    },
                ],
            },
        )
    )

    out = await search_web(WebSearchInput(query="Alcaraz injury", max_results=5))
    assert out.query == "Alcaraz injury"
    assert len(out.results) == 2
    assert out.results[0].title.startswith("Alcaraz withdraws")
    assert out.results[0].published_date == "2026-05-02"
    assert out.results[1].published_date is None
    assert out.cost_usd == pytest.approx(0.005)


@pytest.mark.asyncio
@respx.mock
async def test_search_web_passes_blocked_domains_in_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", "test-key")
    route = respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    await search_web(WebSearchInput(query="anything", max_results=3))
    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert set(body["exclude_domains"]) == set(BLOCKED_DOMAINS)
    assert body["max_results"] == 3
    assert body["search_depth"] == "basic"


@pytest.mark.asyncio
@respx.mock
async def test_search_web_5xx_raises_tavily_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", "test-key")
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(500, text="upstream gateway error")
    )
    with pytest.raises(TavilyError, match="500"):
        await search_web(WebSearchInput(query="anything"))


@pytest.mark.asyncio
@respx.mock
async def test_search_web_empty_results_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tavily returning zero results is a legitimate signal ("no recent news
    surfaced"), not a failure — the tool returns an empty list with the
    standard cost, the agent surfaces it in caveats."""
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", "test-key")
    respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    out = await search_web(WebSearchInput(query="obscure player"))
    assert out.results == []
    assert out.cost_usd == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_search_web_missing_api_key_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", None)
    with pytest.raises(TavilyError, match="TAVILY_API_KEY"):
        await search_web(WebSearchInput(query="anything"))


@pytest.mark.asyncio
@respx.mock
async def test_search_web_network_timeout_raises_tavily_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", "test-key")
    respx.post(TAVILY_SEARCH_URL).mock(side_effect=httpx.ReadTimeout("simulated"))
    with pytest.raises(TavilyError, match="HTTP failure"):
        await search_web(WebSearchInput(query="anything"))
