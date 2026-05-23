"""Unit tests for the Tavily-backed `fetch_url` tool — Phase 5.1.

Same pattern as `test_llm_web_search.py`: mock the Tavily Extract endpoint
via `respx` and assert behaviour on the happy path, paywalled (empty)
response, and HTTP failure.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from tennis_predictor.llm.tools.fetch_url import (
    FETCH_URL_INPUT_SCHEMA,
    FETCH_URL_TOOL,
    FETCH_URL_TOOL_NAME,
    TAVILY_EXTRACT_URL,
    fetch_url,
)
from tennis_predictor.llm.tools.schemas import FetchUrlInput, TavilyError

# ---------------------------------------------------------------------------
# JSON-schema shape
# ---------------------------------------------------------------------------


def test_fetch_url_tool_name_and_schema_shape() -> None:
    assert FETCH_URL_TOOL["name"] == FETCH_URL_TOOL_NAME == "fetch_url"
    assert FETCH_URL_TOOL["input_schema"] is FETCH_URL_INPUT_SCHEMA
    assert FETCH_URL_INPUT_SCHEMA["additionalProperties"] is False
    assert FETCH_URL_INPUT_SCHEMA["required"] == ["url"]
    assert "url" in FETCH_URL_INPUT_SCHEMA["properties"]


# ---------------------------------------------------------------------------
# fetch_url behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_url_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tennis_predictor.llm.tools.fetch_url.TAVILY_API_KEY", "test-key")
    respx.post(TAVILY_EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/article",
                        "raw_content": "Full article body about Alcaraz withdrawal..." * 5,
                    }
                ],
                "failed_results": [],
            },
        )
    )

    out = await fetch_url(FetchUrlInput(url="https://example.com/article"))
    assert out.url == "https://example.com/article"
    assert out.extraction_success is True
    assert "Alcaraz" in out.content
    assert out.cost_usd == pytest.approx(0.005)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_url_failed_extraction_returns_degraded_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paywall / JS-only pages land in `failed_results`. The tool must NOT
    raise — it returns `extraction_success=False` so the agent can mention
    the failure in caveats (CLAUDE.md failure-mode #7)."""
    monkeypatch.setattr("tennis_predictor.llm.tools.fetch_url.TAVILY_API_KEY", "test-key")
    respx.post(TAVILY_EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [],
                "failed_results": [
                    {"url": "https://paywalled.example/article", "error": "paywall"}
                ],
            },
        )
    )

    out = await fetch_url(FetchUrlInput(url="https://paywalled.example/article"))
    assert out.extraction_success is False
    assert out.content == ""
    assert out.cost_usd == pytest.approx(0.005)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_url_caps_content_at_6000_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 50k-token Wikipedia article would blow up the next LLM turn —
    the tool truncates at ~6000 chars."""
    monkeypatch.setattr("tennis_predictor.llm.tools.fetch_url.TAVILY_API_KEY", "test-key")
    huge_body = "x" * 20_000
    respx.post(TAVILY_EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"url": "https://example.com/wiki", "raw_content": huge_body}],
                "failed_results": [],
            },
        )
    )
    out = await fetch_url(FetchUrlInput(url="https://example.com/wiki"))
    assert len(out.content) == 6000
    assert out.extraction_success is True


@pytest.mark.asyncio
@respx.mock
async def test_fetch_url_5xx_raises_tavily_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tennis_predictor.llm.tools.fetch_url.TAVILY_API_KEY", "test-key")
    respx.post(TAVILY_EXTRACT_URL).mock(return_value=httpx.Response(502, text="bad gateway"))
    with pytest.raises(TavilyError, match="502"):
        await fetch_url(FetchUrlInput(url="https://example.com/article"))


@pytest.mark.asyncio
async def test_fetch_url_missing_api_key_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tennis_predictor.llm.tools.fetch_url.TAVILY_API_KEY", None)
    with pytest.raises(TavilyError, match="TAVILY_API_KEY"):
        await fetch_url(FetchUrlInput(url="https://example.com/article"))
