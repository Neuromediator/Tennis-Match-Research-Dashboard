"""Unit tests for the Anthropic LLM client.

The Anthropic SDK is never invoked; every test substitutes a stub
`AsyncAnthropic` that records requests and returns a canned `Message`.
This is the Tier-1 layer of the testing pyramid (CLAUDE.md "Testing the
LLM agent") — runs in CI, costs nothing.

The byte-stability test is the highest-value test in this module: if it
fails, the 5-minute prompt cache went from ~70% input-cost reduction to
0% and we want a red CI immediately, not a $50 surprise at month-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.llm.client import (
    AnthropicLLMClient,
    LLMCallFailure,
    build_web_search_tool_param,
)

# ---------------------------------------------------------------------------
# Fixtures: a fresh DuckDB + a stub Anthropic SDK.
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_db(tmp_path: Path):
    conn = duckdb.connect(str(tmp_path / "llm_client_test.duckdb"))
    schema.create_all_tables(conn)
    yield conn
    conn.close()


class _StubUsage:
    def __init__(self, *, tokens_in: int, tokens_out: int, cache_read: int, cache_create: int):
        self.input_tokens = tokens_in
        self.output_tokens = tokens_out
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_create


class _StubBlock:
    """Mimics the Anthropic SDK content block: has `model_dump()` and
    .type/.name/.input attributes (the agent loop reads via `model_dump`)."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return dict(self._payload)


class _StubMessage:
    def __init__(
        self,
        *,
        content: list[_StubBlock],
        stop_reason: str = "end_turn",
        usage: _StubUsage | None = None,
    ):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


class _StubMessagesAPI:
    """Drop-in for `AsyncAnthropic.messages` — records calls and returns
    `_StubMessage` instances pulled from a queue. Tests prepare the queue
    upfront, then call `acall(...)` and assert against `received_calls`."""

    def __init__(self) -> None:
        self.received_calls: list[dict[str, Any]] = []
        self._queue: list[_StubMessage | Exception] = []

    def enqueue(self, item: _StubMessage | Exception) -> None:
        self._queue.append(item)

    async def create(self, **kwargs: Any) -> _StubMessage:
        self.received_calls.append(kwargs)
        if not self._queue:
            raise AssertionError("StubMessagesAPI.create called with no queued response")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _StubAsyncAnthropic:
    def __init__(self) -> None:
        self.messages = _StubMessagesAPI()


def _basic_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "tool_one",
            "description": "x",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        build_web_search_tool_param(),
        {
            "name": "tool_last",
            "description": "y",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ]


# ---------------------------------------------------------------------------
# Byte-stability of the cacheable prefix.
# ---------------------------------------------------------------------------


def test_cacheable_prefix_marker_attaches_to_last_tool(fresh_db) -> None:
    client = AnthropicLLMClient(fresh_db, client=_StubAsyncAnthropic())  # type: ignore[arg-type]
    tools = _basic_tools()
    _, with_cache = client._build_cacheable_blocks(tools)
    assert with_cache[0].get("cache_control") is None
    assert with_cache[1].get("cache_control") is None
    assert with_cache[-1]["cache_control"] == {"type": "ephemeral"}


def test_cacheable_prefix_does_not_mutate_caller_list(fresh_db) -> None:
    """`_build_cacheable_blocks` must return a deep-copy so the caller's
    tool list stays clean across iterations of the agent loop."""
    client = AnthropicLLMClient(fresh_db, client=_StubAsyncAnthropic())  # type: ignore[arg-type]
    tools = _basic_tools()
    snapshot = json.dumps(tools, sort_keys=True)
    client._build_cacheable_blocks(tools)
    assert json.dumps(tools, sort_keys=True) == snapshot


def test_cacheable_prefix_digest_is_byte_stable_across_calls(fresh_db) -> None:
    """The single most important unit test in this module — if it fails,
    every call after the first becomes a cache miss."""
    client = AnthropicLLMClient(fresh_db, client=_StubAsyncAnthropic())  # type: ignore[arg-type]
    tools_first = _basic_tools()
    tools_second = _basic_tools()
    digest_first = client.cacheable_prefix_digest(tools_first)
    digest_second = client.cacheable_prefix_digest(tools_second)
    assert digest_first == digest_second


def test_cacheable_prefix_digest_changes_when_tool_set_changes(fresh_db) -> None:
    """Sanity check the digest is sensitive to actual content changes."""
    client = AnthropicLLMClient(fresh_db, client=_StubAsyncAnthropic())  # type: ignore[arg-type]
    base = _basic_tools()
    altered = _basic_tools()
    altered[0]["description"] = "z"
    assert client.cacheable_prefix_digest(base) != client.cacheable_prefix_digest(altered)


def test_cacheable_prefix_requires_at_least_one_tool(fresh_db) -> None:
    client = AnthropicLLMClient(fresh_db, client=_StubAsyncAnthropic())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        client._build_cacheable_blocks([])


# ---------------------------------------------------------------------------
# acall() — request body, response parsing, trace logging.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acall_sends_cacheable_prefix_on_the_wire(fresh_db) -> None:
    stub = _StubAsyncAnthropic()
    stub.messages.enqueue(
        _StubMessage(
            content=[_StubBlock({"type": "text", "text": "hi"})],
            usage=_StubUsage(tokens_in=100, tokens_out=50, cache_read=0, cache_create=2000),
        )
    )
    client = AnthropicLLMClient(fresh_db, client=stub)  # type: ignore[arg-type]
    tools = _basic_tools()
    await client.acall(
        messages=[{"role": "user", "content": "predict the match"}],
        tools=tools,
        tool_choice={"type": "auto"},
        max_tokens=1500,
    )
    sent = stub.messages.received_calls[0]
    assert sent["model"] == client.model
    assert sent["system"] == client.system_prompt
    # The wire copy carries the cache marker; the caller's list does not.
    assert sent["tools"][-1].get("cache_control") == {"type": "ephemeral"}
    assert tools[-1].get("cache_control") is None


@pytest.mark.asyncio
async def test_acall_parses_tool_uses_and_text(fresh_db) -> None:
    stub = _StubAsyncAnthropic()
    stub.messages.enqueue(
        _StubMessage(
            content=[
                _StubBlock({"type": "text", "text": "Thinking..."}),
                _StubBlock(
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "get_player_stats",
                        "input": {"player_name": "X", "tour": "ATP", "as_of_date": "2026-01-01"},
                    }
                ),
                _StubBlock(
                    {
                        "type": "server_tool_use",
                        "id": "ws_1",
                        "name": "web_search",
                        "input": {"query": "something"},
                    }
                ),
            ],
            stop_reason="tool_use",
            usage=_StubUsage(tokens_in=200, tokens_out=80, cache_read=1800, cache_create=0),
        )
    )
    client = AnthropicLLMClient(fresh_db, client=stub)  # type: ignore[arg-type]
    response = await client.acall(
        messages=[{"role": "user", "content": "go"}],
        tools=_basic_tools(),
        tool_choice={"type": "auto"},
        max_tokens=1500,
    )
    assert response.stop_reason == "tool_use"
    assert response.text == "Thinking..."
    assert len(response.tool_uses) == 1
    assert response.tool_uses[0].name == "get_player_stats"
    assert response.web_search_count == 1
    assert response.cache_read_tokens == 1800
    assert response.estimated_cost_usd > 0
    assert response.trace_id is not None


@pytest.mark.asyncio
async def test_acall_writes_one_llm_traces_row_per_call(fresh_db) -> None:
    stub = _StubAsyncAnthropic()
    stub.messages.enqueue(
        _StubMessage(
            content=[_StubBlock({"type": "text", "text": "ok"})],
            usage=_StubUsage(tokens_in=100, tokens_out=50, cache_read=0, cache_create=0),
        )
    )
    client = AnthropicLLMClient(fresh_db, client=stub)  # type: ignore[arg-type]
    await client.acall(
        messages=[{"role": "user", "content": "first"}],
        tools=_basic_tools(),
        tool_choice={"type": "auto"},
        max_tokens=1500,
    )
    row = fresh_db.execute(
        "SELECT model, tokens_in, tokens_out, cache_read_tokens, "
        "cache_creation_tokens, web_search_count, estimated_cost_usd, error "
        "FROM llm_traces"
    ).fetchall()
    assert len(row) == 1
    (
        model_name,
        tokens_in,
        tokens_out,
        cache_read,
        cache_creation,
        web_searches,
        cost,
        error,
    ) = row[0]
    assert model_name == client.model
    assert tokens_in == 100
    assert tokens_out == 50
    assert cache_read == 0
    assert cache_creation == 0
    assert web_searches == 0
    assert cost is not None and cost > 0
    assert error is None


@pytest.mark.asyncio
async def test_acall_wraps_api_error_in_llm_call_failure_and_logs_row(fresh_db) -> None:
    from anthropic import APIError

    stub = _StubAsyncAnthropic()
    err = APIError(message="boom", request=MagicMock(), body=None)
    stub.messages.enqueue(err)
    client = AnthropicLLMClient(fresh_db, client=stub)  # type: ignore[arg-type]
    with pytest.raises(LLMCallFailure):
        await client.acall(
            messages=[{"role": "user", "content": "fail"}],
            tools=_basic_tools(),
            tool_choice={"type": "auto"},
            max_tokens=1500,
        )
    rows = fresh_db.execute("SELECT error FROM llm_traces WHERE error IS NOT NULL").fetchall()
    assert len(rows) == 1
    assert "boom" in rows[0][0]
