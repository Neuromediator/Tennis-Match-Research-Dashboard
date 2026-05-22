"""LLMClient abstraction + Anthropic implementation.

# Why an ABC

Vendor flexibility is a Phase-5 contract value — but we don't pay for it
with a third-party abstraction (LangChain, LiteLLM, Managed Agents). A new
provider becomes a new ~100-line `LLMClient` subclass; the agent loop in
`llm/agent.py` and every tool stay unchanged.

# What `acall()` does for every call

1. Builds a request body that is **byte-stable for the cacheable prefix**:
   system prompt + tool definitions in a fixed order, with a single
   `cache_control: {"type": "ephemeral"}` marker on the last tool. The
   marker tells Anthropic to cache everything before-and-including it; the
   user message and per-turn tool results stay uncached.

2. Sends the request via the official `AsyncAnthropic` client, which
   provides built-in retry (`max_retries=2`) for 5xx / 429.

3. Wraps any `APIError` in our typed `LLMCallFailure`. The agent loop
   never sees raw Anthropic exceptions — they would couple our retry
   policy and dashboards to SDK internals.

4. Logs one row to `llm_traces` with token / cache / cost / latency
   stats. Logging happens even on failure (so the dashboard can show
   error rates).

# Cache hit hygiene

A separate `_build_cacheable_blocks()` helper returns the bytes that go
into the cacheable prefix. A unit test (Tier-1) calls it twice and
asserts byte-equality — if that test fails, the cache-hit rate just
dropped to ~0% and we want to know immediately, not at the cost-review
end of the month.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import duckdb
from anthropic import APIError, AsyncAnthropic
from pydantic import BaseModel, ConfigDict, Field

from tennis_predictor.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from tennis_predictor.llm.cost import estimate_call_cost
from tennis_predictor.llm.prompts import SYSTEM_PROMPT, system_prompt_hash

logger = logging.getLogger(__name__)

# Web-search blocklist — the small set of betting / pick-of-the-day domains
# from CLAUDE.md "Web search". Kept centralised so swapping vendor is a
# one-place change.
WEB_SEARCH_BLOCKED_DOMAINS: tuple[str, ...] = (
    "draftkings.com",
    "fanduel.com",
    "betmgm.com",
    "pickwise.com",
    "actionnetwork.com",
)
WEB_SEARCH_MAX_USES: int = 3


class LLMCallFailure(Exception):
    """Wraps an Anthropic APIError. Carries the original exception in
    `__cause__` so debugging is still cheap."""


class LLMToolUse(BaseModel):
    """One client-side tool-use block returned by the model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    input: dict[str, Any]


class LLMResponse(BaseModel):
    """Parsed view of one Anthropic API call.

    `raw_content` is the SDK's content list as JSON-serialisable dicts —
    enough for the agent loop to re-feed it back as an `assistant`
    message on the next turn. `tool_uses` is the parsed client-side
    tool-use blocks the loop dispatches; `text` is the concatenated
    user-visible text. `server_tool_uses` counts e.g. `web_search` blocks
    (the model executes those server-side, no dispatch needed)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stop_reason: str | None
    raw_content: list[dict[str, Any]]
    text: str
    tool_uses: list[LLMToolUse]
    server_tool_uses: list[str] = Field(default_factory=list)

    tokens_in: int
    tokens_out: int
    cache_read_tokens: int
    cache_creation_tokens: int

    web_search_count: int
    estimated_cost_usd: float
    latency_ms: int

    trace_id: int | None = None


class LLMClient(ABC):
    """Abstract base for any vendor-specific LLM implementation."""

    @abstractmethod
    async def acall(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any],
        max_tokens: int,
    ) -> LLMResponse:
        """Send one turn to the model. Implementations MUST log to
        `llm_traces` (success or failure) before returning / raising."""


class AnthropicLLMClient(LLMClient):
    """Direct Anthropic SDK implementation. No abstraction-layer wrappers."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        model: str | None = None,
        client: AsyncAnthropic | None = None,
        max_retries: int = 2,
    ) -> None:
        if client is None and ANTHROPIC_API_KEY is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env or pass a custom "
                "AsyncAnthropic client (tests use this seam)."
            )
        self._conn = conn
        self._model = model or ANTHROPIC_MODEL
        self._client = client or AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            max_retries=max_retries,
        )

    # ------------------------------------------------------------------
    # Cacheable-prefix helpers — exposed as methods so tests can assert
    # byte-stability without reaching into the call path.
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _build_cacheable_blocks(
        self,
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return (system, tools) exactly as they'll go on the wire.

        - `system` is the raw `SYSTEM_PROMPT` string.
        - `tools` is a deep-copy of the caller's list with a single
          `cache_control: {"type": "ephemeral"}` attached to the LAST
          element. The deep-copy is what makes this safe to call from
          tests and from the live path without sharing mutable state.

        Anthropic caches everything *before and including* the marker, so
        one marker on the last tool caches the entire system + tools
        prefix. The user message is intentionally NOT cached — it changes
        per match.
        """
        if not tools:
            raise ValueError("at least one tool required to anchor the cache marker")
        rebuilt: list[dict[str, Any]] = [copy.deepcopy(t) for t in tools]
        rebuilt[-1] = {**rebuilt[-1], "cache_control": {"type": "ephemeral"}}
        return SYSTEM_PROMPT, rebuilt

    def cacheable_prefix_digest(self, tools: list[dict[str, Any]]) -> str:
        """SHA256 of (system + tools_json). The byte-stability test asserts
        two consecutive calls produce the same digest."""
        system, tools_cached = self._build_cacheable_blocks(tools)
        payload = system + "\n---\n" + json.dumps(tools_cached, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def acall(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any],
        max_tokens: int,
    ) -> LLMResponse:
        system, tools_with_cache = self._build_cacheable_blocks(tools)
        request_body: dict[str, Any] = {
            "model": self._model,
            "system": system,
            "messages": messages,
            "tools": tools_with_cache,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
        }

        started_at = time.monotonic()
        error_text: str | None = None
        response = None
        try:
            response = await self._client.messages.create(**request_body)
        except APIError as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            latency_ms = int((time.monotonic() - started_at) * 1000)
            self._log_trace(
                messages=messages,
                response=None,
                error=error_text,
                latency_ms=latency_ms,
                cost_usd=0.0,
                web_search_count=0,
            )
            raise LLMCallFailure(error_text) from exc

        latency_ms = int((time.monotonic() - started_at) * 1000)
        parsed = _parse_response(response, latency_ms=latency_ms, model=self._model)
        trace_id = self._log_trace(
            messages=messages,
            response=parsed,
            error=None,
            latency_ms=latency_ms,
            cost_usd=parsed.estimated_cost_usd,
            web_search_count=parsed.web_search_count,
        )
        return parsed.model_copy(update={"trace_id": trace_id})

    # ------------------------------------------------------------------
    # Trace logging
    # ------------------------------------------------------------------

    def _log_trace(
        self,
        *,
        messages: list[dict[str, Any]],
        response: LLMResponse | None,
        error: str | None,
        latency_ms: int,
        cost_usd: float,
        web_search_count: int,
    ) -> int | None:
        """Insert one row into `llm_traces`. Returns the inserted trace_id,
        or None if the insert itself raises (we still propagate the
        original API error in that case rather than masking it)."""
        try:
            row = self._conn.execute(
                """
                INSERT INTO llm_traces (
                    ts, model, system_prompt_hash, input_messages, tool_calls,
                    output, tokens_in, tokens_out, cache_read_tokens,
                    cache_creation_tokens, latency_ms, error,
                    web_search_count, estimated_cost_usd
                ) VALUES (
                    CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                RETURNING trace_id
                """,
                [
                    self._model,
                    system_prompt_hash(),
                    json.dumps(messages, default=str),
                    json.dumps([t.model_dump() for t in response.tool_uses])
                    if response is not None
                    else None,
                    json.dumps(response.raw_content) if response is not None else None,
                    response.tokens_in if response is not None else None,
                    response.tokens_out if response is not None else None,
                    response.cache_read_tokens if response is not None else None,
                    response.cache_creation_tokens if response is not None else None,
                    latency_ms,
                    error,
                    web_search_count,
                    cost_usd,
                ],
            ).fetchone()
            return int(row[0]) if row is not None else None
        except Exception:
            logger.exception("failed to write llm_traces row; trace dropped")
            return None


def _parse_response(response: Any, *, latency_ms: int, model: str) -> LLMResponse:
    """Translate the SDK's `Message` object into the strongly-typed
    `LLMResponse` the agent loop consumes."""
    usage = getattr(response, "usage", None)
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    tokens_out = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
    cache_creation_tokens = (
        int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
    )

    text_chunks: list[str] = []
    tool_uses: list[LLMToolUse] = []
    server_tool_uses: list[str] = []
    raw_content: list[dict[str, Any]] = []
    web_search_count = 0

    for block in response.content:
        # The SDK's content blocks are pydantic models; `model_dump()` gives
        # the JSON-serialisable form we re-feed on the next turn.
        block_dump = block.model_dump() if hasattr(block, "model_dump") else dict(block)
        raw_content.append(block_dump)

        block_type = block_dump.get("type")
        if block_type == "text":
            text_chunks.append(block_dump.get("text", ""))
        elif block_type == "tool_use":
            tool_uses.append(
                LLMToolUse(
                    id=block_dump["id"],
                    name=block_dump["name"],
                    input=block_dump.get("input", {}) or {},
                )
            )
        elif block_type in ("server_tool_use", "web_search_tool_result"):
            server_tool_uses.append(block_dump.get("name", block_type))
            if block_type == "server_tool_use" and block_dump.get("name") == "web_search":
                web_search_count += 1

    estimated_cost_usd = estimate_call_cost(
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        web_search_count=web_search_count,
    )

    return LLMResponse(
        stop_reason=getattr(response, "stop_reason", None),
        raw_content=raw_content,
        text="".join(text_chunks),
        tool_uses=tool_uses,
        server_tool_uses=server_tool_uses,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        web_search_count=web_search_count,
        estimated_cost_usd=estimated_cost_usd,
        latency_ms=latency_ms,
    )


def build_web_search_tool_param() -> dict[str, Any]:
    """Return the `web_search` server-tool definition with the
    project-wide configuration applied. Excluded from
    `_build_cacheable_blocks` cache-key concerns because it's part of the
    `tools` list — the marker still attaches to the LAST tool in the
    final list, so this is a normal entry."""
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": WEB_SEARCH_MAX_USES,
        "blocked_domains": list(WEB_SEARCH_BLOCKED_DOMAINS),
    }


__all__ = [
    "WEB_SEARCH_BLOCKED_DOMAINS",
    "WEB_SEARCH_MAX_USES",
    "AnthropicLLMClient",
    "LLMCallFailure",
    "LLMClient",
    "LLMResponse",
    "LLMToolUse",
    "build_web_search_tool_param",
]
