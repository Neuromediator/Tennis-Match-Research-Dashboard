"""Tennis agent orchestrator + per-call budget enforcement.

Two responsibilities in this file:

1. `AgentBudget` / `BudgetTracker` — bounded per-call resource accounting.
   CLAUDE.md "Budget discipline" lists this as one of three layers; the
   org-level $20/month hard cap is the wall, this is the per-call ceiling.

2. `TennisAgent.predict()` — the actual agent loop. Calls
   `get_model_prediction` first (Hard Rule #10 — fatal if it fails),
   then runs an Anthropic conversation with hybrid `tool_choice`:
   `"auto"` while gathering data, hard-forced to `submit_analysis` on the
   final iteration. Wraps the whole loop in `asyncio.timeout(120)` so a
   stuck call cannot run forever.

The agent loop never raises a raw LLM error to its caller — it logs the
partial state to `llm_traces` (via `LLMClient._log_trace`) and surfaces a
typed `AgentError` instead. CLAUDE.md "LLM agent failure modes" lists the
six surfaces the loop must distinguish; the dispatch matrix in
`_dispatch_tool` and the error mapping in `predict()` keep them honest.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import duckdb
from pydantic import ValidationError

from tennis_predictor.llm.client import (
    AnthropicLLMClient,
    LLMCallFailure,
    LLMClient,
    LLMResponse,
    LLMToolUse,
    build_web_search_tool_param,
)
from tennis_predictor.llm.tools.db_tools import (
    get_head_to_head,
    get_player_ranking,
    get_player_stats,
    get_recent_form,
)
from tennis_predictor.llm.tools.model_tool import get_model_prediction
from tennis_predictor.llm.tools.schemas import (
    GetHeadToHeadInput,
    GetModelPredictionInput,
    GetPlayerRankingInput,
    GetPlayerStatsInput,
    GetRecentFormInput,
    MatchContext,
    ModelPrediction,
    ModelUnavailableError,
    PlayerResolutionError,
)
from tennis_predictor.llm.tools.submit import (
    SUBMIT_ANALYSIS_TOOL,
    SUBMIT_ANALYSIS_TOOL_NAME,
    AgentResponse,
)

logger = logging.getLogger(__name__)

# Per-iteration token / wall-clock thresholds for "force submit". These
# are softer than the hard limits: we cut over to `tool_choice = submit`
# when fewer than one full extra iteration's worth of budget is left, so
# the LLM still has room to finalise instead of being mid-thought when
# the cap actually trips.
_FORCE_SUBMIT_TOKEN_BUFFER: int = 4_000
_FORCE_SUBMIT_WALL_CLOCK_BUFFER: float = 15.0


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentBudget:
    """Per-`predict()` hard limits.

    The defaults come from CLAUDE.md "Budget discipline" and are tuned for
    Sonnet 4.6 with ~2000-token cacheable prefix. Override at construction
    time only for scoped experiments; never inline."""

    max_tool_iterations: int = 6
    max_total_tokens: int = 30_000
    max_wall_clock_seconds: float = 120.0
    max_web_searches: int = 3
    output_max_tokens: int = 1500


class BudgetExceededError(Exception):
    """Hard-limit overrun. Carries a short reason for the operator."""


@dataclass
class BudgetTracker:
    """Runtime accumulator paired with one `AgentBudget`. One instance
    per `predict()` call.

    `register_iteration` is called after each successful LLM call to
    update counters from the parsed `LLMResponse`. Helper predicates
    expose remaining headroom so the orchestrator can switch to
    `tool_choice = submit_analysis` *before* the next call would breach a
    limit, rather than reactively raising.
    """

    budget: AgentBudget
    iterations_used: int = 0
    tokens_used: int = 0
    web_searches_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # ------------------------------------------------------------------

    def register_iteration(self, response: LLMResponse) -> None:
        # The budget is cost-weighted, not token-weighted: we count fresh
        # input, cache-creation input (full price + 25% surcharge), and
        # output, but NOT cache reads. Cache reads are billed at ~10% and
        # represent the same prefix read repeatedly; counting them would
        # punish the very mechanism prompt caching was designed to enable.
        # The org-level $20/month hard cap remains the real wall — this
        # is just per-call sanity. Live `llm_traces.estimated_cost_usd`
        # captures the full picture for the dashboard.
        self.iterations_used += 1
        self.tokens_used += (
            response.tokens_in + response.tokens_out + response.cache_creation_tokens
        )
        self.web_searches_used += response.web_search_count
        self.check_within_limits()

    # ------------------------------------------------------------------

    def wall_clock_elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def wall_clock_remaining(self) -> float:
        return self.budget.max_wall_clock_seconds - self.wall_clock_elapsed()

    def iterations_remaining(self) -> int:
        return self.budget.max_tool_iterations - self.iterations_used

    def tokens_remaining(self) -> int:
        return self.budget.max_total_tokens - self.tokens_used

    def web_searches_remaining(self) -> int:
        return self.budget.max_web_searches - self.web_searches_used

    # ------------------------------------------------------------------

    def should_force_submit(self) -> bool:
        """True when the next call should hard-force `submit_analysis`.

        We force one iteration early so the LLM doesn't get cut off
        mid-thought. The web-search exhaustion case also forces submit
        — if the agent has used all 3 searches, more data tools won't
        help and we want it to wrap up."""
        return (
            self.iterations_remaining() <= 1
            or self.tokens_remaining() <= _FORCE_SUBMIT_TOKEN_BUFFER
            or self.wall_clock_remaining() <= _FORCE_SUBMIT_WALL_CLOCK_BUFFER
            or self.web_searches_remaining() <= 0
        )

    def check_within_limits(self) -> None:
        """Raise `BudgetExceededError` if any hard limit has been crossed."""
        if self.iterations_used > self.budget.max_tool_iterations:
            raise BudgetExceededError(
                f"max_tool_iterations exceeded: {self.iterations_used} > "
                f"{self.budget.max_tool_iterations}"
            )
        if self.tokens_used > self.budget.max_total_tokens:
            raise BudgetExceededError(
                f"max_total_tokens exceeded: {self.tokens_used} > {self.budget.max_total_tokens}"
            )
        if self.web_searches_used > self.budget.max_web_searches:
            raise BudgetExceededError(
                f"max_web_searches exceeded: {self.web_searches_used} > "
                f"{self.budget.max_web_searches}"
            )
        if self.wall_clock_elapsed() > self.budget.max_wall_clock_seconds:
            raise BudgetExceededError(
                f"max_wall_clock_seconds exceeded: {self.wall_clock_elapsed():.1f} > "
                f"{self.budget.max_wall_clock_seconds}"
            )


# ---------------------------------------------------------------------------
# Agent errors
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Surface error type for `TennisAgent.predict`. Wraps any of the six
    failure surfaces from CLAUDE.md so callers handle one type."""


class AgentTimeoutError(AgentError):
    """Loop exceeded `AgentBudget.max_wall_clock_seconds`."""


class AgentNoSubmissionError(AgentError):
    """Loop terminated without ever calling `submit_analysis`. Should be
    rare — happens when the LLM ignored the forced tool_choice on the
    last iteration."""


# ---------------------------------------------------------------------------
# TennisAgent
# ---------------------------------------------------------------------------


class TennisAgent:
    """Orchestrates the per-match prediction loop.

    Lifecycle of one `predict(match_context)` call:

    1. Run `get_model_prediction` synchronously. Fatal if it raises
       `ModelUnavailableError` — we never invoke the LLM without the
       calibrated number (Hard Rule #10).

    2. Build the cacheable tool list (data tools + web_search +
       submit_analysis, with `submit_analysis` LAST so the cache marker
       attaches to a tool whose schema is stable across releases).

    3. Drive an Anthropic conversation with `tool_choice = "auto"`. After
       each LLM response, dispatch every `tool_use` block. Switch to
       `tool_choice = {"type": "tool", "name": "submit_analysis"}` when
       `BudgetTracker.should_force_submit()` flips True or after we see
       the model emit a submit on its own.

    4. On the submit, construct `AgentResponse` by merging the LLM's
       qualitative fields with the model's probability (from step 1).
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        llm_client: LLMClient | None = None,
        budget: AgentBudget | None = None,
    ) -> None:
        self._conn = conn
        self._llm = llm_client or AnthropicLLMClient(conn)
        self._budget = budget or AgentBudget()

    # ------------------------------------------------------------------

    async def predict(self, match_context: MatchContext) -> AgentResponse:
        """Run the agent for one match. Raises `AgentError` subclasses on
        failure; `ModelUnavailableError` is allowed to bubble (Hard Rule
        #10: the caller decides what to do; we will NOT show a prediction
        without the model number)."""
        model_prediction = self._run_model(match_context)
        try:
            return await asyncio.wait_for(
                self._run_loop(match_context, model_prediction),
                timeout=self._budget.max_wall_clock_seconds,
            )
        except TimeoutError as exc:
            raise AgentTimeoutError(
                f"agent loop exceeded {self._budget.max_wall_clock_seconds}s"
            ) from exc

    # ------------------------------------------------------------------

    def _run_model(self, ctx: MatchContext) -> ModelPrediction:
        """Step 1 — call `get_model_prediction` synchronously. Fatal on
        `ModelUnavailableError` or `PlayerResolutionError` (we can't even
        identify the players).

        `PlayerResolutionError` here aborts the prediction; inside the
        LLM loop the same error is converted to a tool error block (the
        agent might recover by trying a different name).
        """
        return get_model_prediction(
            self._conn,
            GetModelPredictionInput(
                player_a_name=ctx.player_a_name,
                player_b_name=ctx.player_b_name,
                tour=ctx.tour,
                surface=ctx.surface,
                tournament_level=ctx.tournament_level,
                best_of=ctx.best_of,
                match_date=ctx.match_date,
            ),
        )

    # ------------------------------------------------------------------
    # The LLM loop
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        ctx: MatchContext,
        model_prediction: ModelPrediction,
    ) -> AgentResponse:
        tools = _build_tools_list()
        tracker = BudgetTracker(self._budget)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _initial_user_text(ctx, model_prediction)},
        ]

        while True:
            if tracker.iterations_remaining() <= 0:
                raise AgentNoSubmissionError(
                    "agent exhausted iteration budget without calling submit_analysis"
                )

            tool_choice: dict[str, Any] = (
                {"type": "tool", "name": SUBMIT_ANALYSIS_TOOL_NAME}
                if tracker.should_force_submit()
                else {"type": "auto"}
            )
            try:
                response = await self._llm.acall(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=self._budget.output_max_tokens,
                )
            except LLMCallFailure as exc:
                raise AgentError(f"LLM call failed: {exc}") from exc

            tracker.register_iteration(response)

            submit_use, other_uses = _partition_tool_uses(response.tool_uses)
            if submit_use is not None:
                return _build_agent_response(model_prediction, submit_use)

            if not other_uses:
                # The model returned no tool_use and no submit — usually a
                # text-only assistant turn. Nudge it forward by appending
                # an instruction to use submit_analysis next.
                messages.append({"role": "assistant", "content": response.raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Call `submit_analysis` now with your synthesis. "
                            "Do not include a probability field."
                        ),
                    }
                )
                continue

            # Append the assistant turn (verbatim, including server_tool_use
            # blocks which Anthropic re-checks on the next request) and
            # dispatch each client-side tool call.
            messages.append({"role": "assistant", "content": response.raw_content})
            tool_result_blocks = [self._dispatch_tool(ctx, use) for use in other_uses]
            messages.append({"role": "user", "content": tool_result_blocks})

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, ctx: MatchContext, use: LLMToolUse) -> dict[str, Any]:
        """Run one client-side tool and return the `tool_result` block
        that goes back to the model on the next turn. DB tool exceptions
        bubble (Hard rule from CLAUDE.md failure-mode 3); only
        `PlayerResolutionError` is caught and returned as a tool error so
        the LLM can mention the unresolved name in `caveats`."""
        try:
            result = _run_client_tool(self._conn, ctx, use)
        except PlayerResolutionError as exc:
            return {
                "type": "tool_result",
                "tool_use_id": use.id,
                "is_error": True,
                "content": [{"type": "text", "text": str(exc)}],
            }
        return {
            "type": "tool_result",
            "tool_use_id": use.id,
            "content": [{"type": "text", "text": result}],
        }


# ---------------------------------------------------------------------------
# Helpers — built-in tools list, initial user message, tool dispatch.
# ---------------------------------------------------------------------------


# Pydantic input models per tool — used to validate the LLM's JSON before
# we touch the DB. Order doesn't matter for cache hashing because the
# `tools` list (built below) IS what gets hashed.
_CLIENT_TOOL_INPUT_MODELS = {
    "get_player_stats": GetPlayerStatsInput,
    "get_head_to_head": GetHeadToHeadInput,
    "get_recent_form": GetRecentFormInput,
    "get_player_ranking": GetPlayerRankingInput,
}


def _build_tools_list() -> list[dict[str, Any]]:
    """Compose the full tool list. Order matters for cache hashing — the
    LAST tool gets the `cache_control` marker in `LLMClient`, so keep
    `submit_analysis` last and the rest in a stable order."""
    return [
        {
            "name": "get_player_stats",
            "description": (
                "Career win/loss and per-surface tallies for one player as of "
                "a given date. Use to anchor surface fit and overall workload."
            ),
            "input_schema": _strip_default_keys(GetPlayerStatsInput.model_json_schema()),
        },
        {
            "name": "get_head_to_head",
            "description": (
                "Head-to-head record between two players up to a given date. "
                "Returns aggregate wins plus every recorded meeting (oldest first)."
            ),
            "input_schema": _strip_default_keys(GetHeadToHeadInput.model_json_schema()),
        },
        {
            "name": "get_recent_form",
            "description": (
                "Most recent N matches for one player, newest-first. W/L is from "
                "the queried player's perspective. Defaults to 10 matches."
            ),
            "input_schema": _strip_default_keys(GetRecentFormInput.model_json_schema()),
        },
        {
            "name": "get_player_ranking",
            "description": (
                "Singles ranking on or just before a given date. `rank` is null "
                "when the player was unranked at that date."
            ),
            "input_schema": _strip_default_keys(GetPlayerRankingInput.model_json_schema()),
        },
        build_web_search_tool_param(),
        SUBMIT_ANALYSIS_TOOL,
    ]


def _strip_default_keys(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove Pydantic's `title` keys from a generated JSON schema. They
    don't affect validation, drift across minor Pydantic versions, and
    would compromise the cacheable-prefix byte-stability check."""
    cleaned = {k: v for k, v in schema.items() if k != "title"}
    if "properties" in cleaned:
        cleaned["properties"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "title"}
            for k, v in cleaned["properties"].items()
        }
    return cleaned


def _initial_user_text(ctx: MatchContext, prediction: ModelPrediction) -> str:
    """Per-turn user message. The current date and full match context go
    HERE (not in the system prompt) so the cacheable prefix stays
    byte-stable across calls."""
    parts = [
        f"Match: {ctx.player_a_name} vs {ctx.player_b_name} ({ctx.tour}).",
        f"Surface: {ctx.surface}; tournament level: {ctx.tournament_level}; best of {ctx.best_of}.",
        f"Match date: {ctx.match_date.isoformat()}.",
    ]
    if ctx.tournament_name:
        parts.append(f"Tournament: {ctx.tournament_name}.")
    parts.extend(
        [
            "",
            "Model probability (from `get_model_prediction`, ALREADY CALLED for you "
            "before this turn):",
            f"  P({ctx.player_a_name} wins) = {prediction.model_probability_player_a:.3f}",
            f"  P({ctx.player_b_name} wins) = {prediction.model_probability_player_b:.3f}",
            f"  Model artifact: {prediction.model_artifact_version}",
            "",
            "Feature summary the model conditioned on:",
            json.dumps(prediction.feature_summary.model_dump(), indent=2, default=str),
            "",
            "Now build the context picture with the other tools and finish with `submit_analysis`.",
        ]
    )
    return "\n".join(parts)


def _partition_tool_uses(
    uses: list[LLMToolUse],
) -> tuple[LLMToolUse | None, list[LLMToolUse]]:
    submit: LLMToolUse | None = None
    others: list[LLMToolUse] = []
    for u in uses:
        if u.name == SUBMIT_ANALYSIS_TOOL_NAME and submit is None:
            submit = u
        else:
            others.append(u)
    return submit, others


def _run_client_tool(
    conn: duckdb.DuckDBPyConnection,
    ctx: MatchContext,
    use: LLMToolUse,
) -> str:
    """Validate the LLM's JSON, run the tool, JSON-encode the result."""
    if use.name not in _CLIENT_TOOL_INPUT_MODELS:
        raise AgentError(f"unknown tool {use.name!r}")

    input_model = _CLIENT_TOOL_INPUT_MODELS[use.name]
    try:
        validated = input_model.model_validate(use.input)
    except ValidationError as exc:
        # Surface schema mistakes as tool-result errors so the LLM can
        # retry with the right field names rather than crashing the loop.
        raise PlayerResolutionError(f"invalid arguments for {use.name}: {exc.errors()}") from exc

    if use.name == "get_player_stats":
        assert isinstance(validated, GetPlayerStatsInput)
        return get_player_stats(conn, validated).model_dump_json()
    if use.name == "get_head_to_head":
        assert isinstance(validated, GetHeadToHeadInput)
        return get_head_to_head(conn, validated).model_dump_json()
    if use.name == "get_recent_form":
        assert isinstance(validated, GetRecentFormInput)
        return get_recent_form(conn, validated).model_dump_json()
    if use.name == "get_player_ranking":
        assert isinstance(validated, GetPlayerRankingInput)
        return get_player_ranking(conn, validated).model_dump_json()
    raise AgentError(f"unhandled tool {use.name!r}")  # unreachable


def _build_agent_response(
    prediction: ModelPrediction,
    submit_use: LLMToolUse,
) -> AgentResponse:
    """Merge the LLM's qualitative payload with the model's probability.
    Pydantic's `extra="forbid"` rejects any LLM-emitted probability field
    a second time here (the JSON-schema layer is the first wall)."""
    merged = {
        **submit_use.input,
        "model_probability_player_a": prediction.model_probability_player_a,
        "model_probability_player_b": prediction.model_probability_player_b,
    }
    try:
        return AgentResponse.model_validate(merged)
    except ValidationError as exc:
        raise AgentError(f"submit_analysis returned invalid payload: {exc}") from exc


__all__ = [
    "AgentBudget",
    "AgentError",
    "AgentNoSubmissionError",
    "AgentTimeoutError",
    "BudgetExceededError",
    "BudgetTracker",
    "ModelUnavailableError",
    "TennisAgent",
]
