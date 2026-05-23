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
)
from tennis_predictor.llm.tools.db_tools import (
    get_head_to_head,
    get_player_ranking,
    get_player_stats,
    get_recent_form,
)
from tennis_predictor.llm.tools.fetch_url import (
    FETCH_URL_TOOL,
    FETCH_URL_TOOL_NAME,
    fetch_url,
)
from tennis_predictor.llm.tools.model_tool import get_model_prediction
from tennis_predictor.llm.tools.schemas import (
    FetchUrlInput,
    GetHeadToHeadInput,
    GetModelPredictionInput,
    GetPlayerRankingInput,
    GetPlayerStatsInput,
    GetRecentFormInput,
    MatchContext,
    ModelPrediction,
    ModelUnavailableError,
    PlayerResolutionError,
    TavilyError,
    WebSearchInput,
)
from tennis_predictor.llm.tools.submit import (
    SUBMIT_ANALYSIS_TOOL,
    SUBMIT_ANALYSIS_TOOL_NAME,
    AgentResponse,
)
from tennis_predictor.llm.tools.web_search import (
    WEB_SEARCH_TOOL,
    WEB_SEARCH_TOOL_NAME,
    search_web,
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
    Sonnet 4.6 with ~2250-token cacheable prefix (Phase 5.1 added the two
    new Tavily tools, ~250 tokens). Override at construction time only for
    scoped experiments; never inline.

    Phase 5.1 added `max_fetch_urls` — separate from `max_web_searches`
    because they're different kinds of action with different cost profiles
    (snippet vs full body extraction)."""

    max_tool_iterations: int = 6
    max_total_tokens: int = 30_000
    max_wall_clock_seconds: float = 120.0
    max_web_searches: int = 3
    max_fetch_urls: int = 2
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
    fetch_urls_used: int = 0
    # Phase 5.1: Tavily costs accumulated between LLM calls. Consumed on
    # the next acall() so the trace row reflects the iteration's full cost
    # picture (Anthropic line items + Tavily charges).
    pending_tool_cost_usd: float = 0.0
    pending_web_searches: int = 0
    pending_fetch_urls: int = 0
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
        #
        # Phase 5.1 note: `web_searches_used` and `fetch_urls_used` are
        # tracked exclusively by the agent's `reserve_*` reservations.
        # `response.web_search_count` here would double-count (it already
        # includes the agent's extras merged in by `LLMClient.acall`), so
        # this method does NOT touch those counters.
        self.iterations_used += 1
        self.tokens_used += (
            response.tokens_in + response.tokens_out + response.cache_creation_tokens
        )
        self.check_within_limits()

    def reserve_web_search(self) -> bool:
        """Atomically reserve a web_search slot. Returns True on success.

        Reservation is a single sync step (asyncio is single-threaded so
        no race between read + write) — `_dispatch_tool` calls this
        BEFORE awaiting Tavily so parallel dispatches in one turn can't
        all see the same `web_searches_remaining` and overshoot.
        """
        if self.web_searches_used >= self.budget.max_web_searches:
            return False
        self.web_searches_used += 1
        self.pending_web_searches += 1
        return True

    def reserve_fetch_url(self) -> bool:
        """Atomic counterpart for fetch_url — same rationale."""
        if self.fetch_urls_used >= self.budget.max_fetch_urls:
            return False
        self.fetch_urls_used += 1
        self.pending_fetch_urls += 1
        return True

    def register_tool_search(self, cost_usd: float) -> None:
        """Record cost of one already-reserved web_search dispatch. Count
        is incremented separately by `reserve_web_search`."""
        self.pending_tool_cost_usd += cost_usd
        self.check_within_limits()

    def register_tool_fetch(self, cost_usd: float) -> None:
        """Record cost of one already-reserved fetch_url dispatch."""
        self.pending_tool_cost_usd += cost_usd
        self.check_within_limits()

    def refund_web_search(self) -> None:
        """Undo a reservation when the Tavily call failed before incurring
        any cost (e.g., raised before reaching the API)."""
        self.web_searches_used -= 1
        self.pending_web_searches -= 1

    def refund_fetch_url(self) -> None:
        """Undo a fetch_url reservation."""
        self.fetch_urls_used -= 1
        self.pending_fetch_urls -= 1

    def consume_pending(self) -> tuple[float, int, int]:
        """Return (cost_usd, web_search_count, fetch_url_count) accumulated
        since the last LLM call, and reset the pending counters. Called
        right before `LLMClient.acall()` so the trace row attributes the
        Tavily activity to the iteration it preceded."""
        cost = self.pending_tool_cost_usd
        searches = self.pending_web_searches
        fetches = self.pending_fetch_urls
        self.pending_tool_cost_usd = 0.0
        self.pending_web_searches = 0
        self.pending_fetch_urls = 0
        return cost, searches, fetches

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

    def fetch_urls_remaining(self) -> int:
        return self.budget.max_fetch_urls - self.fetch_urls_used

    # ------------------------------------------------------------------

    def should_force_submit(self) -> bool:
        """True when the next call should hard-force `submit_analysis`.

        We force one iteration early so the LLM doesn't get cut off
        mid-thought. Web-search and fetch-url exhaustion also force
        submit — if the agent has used all of either, more data tools
        won't help and we want it to wrap up."""
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
        if self.fetch_urls_used > self.budget.max_fetch_urls:
            raise BudgetExceededError(
                f"max_fetch_urls exceeded: {self.fetch_urls_used} > {self.budget.max_fetch_urls}"
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
            # Hand off the Tavily activity that happened since the last LLM
            # call so the trace row reflects it. consume_pending zeros the
            # counters; if the LLM call fails the cost is logged on the
            # error trace row.
            pending_cost, pending_searches, pending_fetches = tracker.consume_pending()
            try:
                response = await self._llm.acall(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=self._budget.output_max_tokens,
                    extra_tool_cost_usd=pending_cost,
                    extra_web_search_count=pending_searches,
                    extra_fetch_url_count=pending_fetches,
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

            # Append the assistant turn verbatim and dispatch every
            # client-side tool call concurrently. Tavily HTTP dominates
            # the dispatch latency, so parallel makes a real difference
            # when the LLM asks for stats + H2H + web_search in one turn.
            messages.append({"role": "assistant", "content": response.raw_content})
            tool_result_blocks = await asyncio.gather(
                *[self._dispatch_tool(ctx, use, tracker) for use in other_uses]
            )
            blocks = list(tool_result_blocks)
            # Phase 5.1 rolling cache: drop a `cache_control` marker on the
            # LAST tool_result block, BUT only when we expect at least one
            # more data-gathering iteration. Empirically (live smoke
            # 2026-05-23) this saves ~$0.02 per multi-iteration predict;
            # skipping the marker on the iteration that's about to
            # force-submit avoids paying ~$0.02-0.05 of cache_creation
            # surcharge for content that's never read back from cache.
            if blocks and not tracker.should_force_submit():
                blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            messages.append({"role": "user", "content": blocks})

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tool(
        self,
        ctx: MatchContext,
        use: LLMToolUse,
        tracker: BudgetTracker,
    ) -> dict[str, Any]:
        """Run one client-side tool and return the `tool_result` block
        that goes back to the model on the next turn.

        Pre-flight budget check: if dispatching this tool would push the
        agent over its per-call cap (`max_web_searches`, `max_fetch_urls`),
        refuse without calling the vendor. The LLM gets an `is_error`
        tool_result explaining the limit so it can adapt on the next turn
        (typically by calling `submit_analysis`). Without this gate, a
        single LLM turn requesting N+1 searches would crash the loop with
        a `BudgetExceededError`.

        Three error families that fall through to soft failure:
        - `PlayerResolutionError` (DB tools) — name not in `player_aliases`.
        - `TavilyError` (web_search / fetch_url) — Tavily HTTP failure.
        - Pre-flight budget refusal — message tells the LLM what to do next.

        Other exceptions bubble (programming bugs, CLAUDE.md failure-mode #3).
        """
        # Pre-flight: ATOMICALLY reserve a Tavily budget slot. asyncio is
        # single-threaded so reserve_* is race-free; this gates parallel
        # tool dispatches in a single turn from overshooting the cap.
        if use.name == WEB_SEARCH_TOOL_NAME and not tracker.reserve_web_search():
            return _error_block(
                use,
                "web_search budget exhausted for this prediction "
                f"({tracker.budget.max_web_searches} calls used). "
                "Call `submit_analysis` next with what you have.",
            )
        if use.name == FETCH_URL_TOOL_NAME and not tracker.reserve_fetch_url():
            return _error_block(
                use,
                "fetch_url budget exhausted for this prediction "
                f"({tracker.budget.max_fetch_urls} calls used). "
                "Call `submit_analysis` next with what you have.",
            )

        try:
            result = await _run_client_tool(self._conn, ctx, use, tracker)
        except (PlayerResolutionError, TavilyError) as exc:
            # Tavily failed — refund the reservation so a retry could go
            # through (LLM unlikely to retry the same query, but the budget
            # accounting stays honest).
            if use.name == WEB_SEARCH_TOOL_NAME:
                tracker.refund_web_search()
            elif use.name == FETCH_URL_TOOL_NAME:
                tracker.refund_fetch_url()
            return _error_block(use, str(exc))
        return {
            "type": "tool_result",
            "tool_use_id": use.id,
            "content": [{"type": "text", "text": result}],
        }


def _error_block(use: LLMToolUse, message: str) -> dict[str, Any]:
    """Standard tool_result shape for a soft-failure case."""
    return {
        "type": "tool_result",
        "tool_use_id": use.id,
        "is_error": True,
        "content": [{"type": "text", "text": message}],
    }


# ---------------------------------------------------------------------------
# Helpers — built-in tools list, initial user message, tool dispatch.
# ---------------------------------------------------------------------------


# Pydantic input models per tool — used to validate the LLM's JSON before
# dispatch. Order doesn't matter for cache hashing because the `tools`
# list (built below) IS what gets hashed.
_CLIENT_TOOL_INPUT_MODELS = {
    "get_player_stats": GetPlayerStatsInput,
    "get_head_to_head": GetHeadToHeadInput,
    "get_recent_form": GetRecentFormInput,
    "get_player_ranking": GetPlayerRankingInput,
    WEB_SEARCH_TOOL_NAME: WebSearchInput,
    FETCH_URL_TOOL_NAME: FetchUrlInput,
}


def _build_tools_list() -> list[dict[str, Any]]:
    """Compose the full tool list. Order matters for cache hashing — the
    LAST tool gets the `cache_control` marker in `LLMClient`, so keep
    `submit_analysis` last and the rest in a stable order.

    Phase 5.1 added Tavily-backed `web_search` and `fetch_url` as
    client-side tools (replacing Anthropic's server-side native search).
    They sit between the DB tools and `submit_analysis` so the cache
    marker still lands on a stable tool definition."""
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
        WEB_SEARCH_TOOL,
        FETCH_URL_TOOL,
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


async def _run_client_tool(
    conn: duckdb.DuckDBPyConnection,
    ctx: MatchContext,
    use: LLMToolUse,
    tracker: BudgetTracker,
) -> str:
    """Validate the LLM's JSON, run the tool, JSON-encode the result.

    DB tools are synchronous (DuckDB is in-process); Tavily tools are
    async (HTTP). Async signature lets `_dispatch_tool` run multiple
    tools concurrently via `asyncio.gather`.

    `tracker` is mutated for Tavily tools (search/fetch budget + pending
    cost accounting). DB tools don't touch the tracker — they're free
    and not separately budgeted (they count via `register_iteration`
    which sees the overall token usage).
    """
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
    if use.name == WEB_SEARCH_TOOL_NAME:
        assert isinstance(validated, WebSearchInput)
        result = await search_web(validated)
        tracker.register_tool_search(result.cost_usd)
        return result.model_dump_json()
    if use.name == FETCH_URL_TOOL_NAME:
        assert isinstance(validated, FetchUrlInput)
        result = await fetch_url(validated)
        tracker.register_tool_fetch(result.cost_usd)
        return result.model_dump_json()
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
