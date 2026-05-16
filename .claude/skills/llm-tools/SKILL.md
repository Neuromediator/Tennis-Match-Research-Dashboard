---
name: llm-tools
description: Use when defining LLM tools, designing the structured output, working with the LLMClient abstraction, or extending prompt-caching/observability behavior. Establishes the rule that the LLM does not emit a probability.
---

# LLM tool definitions and structured output

## Tools the LLM may call (v1)

| Tool | Purpose |
|---|---|
| `get_player_stats(player_name, as_of_date)` | Career and recent stats. |
| `get_head_to_head(player_a, player_b)` | H2H record and recent meetings. |
| `get_recent_form(player_name, n_matches=10)` | Last N matches with results. |
| `get_model_prediction(player_a, player_b, surface, tournament_level, match_date)` | Calibrated probability from the ML model. |
| `search_tennis_news(query)` | Claude's native `web_search`, scoped to tennis-relevant queries. |
| `get_player_ranking(player_name, as_of_date)` | Current or historical ranking. |

All non-`web_search` tools accept canonical player names; resolution happens inside the tool via `player_aliases`. Tools return JSON-serializable dicts; their schemas are Pydantic-validated on both input and output.

## The probability rule

**The LLM does not produce a probability.** The model's `get_model_prediction` output is the only probability shown to the user.

The structured output schema:

```python
class AgentResponse(BaseModel):
    model_probability_player_a: float = Field(ge=0, le=1)
    model_probability_player_b: float = Field(ge=0, le=1)
    key_factors: list[str]
    narrative: str
    confidence_band: Literal["low", "medium", "high"]
    caveats: list[str]
    tools_used: list[str]
```

`confidence_band` is the LLM's qualitative read on how well-supported the prediction is given what the tools returned (e.g., "low" if recent form is sparse or news surfaces a withdrawal). It is **not** a hidden probability adjustment.

Validation forbids any field named like a probability override (`llm_probability`, `adjusted_probability`, etc.). Tests assert this.

## The `LLMClient` interface

`src/tennis_predictor/llm/client.py`:

```python
class LLMClient(ABC):
    @abstractmethod
    def call(self, system: str, messages: list[Message], tools: list[Tool]) -> Response: ...
```

The Claude implementation:

- Sends `system` and `tools` as cacheable blocks (`cache_control={"type": "ephemeral"}`).
- Logs every call to `llm_traces` with: model, input message digest, tool calls, tool results, output, tokens in/out, cache read/creation tokens, latency, error if any.
- Never raises raw `APIError`; wraps in a typed `LLMCallFailure`.

## Prompt caching

Required from day one. Don't retrofit.

- The system prompt is one cacheable block (it's stable across calls).
- The tool definitions are another cacheable block (also stable).
- Per-call user messages are not cached.

## Observability

Every call goes to `llm_traces`. The Streamlit dashboard (phase 6) reads from this table to show: total calls, p50/p95 latency, cache hit rate, recent tool-call sequences. There is no other logging path — `print` statements and ad-hoc `logger.info` do not count.

## What NOT to do

- Do not let the LLM emit its own win-probability number, in any form, under any rationale.
- Do not bypass `LLMClient` to call `anthropic.Anthropic()` directly elsewhere.
- Do not add a new tool without a Pydantic input/output schema and a test that round-trips through the schema.
- Do not skip prompt caching "to simplify"; it's a 10× cost difference and trivial to enable.
