"""Structured-output collector — the `submit_analysis` tool.

The agent calls every other tool to gather context, then is hard-forced
on the final iteration to call `submit_analysis` with its synthesised
analysis. The tool's `input_schema` mirrors `AgentResponse` exactly, with
`additionalProperties: false` so any hallucinated field — most notably any
`probability`-like field the LLM might emit (CLAUDE.md hard rule #4) — is
rejected at the JSON-schema layer before Pydantic ever sees it.

Two walls of defence against an LLM-emitted probability:

1. **JSON-schema layer:** the `submit_analysis` `input_schema` declares only
   `key_factors`, `narrative`, `confidence_band`, `caveats`, `tools_used`.
   Combined with `additionalProperties: false`, any other field name
   (`probability`, `model_probability_player_a`, `adjusted_probability`,
   `confidence`, …) is rejected by Anthropic's schema validator before
   the tool-use block is dispatched to us.

2. **Pydantic layer:** `AgentResponse` itself is `extra="forbid"`. Even if
   a malformed payload slips past, `AgentResponse.model_validate(...)`
   raises on construction.

The probability values returned to the user come from `get_model_prediction`
and are merged onto `AgentResponse` by the orchestrator (`TennisAgent.predict`)
— never written by the LLM.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ConfidenceBand = Literal["low", "medium", "high"]


class AgentResponse(BaseModel):
    """Final structured output of the LLM agent.

    Two fields are filled by the orchestrator from `get_model_prediction`
    rather than by the LLM: `model_probability_player_a` and
    `model_probability_player_b`. They live here (not on `ModelPrediction`)
    because the user-facing serialisation merges them with the LLM's
    qualitative read; the LLM is forbidden from writing them via the
    `submit_analysis` JSON schema below.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    # --- Populated by the orchestrator from `get_model_prediction`. ---
    model_probability_player_a: float = Field(ge=0.0, le=1.0)
    model_probability_player_b: float = Field(ge=0.0, le=1.0)

    # --- Populated by the LLM through `submit_analysis`. ---
    key_factors: list[str] = Field(min_length=1, max_length=8)
    narrative: str = Field(min_length=1, max_length=2000)
    confidence_band: ConfidenceBand
    caveats: list[str] = Field(default_factory=list, max_length=8)
    tools_used: list[str] = Field(default_factory=list, max_length=12)


# ---------------------------------------------------------------------------
# The tool definition Anthropic receives.
# ---------------------------------------------------------------------------


SUBMIT_ANALYSIS_TOOL_NAME: str = "submit_analysis"

# Property declarations are inlined as a plain dict literal rather than
# generated from `AgentResponse.model_json_schema()`. Two reasons:
#
#   1. `model_json_schema()` includes the orchestrator-filled probability
#      fields, which the LLM must NOT see in this tool — hiding them
#      removes the temptation entirely (and Anthropic's schema validator
#      rejects unknown fields, so a hallucinated probability still fails).
#   2. Byte stability is a hard contract (CLAUDE.md "Cache hit hygiene").
#      Pydantic's JSON-schema generator can shift key order across minor
#      versions; a hand-written literal avoids that risk and lets the
#      cache-stability test be a simple `==` comparison.
SUBMIT_ANALYSIS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["key_factors", "narrative", "confidence_band"],
    "properties": {
        "key_factors": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 280},
            "minItems": 1,
            "maxItems": 8,
            "description": (
                "1-8 short bullets naming the strongest signals you used "
                "to interpret the model's probability. Examples: 'Sinner "
                "leads H2H 5-2 on hard', 'Alcaraz coming off back-to-back "
                "5-setters', 'no recent news surfaced for either player'."
            ),
        },
        "narrative": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2000,
            "description": (
                "Three or four sentences explaining how the model's "
                "probability aligns (or disagrees) with the picture you "
                "built from the other tools and recent news. Refer to "
                "specific facts you fetched — do NOT invent any. Do NOT "
                "emit your own probability number; the model's number is "
                "the only probability shown to the user."
            ),
        },
        "confidence_band": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": (
                "Qualitative read on how well-supported the prediction "
                "feels given the tools' return values. 'low' if recent "
                "form is sparse, the news surfaces a withdrawal hint, or "
                "the H2H is empty on the surface. 'high' only when "
                "rankings, form, and news all point the same way. NOT a "
                "hidden probability adjustment."
            ),
        },
        "caveats": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 280},
            "maxItems": 8,
            "description": (
                "0-8 short bullets flagging anything that makes the "
                "prediction shakier. Must contain 'no recent news "
                "surfaced' when web search returned nothing material. "
                "Never fabricate plausible-sounding news to fill this slot."
            ),
        },
        "tools_used": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
            "maxItems": 12,
            "description": (
                "Names of every tool you called during this prediction, "
                "in call order. Helps the dashboard surface which tools "
                "are most influential."
            ),
        },
    },
}


SUBMIT_ANALYSIS_TOOL: dict[str, Any] = {
    "name": SUBMIT_ANALYSIS_TOOL_NAME,
    "description": (
        "Submit your final analysis of the match. Call this exactly once, "
        "at the end of your reasoning, after you have called every other "
        "tool you need. Do NOT include a probability field — the model's "
        "number is the only probability shown to the user."
    ),
    "input_schema": SUBMIT_ANALYSIS_INPUT_SCHEMA,
}


__all__ = [
    "SUBMIT_ANALYSIS_INPUT_SCHEMA",
    "SUBMIT_ANALYSIS_TOOL",
    "SUBMIT_ANALYSIS_TOOL_NAME",
    "AgentResponse",
    "ConfidenceBand",
]
