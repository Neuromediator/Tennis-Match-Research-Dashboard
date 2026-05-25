"""Structured-output collector — the `submit_analysis` tool (Phase 6.1).

The agent's only deliverable is a list of dated `NewsItem`s + a lookup
status enum. The `submit_analysis` tool's `input_schema` mirrors the
new `AgentResponse` exactly, with `additionalProperties: false` so
ANY hallucinated field — probability-like (CLAUDE.md hard rule #4),
prose-like (`narrative`, `caveats`, `confidence`), or otherwise — is
rejected at the JSON-schema layer before Pydantic ever sees it.

Two walls of defence:

1. **JSON-schema layer:** declares only `news_items`, `news_lookup_status`,
   `tools_used`. `additionalProperties: false` rejects everything else.

2. **Pydantic layer:** `AgentResponse` is `extra="forbid"`. Even if a
   malformed payload slips past, `model_validate` raises on construction.

The probability values returned to the user come from `get_model_prediction`
and are merged onto `AgentResponse` by `TennisAgent.predict` — never
written by the LLM.

# Post-validate filtering (Phase 6.1)

After Pydantic validates the payload, the agent loop applies two
post-validate filters before returning to the caller:

- News items tagged `category="other"` are dropped. The agent uses
  `other` as a fallback for items that don't fit the whitelist; the
  contract is that we surface only whitelisted categories.
- News items whose `published_date` parses to more than 32 days before
  the match date are dropped. Items with `published_date=None` are
  kept (Tavily doesn't always parse publication dates from page meta).

If filtering empties the list AND status was `ok`, status flips to
`no_results` to keep the contract honest with the UI.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tennis_predictor.llm.tools.schemas import NewsItem, NewsLookupStatus


class AgentResponse(BaseModel):
    """Final structured output of the Phase 6.1 LLM news-discovery agent.

    Two fields are filled by the orchestrator from `get_model_prediction`
    rather than by the LLM (and the `submit_analysis` JSON schema below
    deliberately hides them so the LLM is never tempted to write them):

    - `model_probability_player_a`
    - `model_probability_player_b`
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    # --- Populated by the orchestrator from `get_model_prediction`. ---
    model_probability_player_a: float = Field(ge=0.0, le=1.0)
    model_probability_player_b: float = Field(ge=0.0, le=1.0)

    # --- Populated by the LLM through `submit_analysis`. ---
    news_items: list[NewsItem] = Field(default_factory=list, max_length=12)
    news_lookup_status: NewsLookupStatus
    tools_used: list[str] = Field(default_factory=list, max_length=12)


# ---------------------------------------------------------------------------
# The tool definition Anthropic receives.
# ---------------------------------------------------------------------------

SUBMIT_ANALYSIS_TOOL_NAME: str = "submit_analysis"

# Hand-written rather than generated from `AgentResponse.model_json_schema()`
# for the same reasons documented at length in the Phase 5 version of
# this file:
#   1. The orchestrator-filled probability fields are deliberately
#      excluded so the LLM never sees them.
#   2. Byte stability is a hard contract (CLAUDE.md "Cache hit hygiene");
#      `model_json_schema()` can shift key order across Pydantic minor
#      versions.
SUBMIT_ANALYSIS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["news_items", "news_lookup_status"],
    "properties": {
        "news_items": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "title",
                    "url",
                    "snippet",
                    "source_domain",
                    "player_subject",
                    "category",
                ],
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 300},
                    "url": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "snippet": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "published_date": {
                        "type": ["string", "null"],
                        "maxLength": 64,
                        "description": (
                            "ISO-like date string from Tavily when present "
                            "(e.g. '2026-05-15'). Pass through verbatim; "
                            "null if Tavily did not return one. Do NOT "
                            "invent a date — the post-filter checks this "
                            "against the 32-day window."
                        ),
                    },
                    "source_domain": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                        "description": (
                            "Extract the bare host from `url` (e.g. "
                            "'bbc.co.uk', 'tennis.com'). Used by the UI "
                            "to badge the item next to the title."
                        ),
                    },
                    "player_subject": {
                        "type": "string",
                        "enum": ["player_a", "player_b", "both"],
                        "description": (
                            "Which player the item primarily concerns. "
                            "Use 'both' only when the item is genuinely "
                            "about the matchup itself (rare)."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "injury",
                            "withdrawal",
                            "illness",
                            "result",
                            "coach_change",
                            "personal",
                            "other",
                        ],
                        "description": (
                            "MUST be one of the whitelist. Items tagged "
                            "`other` are DROPPED before the response is "
                            "returned, so use sparingly."
                        ),
                    },
                },
            },
        },
        "news_lookup_status": {
            "type": "string",
            "enum": ["ok", "no_results", "failed"],
            "description": (
                "`ok` — found at least one relevant whitelisted item. "
                "`no_results` — searched, found nothing material in the "
                "last 32 days. `failed` — DO NOT emit this yourself; the "
                "agent loop sets it when web_search itself errors."
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
        "Submit your final structured output. Call this exactly once at "
        "the end. Pass the list of NewsItems you discovered (or an empty "
        "list with `news_lookup_status=no_results`). Do NOT include a "
        "probability field — the model's number is the only probability "
        "shown to the user. Do NOT include narrative / caveats / "
        "confidence — those fields no longer exist."
    ),
    "input_schema": SUBMIT_ANALYSIS_INPUT_SCHEMA,
}


__all__ = [
    "SUBMIT_ANALYSIS_INPUT_SCHEMA",
    "SUBMIT_ANALYSIS_TOOL",
    "SUBMIT_ANALYSIS_TOOL_NAME",
    "AgentResponse",
]
