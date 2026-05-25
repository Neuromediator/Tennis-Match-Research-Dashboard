"""LLM tool-schema and structured-output discipline.

Covers the contracts that block CLAUDE.md hard rule #4 (LLM does not emit
a probability) and the byte-stability of the `submit_analysis` JSON
schema. Tooling-side behaviour (DB lookups, model loading) is tested in
the other `test_llm_*` modules — this file is pure schema validation.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from tennis_predictor.llm.tools.schemas import (
    GetHeadToHeadInput,
    GetModelPredictionInput,
    GetPlayerRankingInput,
    GetPlayerStatsInput,
    GetRecentFormInput,
    MatchContext,
)
from tennis_predictor.llm.tools.submit import (
    SUBMIT_ANALYSIS_INPUT_SCHEMA,
    SUBMIT_ANALYSIS_TOOL,
    SUBMIT_ANALYSIS_TOOL_NAME,
    AgentResponse,
)

# ---------------------------------------------------------------------------
# AgentResponse — the structured-output wall (Hard Rule #4).
# ---------------------------------------------------------------------------


def _sample_news_item(**overrides) -> dict:
    base = {
        "title": "Player A withdraws from Madrid SF — ankle",
        "url": "https://bbc.co.uk/sport/tennis/12345",
        "snippet": "Player A withdrew this morning from his Madrid Open semifinal...",
        "published_date": "2026-05-15",
        "source_domain": "bbc.co.uk",
        "player_subject": "player_a",
        "category": "injury",
    }
    base.update(overrides)
    return base


def test_agent_response_valid_payload_round_trips() -> None:
    resp = AgentResponse(
        model_probability_player_a=0.62,
        model_probability_player_b=0.38,
        news_items=[_sample_news_item()],  # type: ignore[list-item]
        news_lookup_status="ok",
        tools_used=["get_model_prediction", "get_head_to_head", "web_search"],
    )
    parsed = AgentResponse.model_validate_json(resp.model_dump_json())
    assert parsed == resp


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "probability",
        "llm_probability",
        "adjusted_probability",
        "model_probability_override",
        "model_prob_player_a",
        "narrative",
        "confidence_band",
        "caveats",
        "key_factors",
        "summary",
    ],
)
def test_agent_response_rejects_forbidden_extra_fields(forbidden_field: str) -> None:
    """Phase 6.1: rejects probability-like fields (hard rule #4) AND
    prose-like synthesis fields (the structural reason 6.1 exists)."""
    payload = {
        "model_probability_player_a": 0.5,
        "model_probability_player_b": 0.5,
        "news_items": [],
        "news_lookup_status": "no_results",
        "tools_used": [],
        forbidden_field: 0.7 if "probability" in forbidden_field else "x",
    }
    with pytest.raises(ValidationError):
        AgentResponse.model_validate(payload)


def test_agent_response_news_lookup_status_constrained_to_three_values() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            model_probability_player_a=0.5,
            model_probability_player_b=0.5,
            news_items=[],
            news_lookup_status="great",  # type: ignore[arg-type]
            tools_used=[],
        )


def test_agent_response_allows_empty_news_list_with_no_results_status() -> None:
    """The most common UX outcome: nothing material was found."""
    resp = AgentResponse(
        model_probability_player_a=0.5,
        model_probability_player_b=0.5,
        news_items=[],
        news_lookup_status="no_results",
        tools_used=["web_search"],
    )
    assert resp.news_items == []


# ---------------------------------------------------------------------------
# submit_analysis JSON schema — first wall, before Pydantic.
# ---------------------------------------------------------------------------


def test_submit_analysis_tool_shape() -> None:
    assert SUBMIT_ANALYSIS_TOOL["name"] == SUBMIT_ANALYSIS_TOOL_NAME
    assert SUBMIT_ANALYSIS_TOOL["input_schema"] is SUBMIT_ANALYSIS_INPUT_SCHEMA


def test_submit_analysis_schema_has_additional_properties_false() -> None:
    """JSON-schema-level wall against any LLM-emitted probability field
    (Hard Rule #4)."""
    assert SUBMIT_ANALYSIS_INPUT_SCHEMA["additionalProperties"] is False


def test_submit_analysis_schema_does_not_declare_probability_fields() -> None:
    properties = SUBMIT_ANALYSIS_INPUT_SCHEMA["properties"]
    for key in properties:
        assert "probability" not in key.lower(), (
            f"submit_analysis must not declare a probability-like field "
            f"(found: {key!r}). The model is the only source."
        )


def test_submit_analysis_does_not_declare_prose_fields() -> None:
    """Phase 6.1 contract: the LLM no longer writes prose. Any
    `narrative`, `confidence_band`, `caveats`, `key_factors` field
    showing up in the schema would silently re-enable the failure mode
    Phase 6.1 was created to eliminate."""
    properties = SUBMIT_ANALYSIS_INPUT_SCHEMA["properties"]
    for forbidden in ("narrative", "confidence_band", "caveats", "key_factors", "summary"):
        assert forbidden not in properties, (
            f"submit_analysis must not declare a prose field (found: {forbidden!r}). "
            "Phase 6.1 retired these — see docs/tutorials/phase_6_1_notes.md."
        )


def test_submit_analysis_news_lookup_status_enumerates_three_values() -> None:
    enum_values = SUBMIT_ANALYSIS_INPUT_SCHEMA["properties"]["news_lookup_status"]["enum"]
    assert sorted(enum_values) == ["failed", "no_results", "ok"]


def test_submit_analysis_news_item_category_whitelist_enumerated() -> None:
    """The whitelist on `category` is enforced at JSON-schema level so
    Anthropic rejects unknown values before our code even sees them."""
    item_schema = SUBMIT_ANALYSIS_INPUT_SCHEMA["properties"]["news_items"]["items"]
    cat_enum = item_schema["properties"]["category"]["enum"]
    assert "injury" in cat_enum
    assert "withdrawal" in cat_enum
    assert "result" in cat_enum
    assert "interview" not in cat_enum, "interview is intentionally excluded"
    assert "sponsorship" not in cat_enum


# ---------------------------------------------------------------------------
# Tool input schemas — fail fast on missing fields / wrong types.
# ---------------------------------------------------------------------------


def test_match_context_minimum_required_fields() -> None:
    ctx = MatchContext(
        tour="ATP",
        player_a_name="Carlos Alcaraz",
        player_b_name="Jannik Sinner",
        surface="Clay",
        tournament_level="Slam",
        best_of=5,
        match_date=date(2026, 6, 8),
    )
    assert ctx.scheduled_match_id is None


def test_match_context_rejects_invalid_tour() -> None:
    with pytest.raises(ValidationError):
        MatchContext(
            tour="ITF",  # type: ignore[arg-type]
            player_a_name="A",
            player_b_name="B",
            surface="Clay",
            tournament_level="Slam",
            best_of=5,
            match_date=date(2026, 6, 8),
        )


def test_match_context_rejects_invalid_best_of() -> None:
    with pytest.raises(ValidationError):
        MatchContext(
            tour="ATP",
            player_a_name="A",
            player_b_name="B",
            surface="Clay",
            tournament_level="Slam",
            best_of=4,  # type: ignore[arg-type]
            match_date=date(2026, 6, 8),
        )


def test_get_recent_form_input_clips_n_matches_to_bounds() -> None:
    GetRecentFormInput(player_name="A", tour="ATP", as_of_date=date(2026, 1, 1), n_matches=25)
    with pytest.raises(ValidationError):
        GetRecentFormInput(player_name="A", tour="ATP", as_of_date=date(2026, 1, 1), n_matches=100)


def test_all_tool_inputs_forbid_extra_fields() -> None:
    """Every tool input model must `extra="forbid"` so a hallucinated
    field isn't silently ignored at the boundary."""
    payload = {"unknown_field": "x"}
    for model_cls, baseline in [
        (
            GetPlayerStatsInput,
            {"player_name": "A", "tour": "ATP", "as_of_date": date(2026, 1, 1)},
        ),
        (
            GetHeadToHeadInput,
            {
                "player_a_name": "A",
                "player_b_name": "B",
                "tour": "ATP",
                "as_of_date": date(2026, 1, 1),
            },
        ),
        (
            GetRecentFormInput,
            {"player_name": "A", "tour": "ATP", "as_of_date": date(2026, 1, 1)},
        ),
        (
            GetPlayerRankingInput,
            {"player_name": "A", "tour": "ATP", "as_of_date": date(2026, 1, 1)},
        ),
        (
            GetModelPredictionInput,
            {
                "player_a_name": "A",
                "player_b_name": "B",
                "tour": "ATP",
                "surface": "Hard",
                "tournament_level": "Slam",
                "best_of": 5,
                "match_date": date(2026, 1, 1),
            },
        ),
    ]:
        with pytest.raises(ValidationError):
            model_cls.model_validate({**baseline, **payload})
