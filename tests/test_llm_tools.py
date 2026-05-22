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


def test_agent_response_valid_payload_round_trips() -> None:
    resp = AgentResponse(
        model_probability_player_a=0.62,
        model_probability_player_b=0.38,
        key_factors=["surface edge", "fresh from a long rest"],
        narrative="Player A has a clear surface-Elo lead and won their last meeting.",
        confidence_band="medium",
        caveats=["no recent news surfaced"],
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
    ],
)
def test_agent_response_rejects_probability_like_extra_fields(forbidden_field: str) -> None:
    payload = {
        "model_probability_player_a": 0.5,
        "model_probability_player_b": 0.5,
        "key_factors": ["a"],
        "narrative": "n",
        "confidence_band": "low",
        "caveats": [],
        "tools_used": [],
        forbidden_field: 0.7,
    }
    with pytest.raises(ValidationError):
        AgentResponse.model_validate(payload)


def test_agent_response_confidence_band_constrained_to_three_values() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            model_probability_player_a=0.5,
            model_probability_player_b=0.5,
            key_factors=["a"],
            narrative="n",
            confidence_band="very high",  # type: ignore[arg-type]
            caveats=[],
            tools_used=[],
        )


def test_agent_response_requires_at_least_one_key_factor() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            model_probability_player_a=0.5,
            model_probability_player_b=0.5,
            key_factors=[],
            narrative="n",
            confidence_band="low",
            caveats=[],
            tools_used=[],
        )


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


def test_submit_analysis_confidence_band_enumerates_three_values() -> None:
    enum_values = SUBMIT_ANALYSIS_INPUT_SCHEMA["properties"]["confidence_band"]["enum"]
    assert sorted(enum_values) == ["high", "low", "medium"]


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
