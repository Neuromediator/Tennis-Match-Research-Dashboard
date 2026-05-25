"""Phase 6.1 hand-crafted unit tests for the slimmed agent loop.

Targets the parts of `TennisAgent` that changed in Phase 6.1:

- New tool surface (`get_head_to_head`, `get_surface_elo`, `web_search`,
  `submit_analysis` — and nothing else).
- New `AgentResponse` shape (news_items + status, no prose).
- Post-validate filters: `other`-category items dropped; items older
  than 32 days dropped; empty result after filtering flips status from
  `ok` to `no_results`.

The fixture-replay e2e tests (`test_agent_e2e.py`) are skipped until
their JSON files are re-recorded against the new schema (task #34).
These tests cover the same agent surface with hand-built stubs so we
don't lose coverage in the meantime.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from tennis_predictor.llm.agent import (
    AgentError,
    _build_agent_response,
    _filter_news_items,
    _parse_iso_date_lenient,
)
from tennis_predictor.llm.client import LLMToolUse
from tennis_predictor.llm.tools.schemas import (
    ModelFeatureSummary,
    ModelPrediction,
    NewsItem,
)
from tennis_predictor.llm.tools.submit import (
    SUBMIT_ANALYSIS_INPUT_SCHEMA,
    SUBMIT_ANALYSIS_TOOL_NAME,
)


def _stub_prediction() -> ModelPrediction:
    return ModelPrediction(
        player_a_name="Alice Alpha",
        player_b_name="Bob Beta",
        tour="ATP",
        surface="Clay",
        tournament_level="ATP250",
        best_of=3,
        match_date=date(2026, 5, 24),
        model_probability_player_a=0.62,
        model_probability_player_b=0.38,
        model_artifact_version="20260520-1200",
        feature_summary=ModelFeatureSummary(
            elo_player_a=1900.0,
            elo_player_b=1700.0,
            elo_diff_a_minus_b=200.0,
            rank_player_a=10,
            rank_player_b=25,
            h2h_player_a_wins=2,
            h2h_player_b_wins=1,
            fatigue_matches_7d_player_a=0,
            fatigue_matches_7d_player_b=1,
        ),
    )


def _news_item(**overrides: object) -> NewsItem:
    base: dict[str, object] = {
        "title": "Player A withdraws - ankle",
        "url": "https://bbc.co.uk/sport/x",
        "snippet": "Player A withdrew from Madrid SF this morning citing an ankle injury.",
        "published_date": "2026-05-15",
        "source_domain": "bbc.co.uk",
        "player_subject": "player_a",
        "category": "injury",
    }
    base.update(overrides)
    return NewsItem.model_validate(base)


def _submit_use(news_items: list[object], status: str = "ok") -> LLMToolUse:
    return LLMToolUse(
        id="toolu_test",
        name=SUBMIT_ANALYSIS_TOOL_NAME,
        input={
            "news_items": news_items,
            "news_lookup_status": status,
            "tools_used": ["get_head_to_head", "get_surface_elo", "web_search"],
        },
    )


# ---------------------------------------------------------------------------
# _parse_iso_date_lenient
# ---------------------------------------------------------------------------


def test_parse_iso_date_lenient_handles_common_forms() -> None:
    assert _parse_iso_date_lenient("2026-05-15") == date(2026, 5, 15)
    assert _parse_iso_date_lenient("2026-05-15T10:30:00Z") == date(2026, 5, 15)
    assert _parse_iso_date_lenient("2026-05") == date(2026, 5, 1)
    assert _parse_iso_date_lenient(None) is None
    assert _parse_iso_date_lenient("") is None
    assert _parse_iso_date_lenient("not-a-date") is None


# ---------------------------------------------------------------------------
# _filter_news_items — post-validate filter
# ---------------------------------------------------------------------------


def test_filter_news_items_drops_other_category() -> None:
    items = [
        _news_item(category="injury"),
        _news_item(category="other", title="Random podcast appearance"),
    ]
    out = _filter_news_items(items, date(2026, 5, 24))
    assert len(out) == 1
    assert out[0].category == "injury"


def test_filter_news_items_drops_older_than_32_days() -> None:
    items = [
        _news_item(published_date="2026-05-20"),  # 4 days before match date
        _news_item(published_date="2026-03-15", title="Stale article"),  # 70 days
    ]
    out = _filter_news_items(items, date(2026, 5, 24))
    assert len(out) == 1
    assert out[0].title != "Stale article"


def test_filter_news_items_keeps_items_without_date() -> None:
    """Tavily often returns no published_date for legitimate fresh items;
    we keep them rather than dropping on absence."""
    items = [_news_item(published_date=None)]
    out = _filter_news_items(items, date(2026, 5, 24))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# _build_agent_response — Pydantic + filter + status-downgrade
# ---------------------------------------------------------------------------


def test_build_agent_response_merges_model_probability() -> None:
    prediction = _stub_prediction()
    use = _submit_use(
        news_items=[_news_item().model_dump(mode="json")],
        status="ok",
    )
    resp = _build_agent_response(prediction, use, date(2026, 5, 24))
    assert resp.model_probability_player_a == pytest.approx(0.62)
    assert resp.model_probability_player_b == pytest.approx(0.38)
    assert resp.news_lookup_status == "ok"
    assert len(resp.news_items) == 1


def test_build_agent_response_downgrades_to_no_results_when_filter_empties_list() -> None:
    """If the LLM said `ok` but every item was dropped (all `other` or
    all too old), the UI's empty-state should match: status → no_results."""
    prediction = _stub_prediction()
    use = _submit_use(
        news_items=[
            _news_item(category="other").model_dump(mode="json"),
            _news_item(published_date="2026-01-01").model_dump(mode="json"),  # >32d old
        ],
        status="ok",
    )
    resp = _build_agent_response(prediction, use, date(2026, 5, 24))
    assert resp.news_items == []
    assert resp.news_lookup_status == "no_results"


def test_build_agent_response_keeps_failed_status_through_filter() -> None:
    """`failed` is set by the agent loop (Tavily error); filter should
    not silently overwrite it with `no_results`."""
    prediction = _stub_prediction()
    use = _submit_use(news_items=[], status="failed")
    resp = _build_agent_response(prediction, use, date(2026, 5, 24))
    assert resp.news_lookup_status == "failed"


def test_build_agent_response_rejects_legacy_narrative_field() -> None:
    """The Phase 5 LLM emitted `narrative`; the new schema forbids it
    at the Pydantic layer. A malformed payload from a confused LLM
    must fail loud, not be silently accepted."""
    prediction = _stub_prediction()
    bad_use = LLMToolUse(
        id="toolu_bad",
        name=SUBMIT_ANALYSIS_TOOL_NAME,
        input={
            "news_items": [],
            "news_lookup_status": "no_results",
            "narrative": "this should not be allowed",
        },
    )
    with pytest.raises(AgentError):
        _build_agent_response(prediction, bad_use, date(2026, 5, 24))


# ---------------------------------------------------------------------------
# Schema invariants the agent depends on
# ---------------------------------------------------------------------------


def test_submit_analysis_schema_news_category_includes_required_values() -> None:
    """The agent prompt promises these whitelist values to the LLM; if
    they ever drift, the prompt should drift too."""
    item_schema = SUBMIT_ANALYSIS_INPUT_SCHEMA["properties"]["news_items"]["items"]
    categories = set(item_schema["properties"]["category"]["enum"])
    assert {"injury", "withdrawal", "illness", "result", "coach_change", "personal", "other"} <= (
        categories
    )


def test_news_item_construction_validates_url_and_snippet_lengths() -> None:
    """Defensive: a bad LLM giving us empty strings should fail at
    Pydantic, not silently render an empty card in the UI."""
    with pytest.raises(ValidationError):
        NewsItem(
            title="",
            url="https://x.com/y",
            snippet="x",
            source_domain="x.com",
            player_subject="player_a",
            category="result",
        )
