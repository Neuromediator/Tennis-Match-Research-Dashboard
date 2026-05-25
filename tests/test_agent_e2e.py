"""Tier-2 e2e tests for the LLM agent.

Phase 6.1 status: the Phase 5 fixtures in `tests/fixtures/llm/*.json`
reference the retired `narrative` / `confidence_band` / `caveats` /
`key_factors` schema and the retired `get_player_stats` /
`get_recent_form` / `get_player_ranking` / `fetch_url` tool surface.
Re-recording them requires a live API run (Phase 6.1 task #34) which
is the user's responsibility (paid).

Until the fresh fixtures land, this whole module is skipped to keep
the unit-test tier green. The new agent surface IS covered by
hand-crafted unit tests in `tests/test_agent_loop_phase_6_1.py`.
"""

# pyright: reportAttributeAccessIssue=false
# Phase 6.1 retired the AgentResponse fields this file's assertions still
# read (`narrative`, `confidence_band`, `caveats`, `key_factors`). The
# whole module is `pytest.mark.skip`'d at the bottom pending fixture
# re-record (task #34); this directive prevents pyright from blocking
# the gate on dead-but-still-present assertions.
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pytest
import respx

from tennis_predictor.data import schema
from tennis_predictor.data.reconcile import seed_aliases_from_players
from tennis_predictor.llm import agent as agent_module
from tennis_predictor.llm.agent import TennisAgent
from tennis_predictor.llm.client import AnthropicLLMClient
from tennis_predictor.llm.tools.fetch_url import TAVILY_EXTRACT_URL
from tennis_predictor.llm.tools.schemas import (
    MatchContext,
    ModelFeatureSummary,
    ModelPrediction,
)
from tennis_predictor.llm.tools.web_search import TAVILY_SEARCH_URL

pytestmark = pytest.mark.skip(
    reason=(
        "Phase 5 fixtures reference retired AgentResponse fields "
        "(narrative / confidence_band / caveats / key_factors) and the "
        "retired tool surface. Re-record under Phase 6.1 task #34."
    )
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llm"

PLAYER_A_ID = "ATP_900001"
PLAYER_B_ID = "ATP_900002"


# ---------------------------------------------------------------------------
# Fixture replay machinery — mirrors test_llm_client.py but loads from JSON.
# ---------------------------------------------------------------------------


class _Block:
    """Minimal Anthropic content block — exposes `model_dump()` like the SDK."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return dict(self._payload)


class _Usage:
    def __init__(self, raw: dict[str, int]):
        self.input_tokens = raw.get("input_tokens", 0)
        self.output_tokens = raw.get("output_tokens", 0)
        self.cache_read_input_tokens = raw.get("cache_read_input_tokens", 0)
        self.cache_creation_input_tokens = raw.get("cache_creation_input_tokens", 0)


class _Message:
    def __init__(self, raw: dict[str, Any]):
        self.content = [_Block(b) for b in raw["content"]]
        self.stop_reason = raw.get("stop_reason", "end_turn")
        self.usage = _Usage(raw.get("usage", {}))


class _FixtureMessagesAPI:
    def __init__(self, responses: list[dict[str, Any]]):
        self._queue = [_Message(r) for r in responses]
        self.received_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Message:
        self.received_calls.append(kwargs)
        if not self._queue:
            raise AssertionError("fixture exhausted: agent looped more than expected")
        return self._queue.pop(0)


class _FixtureAnthropic:
    def __init__(self, responses: list[dict[str, Any]]):
        self.messages = _FixtureMessagesAPI(responses)


def _load_fixture(name: str) -> list[dict[str, Any]]:
    raw = json.loads((FIXTURES_DIR / f"{name}.json").read_text())
    return raw["responses"]


# ---------------------------------------------------------------------------
# Seeded DB so the DB tools and alias resolver have data to read.
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_db(tmp_path: Path):
    conn = duckdb.connect(str(tmp_path / "e2e.duckdb"))
    schema.create_all_tables(conn)
    conn.executemany(
        """
        INSERT INTO players (
            player_id, tour, sackmann_id, name_first, name_last, full_name,
            hand, dob, ioc, height
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                PLAYER_A_ID,
                "ATP",
                900001,
                "Carlos",
                "Alcaraz",
                "Carlos Alcaraz",
                "R",
                date(2003, 5, 5),
                "ESP",
                183,
            ),
            (
                PLAYER_B_ID,
                "ATP",
                900002,
                "Jannik",
                "Sinner",
                "Jannik Sinner",
                "R",
                date(2001, 8, 16),
                "ITA",
                188,
            ),
        ],
    )
    conn.execute(
        """
        INSERT INTO matches (
            match_id, source, match_external_id, tour, match_tier,
            tourney_id, tourney_name, tourney_level, tourney_date, surface,
            match_num, round, best_of, match_status,
            winner_player_id, loser_player_id, score
        ) VALUES (
            'stub::M1', 'stub', 'M1', 'ATP', 'main', 'stub-1', 'Madrid', 'M',
            DATE '2024-05-08', 'Clay', 1, 'F', 3, 'completed',
            ?, ?, '6-4 6-3'
        )
        """,
        [PLAYER_A_ID, PLAYER_B_ID],
    )
    conn.execute(
        "INSERT INTO rankings (ranking_date, player_id, rank, points) "
        "VALUES (DATE '2026-05-01', ?, 1, 11000), (DATE '2026-05-01', ?, 2, 10500)",
        [PLAYER_A_ID, PLAYER_B_ID],
    )
    seed_aliases_from_players(conn, "ATP")
    yield conn
    conn.close()


def _match_context() -> MatchContext:
    return MatchContext(
        tour="ATP",
        player_a_name="Carlos Alcaraz",
        player_b_name="Jannik Sinner",
        surface="Clay",
        tournament_level="Slam",
        tournament_name="Roland Garros",
        best_of=5,
        match_date=date(2026, 6, 8),
    )


def _stub_model_prediction() -> ModelPrediction:
    return ModelPrediction(
        player_a_name="Carlos Alcaraz",
        player_b_name="Jannik Sinner",
        tour="ATP",
        surface="Clay",
        tournament_level="Slam",
        best_of=5,
        match_date=date(2026, 6, 8),
        model_probability_player_a=0.6234,
        model_probability_player_b=0.3766,
        model_artifact_version="stub-20260522",
        feature_summary=ModelFeatureSummary(
            elo_player_a=2120.0,
            elo_player_b=2050.0,
            elo_diff_a_minus_b=70.0,
            rank_player_a=1,
            rank_player_b=2,
            h2h_player_a_wins=2,
            h2h_player_b_wins=1,
            win_pct_last10_player_a=0.8,
            win_pct_last10_player_b=0.8,
            win_pct_last25_surface_player_a=0.85,
            win_pct_last25_surface_player_b=0.7,
            fatigue_matches_7d_player_a=3,
            fatigue_matches_7d_player_b=2,
            days_since_last_match_player_a=2,
            days_since_last_match_player_b=4,
        ),
    )


def _build_agent(conn, stub_client_responses) -> TennisAgent:
    stub = _FixtureAnthropic(stub_client_responses)
    llm = AnthropicLLMClient(conn, client=stub)  # type: ignore[arg-type]
    return TennisAgent(conn, llm_client=llm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _mock_tavily_search_ok(payload: dict[str, Any] | None = None) -> respx.Route:
    """Default Tavily search response — two plausible Alcaraz/Sinner hits."""
    default = {
        "results": [
            {
                "title": "Alcaraz withdraws from Roland Garros",
                "url": "https://example.com/alcaraz-rg-withdrawal",
                "content": "Carlos Alcaraz pulled out of Roland Garros 2026 with a wrist injury...",
                "published_date": "2026-05-02",
            },
            {
                "title": "Sinner top seed at Paris",
                "url": "https://example.com/sinner-paris-seed",
                "content": "Jannik Sinner confirmed as top seed at Roland Garros 2026.",
                "published_date": "2026-05-15",
            },
        ]
    }
    return respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=payload or default)
    )


@respx.mock
async def test_happy_path_fixture_replays_to_valid_agent_response(seeded_db, monkeypatch) -> None:
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        agent_module, "get_model_prediction", lambda *a, **k: _stub_model_prediction()
    )
    _mock_tavily_search_ok()
    agent = _build_agent(seeded_db, _load_fixture("happy_path"))
    resp = await agent.predict(_match_context())
    assert resp.confidence_band == "medium"
    assert resp.model_probability_player_a == pytest.approx(0.6234)
    assert resp.model_probability_player_b == pytest.approx(0.3766)
    assert any("clay" in factor.lower() for factor in resp.key_factors)
    assert "no recent news surfaced" in resp.caveats


@respx.mock
async def test_web_search_error_fixture_still_produces_valid_response(
    seeded_db, monkeypatch
) -> None:
    """Tavily returns 5xx → agent surfaces 'news lookup unavailable' in caveats."""
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        agent_module, "get_model_prediction", lambda *a, **k: _stub_model_prediction()
    )
    respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(503, text="service unavailable"))
    agent = _build_agent(seeded_db, _load_fixture("web_search_error"))
    resp = await agent.predict(_match_context())
    assert resp.confidence_band == "low"
    assert any("news lookup unavailable" in c for c in resp.caveats)


async def test_empty_h2h_fixture_surfaces_fresh_pairing_in_caveats(seeded_db, monkeypatch) -> None:
    """No web_search calls in this fixture — Tavily is not touched."""
    monkeypatch.setattr(
        agent_module, "get_model_prediction", lambda *a, **k: _stub_model_prediction()
    )
    agent = _build_agent(seeded_db, _load_fixture("empty_h2h"))
    resp = await agent.predict(_match_context())
    assert resp.confidence_band == "low"
    assert any("head-to-head" in c.lower() for c in resp.caveats)


@respx.mock
async def test_fetch_url_used_fixture_dispatches_and_counts(seeded_db, monkeypatch) -> None:
    """Phase 5.1 path: agent calls web_search, then fetch_url for one snippet
    that needed the full body. Asserts both tools were dispatched and the
    fetch_url_count column on the final trace row is > 0."""
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("tennis_predictor.llm.tools.fetch_url.TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        agent_module, "get_model_prediction", lambda *a, **k: _stub_model_prediction()
    )
    _mock_tavily_search_ok()
    respx.post(TAVILY_EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/kasatkina-interview-2026",
                        "raw_content": "Kasatkina spoke about her break and intentions to return slowly...",
                    }
                ],
                "failed_results": [],
            },
        )
    )
    agent = _build_agent(seeded_db, _load_fixture("fetch_url_used"))
    resp = await agent.predict(_match_context())
    assert resp.confidence_band == "medium"
    assert "fetch_url" in resp.tools_used
    row = seeded_db.execute(
        "SELECT max(fetch_url_count), max(web_search_count) FROM llm_traces"
    ).fetchone()
    assert row is not None
    max_fetch_count, max_search_count = row
    assert (max_fetch_count or 0) >= 1
    assert (max_search_count or 0) >= 1


@respx.mock
async def test_agent_response_round_trips_through_llm_traces_output(seeded_db, monkeypatch) -> None:
    """Pulls the output JSON we logged on the final iteration and
    re-validates it through `AgentResponse` — Phase 5 acceptance
    criterion #4."""
    monkeypatch.setattr("tennis_predictor.llm.tools.web_search.TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        agent_module, "get_model_prediction", lambda *a, **k: _stub_model_prediction()
    )
    _mock_tavily_search_ok()
    agent = _build_agent(seeded_db, _load_fixture("happy_path"))
    response = await agent.predict(_match_context())
    rows = seeded_db.execute(
        "SELECT output FROM llm_traces ORDER BY trace_id DESC LIMIT 1"
    ).fetchall()
    assert rows, "agent should have logged at least one llm_traces row"
    parsed_output = json.loads(rows[0][0])
    assert isinstance(parsed_output, list)
    submit_block = next(b for b in parsed_output if b.get("type") == "tool_use")
    merged = {
        **submit_block["input"],
        "model_probability_player_a": response.model_probability_player_a,
        "model_probability_player_b": response.model_probability_player_b,
    }
    from tennis_predictor.llm.tools.submit import AgentResponse

    reparsed = AgentResponse.model_validate(merged)
    assert reparsed == response
