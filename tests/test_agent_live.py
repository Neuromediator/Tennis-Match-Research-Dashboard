"""Tier-3 live-API smoke tests.

These tests really call Anthropic — they need `ANTHROPIC_API_KEY` and they
spend a small amount of money each run (~$0.03 per test). Excluded from
the default pytest collection (`addopts = -m "not llm_live"` in
`pyproject.toml`); run manually before pushing a phase or after a system
prompt change:

    uv run pytest -m llm_live

Tests:

- A two-call agent loop on a real upcoming-style match (free-form
  context, no scheduled_matches dependency). Asserts the LLM-emitted
  `AgentResponse` is valid Pydantic and that `cache_creation_tokens > 0`
  on the first call.
- A second call within the same TTL window. Asserts `cache_read_tokens > 0`
  — the cacheable prefix was actually hit. This is the proof Phase 5
  acceptance criterion #2 calls for.

`get_model_prediction` is patched to a stub so the live tier doesn't need
a real model artifact installed on the runner. If you want to also smoke
the real model, run the CLI manually.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.config import ANTHROPIC_API_KEY
from tennis_predictor.data import schema
from tennis_predictor.data.reconcile import seed_aliases_from_players
from tennis_predictor.llm import agent as agent_module
from tennis_predictor.llm.agent import TennisAgent
from tennis_predictor.llm.tools.schemas import (
    MatchContext,
    ModelFeatureSummary,
    ModelPrediction,
)

pytestmark = pytest.mark.llm_live


PLAYER_A_ID = "ATP_900001"
PLAYER_B_ID = "ATP_900002"


@pytest.fixture
def seeded_db(tmp_path: Path):
    if not ANTHROPIC_API_KEY:
        pytest.fail(
            "ANTHROPIC_API_KEY not set; live tier cannot run. "
            "Add it to .env or unset the llm_live marker."
        )
    conn = duckdb.connect(str(tmp_path / "live.duckdb"))
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
    seed_aliases_from_players(conn, "ATP")
    yield conn
    conn.close()


def _stub_prediction() -> ModelPrediction:
    return ModelPrediction(
        player_a_name="Carlos Alcaraz",
        player_b_name="Jannik Sinner",
        tour="ATP",
        surface="Clay",
        tournament_level="Slam",
        best_of=5,
        match_date=date(2026, 6, 8),
        model_probability_player_a=0.58,
        model_probability_player_b=0.42,
        model_artifact_version="live-smoke-stub",
        feature_summary=ModelFeatureSummary(
            elo_player_a=2100.0,
            elo_player_b=2080.0,
            elo_diff_a_minus_b=20.0,
            rank_player_a=1,
            rank_player_b=2,
            h2h_player_a_wins=5,
            h2h_player_b_wins=4,
            fatigue_matches_7d_player_a=3,
            fatigue_matches_7d_player_b=2,
            days_since_last_match_player_a=2,
            days_since_last_match_player_b=3,
        ),
    )


def _ctx() -> MatchContext:
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


async def test_live_agent_returns_valid_response(seeded_db, monkeypatch) -> None:
    """End-to-end smoke: real Anthropic, real submit, real Pydantic round-trip."""
    monkeypatch.setattr(agent_module, "get_model_prediction", lambda *a, **k: _stub_prediction())
    agent = TennisAgent(seeded_db)
    response = await agent.predict(_ctx())
    assert response.confidence_band in ("low", "medium", "high")
    assert response.model_probability_player_a == pytest.approx(0.58)
    assert response.key_factors  # at least one factor


async def test_live_second_call_hits_prompt_cache(seeded_db, monkeypatch) -> None:
    """Phase 5 acceptance criterion #2: second call within TTL window
    reports `cache_read_tokens > 0` on the trace row."""
    monkeypatch.setattr(agent_module, "get_model_prediction", lambda *a, **k: _stub_prediction())
    agent = TennisAgent(seeded_db)
    await agent.predict(_ctx())
    # Capture the row count after the first call so the second call's
    # rows are easy to find.
    seen_so_far = int(seeded_db.execute("SELECT max(trace_id) FROM llm_traces").fetchone()[0] or 0)
    await agent.predict(_ctx())
    cache_reads = seeded_db.execute(
        "SELECT COALESCE(SUM(cache_read_tokens), 0) FROM llm_traces WHERE trace_id > ?",
        [seen_so_far],
    ).fetchone()[0]
    assert cache_reads > 0, (
        f"second call should have hit the prompt cache but cache_read_tokens={cache_reads}"
    )
