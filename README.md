# Tennis Match Research Dashboard

**Production-grade LLM agent engineering on a real domain.** The dashboard renders four independent signals side by side for every upcoming ATP / WTA tour-level singles match — market consensus odds, a trained LightGBM probability, a surface-Elo baseline, an LLM-discovered news block — plus a deterministic *"why model differs"* panel whenever the model-vs-market gap exceeds 10pp. The purpose of project is to demonstrate end-to-end ML+LLM engineering: data ingestion, feature engineering, model training, evaluation, LLM tool-calling integration, deployable interface. **Not a betting tool.**


## Quick start

```bash
# Python 3.12+ pinned via .python-version. Install with uv.
uv sync

# Build everything from public data (~30 min cold start).
uv run python scripts/refresh_data.py            # Sackmann historical
uv run python scripts/refresh_hot.py             # matchstat fixtures + rankings
uv run python scripts/refresh_pre_match_odds.py  # The Odds API pre-match h2h
uv run python scripts/build_features.py          # training_features + elo_state
uv run python scripts/train_models.py            # 4 artifacts: ATP/WTA × Elo/LightGBM

# Run the app.
uv run streamlit run src/tennis_predictor/app/main.py
```

Env vars (in `.env`, template in `.env.example`): `ANTHROPIC_API_KEY`, `X_RAPIDAPI_KEY` (matchstat), `THE_ODDS_API_KEY`, `TAVILY_API_KEY`. Quality gates: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`.

## What's inside

- **LLM agent** — direct Anthropic SDK, prompt caching (~70% input savings), bounded budget (4 iter / 30k tok / 120s / 2 searches), `tool_use` structured output (schema forbids LLM-emitted probability + free-text synthesis), Tavily news search with server-side recency filter, full per-call observability in `llm_traces`.
- **Data engineering** — three flaky sources reconciled. Sackmann cold (1.7M matches), matchstat hot (per-tournament endpoint + 4 prune passes — stale / round-contradicted / duplicate-matchups / completed-Slam cross-check), The Odds API with hyphen-normalised name matching + Tavily fallback.
- **Model** — LightGBM v3 (44 features), walk-forward 8-fold + isotonic calibration. Last-5-fold Brier (post-cal): ATP **0.2087** / WTA **0.1959** vs Surface-Elo baseline 0.2220 / 0.2180 and market ~0.20. Betting odds are **never** training features.
