---
title: Tennis Match Research Dashboard
emoji: 🎾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8080
pinned: false
---

<!-- The YAML block above is Hugging Face Space metadata (Docker SDK),
     required by HF Spaces and rendered as a table by GitHub. -->

# Tennis Match Research Dashboard

**Production-grade LLM agent engineering on a real domain.** The dashboard renders four independent signals side by side for every upcoming ATP / WTA tour-level singles match — market consensus odds, a trained LightGBM probability, a surface-Elo baseline, an LLM-discovered news block — plus a deterministic *"why model differs"* panel whenever the model-vs-market gap exceeds 10pp. The purpose of project is to demonstrate end-to-end ML+LLM engineering: data ingestion, feature engineering, model training, evaluation, LLM tool-calling integration, deployable interface. **Not a betting tool.**

## Live demo

**[https://tennis-research-dashboard.fly.dev/](https://tennis-research-dashboard.fly.dev/)**

A single click on a fixture takes ~40 seconds the first time anyone in the world looks at it (LightGBM inference + LLM news lookup + Tavily searches) and is instant for the next 24 hours for anyone else (DuckDB-backed prediction cache). The dashboard runs on a single Fly.io Machine with daily APScheduler-driven fixture + odds refresh; no separate worker process. A global `DAILY_LLM_BUDGET` caps Anthropic spend at ≈ $1-2/day — past that limit predictions still render (model + market + surface-Elo) but the LLM news block shows a "paused until 00:00 UTC" message.


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

## Deployment

The app ships as a **single Docker image** to **Fly.io**, one Machine (`shared-cpu-1x` / 2 GB RAM), one 5 GB persistent volume mounted at `/data`. DuckDB lives on the volume; Sackmann submodules, model artefacts, and the prediction cache table sit alongside it. No microservices, no model server, no message queue.

- **Daily refresh** runs in-process via APScheduler (`src/tennis_predictor/app/scheduler.py`) — a background thread fires `CronTrigger(hour=21, minute=0, UTC)` and calls `refresh_hot.py` + `refresh_pre_match_odds.py` against a fresh DuckDB connection. No separate cron Machine: DuckDB does not support multi-process writes, so two Machines sharing one volume would deadlock the file lock.
- **Cost defenses:** `st.cache_data(ttl=300)` (per-process) → `prediction_cache` DuckDB table (24 h, cross-session) → `DAILY_LLM_BUDGET=60` traces/day. Past the cap, predictions still render without the news block.
- **Secrets** (`fly secrets set`): `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `X_RAPIDAPI_KEY`, `THE_ODDS_API_KEY`.
- **Bootstrap** is one-shot: see `docs/phase7_plan.md` for the `sleep infinity` CMD trick that lets you `fly ssh console` into the Machine without Streamlit grabbing the DuckDB file lock. Production `tennis.duckdb` and `models/` are built locally and uploaded via `fly ssh sftp put` because Fly's shared CPU is too slow for `build_features.py` / `train_models.py`.
- **Manual ops:** weekly Sackmann cold refresh is operator-driven (`fly machine stop` → `git pull` submodules → `refresh_data.py --skip-submodules` → `fly machine start`). Hot fixtures and odds run automatically.

Cost on Fly.io: ~$3-4/month (Machine + volume; bandwidth well under the 100 GB free tier).
