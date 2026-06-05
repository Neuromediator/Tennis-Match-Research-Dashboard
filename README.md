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

**[https://neuromediator-tennis-research-dashboard.hf.space/](https://neuromediator-tennis-research-dashboard.hf.space/)**

A single click on a fixture takes ~30–60 seconds the first time anyone in the world looks at it (LightGBM inference + LLM news lookup + Tavily searches) and is instant afterwards (DuckDB-backed prediction cache). The dashboard runs on a single free Hugging Face Space (Docker, CPU basic) with an in-process daily fixture + odds refresh; no separate worker process. A global `DAILY_LLM_BUDGET` caps Anthropic spend at ≈ $1-2/day — past that limit predictions still render (model + market + surface-Elo) but the LLM news block shows a "paused until 00:00 UTC" message.


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

The app ships as a **single Docker image** to a free **Hugging Face Space** (Docker SDK, CPU basic — 2 vCPU / 16 GB RAM). No microservices, no model server, no message queue.

- **Data on boot:** the Space has no persistent disk (HF retired the flat-rate storage tier; only object-storage buckets remain, which don't suit DuckDB's file locking + random I/O). Instead the 1.3 GB `tennis.duckdb` + model artefacts are pulled on container start from a companion HF Dataset (`Neuromediator/tennis-dashboard-data`) onto the container's local filesystem — a real FS, so DuckDB stays fast. See `scripts/hf_bootstrap.py`.
- **Daily refresh** runs in-process via APScheduler (`src/tennis_predictor/app/scheduler.py`) — a background thread fires `CronTrigger(hour=21, minute=0, UTC)` and calls `refresh_hot` + odds refresh against a fresh DuckDB connection.
- **Stays warm:** a twice-daily GitHub Actions ping (`.github/workflows/keepalive.yml`) keeps the Space from sleeping (free Spaces sleep after 48 h idle), so the in-memory DB + prediction cache persist for the uptime and visitors skip the cold-start re-download. An involuntary reset (HF rebuild/migration) is recovered by bootstrap + catch-up-on-wake (`maybe_catch_up_refresh` in `app/scheduler.py`).
- **Cost defenses:** `st.cache_data(ttl=300)` (per-process) → `prediction_cache` DuckDB table → `DAILY_LLM_BUDGET=60` traces/day. Past the cap, predictions still render without the news block.
- **Secrets** (HF Space secrets): `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `X_RAPIDAPI_KEY`, `THE_ODDS_API_KEY`. Non-secret config via Space variables (`ENABLE_SCHEDULER`, `REFRESH_HOUR_UTC`, `HF_DATA_REPO`, …).
- **Manual ops:** Sackmann cold refresh is operator-driven (rebuild the DuckDB locally, re-upload to the companion dataset). Hot fixtures and odds refresh automatically.

Cost: **$0/month** (free CPU, no persistent storage). Tradeoff: writes don't survive an involuntary container reset — refresh data is re-fetched and the prediction cache re-computes on demand.

> Previously deployed on Fly.io (single Machine + 5 GB volume, ~$4/month); migrated to Hugging Face for more RAM (16 GB → no swap → faster cold path) at no cost. The Fly deployment notes are kept as history in `docs/phases.md` Phase 7 + `docs/phase7_plan.md`; the migration is Phase 8.
