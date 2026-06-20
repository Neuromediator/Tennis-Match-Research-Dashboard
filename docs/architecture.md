# Architecture

A single-process Python application with five logical layers and one DuckDB file as the system of record. No microservices, no message queue, no separate model server.

```
┌────────────────────────────────────────────────────────────────────┐
│ Interface (Streamlit)                                              │
│   - Home                — upcoming matches, tour filter, date-only │
│   - Match dashboard     — signal comparison (market / model /      │
│                           surface-Elo) + "why model differs"       │
│                           panel + H2H + recent form + news block   │
│   - Custom match        — 3-input what-if form                     │
│   - Model evaluation    — calibration plots, scoreboard, quotas,   │
│                           cost monitor, llm_traces                 │
└────────────────────┬───────────────────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────────────────┐
│ LLM agent                                                          │
│   - LLMClient (Anthropic SDK direct)                               │
│   - Prompt caching with byte-stable cacheable prefix               │
│   - Bounded budget: 4 iter / 30k tok / 120s / 2 web searches       │
│   - Tools: get_model_prediction, get_head_to_head, get_surface_elo,│
│            web_search (Tavily), submit_analysis                    │
│   - Output: news_items (typed, dated, categorised) + status enum.  │
│     No narrative, no confidence_band — view layer renders details. │
│   - Every call logged to llm_traces                                │
└──────────┬─────────────────────────────────┬───────────────────────┘
           │                                 │
┌──────────▼──────────────┐    ┌─────────────▼──────────────────────┐
│ Modeling                │    │ Feature engineering                │
│   - 4 trained artifacts │    │   - build_training_features()      │
│     (ATP/WTA × Elo /    │    │     (chronological replay)         │
│     LightGBM)           │    │   - compute_features(...)          │
│   - Walk-forward CV     │    │     → FeatureVector (44 fields)    │
│   - Isotonic / Platt    │    │   - State objects: EloState,       │
│   - Market overlay on   │    │     LastMatchState,                │
│     every report        │    │     LastMatchPerSurfaceState, etc. │
└──────────┬──────────────┘    └─────────────┬──────────────────────┘
           │                                 │
┌──────────▼─────────────────────────────────▼──────────────────────┐
│ Data layer — DuckDB (single file)                                 │
│   matches            scheduled_matches    players                 │
│   rankings           player_aliases       market_implied_probs    │
│   elo_state          last_match_state     last_match_per_surface  │
│   training_features  llm_traces           ingestion_runs          │
│   matchstat_*_cache  matchstat_quota      pre_match_odds          │
│   odds_api_quota     prediction_log                               │
└───────────────────────────────────────────────────────────────────┘
```

## Data sources

| Source | Role | Refresh cadence | Quota |
|---|---|---|---|
| **Sackmann** (was `tennis_atp` / `tennis_wta`) | Historical match record. Source of truth for `matches`, used to train the model and feed Elo state. | **Frozen** — the upstream GitHub repos were removed (now 404), so the cold layer is fixed at the last snapshot (~mid-2026) shipped in the HF Dataset. See note below. | Free, no rate limit (when it existed). |
| **matchstat** (RapidAPI) | Upcoming fixtures, current rankings, on-demand H2H + per-player past matches for the prediction view. | Daily evening UTC + lazy per-prediction. | 500 req/month free — resets on the RapidAPI **subscription billing cycle**, not the calendar 1st. |
| **The Odds API** | Pre-match h2h odds for active tour-level tournaments. Aggregated to median + Pinnacle subtitle. | Daily + lazy refresh on Prediction-page load when cache > 24h old. | 500 credits/calendar month free. |
| **tennis-data.co.uk** | Historical closing-price implied probabilities. Calibration benchmark on training reports — not a feature. | Same orchestrator as Sackmann. | Free, manual download. |
| **Anthropic** | LLM agent (news discovery + categorisation). | Per Match-dashboard render (cached in `st.session_state` after first run). | $20/month workspace cap. |
| **Tavily** | News-snippet search. Called from inside the agent's `web_search` tool. | Per agent call (max 2 searches each). | 1000 searches/month free. |

## Module layout

```
src/tennis_predictor/
├── app/                    Streamlit interface
│   ├── main.py             entry point + sidebar
│   ├── views/              one file per page
│   ├── widgets.py          shared widgets (cost monitor, quota blocks, etc.)
│   ├── context.py          MatchContext builders (from scheduled / freeform)
│   ├── db.py               session-scoped DuckDB connection
│   └── why_differs.py      6 deterministic rules + generic fallback
├── data/
│   ├── ingest_sackmann.py  cold-layer ingestion
│   ├── load_market.py      tennis-data.co.uk loader
│   ├── matchstat.py        matchstat API client (typed)
│   ├── matchstat_live.py   on-demand fetcher with 24h DuckDB cache
│   ├── odds_api.py         The Odds API client + aggregator
│   ├── odds_fallback.py    Tavily-regex odds extraction
│   ├── pre_match_odds.py   persistence + name reconciliation
│   ├── recent_form_live.py view-layer H2H + last-N helpers
│   ├── load_hot.py         scheduled_matches / rankings persistence
│   ├── refresh_hot.py      daily orchestrator + 4 prune passes
│   ├── reconcile.py        AliasIndex + fuzzy resolution
│   └── schema.py           DDL + idempotent migrations
├── features/               build_training_features + compute_features
├── models/                 walk-forward, calibration, artifact I/O
├── llm/
│   ├── client.py           LLMClient ABC + AnthropicLLMClient
│   ├── agent.py            TennisAgent.predict + AgentBudget
│   ├── prompts.py          system prompt (byte-stable)
│   ├── tools/              per-tool input/output schemas + dispatch
│   └── cost.py             pricing + cache hit rate
└── config.py               env vars, paths, model defaults
```

```
scripts/
├── refresh_data.py             — cold layer (Sackmann + market)
├── refresh_hot.py              — matchstat fixtures + rankings (daily)
├── refresh_pre_match_odds.py   — The Odds API (daily + lazy)
├── apply_aliases_review.py     — promote manual-review CSV
├── find_duplicate_players.py   — Sackmann same-name-same-DOB detection
├── apply_player_dedupe.py      — repoint stale IDs → canonical
├── build_features.py           — training_features rebuild
├── train_models.py             — 4 production artifacts
├── predict_match.py            — CLI prediction
└── clear_scheduled_matches.py  — one-off reset utility
```

## Cross-cutting concerns

- **Provenance** — every match row carries `source` + `match_external_id`.
- **Player ID reconciliation** — one canonical `player_id` per player, aliases tracked in `player_aliases` with `source` annotated. Same-name-same-DOB Sackmann duplicates are surfaced and merged via the dedupe scripts.
- **Audit artefacts** — `aliases_review.csv` (low-confidence resolutions awaiting human verdict) and `duplicate_players_review.csv` (Sackmann roster dedupe candidates) live under `data/processed/`. Both are append-only / regenerable.
- **Point-in-time correctness** — enforced by `tests/test_feature_leakage.py`. A tampered future row may not change any earlier feature value.
- **Observability** — every LLM call logged to `llm_traces` (tokens, cache stats, cost, latency, tool sequence). Every refresh logged to `ingestion_runs` (rows added/skipped/failed, requests_used, status).
- **Configuration** — all paths and env vars flow through `src/tennis_predictor/config.py`. Resolves both local dev and containerised deploy.

## Deployment shape

- **Local dev:** `uv run streamlit run src/tennis_predictor/app/main.py`. DuckDB at `data/processed/tennis.duckdb`.
- **Production:** free **Hugging Face Space** (Docker SDK, CPU basic — 2 vCPU / 16 GB RAM). Live at https://neuromediator-tennis-research-dashboard.hf.space/.
  - One Dockerfile, two-stage build (`python:3.12-slim` + uv), ~270 MB image. Same image runs on HF and (historically) Fly.
  - **No persistent disk.** HF retired the flat-rate storage tier; only object-storage buckets remain, which break DuckDB's file locking + random I/O. Instead the 1.3 GB `tennis.duckdb` + `models/` are pulled on container boot from the companion HF Dataset `Neuromediator/tennis-dashboard-data` (`scripts/hf_bootstrap.py`) onto the container's **local ephemeral FS** — a real filesystem, so DuckDB stays fast. 16 GB RAM keeps the working set page-cached.
  - Daily refresh runs **in-process** via APScheduler (`app/scheduler.py`) on a background thread. `maybe_catch_up_refresh` additionally triggers a background refresh on app start when data is stale — the primary freshness path on a host that can sleep/reset.
  - **Stays warm:** a twice-daily GitHub Actions ping (`.github/workflows/keepalive.yml`) keeps the Space from sleeping (free Spaces sleep after 48 h idle), so the in-memory DB + prediction cache persist across the uptime. An involuntary reset (HF rebuild/migration) wipes the ephemeral FS; bootstrap + catch-up-on-wake restore it.
  - Three-layer prediction cache: `st.session_state` (per-tab) → `@st.cache_data(ttl=300)` (per-process) → `prediction_cache` DuckDB table (cross-session, persists for the container uptime). Repeat clicks on the same fixture cost $0.
  - Global daily LLM trace cap (`DAILY_LLM_BUDGET=60`) caps Anthropic spend at ≈ $1-2/day.
  - Secrets via HF Space secrets: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `X_RAPIDAPI_KEY`, `THE_ODDS_API_KEY`. Non-secret env via Space variables: `ENABLE_SCHEDULER=true`, `REFRESH_HOUR_UTC=5`, `HF_DATA_REPO`, `MODELS_DIR=/data/models`.
  - Bootstrap: `tennis.duckdb` and `models/` are built locally and uploaded to the companion HF Dataset; the Space pulls them on boot. Cost: **$0/month** (free CPU, no storage).
  - Cold (Sackmann) layer is **frozen** — the upstream repos were removed, so there is no routine cold refresh anymore (the historical record is fixed at the shipped snapshot). Hot fixtures/odds still refresh automatically while warm. See the note below.
  - Prior Fly.io deployment (single Machine + volume) kept as history in `docs/phases.md` Phase 7; the migration is Phase 8.

### Cold data (Sackmann) is frozen — upstream removed

**The Sackmann source no longer exists.** The `JeffSackmann/tennis_atp` and
`tennis_wta` GitHub repos that fed the cold layer were removed (the URLs now
404; the author published a new, differently-structured point-by-point repo
we don't use). Practical consequences:

- The historical match record (`matches`, and the Elo / form / H2H state
  derived from it) is **frozen at the last snapshot** (~mid-2026), shipped in
  the HF Dataset and loaded on boot by `scripts/hf_bootstrap.py`.
- There is **no routine cold refresh** anymore — the old "`git -C
  data/raw/... pull` → `refresh_data.py` → re-upload" runbook is dead because
  the pull has no remote. (`git`-related steps in `scripts/refresh_all.py`
  are kept only for an existing local clone.)
- **No immediate impact**: the trained model artifacts and all existing
  features are unaffected. Over a season, rankings/Elo for active players
  gradually go stale (the matchstat *hot* layer still keeps the upcoming-
  fixture window current — that path is unaffected). For a research demo this
  is acceptable; a long-lived product would need a new historical source.
- To rebuild/retrain you are limited to the existing snapshot (saved in the
  dataset and locally), or you would have to adapt the new Sackmann schema —
  out of scope. If you do rebuild from the local snapshot, re-upload the DB
  (and, if retrained, the symlink-dereferenced `models/`) to the dataset and
  `hf spaces restart --factory-reboot`.

## What's intentionally absent

- No microservices, no message queue, no Redis, no managed Postgres.
- No model server — predictions run in-process via `joblib.load`.
- No user accounts, no session storage beyond Streamlit's per-tab state.
- No realtime data path — all ingestion is batch.
