# Phase 7 — Public deployment plan

Public deployment of the research dashboard to Fly.io. This document captured the design decisions agreed during planning and the concrete steps to execute them. **Phase 7 shipped 2026-05-29** — live at https://tennis-research-dashboard.fly.dev/. See the *Execution notes* section at the bottom for what changed between the plan and reality.

## Goals

- A public URL anyone can open to see upcoming matches, signals, and the model evaluation.
- Hard ceiling on external-API spend (Anthropic / Tavily / matchstat / The Odds API).
- Daily fixtures + odds refresh runs on a schedule, not on user clicks.
- Single Dockerfile, single DuckDB file, no microservices.

## Non-goals (v1)

- High availability / horizontal scaling.
- Multi-user accounts, profiles, history.
- Live (in-match) updates.
- Custom domain (uses `.fly.dev` subdomain).
- Per-IP rate limiting (deferred to Cloudflare if abuse appears).
- **Automated weekly cold (Sackmann) refresh.** The cold layer is updated manually via `fly machine stop` → `fly ssh console` → `git pull` Sackmann clones + `python scripts/refresh_data.py --skip-submodules` → `fly machine start`. Refactoring `scripts/refresh_data.py` into a library function is meaningful work and the cold layer is not user-facing critical (matchstat covers the rolling 7-day window).

## Architecture on Fly.io

One always-on Machine. The daily refresh runs as a background thread
inside the Streamlit process via APScheduler — DuckDB does not support
multi-process write access, so a separate cron Machine would deadlock
on the file lock. Single Machine + APScheduler keeps the writer count
at one.

```
┌──────────────────────────────────────────────────────────────┐
│ Streamlit Machine (always-on, min=1)                         │
│                                                              │
│   ┌────────────────────────┐  ┌─────────────────────────┐    │
│   │ Streamlit main thread  │  │ APScheduler bg thread   │    │
│   │   serves HTTP traffic  │  │   CronTrigger(21:00 UTC)│    │
│   │   read+write DuckDB    │  │   opens fresh conn,     │    │
│   │   conn (st.cache_res.) │  │   runs refresh_jobs,    │    │
│   │                        │  │   closes conn           │    │
│   └───────────┬────────────┘  └────────────┬────────────┘    │
│               │                            │                 │
│               └─────────┬──────────────────┘                 │
│                         │ (same process, in-DB coordination) │
└─────────────────────────┼────────────────────────────────────┘
                          ▼
                ┌─────────────────────────────┐
                │ Fly persistent volume       │
                │ /data/processed/tennis.duck │
                │ /data/models/               │
                │ /data/raw/tennis_atp        │
                │ /data/raw/tennis_wta        │
                └─────────────────────────────┘
```

**Refresh timing:** APScheduler `CronTrigger(hour=21, minute=0)` fires at
exact 21:00 UTC (configurable via `REFRESH_HOUR_UTC` env var). On Machine
restart the scheduler reconstructs the next fire time from scratch;
`misfire_grace_time=3600` lets it catch a run missed by up to 1h.

**Bootstrap and manual refreshes** still use `scripts/refresh_all.py`
(subprocess-based CLI) — but only when Streamlit is stopped (else the
subprocess CLI scripts collide with the running app on the DuckDB file
lock). Operator pattern: `fly machine stop` → `fly ssh console` → run
scripts → `fly machine start`.

## Decisions

### 1. Access control + cost ceiling

- **Home / Model evaluation** — public, no auth. DB-only, no external API cost.
- **Match dashboard** — public, with three layered defenses:
  - **Layer 1** — Streamlit `@st.cache_data(ttl=300)` (in-memory, per-process, 5 min). Fast path inside a single visit.
  - **Layer 2** — DuckDB `prediction_cache` table (cross-session, 24h TTL, shared by all visitors, survives Machine restarts). Keyed by `scheduled_match_id`. Schema: `(scheduled_match_id PK, cached_at, agent_response_json)`. Custom predictions skip this layer.
  - **Layer 3** — Global daily LLM trace cap. `DAILY_LLM_BUDGET` env var (default 60 traces ≈ 15-20 unique predictions/day, ≈ $1-2/day Anthropic spend). Counts `llm_traces` rows since UTC midnight. When the cap is reached, `_cached_predict` skips the LLM and returns a model-only `AgentResponse` (direct `get_model_prediction` call) with `news_lookup_status="budget_exhausted"`. The news block renders "Daily LLM news-lookup budget reached, paused until 00:00 UTC"; all other blocks (market / model / surface-Elo / H2H / recent form) render normally. Budget-exhausted responses are NOT written to Layer 2 so the next-day visitor gets a real lookup.
- **Custom prediction** — public in v1. The global `DAILY_LLM_BUDGET` cap protects all paths (including Custom) from cost abuse, and the audience is small. Basic-auth gate is deferred to follow-ups if traffic warrants.

### 2. DuckDB concurrency

- **One process, one Machine, one volume.** DuckDB's file lock is per-process and prevents multi-process write access — so the daily refresh runs *inside* the Streamlit process as an APScheduler background thread.
- APScheduler job opens a fresh `duckdb.connect()` per run; DuckDB allows multiple connections within one process (they share the database instance internally, no lock conflict).
- The refresh writes for 30s-2min/day; user-facing read queries during that window may see slightly slower responses but no errors (DuckDB serialises writes within the process).
- No retry wrapper needed.
- Env var `ENABLE_SCHEDULER` (default false locally, set to true in Fly) gates the scheduler so local `streamlit run` doesn't hit production APIs.

### 3. Bootstrap DB

- Image ships code + Python deps only. **No DuckDB file, no Sackmann data, no models in the image.**
- After first `fly deploy`, populate volume once via `fly ssh console`:
  ```bash
  # Clone Sackmann submodules onto the volume (one-time, ~450MB total).
  mkdir -p /data/raw /data/processed /data/models
  git clone --depth 50 https://github.com/JeffSackmann/tennis_atp.git /data/raw/tennis_atp
  git clone --depth 50 https://github.com/JeffSackmann/tennis_wta.git /data/raw/tennis_wta

  # Cold ingest + features + models + first hot refresh.
  python scripts/refresh_data.py --clean --skip-submodules
  python scripts/build_features.py
  python scripts/train_models.py
  python scripts/refresh_hot.py
  python scripts/refresh_pre_match_odds.py
  ```
- Weekly Sackmann updates are **manual** (see Non-goals). When the operator wants fresh cold data: `fly machine stop` → `fly ssh console` → `git -C /data/raw/tennis_atp pull --ff-only && git -C /data/raw/tennis_wta pull --ff-only` → `python scripts/refresh_data.py --skip-submodules` → `fly machine start`.

### 4. Cron runner

- **In-process APScheduler.** Module `src/tennis_predictor/app/scheduler.py` builds a singleton `BackgroundScheduler` via `@st.cache_resource`, started from `app/main.py`.
- Daily job (`CronTrigger(hour=21, minute=0, timezone=UTC)`) calls `run_daily_refreshes()` from `src/tennis_predictor/data/refresh_jobs.py`.
- Inside the job: open a **fresh** DuckDB connection (not shared with the Streamlit main thread), call the library function `refresh_hot.refresh_hot(...)` + `odds_refresh.refresh(...)`, close the connection.
- **Daily scope only.** Weekly Sackmann cold ingest is intentionally NOT in the scheduler (see Non-goals / Section 3).
- No `fly machine run --schedule` is used. No subprocesses. No second Machine.
- Logging: `[scheduler]` / `[refresh_jobs]` INFO messages land in `fly logs` via `logging.basicConfig(level=INFO)` set in `app/main.py`.
- Local dev: scheduler is gated by `ENABLE_SCHEDULER=true` (defaults off) so `uv run streamlit` doesn't hit production APIs. Overrides: `REFRESH_HOUR_UTC`, `REFRESH_MINUTE_UTC`.

### 5. Misc

- **Secrets** via `fly secrets set` (names match `config.py` env-var reads):
  - `ANTHROPIC_API_KEY`
  - `TAVILY_API_KEY`
  - `X_RAPIDAPI_KEY` (matchstat via RapidAPI)
  - `THE_ODDS_API_KEY`
  - `DAILY_LLM_BUDGET` (optional override; default in code)
- **Rate limiting** — skipped for v1.
- **README** — add "Live demo" section with public URL + short "Deployment" block (~30 lines).
- **Domain** — `.fly.dev` subdomain only.

## Implementation order (tasks)

1. **Dockerfile** — multi-stage build, Python 3.12, uv-installed deps, no DB in image. ✅
2. **fly.toml** — single always-on app, volume mount, health check. ✅
3. **APScheduler integration** — `refresh_jobs.py` (in-process driver) + `scheduler.py` (BackgroundScheduler singleton) + hook in `main.py`. Added `apscheduler` to deps. ✅
4. **Prediction cache table** — schema migration + read/write helpers + integration with `_cached_predict`. ✅
5. **Global LLM cap** — daily counter from `llm_traces` + check before `agent.predict` + graceful fallback rendering. ✅
6. **Fly setup** — `fly launch`, `fly volumes create`, `fly secrets set`, first deploy. ✅
7. **Volume bootstrap** — `fly ssh console` + sftp upload of pre-built `tennis.duckdb` and `models/` (Fly's CPU is too slow for `build_features.py` / `train_models.py`). ✅
8. **Cache-routing bug fix** — `views/prediction.py:_run_agent` was calling `TennisAgent.predict()` directly, bypassing all three cache layers. Routed through `widgets._cached_predict` so cross-session L2 hits work. ✅
9. **Verify** — APScheduler started, predictions render, L2 cache hits cross-session (desktop + phone show identical result). ✅
10. **README update** — live demo + deployment section. ✅

(Basic-auth on Custom page was originally task 6, dropped from v1 — see Decisions §1 and Open follow-ups.)

## Acceptance criteria

- Public URL serves the dashboard.
- Home / Match dashboard / Model evaluation pages render without auth.
- Re-clicking Predict on the same fixture within 24h does not increment LLM spend (Layer 2 cache hit; `llm_traces` row count does not grow).
- After hitting `DAILY_LLM_BUDGET`, predictions still render (model + market + Elo + H2H + recent form) — no 500s. The news block shows the budget-exhausted message.
- APScheduler's daily refresh appears in `fly logs` as `[scheduler] daily refresh finished` and writes one row to `ingestion_runs` per source (matchstat + the_odds_api).
- Monthly spend ceilings: Anthropic ≤ $20 workspace cap, Tavily within 1000/mo, matchstat within 500/mo, Odds API within 500/mo.

## Open follow-ups (not in v1 scope)

- **Automated weekly Sackmann refresh** in APScheduler (requires refactoring `scripts/refresh_data.py` into a library function callable in-process). Tracked separately as Phase 7.1.
- **Visible cold-data freshness warning** — surface "Sackmann last updated N days ago" in the recent-form block footer so users know when matchstat-fallback may show stale matches.
- **Basic-auth gate on Custom page** — `st.text_input(type="password")` against `CUSTOM_PAGE_PASSPHRASE` env. Add if traffic / abuse becomes a concern.
- Cloudflare in front for L7 protection if traffic warrants.
- Custom domain.
- Per-IP rate limit.
- Backups of DuckDB volume (currently regenerable from cold layer; consider weekly snapshot if state diverges from Sackmann).

---

## Execution notes (what changed from plan to reality)

The plan above is the intended design. A few decisions were tightened or revised during execution; this section is the honest delta.

### Memory: 1 GB → 2 GB

The plan called for `shared-cpu-1x` / 1 GB ≈ $1.94/mo. First real deploy OOM-killed the Machine within minutes: Streamlit + DuckDB metadata cache against a 1.3 GB DB + LightGBM artifacts + the page render pipeline doesn't fit. Scaled to 2 GB (`fly scale memory 2048`, also reflected in `fly.toml`). New cost: ~$3.89/mo.

### Bootstrap: built locally, uploaded via sftp

The plan was to run `python scripts/refresh_data.py --clean --skip-submodules`, then `build_features.py`, then `train_models.py` inside the bootstrap Machine. In practice:
- `refresh_data.py` on Fly's shared CPU got hung on `tennis-data.co.uk` market ingest — rapidfuzz alias matching against the year-by-year ATP/WTA xlsx is heavily CPU-bound. Workaround: `--skip-market`. Market data is not loaded in production (only used as a calibration overlay).
- `build_features.py` OOM-killed after ~20 min on 1 GB. Even at 2 GB the wall time on shared CPU was excessive.
- **Final path:** bootstrap with `refresh_data.py --clean --skip-submodules --skip-market` for the cold layer schema, then **upload the local `tennis.duckdb` (1.3 GB) and `models/` (15 MB) via `fly ssh sftp put`**. Subsequent fresh data comes from the daily APScheduler runs.

### CPU performance varies by time-of-day

`shared-cpu-1x` means the vCPU is shared with other tenants on the same host. Cold-path prediction takes ~40 s during EU morning / US night and ~80 s during EU night / US day. The L2 cache (24 h) absorbs this — repeat clicks on the same fixture are instant — so the variance only hits the first visitor per fixture per day.

### `fly ssh console -C "..."` is not a shell

`-C "rm /x && ls /y"` passes `-la` as an arg to `rm`. Wrap in `bash -c '...'` to get shell semantics (`&&`, redirection, glob).

### Cache integration bug (caught post-deploy)

The three-layer cache was wired inside `widgets._cached_predict`, but the Prediction page's `_run_agent` was calling `TennisAgent.predict()` directly — bypassing all three layers. Users saw "phone gives different news than desktop" for the same fixture. Fix: route `_run_agent` through `_cached_predict`. **Lesson:** the plan should have included an integration test for "same `scheduled_match_id` + fresh session → L2 hit, no new `llm_traces` row". Wired-but-uncalled caching is the worst kind of false confidence.
