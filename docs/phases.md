# Phased roadmap

What was built, in the order it was built. Each phase has an entry condition, deliverables, and an exit signal. Phases are not started until the previous one is green.

---

## Phase 0 — Bootstrap

Empty project → opinionated skeleton. Repo layout, `pyproject.toml`, `.gitignore`, `.python-version`, `.env.example`, CI workflow, pre-commit, this docs tree.

**Exit:** repo can be opened by any contributor and conventions are clear.

---

## Phase 1 — Cold data layer ✅

**Goal:** a queryable historical record of professional tennis.

**Built:**
- Jeff Sackmann's `tennis_atp` and `tennis_wta` git submodules ingested into DuckDB.
- 8-table schema (`matches`, `players`, `rankings`, `player_aliases`, `market_implied_probabilities`, etc.).
- Player reconciliation: fuzzy alias index with a manual-review checkpoint at confidence < 0.90.
- tennis-data.co.uk loader for historical closing odds (2013-current), JOIN-resolved against matches.
- `scripts/refresh_data.py` orchestrator — idempotent / incremental / `--clean`.

**Headline numbers after a clean rebuild:**
- 137 318 players (composite IDs `ATP_<id>` / `WTA_<id>`)
- 1.7M matches across all tiers (~360k tour-level singles)
- 5.6M weekly rankings
- ~52k market-implied probability rows
- Match rate vs tennis-data.co.uk: ~75% median

**Known limit (later resolved):** ~840 ATP / ~1945 WTA players share `full_name` in Sackmann's roster. Phase 4 added `scripts/find_duplicate_players.py` to surface and merge these.

---

## Phase 2 — Hot data layer ✅

**Goal:** know what matches are coming up and surface current rankings.

**Built:**
- matchstat Tennis API client (RapidAPI, "Tennis API - ATP WTA ITF", 500 req/month free tier).
- `scheduled_matches` table for upcoming fixtures.
- `ingestion_runs` table — one row per refresh, source of the "data is stale" signal.
- `scripts/refresh_hot.py` — daily orchestrator.
- Source-specific player mapping via `player_aliases.source = 'matchstat'`.

**Initial design** (Path C — chosen pragmatically): completed matches come from Sackmann cold layer, not matchstat. matchstat's `calendar/{year}` is forward-only and drops currently-active tournaments, so the "discover seasonid → fetch results" path is unreliable. Trade-off: 1-7 day lag on newly-finished matches, well within model noise.

**Exit:** daily refresh runs end-to-end, fixtures populated, freshness signal works.

---

## Phase 3 — Feature engineering ✅

**Goal:** convert match history into a point-in-time-correct feature table the model can train on.

**Built:**
- `FeatureVector` Pydantic schema (v1: 28 fields; later v3: 44).
- `build_training_features()` — single-pass chronological replay producing the `training_features` table.
- `compute_features(player, opponent, surface, tour, as_of_date) → FeatureVector` — inference-time API. Identical values to replay for the same key.
- In-memory state objects: `EloState` (persisted), `RollingFormState`, `H2HState`, `FatigueState`, `ServeReturnState`, `RankingLookup`.
- Surface-Elo (start 1500, K=32, per-(player, surface) pair) with indoor/Carpet normalisation.
- Leakage tests in `tests/test_feature_leakage.py` — tamper a future row, assert no earlier feature value changes.

**Headline numbers:**
- 369 071 training_features rows from 1.7M scanned matches.
- 106k `elo_state` rows.
- ~5 min wall-time for full replay.

---

## Phase 4 — Modeling ✅

Single combined phase covering the original Phase 4, the v2 feature expansion (Phase 4.1), the v3 surface-recovery features (Phase 4.2), and the Sackmann player-roster dedupe pass.

**Goal:** trained, calibrated model + walk-forward evaluation, iteratively improved.

**Built:**
- Walk-forward validation harness (`features/walk_forward.py`): train ≤ year `Y-2`, calibrate on `Y-1`, validate on `Y`.
- Per-tour LightGBM (production) + per-tour surface-Elo (baseline) — four model artifacts under `models/<tour>/<type>/<YYYYMMDD-HHMM>/` with `latest` symlink.
- Each artifact has `model.joblib`, `metadata.json` (training date, data range, features, walk-forward metrics, git commit), `report.md`, `calibration_plot.png`.
- Isotonic calibration when held-out set ≥ 1000, Platt otherwise.
- Market overlay on every calibration plot (closing odds from tennis-data.co.uk).
- Round-trip serialization test (catches lightgbm / sklearn version drift).

**Feature evolution:**
- v1 (28 fields): surface-Elo, recent form, serve/return rolling, H2H, fatigue, ranking, tournament context.
- v2 (39 fields): added handedness, age, height, tournament altitude, travel jet-lag, days-since-last-match recovery.
- v3 (44 fields): added `days_since_last_match_surface_p1/p2` — same recovery signal but surface-specific. LightGBM ranks the new features top-5 by gain on both tours.

**Player roster dedupe** (Phase 4 follow-up):
- `scripts/find_duplicate_players.py` finds Sackmann's same-name-same-DOB-same-country duplicates.
- `scripts/apply_player_dedupe.py` repoints aliases / scheduled_matches / matches / rankings to the canonical player_id and drops the stale row.
- First pass found 209 duplicate groups, 214 stale rows merged.

**Headline metrics (last 5 walk-forward folds, post-calibration Brier, sample-weighted):**
| Tour | Surface-Elo baseline | LightGBM v3 | Market |
|---|---|---|---|
| ATP | 0.2220 | **0.2087** | ~0.20 |
| WTA | 0.2180 | **0.1959** | ~0.20 |

Model is "approaching the market in aggregate but not beating it" — and that average masks tail failures on top matches (the motivation for Phase 6's reframe).

---

## Phase 5 — LLM agent ✅

Combined original Phase 5 + the Phase 5.1 search-provider swap.

**Goal:** an Anthropic-SDK agent that enriches each prediction with news context — bounded, observable, never overriding the model's probability.

**Built:**
- `LLMClient` abstract base + `AnthropicLLMClient` implementation. Direct SDK, no framework wrappers.
- Prompt caching by default — one `cache_control` marker on the last tool definition; cacheable prefix is byte-stable across calls.
- `TennisAgent.predict(MatchContext) → AgentResponse` orchestrator with hard budget caps.
- Tool surface:
  - `get_model_prediction` (mandatory, the only source of the win probability).
  - `get_head_to_head` (detailed H2H against matchstat `/h2h/matches/{a}/{b}`).
  - `get_surface_elo`.
  - `web_search` against Tavily (`topic="news"`, `days=32`, betting-site exclude list).
  - `submit_analysis` — structured-output collector with `additionalProperties: false`.
- `AgentResponse` = `news_items: list[NewsItem]` + `news_lookup_status` enum. Schema rejects any LLM-emitted probability and any free-text synthesis field.
- Three-tier tests: mocked-Anthropic unit tests + recorded-fixture e2e + `@pytest.mark.llm_live` (excluded from CI).
- `scripts/predict_match.py` — CLI entry point.

**Provider swap (was Phase 5.1):** Tavily snippet-only replaced Anthropic native `web_search_20250305` after an A/B test showed ~9× cheaper / ~3× faster with comparable niche-source recall.

**Cost shape:** ~$0.10 per prediction (Sonnet 4.6 + Tavily basic). $20/month workspace cap in the Anthropic console is the ultimate wall.

---

## Phase 6 — Streamlit dashboard ✅

Combined original Phase 6 + the Phase 6.1 LLM-analyst v2 + the Phase 6.2 reframe + all post-launch follow-ups (per-tournament refresh, date-only Home, hyphen-normalised odds matching, etc.).

**Goal:** ship a usable interface, observe failure modes live, iterate honestly.

**Built:**
- **Streamlit app** at `src/tennis_predictor/app/`:
  - **Home** — upcoming matches, two-level grouping (tour → tournament), tour filter (All/ATP/WTA), date-only display.
  - **Match dashboard** — signal comparison row (market / model / surface-Elo), deterministic "why model differs" panel when |gap| > 10pp, H2H detail, surface-Elo block, two-column recent form, LLM news block.
  - **Custom match** — 3-input form with player autocomplete for what-if prediction.
  - **Model evaluation** — last-20 predictions vs market scoreboard, external API usage (matchstat + Odds API quotas), cost monitor, calibration plots, recent LLM traces.

- **Live odds integration** (was Phase 6.2): The Odds API (`the-odds-api.com`, 500 credits/month free) with daily refresh + lazy on-demand. Per-tournament sport keys. Tavily-snippet fallback for missing rows. Stored in `pre_match_odds`, joined to scheduled fixtures by hyphen-normalised name match.

- **LLM analyst v2** (was Phase 6.1): dropped narrative / confidence_band / caveats / key_factors from `AgentResponse`. The LLM now emits only dated, attributed `NewsItem`s tagged with a category from a closed whitelist (`injury / withdrawal / illness / result / coach_change / personal`). Determinism is rendered by the view layer from typed tool outputs; the LLM's job is news discovery + categorisation only.

- **Product reframe** (was Phase 6.2): UI text, page titles, README, sidebar tagline all say "research dashboard" / "match dashboard" — never "calibrated prediction". Phase 6.1 close-out exposed that the model's top-match failures (inverted favourites) embarrass the product; reframing made the limits visible instead of hidden.

- **"Why model differs" panel** — six deterministic rules: activity gap, stale surface-Elo (> 180 days), career asymmetry, surface-Elo + model both lean against the market, recent-form gap on the surface, generic fallback when none fire. Panel never goes silent on a > 10pp gap.

- **matchstat hot-refresh hardening** — 4 post-refresh prune passes catching matchstat's known data quirks: stale-fixture cleanup, round-contradiction prune (player can't be in both R1 and R2 at the same tournament), duplicate-matchup prune (matchstat re-uses `fixture_external_id`), completed-Slam cross-check via `/tournament/results`. Fixture refresh switched from per-date to per-tournament endpoint (`/fixtures/tournament/{id}`) — fewer credits + more complete data.

- **Display polish** — Home shows date only (matchstat's `T12:00:00Z` is ambiguous between real 11:00 CEST and day-level placeholder); time-zone-honest match-page header; in-app navigation with explicit "back to home" buttons; `st.session_state` persistence so the agent doesn't re-fire on page revisits.

**Exit signal:** dashboard ships, used live during Roland Garros 2026, iteration cycle observed → fixed → tested → committed for ~10 distinct issues.

---

## Phase 7 — Deployment ✅

Public deployment to **Fly.io**, single-Machine, single-DuckDB-file. Live at
**https://tennis-research-dashboard.fly.dev/**.

**Built:**
- **Dockerfile** — two-stage build, `python:3.12-slim` + uv. Final image ~270 MB. Includes `libgomp1` (lightgbm runtime) and `git` (used by the deferred weekly Sackmann pull).
- **fly.toml** — one always-on app process (Streamlit on port 8080), `shared-cpu-1x` / 2 GB RAM, 5 GB persistent volume at `/data`, force-https, health check on `/_stcore/health`.
- **APScheduler in-process daily refresh** (`src/tennis_predictor/app/scheduler.py` + `data/refresh_jobs.py`). Background-thread `CronTrigger(hour=21, minute=0, timezone=UTC)`. Each fire opens a fresh DuckDB connection (same process, multi-connection inside DuckDB is fine), runs `refresh_hot.refresh_hot(...)` + `odds_refresh.refresh(...)`, closes. Gated on `ENABLE_SCHEDULER=true`. **No separate cron Machine** — DuckDB does not support multi-process write access, so two Machines on one volume would deadlock the file lock.
- **Three-layer prediction cache:**
  - L0 `st.session_state` — per-tab, instant.
  - L1 `@st.cache_data(ttl=300)` — per-process, 5 min.
  - L2 `prediction_cache` DuckDB table — cross-session, 24 h, shared across all visitors and devices. Keyed by `scheduled_match_id`.
- **Global daily LLM budget cap** — `DAILY_LLM_BUDGET=60` traces/day (≈ 15-20 predictions ≈ $1-2 Anthropic). When exceeded, `_cached_predict` returns a synthetic AgentResponse via direct `get_model_prediction` call with `news_lookup_status="budget_exhausted"`; news block renders a "paused until 00:00 UTC" message, all other blocks unaffected.
- **Volume bootstrap recipe** — temporarily replace CMD with `sleep infinity`, deploy, `fly ssh console`, clone Sackmann submodules to `/data/raw/`, run cold ingest + features + train + first hot/odds refresh, revert CMD, redeploy.
- **`scripts/refresh_all.py`** — manual subprocess-based runner for bootstrap and offline ops (the in-process job uses `data/refresh_jobs.py` instead).

**Surprises during execution** (worth knowing for v2 / similar deployments):
- **1 GB RAM is too tight.** Streamlit + LightGBM + DuckDB metadata cache on a 1.3 GB DB + render pipeline OOMs under load. 2 GB is the working size; cost goes $1.94 → $3.89/mo.
- **`shared-cpu-1x` performance varies by time-of-day.** Cold-path prediction is ~40 s during EU morning / US night and ~80 s during EU night / US day (host contention with other tenants). Acceptable because cache hits dominate.
- **`build_features.py` and `train_models.py` will not complete on Fly's shared CPU.** Bootstrap pattern: build locally, upload `tennis.duckdb` (1.3 GB) and `models/` to the volume via `fly ssh sftp put`.
- **tennis-data.co.uk loader is CPU-bound** (rapidfuzz alias matching against ~5 k names per year × 25 years). Skip it during bootstrap (`refresh_data.py --skip-market`) — market data is only used as a calibration overlay, not in production inference.
- **Three-layer cache was wired in `widgets._cached_predict` but NOT called from `views/prediction._run_agent`.** Cross-session L2 hits silently failed (phone and desktop got independent LLM runs). Found by user reporting "phone gave different news than desktop". Fix: route `_run_agent` through `_cached_predict`. Lesson: integration tests for cache hit on a fresh session would have caught it.
- **`fly ssh console -C "cmd && cmd"` does not run through a shell** — `&&` is passed literally to `cat`/`rm`/etc. Wrap in `bash -c '...'`.

**Open follow-ups (Phase 7.1+):**
- Automated weekly Sackmann refresh in APScheduler (requires refactoring `scripts/refresh_data.py` into a library function).
- Visible cold-data freshness warning in the recent-form block footer.
- Basic-auth gate on Custom page (if traffic warrants).
- Cloudflare in front for L7 protection.
- Custom domain.

> **Superseded by Phase 8.** The Fly.io deployment was retired in favour of a
> free Hugging Face Space. This section is kept as historical record.

---

## Phase 8 — Migration to Hugging Face Spaces ✅

Public deployment moved off Fly.io onto a **free Hugging Face Space**. Live at
**https://neuromediator-tennis-research-dashboard.hf.space/**. Cost dropped from
~$4/month to **$0**.

**Why move:**
- **Speed.** Fly's `shared-cpu-1x` / 2 GB OOM-swapped under load; cold-path
  prediction reached ~4 min during contended hours. HF free CPU gives **16 GB
  RAM** (no swap) → the working set stays page-cached and the cold path is back
  to ~30-60 s.
- **Cost.** Free CPU basic vs Fly's paid Machine + volume.

**The storage saga (the hard part):**
- HF has **retired the flat-rate persistent-storage tier** (the old
  `request_space_storage(small)` API now 404s). The only durable read+write
  storage left is **object-storage buckets**, which break DuckDB — the DB file
  needs real file locking + random I/O + `fsync`/WAL semantics that an S3-FUSE
  mount can't provide, and random reads of a 1.3 GB file over object storage
  would be *slower* than the Fly disk we were leaving.
- **Resolution: no persistent storage at all.** The DuckDB + model artifacts
  live on the container's **local ephemeral filesystem** (a real, fast FS).
  `scripts/hf_bootstrap.py` pulls the 1.3 GB snapshot from a companion HF
  Dataset (`Neuromediator/tennis-dashboard-data`) on container boot. Writes
  (daily refresh, `prediction_cache`, `llm_traces`) persist for the container's
  uptime but not across an involuntary reset — acceptable for a research demo
  where the dataset is the source of truth.

**Keeping it usable at low traffic:**
- Free Spaces sleep after 48 h idle; a cold start would re-download the 1.3 GB
  snapshot (~1-3 min — bad for the first visitor). Fix: a **twice-daily GitHub
  Actions ping** (`.github/workflows/keepalive.yml`) keeps the container warm,
  so the in-memory DB + prediction cache persist and the 05:00 UTC APScheduler
  refresh runs. (GitHub disables scheduled workflows after 60 days of repo
  inactivity — the Space then simply sleeps again; no data loss.)
- **`maybe_catch_up_refresh`** (`app/scheduler.py`): on app start, if hot data
  is >24 h stale, schedule an immediate background refresh. This is what
  recovers freshness after an involuntary reset, and what the original
  "scheduler silently stopped" incident motivated.

**Also shipped (carried over from the incident that triggered the migration):**
- **Visible 429 signal.** matchstat's monthly quota (RapidAPI, resets on the
  subscription billing cycle ~the 18th, *not* the calendar 1st) had silently
  exhausted; the hot refresh failed on its first call every night for 3 days
  and the UI showed only a generic "stale" banner. `widgets.is_quota_error` +
  `query_last_hot_run_error` now surface "matchstat monthly quota exhausted
  (429)" distinctly.

**Built:**
- `scripts/hf_bootstrap.py` + `scripts/docker-entrypoint.sh` (bootstrap-on-boot;
  no-op when the volume is already populated or `HF_DATA_REPO` unset, so the
  same image still runs anywhere).
- HF Space front-matter in `README.md` (`sdk: docker`, `app_port: 8080`).
- `.github/workflows/keepalive.yml` keep-warm pinger.
- `app/scheduler.py` catch-up-on-wake + a process-wide refresh lock.

**Open follow-ups:**
- All Phase 7.1+ items still stand (automated Sackmann refresh, cold-data
  freshness warning, basic-auth on Custom, custom domain).
- If matchstat quota proves too tight (daily refresh ~450/mo of 500), move
  per-prediction H2H / recent-form off live matchstat onto the local Sackmann
  data so predictions cost zero quota.
- Consider a smaller bootstrap DB (prune history not needed for inference) to
  shrink the cold-start re-download after an involuntary reset.
