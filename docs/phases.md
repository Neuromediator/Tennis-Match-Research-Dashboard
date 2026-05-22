# Phased roadmap

Each phase has entry criteria (what must be true before starting), deliverables (what is produced), and exit criteria (what must be true before moving on). Phases are not started until the previous phase's exit criteria are green.

---

## Phase 0 — Bootstrap

**Entry:** empty project directory, agreed scope.

**Deliverables:**
- Repo skeleton and directory tree.
- `CLAUDE.md`, four skill files, three docs (this file, architecture, methodology), README.
- `pyproject.toml` with declared dependencies (not yet installed).
- `.gitignore`, `.env.example`, `.python-version`.
- CI workflow (`.github/workflows/ci.yml`), `.pre-commit-config.yaml`.

**Exit:** repo can be opened by any contributor and conventions are clear; nothing is yet implemented.

---

## Phase 1 — Cold data layer  ✅ complete

**Entry:** phase 0 complete; `uv sync` succeeds.

**Deliverables:**
- Sackmann `tennis_atp` and `tennis_wta` added as git submodules under `data/raw/`.
- DuckDB schema (`schema.py`) for all 8 tables.
- Sackmann ingestion (`ingest_sackmann.py`): players, matches (3 tiers ATP / 2 tiers WTA), rankings.
- Player reconciliation (`reconcile.py`): `normalize_name`, `seed_aliases_from_players` (three forms per player), `AliasIndex` with fuzzy lookup, manual-review threshold.
- tennis-data.co.uk loader (`load_market.py`): download, parse, overround normalization, JOIN-based match resolution.
- Orchestration script (`scripts/refresh_data.py`) with `--clean` / `--skip-submodules` / `--skip-market` / `--tours` / `--market-years` flags.
- 42 unit tests across 7 module-specific test files.

**Exit (all green):**
- ✅ `uv run python scripts/refresh_data.py` is idempotent — incremental on a populated DB (cheap, daily-runnable), full build on an empty one. `--clean` forces a from-scratch rebuild (~5 min Sackmann + ~25 min market data 2013–current).
- ✅ All phase-1 tests pass locally and in CI.
- ✅ `aliases_review.csv` exists (~800 rows queued for manual review on a clean rebuild).
- ✅ Coverage report printed at end of refresh: per-tour-per-year match counts and market overlap.

**Headline numbers (full rebuild):**
- 137,318 players (ATP + WTA combined, composite IDs `ATP_<id>` / `WTA_<id>`)
- 1,701,617 matches across all tiers (~360k tour-level singles)
- 5,559,400 weekly rankings
- 360,676 player_aliases (canonical + reversed + abbreviated forms per player)
- ~52,000 market-implied-probability rows across ATP+WTA 2013–current (older years require legacy `.xls` parsing — deferred)
- Match rate vs tennis-data.co.uk: ~65–85% per year (median ~75%)

**Known limits, documented for later phases:**
- tennis-data.co.uk pre-2013 archive: we currently download `.xlsx`, and that format is not published for earlier years (the site detects an `.xls`-only landing page and our loader skips with a warning). **CORRECTION (Phase 3 review):** tennis-data.co.uk also publishes a CSV variant for every year, including pre-2013. Switching the loader from `.xlsx` to CSV is a small future task that unlocks ~10 extra years of market benchmark coverage. Odds are not training features (CLAUDE.md hard rule #3), so adding them does NOT require a rebuild of `training_features` / `elo_state` — only the `market_implied_probabilities` table grows. Deferred until Phase 4 calibration reporting actually needs the older years.
- ~840 ATP and ~1945 WTA players share a `full_name` in Sackmann's roster (different IDs). `find_namesakes()` surfaces them; phase 2/3 needs a per-source disambiguator before merging.

---

## Phase 2 — Hot data layer  ✅ complete

**Entry:** phase 1 exit criteria met.

**Deliverables:**

- **Hot API provider: matchstat Tennis API.** Published as "Tennis API - ATP WTA ITF" on RapidAPI; reference docs at `tennisapidoc.matchstat.com`. Free tier: 500 requests / month, hard cap. Documented fallback if matchstat is retired or its free tier shrinks: any RapidAPI tennis API exposing fixtures-by-date, completed matches with scores, and ATP/WTA rankings on a free tier ≥ 1500 req/month.

- **Endpoint-to-responsibility map** (all under `https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2`):

  | Endpoint | Purpose | Frequency |
  |---|---|---|
  | `/{tour}/tournament/calendar/{year}` | Tier-by-tournament-id lookup so `scheduled_matches.tournament_tier` can be populated. | Daily per tour (1 call). |
  | `/{tour}/fixtures/{date}?include=tournament.court,tournament.rank,round&filter=PlayerGroup:singles` | Upcoming fixtures → `scheduled_matches`. `include` brings surface (`court.name`) and round name; `filter` drops doubles. | Daily per tour, paginated. |
  | `/{tour}/ranking/singles?pageSize=100` | Inter-week ranking overlay between weekly Sackmann snapshots. | Daily per tour. |

  **Completed matches are NOT pulled from matchstat.** The first live smoke test surfaced that matchstat's `calendar/{year}` is forward-only: tournaments disappear from the listing once they start, so the "discover seasonid via calendar → fetch results" path silently misses the current week's events. Path C (chosen): completed matches come from Sackmann (cold, weekly git submodule). The trade-off is a 1–7 day lag for newly-finished matches in `matches`; for feature engineering this is a ~10% off-by-one in `last_10` form for active top players, well within model noise. The `tournament/results` endpoint and `insert_completed_matches` code stay in the codebase, exercised by unit tests, available if Path B (discover seasonid via fixtures' `tournamentId`) is wired in later.

  Tour-level whitelist on `tier`: `{"Grand Slam", "ATP Masters 1000", "ATP 500", "ATP 250", "WTA Masters 1000", "WTA 1000", "WTA 500", "WTA 250", "Finals"}`. Strings are matchstat's literal values, observed via the live API — the Masters tier uses the full `"ATP Masters 1000"` form, not bare `"ATP 1000"`. Everything outside this set (Challengers, ITF tiers like `"Future"`/`"M15"`/`"M25"`) is dropped.

- **Budget.** Typical day under Path C: ~4 calls per tour (1 calendar + ~2 fixtures pages + 1 rankings). Both tours daily: ~8. Bursty days with full Slam draws: ~12. Steady-state monthly: ~150–250 against the 500/month cap. Refresh script must avoid naive retry loops; per-call counts logged to `ingestion_runs`.

- **Upcoming fixtures lookahead is naturally short.** Tennis draws fix round 1 at the start of the week (Sun/Mon), and each subsequent round becomes known only once the previous round completes. The `scheduled_matches` table reflects whatever the API knows at refresh time: full round 1 right after a draw, rolling down to "today plus part of tomorrow" by mid-tournament. This is what the product lets users predict against; without it the app cannot surface "tonight's matches."

- **Cross-source key.** Fixture `id` from `/fixtures/...` (small integer) and match `id` from `/tournament/results/...` (8-digit string) are **not** the same identifier. The link between a `scheduled_matches` row and the `matches` row produced when the match completes is the composite `(tournamentId, player1Id, player2Id, roundId)`, not a shared external id. Under Path C this is currently a near-no-op (Sackmann tourney_id doesn't match matchstat tournament_id), but `promote_completed_fixtures` remains wired for the future Path B case.

- **New tables.** `scheduled_matches` (upcoming fixtures) and `ingestion_runs` (one row per refresh: run id, source, started_at, finished_at, rows added/skipped/failed, error if any) — the UI reads the freshness signal from `ingestion_runs`.

- **Source-specific player mapping** integrated into `player_aliases` (`source='matchstat'`). Same manual-review checkpoint as cold data — no silent ambiguous merges.

- **Error budget.** When the hot API is unreachable or partial, the app degrades gracefully: predictions remain available against the last cached fixtures with a visible "data is N hours stale" warning, sourced from `ingestion_runs`.

- **Tests.** Hot rows do not duplicate cold rows; daily refresh is idempotent; ranking overlay never reads from the future; the tier-whitelist filter drops Challenger/ITF rows; orchestrator under Path C does NOT call `/tournament/results`.

**Exit:**
- `uv run python scripts/refresh_hot.py` runs end-to-end and updates DuckDB.
- `scheduled_matches` contains every fixture the hot API knows about at the moment the script runs (in practice: full round 1 right after a draw, or today/tomorrow's matches mid-tournament).
- `ingestion_runs` records the run with row counts (added / skipped / failed).
- A simulated API outage (mocked failure) leaves the app usable on the last cache, surfacing the staleness warning.

---

## Phase 3 — Feature engineering  ✅ complete

**Entry:** phase 2 exit criteria met.

**Deliverables:**
- `build_training_features()` producing `training_features` rows for every eligible match.
- `compute_features(...) -> FeatureVector` for inference.
- Pydantic `FeatureVector` schema — **28 fields in v1** (26 numeric + 2 categorical) across seven families: Surface-Elo, recent form, serve/return rolling, H2H, fatigue, ranking, tournament context. Exact field-by-field list and the rationale for the non-obvious serve/return choices (why `first_serve_win_pct` over `first_serve_in_pct`, why `aces_per_game` and `double_faults_per_game` are excluded in v1, etc.) live in `.claude/skills/feature-engineering/SKILL.md`.
- Surface-Elo ratings pipeline with persistent `elo_state` — default rating 1500, K-factor 32, one row per `(player, surface)` pair.
- Rolling, H2H, fatigue, and ranking features.
- Leakage tests: tampered-future-rows fixtures, asserting no past feature value moves.

**Exit (all green):**
- ✅ All 12 anti-leakage tests pass in CI (tampered-future-rows: winner/loser swap, score, stats, surface, indoor-name promotion, ranking insert, INSERT/DELETE).
- ✅ `training_features` has one row per eligible match, with no nulls in required feature columns (verified by `test_required_columns_non_null`).
- ✅ `compute_features` returns identical values for the same `(player, opponent, surface, as_of_date)` whether reached via training replay or inference path (verified by `test_equivalence_with_training_replay`).

**Headline numbers (full DB):**
- 1,701,617 matches scanned, 1,635,925 state updates applied.
- **369,064 training_features rows** written (ATP main 169k + ATP qualifying 30k + WTA main 131k + WTA qualifying 38k).
- 106,404 `elo_state` rows persisted (one per `(player_id, surface)` pair with at least one completed match).
- ~5 min wall-time for a full replay on the populated DB.
- Skip breakdown: 56,745 non-completed (RET/W/O/DEF), 8,947 null-surface, 1,220,289 non-main-tier (Challengers/Futures/ITF — feed state, no labels), 28,239 excluded level (Davis Cup, Olympics, WTA OOS, WTA 125), 18,214 below history floor.
- Label balance ~53/47 in favour of `label=1` — lex-ordering by `player_id` correlates lightly with career length; LightGBM handles class imbalance natively.
- 271 tests pass (Phase 1+2+3 combined).

**Implementation notes (post-design):**
- 5 in-memory state objects + 1 in-memory lookup: `EloState` (persisted), `RollingFormState`, `H2HState`, `FatigueState`, `ServeReturnState` (rebuilt each run) and `RankingLookup` (bisect over `rankings` table).
- Surface taxonomy: `{Hard, IHard, Clay, Grass}`. Carpet → IHard. Indoor whitelist (`src/tennis_predictor/features/indoor_tournaments.py`) lifts Paris Bercy / Vienna / Rotterdam / etc. from Hard to IHard.
- Tournament-level normalization: 7 canonical values (Slam, M1000, ATP500/250, WTA500/250, Finals). ATP `A` disambiguated via hardcoded 500-list; WTA legacy Tier I-V mapped to modern equivalents; D/O/WTA-125 excluded.
- History floor: both players must have ≥5 completed matches; canonical `(p1, p2)` is lex-smaller `player_id` first.
- Tour-level main-draw qualifying (Q1/Q2/Q3 at Slams/Masters/250-500) is label-eligible per user decision — Sackmann stores them inside `qual_chall` / `qual_itf` files mixed with Challengers/ITF, whitelisted by per-tour level codes.
- `compute_features` rebuilds Elo from scratch when `as_of_date ≤ persisted snapshot date` — necessary for historical inference (e.g., the equivalence test) since the snapshot reflects state **after** every DB match.

**Known limits, documented for later phases:**
- Market odds for qualifying matches: currently 0 rows in `market_implied_probabilities` for qual_chall / qual_itf tiers. Likely a `load_market.py` JOIN bug (tennis-data.co.uk does publish qualifying odds). Phase 4 calibration plots will be main-draw only until this is fixed.

---

## Phase 4 — Modeling  ✅ complete

**Entry:** phase 3 exit criteria met.

**Deliverables:**
- Two trained models per tour, four artifacts total:
  - **Surface-Elo baseline** — no learning, pure rating-based prediction. Kept as the honest reference floor: any shipped model must beat this Brier score on walk-forward.
  - **LightGBM** — gradient-boosted, the production model.
  - Logistic regression is fine as exploratory work during development but is not a shipped artifact.
- Walk-forward validation harness with per-fold metrics.
- Calibration applied per the isotonic/Platt decision rule (`docs/methodology.md`).
- Per-model artifact directory with `model.joblib`, `metadata.json`, `report.md`, `calibration_plot.png`.
- Market-benchmark calibration overlay in every report.
- Round-trip serialization test.

**Exit (all green):**
- ✅ Four fresh model artifacts exist in `models/` (Elo baseline + LightGBM per tour), with `latest` symlinks pointing at the production run.
- ✅ LightGBM Brier score beats the Elo baseline on each tour over the most-recent 5 walk-forward folds (post-calibration, sample-weighted): ATP 0.2105 vs 0.2220 (Δ +0.0115); WTA 0.2026 vs 0.2180 (Δ +0.0154).
- ✅ Market-benchmark plot is visible in every report; market remains slightly better-calibrated than our model (ATP recent-fold market Brier ~0.20 vs our 0.21), in line with the "approaching but not reaching" framing.

**Headline numbers:**
- 8 walk-forward folds per tour: validate years 2018–2025. Each fold splits train (≤ V−2), calibrate (V−1), validate (V).
- Production model: train ≤ 2024, calibrate on 2025; isotonic on both tours (calibration set ~3.6–3.7k matches, well over the 1000-row threshold).
- LightGBM hyperparameters: 1500 trees max, lr 0.03, num_leaves 63, min_child_samples 50, feature/bagging fractions 0.9, early stopping after 75 rounds on calibration-set log loss. Categorical features (`tournament_level`, `surface`) handled natively (no one-hot).
- Wall-time end-to-end: ~55 seconds for all four artifacts on a workstation CPU.
- Per-fold market overlay covers 7/8 folds (2020 ATP fold has only 969 market rows, below the 1000-row overlay threshold).

**Implementation notes (post-design):**
- All four artifacts share the `CalibratedPredictor` wrapper (base estimator + 1-D calibrator), so the joblib round-trip is shape-uniform and the serialization test covers both Elo and LightGBM with the same code path.
- The Elo baseline is also post-calibrated. The raw formula is well-ordered but isotonic nudges it onto the diagonal — costs nothing and keeps the calibration plot honest.
- LightGBM early-stopping uses the calibration set (also used later for post-hoc calibration). Validation set stays held out for reported metrics.
- 16-row `roundtrip_fixture.json` saved next to each `model.joblib`: catches lightgbm / sklearn version drift before it silently changes predictions.

---

## Phase 4.1 — Feature expansion (planned)

**Entry:** Phase 4 exit criteria met. Full design document: `docs/tutorials/phase_4_1_notes.md`.

**Motivation.** Phase 4 LightGBM closed about half the Brier gap to the closing market on each tour. Several low-cost signals are sitting in the cold DB unused (`players.hand` 100% / `dob` ~95% / `height` 25–57% coverage on active players), plus one static external lookup (tournament geography) unlocks travel context. These are table-stakes features in every public tennis-prediction reference; their absence in v1 is what bounds the gap, not the model class.

**Deliverables:**
- v2 FeatureVector — 28 v1 fields + **14 new fields** = **42 total**:
  - Handedness match-up (2): `hand_p1`, `hand_p2` (categorical R / L / A / U).
  - Age (4): `age_p1`, `age_p2`, `age_vs_peak_p1`, `age_vs_peak_p2` (peak ATP=26.0, WTA=24.0).
  - Height (3): `height_p1`, `height_p2`, `height_diff_cm`.
  - Travel jet-lag (4): `tz_shift_hours_{p1,p2}`, `days_since_last_match_{p1,p2}`.
  - Tournament altitude (1): `altitude_meters`.
- New static lookup `data/static/tournament_locations.csv` covering tour-level main-draw tournaments active 2018+ (~200 distinct rows). Loaded into a `tournament_locations` DuckDB table by `refresh_data.py`.
- New state object `TravelState` (mirrors `EloState`): persisted snapshot, rebuilt forward at inference if the requested `as_of_date` precedes the snapshot.
- Schema migration: `training_features.schema_version` 1 → 2; table drop-and-recreate, full feature rebuild (~5 min on the populated DB).
- Extended leakage tests: tampered-future-row fixtures assert no travel/age/etc. feature for an earlier date changes when a later row is altered.
- Smoke test extended to populate `players` + `tournament_locations` + the 14 new columns.
- Re-trained 4 production artifacts with the v2 feature set; old Phase 4 artifacts kept until the validation gate clears.

**Exit:**
- 4 fresh artifacts on disk (`models/<tour>/<{elo,lightgbm}>/latest/`) with feature count 42 in `metadata.json`.
- Tournament-locations CSV JOIN covers ≥95% of `training_features` rows with `match_date >= 2018-01-01`.
- All Phase 3 leakage tests + new Phase 4.1 leakage tests pass.
- Round-trip serialization tests pass on the new artifacts.
- LightGBM post-calibration Brier improves on at least one tour by ≥0.001 over Phase 4 on the most recent 5 walk-forward folds. The per-tour delta is documented in `docs/tutorials/phase_4_1_results.md` either way — if neither tour improves, the phase ends with a roll-back and a documented null result.

**Out of scope (decided up front):**
- Court Pace Index (no public source; serve/return rolling features already capture surface-speed signal).
- Per-player altitude adaptation (literature support too thin to justify the feature count).
- Playing-style tags (defender/attacker) — derivable from existing stats, no new signal.

---

## Phase 5 — LLM agent

**Entry:** Phase 4.1 exit criteria met. Full design document: `docs/tutorials/phase_5_notes.md`.

**Architectural contract** (locked in `CLAUDE.md` before implementation; see sections "Anthropic SDK", "Web search", "Structured output discipline", "LLM agent failure modes", "Testing the LLM agent", "Budget discipline"):
- Direct `anthropic` SDK only — no LangChain / LiteLLM / OpenRouter / Managed Agents.
- Sonnet 4.6 default (`ANTHROPIC_MODEL`); not Opus (cost), not Haiku (synthesis).
- Tool-use pattern for structured output: single `submit_analysis` tool with `additionalProperties: false`; hybrid `tool_choice` (auto → forced on final iteration).
- One `cache_control` marker on the last tool definition; system prompt + tool defs are byte-stable.
- Web search: native `web_search`, no `allowed_domains`, small `blocked_domains` (betting sites), preferred sources noted in system prompt (ESPN / BBC / tennis.com / tennis365.com).
- `AgentBudget` with four hard limits per call (6 tool iterations, 30k tokens, 120 s wall clock, 3 web searches). Org-level $20/month cap in Anthropic console as the final wall.
- New hard rule #10: `get_model_prediction` is mandatory; agent never invoked without it.

**Deliverables:**
- `LLMClient` abstract base + `AnthropicLLMClient` implementation in `src/tennis_predictor/llm/client.py`. Prompt caching, byte-stability test, every call logged to `llm_traces`.
- Tools wired up with Pydantic input/output schemas:
  - `get_model_prediction` — the **only** source of the win probability shown to the user.
  - `get_player_stats`, `get_head_to_head`, `get_recent_form`, `get_player_ranking` — DuckDB-backed.
  - `web_search` — Anthropic's native, with `max_uses=3`, `blocked_domains` list, 14-day recency enforced via system prompt.
  - `submit_analysis` — structured-output collector mirroring `AgentResponse`.
- `TennisAgent.predict(match_context) -> AgentResponse` orchestrator with `AgentBudget` enforcement and 120 s `asyncio.timeout` wrap.
- `AgentResponse` Pydantic model (no LLM-emitted probability allowed; schema rejects any `probability`-like field via `additionalProperties: false`).
- `llm_traces` schema migration: add `web_search_count INTEGER` + `estimated_cost_usd DOUBLE` columns. `ALTER TABLE ADD COLUMN` (DuckDB-native, preserves existing rows).
- Three-tier test suite: unit tests (mocked Anthropic, run in CI), recorded-fixture e2e tests in `tests/fixtures/llm/` (run in CI), live-API tests under `@pytest.mark.llm_live` (run locally only).
- CLI: `uv run python scripts/predict_match.py --match-id <id>` produces a valid `AgentResponse`, prints it, writes to `llm_traces`.

**Exit:**
- CLI runs end-to-end on a sample upcoming match, produces valid `AgentResponse` with `key_factors` / `caveats` reflecting real news (or "no recent news surfaced" — both acceptable).
- `llm_traces` row exists for that call with non-zero `cache_read_tokens` on the second invocation within 5 minutes.
- All quality gates green; live-API smoke test passes locally.

---

## Phase 6 — Streamlit app

**Entry:** phase 5 exit criteria met.

**Deliverables:**
- **Home page — upcoming matches.** Sourced from `scheduled_matches`. Grouped by tournament, sorted by scheduled start. Each row links to its prediction page. This is the primary entry point — most users arrive wanting "predict tonight's matches", not "type in two player names."
- **Prediction page.** Shows: model probability (with the LLM's `confidence_band` as a qualitative tag), `key_factors` and `narrative`, `caveats`, news links surfaced by `search_tennis_news`, and a freshness indicator from `ingestion_runs`.
- **Custom prediction page (secondary).** Manual player + tournament + surface + date entry, for matches not in `scheduled_matches` or for "what-if" questions.
- **Dashboard page.** Per-model calibration plots (model vs market overlay), headline metrics over walk-forward folds, recent `llm_traces` browser. This is the "trust" tab — a curious user opens it to see why they should trust the number on the prediction page.
- Sensible empty / error states. Stale-data warning shown when `ingestion_runs` reports the last successful hot refresh is over 24h old.

**Exit:**
- `uv run streamlit run src/tennis_predictor/app/main.py` works end to end.
- Manual smoke test of the golden path (open home → pick a fixture → see prediction + news) and at least two edge cases: a player with no recent matches; hot API marked stale.

---

## Phase 7 — Deployment

**Entry:** phase 6 exit criteria met.

**Deliverables:**
- Dockerfile producing an image that runs the Streamlit app, with the DuckDB file mounted from a volume.
- Fly.io or Railway deployment configuration committed (provider chosen by start of this phase).
- Daily hot refresh runs on a scheduler (Fly cron / Railway cron / GitHub Actions) — not manually.
- **Cost discipline.** Anthropic API calls are rate-limited per session (or per IP if anonymous). Daily Anthropic-spend cap configured; on overrun the app shows a clean "service unavailable, daily limit reached" state instead of failing requests one by one. The cap is documented in README.
- README polished: setup, run, deploy, public URL, links to docs.
- `.env.example` exhaustively updated.

**Exit:**
- App is reachable at a public URL.
- A teardown procedure is documented (so the project doesn't quietly burn budget after attention shifts).
- First-load latency on a cold container: a real prediction for an upcoming match renders within a few seconds.
