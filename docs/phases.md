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

## Phase 4.1 — Feature expansion (complete)

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

## Phase 5 — LLM agent  ✅ complete

**Entry:** Phase 4.1 exit criteria met. Full design document: `docs/tutorials/phase_5_notes.md`. Results: `docs/tutorials/phase_5_results.md`.

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

**Exit (all green):**
- ✅ CLI runs end-to-end on a sample upcoming match, produces valid `AgentResponse` with `key_factors` / `caveats` reflecting real news (or "no recent news surfaced" — both acceptable). Live smoke surfaced an Alcaraz wrist-injury Roland Garros withdrawal that the trained model could not see.
- ✅ `llm_traces` row exists for that call with non-zero `cache_read_tokens` on the second invocation within 5 minutes (observed `cache_read_tokens = 4 719` on the second prediction).
- ✅ All quality gates green; live-API smoke test passes locally.

**Headline numbers (live smoke 2026-05-22):**
- 365 default tests + 2 live tests, all passing.
- End-to-end cost per prediction: **~$0.10** (vs the $0.03 pre-implementation estimate — the gap is web_search response payloads showing up as `cache_creation_input_tokens` on the next turn).
- Wall-clock per prediction: ~36-38 s for a 2-3 turn loop.
- Per-call `AgentBudget` accounting refined post-smoke: `tokens_used` now counts `tokens_in + tokens_out + cache_creation` (cost-weighted subset), excluding the heavily-discounted `cache_read`. Numeric limits unchanged.
- Org-level $20/month cap remains comfortably above expected 5-10/day personal-use load.

---

## Phase 5.1 — Search provider swap  ✅ complete

**Entry:** Phase 5 complete. Full design document: `docs/tutorials/phase_5_1_notes.md`. Results: `docs/tutorials/phase_5_1_results.md`.

**Motivation.** Phase 5 closed with observed cost of ~$0.10 / prediction, ~3x the pre-implementation estimate. The driver is Anthropic native `web_search` returning full page bodies (~14k tokens of `cache_creation` per call) most of which the agent does not read past the first paragraph. An A/B test (`scripts/compare_search_providers.py`) on 2026-05-23 showed Tavily snippet-only search is 9.5x cheaper / 3.5x faster, surfaces the same niche journalism (Kasatkina, Tatjana Maria) at parity, and naturally returns the diverse non-official sources CLAUDE.md "Web search" prefers.

**Deliverables:**
- New `src/tennis_predictor/llm/tools/web_search.py` — Tavily wrapper + tool definition. Replaces Anthropic native `web_search_20250305`.
- New `src/tennis_predictor/llm/tools/fetch_url.py` — Tavily Extract for the ~5% case where snippets truncate a key detail. `max_fetch_urls = 2` per call.
- Pydantic schemas: `WebSearchInput / Hit / Output`, `FetchUrlInput / Output`, `TavilyError`.
- `llm_traces` migration: new `fetch_url_count INTEGER` column (idempotent ALTER).
- `LLMClient.acall` gained `extra_tool_cost_usd / extra_web_search_count / extra_fetch_url_count` kwargs so Tavily activity between LLM calls is attributed to the next trace row.
- `BudgetTracker` gained `reserve_web_search / reserve_fetch_url` (atomic, race-free against parallel tool dispatches in one turn).
- Rolling cache marker on the last `tool_result` block when more iterations are expected (saves ~$0.02/predict).
- 22 new Tier-1 unit tests, 1 new Tier-2 e2e fixture, 2 Tier-3 live tests confirming cost ceiling.

**Exit (all green):**
- ✅ CLI runs end-to-end against a real upcoming match and produces a valid `AgentResponse` using Tavily-sourced context.
- ✅ Tier-3 live tests pass — `web_search_count > 0` via the client path, second-call cache hit observed.
- ✅ 387 default tests pass; quality gates clean.
- ⚠️ **Cost did NOT drop to projected $0.025**. Observed reality: ~$0.10-0.12/predict (similar to Phase 5). The 4x cost-saving projection over-modelled per-search savings while missing that Sonnet runs MORE iterations when each one is cheaper. See `phase_5_1_results.md` for the honest breakdown and Phase 5.2 plan.

**Real wins** (qualitative, not on the dollar line):
- **3x faster per-search latency** (1.2s vs 4.3s) — visible UX win for Phase 6.
- **Snippet visibility in `llm_traces`** — Phase 6 dashboard can show actual content the agent saw (Anthropic native stored `encrypted_content`).
- **Source diversity matches CLAUDE.md preferences** — Yahoo, BBC, Reddit, niche tennis journalism instead of Anthropic's tour-official bias.
- **fetch_url escape hatch** ready for Kasatkina-style interviews where the snippet truncates a quote.
- **Vendor neutrality** — search provider is now a single ~150-LOC file, swappable for Brave / Serper if Tavily quality regresses.

**Out of scope (intentional):**
- Twitter/X granularity. Neither Anthropic native nor Tavily solves "Djokovic missed morning training" — that requires separate X API integration ($200/month) deferred indefinitely.
- Tavily `search_depth: "advanced"`. Basic-tier ranking is adequate per the A/B test; advanced is a future fallback if quality regresses.
- Hitting the $0.04 cost target. Deferred to Phase 5.2 — pre-load DB context into the initial user message so the agent's iteration count drops from 4 to 2.

---

## Phase 6 — Streamlit app  ✅ complete (superseded by 6.1 for the LLM-analyst surface)

**Entry:** phase 5.1 exit criteria met. Pre-implementation contract:
`docs/tutorials/phase_6_notes.md`. Implementation results:
`docs/tutorials/phase_6_results.md`.

**Deliverables:**
- **Home page — upcoming matches.** Sourced from `scheduled_matches`. Grouped by tournament, sorted by scheduled start. Each row links to its prediction page. This is the primary entry point — most users arrive wanting "predict tonight's matches", not "type in two player names."
- **Prediction page.** Shows: model probability (with the LLM's `confidence_band` as a qualitative tag), `key_factors` and `narrative`, `caveats`, news links surfaced by Tavily `web_search` (and optionally `fetch_url` content for the ~5% deep-dive cases), and a freshness indicator from `ingestion_runs`.
- **Custom prediction page (secondary).** Manual player + tournament + surface + date entry, for matches not in `scheduled_matches` or for "what-if" questions.
- **Dashboard page.** Per-model calibration plots (model vs market overlay), headline metrics over walk-forward folds, recent `llm_traces` browser, and a **cost monitor** (today's $ spend / month-to-date / cache hit rate / breakdown of `web_search_count` vs `fetch_url_count`) sourced from `llm_traces`. This is the "trust" tab — a curious user opens it to see why they should trust the number on the prediction page AND that we aren't accidentally burning budget.
- Sensible empty / error states. Stale-data warning shown when `ingestion_runs` reports the last successful hot refresh is over 24h old.

**Implementation notes:**
- `TennisAgent.predict()` is async; Streamlit pages call it via `asyncio.run(...)` (or `nest_asyncio.apply()` if needed). The DuckDB connection is held in `st.session_state` so it survives Streamlit's per-interaction re-runs without repeatedly opening / closing the file.
- All four pages read from the same DuckDB file (Phase 1 cold + Phase 2 hot + Phase 5 `llm_traces`). No new tables.

**Exit:**
- `uv run streamlit run src/tennis_predictor/app/main.py` works end to end.
- Manual smoke test of the golden path (open home → pick a fixture → see prediction + news) and at least two edge cases: a player with no recent matches; hot API marked stale.
- Cost monitor on the Dashboard page reads non-zero data from `llm_traces` for the smoke-test predictions.

**Implementation status (2026-05-23):**
- App package shipped at `src/tennis_predictor/app/` — four pages
  (`home`, `prediction`, `custom`, `dashboard`), shared helpers
  (`db.py`, `context.py`, `widgets.py`), router in `main.py`.
- `scripts/predict_match.py` now imports context builders from
  `app/context.py` (single source of truth).
- 423 default tests pass (387 → 423, +36 covering context + widgets + TZ + dtype).
- Quality gates clean: ruff / format / pyright / pytest.
- Manual smoke surfaced LLM-analyst quality problems (year-mixing,
  stale "current form", hallucinated "defended title") that prompt
  rules could not eliminate — see Phase 6.1 for the structural fix.

---

## Phase 6.1 — LLM analyst v2 + UI refactor  ✅ implementation complete; superseded by 6.2 product re-scope

**Entry:** Phase 6 implementation complete; user feedback documented the LLM-analyst failure modes that motivate this phase. Full design document: `docs/tutorials/phase_6_1_notes.md`. Results: `docs/tutorials/phase_6_1_results.md` (skeleton).

**Motivation.** Phase 6 shipped an LLM that produced freeform `narrative` over a generic tool set. Live use showed three classes of failure that no prompt iteration fixed: (a) snippet verbs read as present-tense made articles from prior seasons appear current ("Ruud just won Geneva" — 2024 article), (b) DB recent-form was synthesised around even when `data_freshness_warning` was set (Safiullin shown 10-0 missing his RG-quali loss), (c) plausible-sounding prose like "defended his title" appeared without backing. The structural fix: take prose synthesis out of the LLM's hands; render H2H, surface-Elo, and last-8-matches deterministically from typed tool outputs; constrain the LLM to **only** dated, attributed news items with a category whitelist.

**Architectural contract** (locked in `docs/tutorials/phase_6_1_notes.md` before implementation):

- `AgentResponse` drops `narrative`, `confidence_band`, `caveats`, `key_factors`. New shape: `news_items: list[NewsItem] + news_lookup_status: Literal["ok","no_results","failed"]`.
- LLM-callable tool surface shrinks to: `get_model_prediction` (mandatory), `get_head_to_head` (now with per-match detail + odds + completion status), **new** `get_surface_elo` (one call returns both players + diff + baseline%), `web_search` (scoped 32-day window + category whitelist), `submit_analysis`.
- LLM-callable tools removed: `get_player_stats`, `get_recent_form`, `get_player_ranking`, `fetch_url`.
- Recent form (8 last matches per player) is rendered **directly by the view layer** — not via an LLM tool call.
- **matchstat second use case**: per-player past-matches + H2H endpoints added to `MatchstatClient`. On-demand with 24h DuckDB cache (`matchstat_player_recent_cache`, `matchstat_h2h_cache`) and quota tracking (`matchstat_quota`, raising `MatchstatBudgetExceeded` at 480/500). On exhaustion → graceful Sackmann fallback with visible banner.
- Category whitelist for `NewsItem`: `injury / withdrawal / illness / result / coach_change / personal`. `other` is a filtered-out sentinel.
- `MATCHSTAT_SOURCE_TZ` default changes `UTC → Europe/Moscow` (matchstat empirically sends Moscow-local time labelled as Z). One-shot migration: `scripts/clear_scheduled_matches.py` + `refresh_hot.py` re-run.
- Display format for match times: `"Sun, May 25 — 11:00 CEST (09:00 UTC)"`.

**Deliverables:**

- Schema migrations: `matchstat_player_recent_cache`, `matchstat_h2h_cache`, `matchstat_quota` tables.
- `MatchstatClient.player_past_matches(tour, player_id)` + `MatchstatClient.h2h(tour, p1_id, p2_id)`.
- `MatchstatLiveFetcher` (cache + quota + fallback) in `src/tennis_predictor/data/matchstat_live.py`.
- Rewritten `get_head_to_head` LLM tool returning detailed H2H matches with odds + completion status.
- New `get_surface_elo` LLM tool.
- New `fetch_recent_n_matches` helper for the view layer (matchstat-first, Sackmann fallback).
- `AgentResponse` rewritten per drop list. `submit_analysis` schema updated.
- `SYSTEM_PROMPT` rewritten with the new bounded scope, 32-day window, category whitelist.
- `TennisAgent` budget tightened: `max_tool_iterations=4`, `max_web_searches=2`, `fetch_url` retired from this path.
- Streamlit views: `prediction.py` redesigned per the layout in `phase_6_1_notes.md` (Step 8); `custom.py` reduced to 3 inputs with player autocomplete (Step 9, 10); Home page TZ labelling fixed (Step 11).
- New widgets in `app/widgets.py`: `player_autocomplete`, `recent_form_table_two_column`, `h2h_block`, `news_block`, `surface_elo_block`, `format_match_time_for_display`.
- Re-recorded LLM fixture tests (`tests/fixtures/llm/*.json`) for the new 2-3-call agent loop and new `AgentResponse` schema.

**Exit:**

- All quality gates green (ruff / format / pyright / pytest); test count documented in `phase_6_1_results.md`.
- Manual smoke: predicting the Ruud-Safiullin / Jacquet-Trungelliti fixtures that motivated this phase yields output where every fact has a visible source + date and no year-mixing.
- Free-tier matchstat budget verified: a fresh prediction burns ≤ 3 reqs; cache hit on retry within 24h burns 0; quota counter shows accurate `requests_used`.
- TZ correctness: Home page shows a match time that matches the corresponding `atptour.com` / `wtatennis.com` schedule entry minute-for-minute.

**Out of scope** (decided in `phase_6_1_notes.md` Part 3): live X/Twitter, per-match aces/BP stats, per-tournament TZ inference, confidence intervals on the model probability, any return of narrative synthesis under stricter rules.

---

## Phase 6.2 — Re-scope: predictor → tennis context dashboard  ✅ implementation complete (10-match acceptance pending)

**Entry:** Phase 6.1 close-out manual smoke (2026-05-25) exposed that the calibrated LightGBM probability is unreliable on real top matches:
- Cina-Opelka: model 37% on the actual favourite (Cina), market 64% — **inverted favourite**
- Sinner-Djokovic Clay: model 67% Sinner, market 93% — **26pp gap**
- Kasatkina vs X: model 55%, market 69% — 14pp gap

Average Brier 0.21 vs market 0.20 in Phase 4 masked these tail failures because Brier averages across thousands of matches and is dominated by easy predictions (top-10 vs WTA #80). The trained model **cannot decay surface-Elo for inactive players** (Djokovic's clay-Elo froze 2025-05-26), **cannot accelerate for hot streaks** (Sinner's 5 Masters in a row at K=32 only added ~+100), **cannot handle injury returns** (Opelka's 2022 Elo still anchors). These are structural Elo limits, not training data issues; they require feature-engineering work that is deferred.

**Strategy:** instead of trying to fix the model, change what the product IS. Phase 6.2 re-scopes from "predictor" to "tennis match research dashboard." Full design document: `docs/tutorials/phase_6_2_notes.md`. Results: `docs/tutorials/phase_6_2_results.md` (TBD).

**Product framing shift:**
- "A working tool that gives calibrated win probabilities" → "Tennis match research dashboard — see the model's view alongside market, surface-Elo, recent form, and LLM news context"
- The model probability stops being THE answer; it becomes one column in a comparison row.
- When model disagrees with market by > 10pp, the dashboard renders a structural explanation (stale surface-Elo, activity gap, returning veteran). Deterministic checks, not LLM-generated.

**Architectural contract** (locked in `docs/tutorials/phase_6_2_notes.md`):

- **New data source: The Odds API** (`the-odds-api.com`). Free 500 credits/month, email-only signup. Auto-fetched pre-match odds; user never enters odds manually. New `pre_match_odds` DuckDB table.
- **Pinnacle wrapper on RapidAPI considered and rejected** — no readable public docs, live probing surfaced only 2 of N endpoints. The Odds API exposes Pinnacle's price as one of its `bookmakers[]` for EU region, so the sharp-line preference is preserved indirectly (we extract Pinnacle into dedicated columns alongside the median across books).
- **Per-tournament sport keys** (`tennis_atp_french_open`, `tennis_wta_madrid`, etc.) — daily refresh discovers active keys via `GET /v4/sports/?all=false`, then iterates each.
- **Quota strategy:** daily batch + lazy 24h cache on Prediction-page load. NO per-match retry when match < 1h to start. Expected 150-210 credits/month against 500 cap.
- **Fallback chain** when The Odds API has no row: (1) Tavily search + regex decimal-odds extract, flagged `source='tavily'` in UI, (2) UI shows "Market: odds unavailable" explicitly.
- **matchstat H2H endpoint was wrong all of Phase 6.1.** Correct path per `tennisapidoc.matchstat.com/h2h` is `GET /atp/h2h/matches/{a}/{b}` (returns `data:` with `match_winner` + `result` + `odd1/2`), not `/atp/fixtures/h2h/{a}/{b}` (which returns upcoming fixtures and silently produced `data:[]` for Sinner-Djokovic). One-line URL fix + `result_type=='completed'` defensive filter eliminates the Svitolina-Bondar "score unknown" bug.
- **Navigation contract:** "← Back to home" button at top of every non-Home page; clicking calls `st.switch_page("views/home.py")` explicitly. Browser back continues to work but no longer re-triggers `agent.predict()` because `AgentResponse` is persisted in `st.session_state[f"prediction::{match_id}"]` on first run.
- **Comparison row UI:** four-line table per Prediction-page header — Pinnacle (market), our model, surface-Elo, diff-to-market column. Margin-stripped probabilities so all three rows sum to 1.
- **"Why model differs" panel** rendered when |model - market| > 10pp. Three deterministic causes checked in order: (1) activity asymmetry, (2) stale surface-Elo (> 180 days), (3) career-length asymmetry. No LLM-generated explanation — would re-introduce Phase 6 narrative bias.
- **Hard rule changes in CLAUDE.md:**
  - #3 broadened to cover `pre_match_odds` (display + calibration only, never training).
  - #10 relaxed for UI: deterministic blocks render even if agent fails; only the agent invocation itself is blocked on missing model.
  - #11 **NEW:** honest framing in copy (never claim "calibrated prediction" in UI text).
  - #12 **NEW:** acceptance test must include 10-match reality walkthrough (model vs market vs surface-Elo vs subjective rating ≥ 3.5/5; any unexplained inverted favourite = phase reopens).

**Deliverables:**

- Step 1: copy / framing rewrite — page titles, sidebar tagline, README, CLAUDE.md.
- Step 2: The Odds API integration — `src/tennis_predictor/data/odds_api.py`, `pre_match_odds` schema, `scripts/refresh_pre_match_odds.py`, sport-key discovery via `/v4/sports/?all=false`, name-reconciliation matcher (`home_team`/`away_team` vs `scheduled_matches.player1_name`/`player2_name`), Tavily fallback path.
- Step 3: comparison-row layout in `views/prediction.py` + "why model differs" deterministic panel.
- Step 4.1: matchstat H2H endpoint fix in `matchstat.py` (`/fixtures/h2h/` → `/h2h/matches/`) + `result_type` filter.
- Step 4.2: completed-in-upcoming filter in `views/home.py` (filter `scheduled_start_utc > now - 1h`).
- Step 4.3: `session_state[f"prediction::{match_id}"]` persistence + "← Back to home" buttons on all non-Home pages.
- Step 5: news block repositioning (move above recent form, closer to comparison row).
- Step 6: header rewrites + README rewrite for honest framing.
- Step 7: Dashboard scoreboard reading last 20 model-vs-market gaps from `pre_match_odds` × `llm_traces`.

**Exit (all required):**

- All quality gates green (ruff / format / pyright / pytest).
- **10-match reality test passed** — user walks through 10 current upcoming matches, records (model, market, surface-Elo, "why differs" triggered, 1-5 rating). Average rating ≥ 3.5 AND no unexplained inverted favourite.
- All three UI bugs fixed and verified in browser (matchstat endpoint, completed-in-upcoming, navigation persistence).
- README + LinkedIn / portfolio framing updated.
- Pinnacle quota status visible on Dashboard.

**Effort estimate:** 22-32 hours (~3-4 days).

**Out of scope** (decided in `phase_6_2_notes.md` Part 4): surface-Elo time decay (real fix for Sinner-Djokovic — separate phase, requires retrain + walk-forward), market-implied calibration target during training, live odds API integration (Pinnacle covers the need), mobile-responsive layout, public deployment.

**Pre-implementation handoff for new session:**
- All design decisions locked in `docs/tutorials/phase_6_2_notes.md`. **Read that first.**
- User has signed up at `the-odds-api.com` (email-only, no credit card). `THE_ODDS_API_KEY` is already in `.env` (do NOT read `.env` — that's a hard rule; just assume it's there).
- DataMenu Pinnacle Odds API on RapidAPI is **rejected** (no readable docs); design doc records this decision.
- Sample response shape captured in `phase_6_2_notes.md` Step 2 — parser can be written without further probing.
- Start order:
  - **Step 4.1 first** (matchstat H2H endpoint fix — 1 hour, instant win, unblocks user trust in H2H block).
  - **Step 4.2** (completed-in-upcoming filter — 30 min).
  - **Step 4.3** (session_state prediction persistence + Back-to-home buttons — 2-3 hours).
  - Then **Step 2** (The Odds API integration — 6-8 hours; can be done in parallel by a sub-agent if desired).
  - Then UI: Step 3, 5, 6, 7.
  - Final: Step Phase-3 acceptance test (10 real matches walkthrough). Phase does not close until this is green.

---

## Phase 6.2 follow-up — refresh refactor, UX fixes, prompt hardening  ✅ complete

**Entry:** Phase 6.2 implementation shipped; first-day live use on Roland Garros 2026 exposed a cluster of data-layer and UX issues. All addressed in a single follow-up pass without changing the product framing.

**What changed:**

1. **matchstat fixture refresh switched from per-date to per-tournament.**
   - Phase 6.2 implementation pulled `/{tour}/fixtures/{date}` for `today + lookahead` days. matchstat only returns matches whose Order of Play has been published for each specific day, so Slam R1 fixtures whose OoP gets announced later in the day were systematically missed.
   - New flow: `_discover_active_tournament_ids` probes `/fixtures/{today}` once per tour, filters to `tournament.rank.name in {"Grand Slam", "Main tour"}`, then `_refresh_fixtures_for_tournament` pulls `/fixtures/tournament/{id}` for each active tournament. One credit per tournament, returns every round matchstat knows about (R1 + R2 + R3 + ... + F).
   - Net cost: ~7-9 credits per refresh (down from ~13), with strictly more data.
   - `fixture_lookahead_days` parameter kept for the prune-window only; no longer drives the fetch loop.
   - Tests: new `FakeMatchstatClient.set_tournament_fixtures`; regression test `test_refresh_hot_per_tournament_pulls_rounds_not_in_today_payload`.

2. **Four prune passes — robustness fixes:**
   - `_prune_stale_scheduled_matches` (was already there) — drop rows matchstat stopped returning, scoped to the active date window.
   - `_prune_contradicted_round_fixtures` (new) — if player X is in both R1 and R2 at the same tournament, R1 is stale (Bonzi-Zverev case).
   - `_prune_duplicate_matchups` (new) — same matchup under two different `fixture_external_id`s (Sinner-Tabur 1294 + 1295) → keep most-recently-ingested.
   - `_prune_completed_slam_fixtures` (new) — for active Grand Slam tournaments, call `/tournament/results/{id}` and DELETE rows whose pair matchstat lists as completed (Duckworth-Diallo case). 1-2 extra API calls in a Slam week.
   - `ON CONFLICT DO UPDATE` clause in `insert_scheduled_matches` extended to refresh player names + external IDs (matchstat **re-uses fixture_external_id** for different matchups — Popyrin-Svajda took over 1271 from Griekspoor-Arnaldi).

3. **H2H endpoint URL fix + winner-from-score fallback.**
   - URL changed from `/{tour}/fixtures/h2h/{a}/{b}` (which returned upcoming fixtures only) to `/{tour}/h2h/matches/{a}/{b}` (completed history). Eliminates the Svitolina-Bondar "score unknown" pattern from Phase 6.1.
   - `RichMatch.result_type` is read and the consumer defensively keeps only `result_type in (None, "completed")` rows.
   - New helpers `infer_winner_from_score(result)` + `winner_index(match_winner, result)`: matchstat returns `matchWinner: null` on many older H2H rows; we now parse the score string to determine the winner (Khachanov-Trungelliti 2016 case where matchstat said `matchWinner: null` and our code defaulted to "player 2 won").

4. **Time display — DATE only on Home page upcoming list.**
   - matchstat returns `T12:00:00.000Z` for most Slam outside-court fixtures as a day-level placeholder, indistinguishable from a genuine 11:00 CEST start. Rendering all of them as "11:00 CEST" misled — 17 matches at the same hour is physically impossible across the available courts.
   - Home view now renders just `"Tue, May 26"` for every row. No `time_confirmed` heuristic, no "TBD" branch.
   - Lookback filter switched from time-based `now - 5h` to date-based `DATE(scheduled_start_utc) >= today_utc`. Today's matches stay visible all day regardless of when matchstat claims they start.
   - Match Dashboard (per-fixture page) still calls `format_match_time_for_display` when relevant.

5. **Tour filter + tournament grouping on Home.**
   - Horizontal radio `All / ATP / WTA` at top of upcoming list.
   - Two-level grouping: tour → tournament. ATP renders before WTA. Same tournament-name across tours (e.g., French Open) no longer shares a header.

6. **"Refresh fixtures" button on Home.**
   - Synchronous button that runs `refresh_hot(...)` on demand. Lets the user pull mid-day schedule updates without waiting for the next cron.

7. **`why_model_differs` panel — two new rules + generic fallback.**
   - Phase 6.2 original rules (activity gap / stale Elo / career asymmetry) didn't fire on cases like Baez-Burruchaga, where both players are active, both have fresh Elo, both are mid-tour, yet model + surface-Elo both pick the lower-ranked underdog against a rank-driven market.
   - Added: (a) `check_surface_elo_agrees_with_model` — when model AND surface-Elo both lean ≥ 5pp against the market; (b) `check_recent_form_gap` — surface-specific win-rate gap ≥ 30pp over last 60 days; (c) explicit "unexplained" fallback that fires when the page-level 10pp trigger asked for an explanation but none of the structural rules matched. The panel can no longer go silent on a > 10pp gap (Hard rule #12 structurally enforceable).

8. **LLM stale-news rules tightened.**
   - Tavily request now uses `topic="news"` + `days=32` (server-side recency filter).
   - System prompt rewritten with **generic** drop patterns (no longer name-specific examples that overfit): tournament-calendar-slot inference, "withdraws at <tournament>" + null date, multi-year retrospectives, name-collision sanity check (Arthur Fils ≠ Gael Monfils).
   - Date-unknown items now require explicit recency anchors in snippet/title; previously they were kept by default.

**Effort:** ~6 hours (single afternoon iteration). All tests green (484 pass, 5 skipped Phase 5 fixtures).

**Open follow-up** (not blocking phase close):
- `time_confirmed` column on `scheduled_matches` is now dead code; harmless to keep, can be dropped in a future cleanup.
- 10-match acceptance walkthrough using the new "why differs" rules — pending; the user has the data exported (`data/processed/2026-05-25T20-22_export.csv`).

---

## Phase 4.2 — Surface-specific recovery feature  🟡 design locked, implementation pending

**Entry:** Phase 6.2 + follow-up closed. Top-match accuracy still constrained by structural Elo limits (Djokovic clay frozen, Opelka post-injury, etc.). Phase 4.2 adds two FeatureVector fields (surface-specific days-since-last-match) so LightGBM can learn the staleness pattern empirically — cheaper and lower-risk than full surface-Elo time decay or Glicko-2 conversion.

**Design doc:** `docs/tutorials/phase_4_2_notes.md`. Read before starting.

**Strategy:** Variant A from the "what to do about inverted favourites" conversation. Two features only, no Elo restructuring. If LightGBM doesn't pick up the new signal (validation gate #2 below fails), this becomes a null result and we escalate to Variant B (Elo decay) or Variant C (Glicko-2).

**Deliverables:**

- New `LastMatchPerSurfaceState` class mirroring existing `LastMatchState` (Phase 4.1) but keyed on `(player_id, surface)`.
- New `last_match_per_surface_state` table.
- `training_features.schema_version` 2 → 3 with two new columns: `days_since_last_match_surface_p1`, `days_since_last_match_surface_p2` (both nullable, capped 365).
- `FeatureVector` v3 — 42 fields → 44 fields. Same Pydantic constraints as the existing recovery features.
- Wire state into `build_training_features` (replay) and `compute_features` (inference); equivalence test must still pass.
- Leakage tests: tampered-future-row + surface-swap.
- Re-train 4 production artifacts; v3 metadata.

**Validation gates** (Phase 4.2 doesn't close until all pass):

1. Average Brier on each tour: must not be worse than v2 by > 0.005.
2. Tail Brier on inactivity-asymmetric subset (one player surface-gap ≥ 180d, other < 30d): v3 must be ≥ 0.01 better than v2.
3. 10-match acceptance walkthrough — hard rule #12.
4. `phase_4_2_results.md` written, including SHAP/importance for new features.

**Out of scope:** surface-Elo time decay (Variant B), Glicko-2 (Variant C), market-distillation (rejected on hard rule #3).

**Effort estimate:** ~8-9 hours (~1.5 days).

**Pre-implementation handoff:** at the bottom of `docs/tutorials/phase_4_2_notes.md`.

---

## Phase 7 — Deployment (Fly.io, fully public)

**Entry:** Phase 6 exit criteria met.

**Provider locked:** Fly.io. Reasoning: first-class persistent volumes (DuckDB needs one — Railway / Streamlit Cloud don't offer this cleanly), scheduled-machines for the daily hot refresh, generous free tier that covers our load, single-region deployment is fine for a personal project.

**Auth strategy:** **none — fully public URL.** Implications:

- Anyone who finds the URL can spend our Anthropic budget.
- Rate limiting + hard daily cost cap is the ONLY abuse protection. Get this right.
- We accept the risk in exchange for shareable-link convenience. The Anthropic-console $20/month hard cap (Phase 5 deliverable) is the ultimate wall — no software bug can spend past it.

**Deliverables:**

1. **Dockerfile** — `python:3.12-slim` base, `uv` for dependency install, multi-stage so the runtime image excludes dev deps and source tarballs. Final image ~300-400 MB. Streamlit listens on `$PORT` (Fly.io convention).

2. **`fly.toml`** — single app, 1 small machine + auto-stop after idle (free-tier-friendly), mounted volume at `/data` for `tennis.duckdb` (3 GB), HTTP health check on `/_stcore/health` (Streamlit's built-in), auto-scaling `0 → 1` machine when traffic arrives.

3. **Secrets via `fly secrets set`** — `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `X_RAPIDAPI_KEY`. Never committed.

4. **Daily hot refresh — separate Fly.io scheduled machine** running `uv run python scripts/refresh_hot.py` once per day (~03:00 UTC, off-peak). Mounts the same `/data` volume as the app. Writes to `ingestion_runs`; the app reads freshness from there. Why scheduled-machine over GitHub Actions: tight coupling to the same volume, no need to set up SSH / a refresh-trigger HTTP endpoint in the app, ~$0.50/month extra at most.

5. **Rate limiting + hard cap** (the only abuse protection, see "Auth strategy" above):
   - **Per-IP:** 5 predictions / IP / 24h, stored in a small JSON file on the volume. Sufficient to block single-IP scrapers while letting your own daily use through.
   - **Hard daily $ cap:** before invoking `TennisAgent.predict()`, query `SUM(estimated_cost_usd) WHERE ts >= today UTC` from `llm_traces`. If >= `DAILY_LLM_USD_CAP` (default $2.00, configurable), return a clean "service has hit its daily limit, please come back tomorrow" page instead of letting the prediction proceed.
   - **Footer indicator** showing today's $ spend / monthly $ spend on every page (sourced from `llm_traces.estimated_cost_usd`).

6. **README polish** — setup (local), deploy (`fly launch` walkthrough), public URL, monthly cost breakdown, how to rotate keys, how to bump the daily cap.

7. **`.env.example` exhaustively updated** — every env var the app reads, with a short description and where to get the key.

8. **Teardown procedure documented** — one paragraph: `fly apps destroy tennis-predictor` + `fly volumes destroy <vol-id>` + revoke Anthropic workspace + revoke Tavily key. Important so the project doesn't quietly burn budget after attention shifts.

**Exit:**

- App reachable at a public URL (e.g. `tennis-predictor.fly.dev`).
- Daily refresh ran 3+ consecutive days without manual touch — confirmed via `SELECT * FROM ingestion_runs WHERE source='matchstat' ORDER BY started_at DESC LIMIT 3`.
- Rate-limit verified: a single IP hitting prediction #6 on the same day gets the rate-limit page; first 100 unique IPs are served normally.
- Hard cap verified: simulating $2+ spend in a dev run triggers the limit-reached page; recovery the next UTC day works.
- First-load latency on a cold container: <8s for the home page, <15s for a real prediction (Tavily + Anthropic roundtrips dominate).
- Monthly cost observation documented in README across the first 30 days post-launch.

**Cost expectation (monthly):**

- Fly.io app machine + worker machine + 3 GB volume: ~$0-3 (free tier covers most of it; volume is $0.15/GB/mo = $0.45).
- Anthropic at typical load (5-10 predicts/day × $0.10): $5-10. Hard daily cap blocks runaway scenarios.
- Tavily: $0 (free tier covers ~300 searches/month vs our typical ~150).
- **Total expected: ~$5-10/month.** Worst case if URL gets shared widely and hard caps hold: ~$30 (Anthropic dominates, capped by org wall at $20).
