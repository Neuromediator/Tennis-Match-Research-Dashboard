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
- tennis-data.co.uk archive prior to ~2013 is served as `.xls` (legacy binary). The downloader detects and skips these years cleanly. Adding `xlrd` to read them is a small future task.
- ~840 ATP and ~1945 WTA players share a `full_name` in Sackmann's roster (different IDs). `find_namesakes()` surfaces them; phase 2/3 needs a per-source disambiguator before merging.

---

## Phase 2 — Hot data layer

**Entry:** phase 1 exit criteria met.

**Deliverables:**

- **Hot API provider: matchstat Tennis API.** Published as "Tennis API - ATP WTA ITF" on RapidAPI; reference docs at `tennisapidoc.matchstat.com`. Free tier: 500 requests / month, hard cap. Documented fallback if matchstat is retired or its free tier shrinks: any RapidAPI tennis API exposing fixtures-by-date, completed matches with scores, and ATP/WTA rankings on a free tier ≥ 1500 req/month.

- **Endpoint-to-responsibility map** (all under `https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2`):

  | Endpoint | Purpose | Frequency |
  |---|---|---|
  | `/{tour}/tournament/calendar/{year}` | Season inventory — drives the active-tournament filter via the `tier` field. | Weekly per tour, cached. |
  | `/{tour}/fixtures/{date}?include=tournament.court,tournament.rank,round&filter=PlayerGroup:singles` | Upcoming fixtures → `scheduled_matches`. `include` brings surface (`court.name`) and round name; `filter` drops doubles. | Daily per tour. |
  | `/{tour}/tournament/results/{seasonid}` | Completed matches + scores → `matches`. Pre-match odds (`odd1`/`odd2`) → `market_implied_probabilities`. | Daily per active tour-level event. |
  | `/{tour}/ranking/singles?pageSize=100` | Inter-week ranking overlay between weekly Sackmann snapshots. | Daily per tour. |

  Tour-level whitelist on `tier`: `{"Grand Slam", "ATP 1000", "ATP 500", "ATP 250", "WTA 1000", "WTA 500", "WTA 250", "Finals"}`. Everything else (Challengers, ITF tiers like `"M15"`, `"M25"`) is dropped.

- **Budget.** Typical day: ~7–10 calls. Slam weeks: ~12–15. Bootstrap (back-fill the last ~30 days of completed results across ~20–30 recently-finished tour-level events): ~50 calls, one-off. Steady-state monthly: ~250–350 against the 500/month cap. Refresh script must avoid naive retry loops; per-call counts logged to `ingestion_runs`.

- **Upcoming fixtures lookahead is naturally short.** Tennis draws fix round 1 at the start of the week (Sun/Mon), and each subsequent round becomes known only once the previous round completes. The `scheduled_matches` table reflects whatever the API knows at refresh time: full round 1 right after a draw, rolling down to "today plus part of tomorrow" by mid-tournament. This is what the product lets users predict against; without it the app cannot surface "tonight's matches."

- **Cross-source key.** Fixture `id` from `/fixtures/...` (small integer) and match `id` from `/tournament/results/...` (8-digit string) are **not** the same identifier. The link between a `scheduled_matches` row and the `matches` row produced when the match completes is the composite `(tournamentId, player1Id, player2Id, roundId)`, not a shared external id.

- **Pre-match odds bonus.** `odd1`/`odd2` from `tournament/results` feed `market_implied_probabilities` for current events after overround normalization — the same pipeline as the cold tennis-data.co.uk loader. This removes the need to scrape tennis-data.co.uk for the trailing market window.

- **New tables.** `scheduled_matches` (upcoming fixtures) and `ingestion_runs` (one row per refresh: run id, source, started_at, finished_at, rows added/skipped/failed, error if any) — the UI reads the freshness signal from `ingestion_runs`.

- **Source-specific player mapping** integrated into `player_aliases` (`source='matchstat'`). Same manual-review checkpoint as cold data — no silent ambiguous merges.

- **Error budget.** When the hot API is unreachable or partial, the app degrades gracefully: predictions remain available against the last cached fixtures with a visible "data is N hours stale" warning, sourced from `ingestion_runs`.

- **Tests.** Hot rows do not duplicate cold rows; daily refresh is idempotent; a fixture row promotes to a `matches` row (with result) when the match completes (composite-key match); ranking overlay never reads from the future; the tier-whitelist filter drops Challenger/ITF rows.

**Exit:**
- `uv run python scripts/refresh_hot.py` runs end-to-end and updates DuckDB.
- `scheduled_matches` contains every fixture the hot API knows about at the moment the script runs (in practice: full round 1 right after a draw, or today/tomorrow's matches mid-tournament).
- `ingestion_runs` records the run with row counts (added / skipped / failed).
- A simulated API outage (mocked failure) leaves the app usable on the last cache, surfacing the staleness warning.

---

## Phase 3 — Feature engineering

**Entry:** phase 2 exit criteria met.

**Deliverables:**
- `build_training_features()` producing `training_features` rows for every eligible match.
- `compute_features(...) -> FeatureVector` for inference.
- Pydantic `FeatureVector` schema.
- Surface-Elo ratings pipeline with persistent `elo_state`.
- Rolling, H2H, fatigue, and ranking features.
- Leakage tests: tampered-future-rows fixtures, asserting no past feature value moves.

**Exit:**
- All leakage tests pass in CI.
- `training_features` has one row per eligible match, with no nulls in required feature columns.
- `compute_features` returns identical values for the same `(player, opponent, surface, as_of_date)` whether reached via training replay or inference path.

---

## Phase 4 — Modeling

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

**Exit:**
- Four fresh model artifacts exist in `models/` (Elo baseline + LightGBM per tour).
- The LightGBM Brier score beats the Elo baseline on each tour. If it does not, ship the baseline and document the result honestly — the product surfaces the better-calibrated number, not the more complex one.
- Market-benchmark plot is visible in every report.

---

## Phase 5 — LLM agent

**Entry:** phase 4 exit criteria met.

**Deliverables:**
- `LLMClient` abstract base with Anthropic implementation; prompt caching enabled (system prompt and tool definitions as cacheable blocks).
- Tools wired up with Pydantic input/output schemas:
  - `get_model_prediction` — the **only** source of the win probability shown to the user.
  - `get_player_stats`, `get_head_to_head`, `get_recent_form`, `get_player_ranking` — DuckDB-backed.
  - `search_tennis_news` — Claude's native `web_search`, first-class tool. For each prediction the agent is expected to query for **withdrawals, injuries, and personal events affecting either player in the last ~14 days**. This is the user-facing differentiator of the product: news the user might otherwise miss.
- `AgentResponse` Pydantic model (no LLM-emitted probability allowed).
- `llm_traces` table populated by every call.
- Tests: tool schemas validate, structured output schema rejects banned fields, end-to-end agent call against a recorded fixture, second invocation shows non-zero cache stats.

**Exit:**
- A single CLI command runs the agent against a sample upcoming match and produces a valid `AgentResponse` whose `key_factors` and `caveats` reflect news the tool actually surfaced.
- `llm_traces` row exists for that call with non-zero cache stats on the second invocation.

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
