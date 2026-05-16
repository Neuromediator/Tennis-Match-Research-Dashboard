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
- Hot API source chosen and documented (recommended starting point: api-tennis.com).
- Daily refresh script appending the last ~30 days of completed matches to `matches`.
- Source-specific player mapping integrated into `player_aliases`.
- Tests: hot rows do not duplicate cold rows; daily refresh is idempotent.

**Exit:**
- A "yesterday's matches" command runs and updates DuckDB.
- Refresh script logs row counts added/skipped.
- Coverage stays unbroken when switching the active provider (documented fallback).

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
- Six trained models (ATP/WTA × {surface-Elo baseline, logistic, LightGBM}).
- Walk-forward validation harness with per-fold metrics.
- Calibration applied per the isotonic/Platt decision rule.
- Per-model artifact directory with `model.joblib`, `metadata.json`, `report.md`, `calibration_plot.png`.
- Market-benchmark calibration overlay in every report.
- Round-trip serialization test.

**Exit:**
- Six fresh model artifacts exist in `models/`.
- The Brier score of the best ATP model and the best WTA model is reported and recorded.
- Market-benchmark plot is visible in every report.

---

## Phase 5 — LLM agent

**Entry:** phase 4 exit criteria met.

**Deliverables:**
- `LLMClient` abstract base with Anthropic implementation; prompt caching enabled.
- Six tools wired up with Pydantic input/output schemas.
- `AgentResponse` Pydantic model (no LLM-emitted probability allowed).
- `llm_traces` table populated by every call.
- Tests: tool schemas validate, structured output schema rejects banned fields, end-to-end agent call against a recorded fixture.

**Exit:**
- A single CLI command runs the agent against a sample query and produces a valid `AgentResponse`.
- `llm_traces` row exists for that call with non-zero cache stats on the second invocation.

---

## Phase 6 — Streamlit app

**Entry:** phase 5 exit criteria met.

**Deliverables:**
- Prediction page: enter two players + tournament + surface + date → display agent response.
- Dashboard page: per-model calibration plots, headline metrics over time, feature importance (where applicable), recent `llm_traces` browser.
- Sensible empty / error states.

**Exit:**
- `uv run streamlit run src/tennis_predictor/app/main.py` works end to end.
- Manual smoke test of golden path + at least two edge cases (missing player, no recent matches).

---

## Phase 7 — Deployment

**Entry:** phase 6 exit criteria met.

**Deliverables:**
- Dockerfile producing an image that runs the Streamlit app.
- Fly.io or Railway deployment configuration committed.
- README polished: setup, run, deploy, links to docs.
- `.env.example` exhaustively updated.

**Exit:**
- App is reachable at a public URL.
- A teardown procedure is documented.
