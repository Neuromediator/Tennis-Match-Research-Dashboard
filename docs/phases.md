# Phased roadmap

Each phase has entry criteria (what must be true before starting), deliverables (what is produced), and exit criteria (what must be true before moving on). Phases are not started until the previous phase's exit criteria are green.

---

## Phase 0 — Bootstrap (this session)

**Entry:** empty project directory, agreed scope.

**Deliverables:**
- Repo skeleton and directory tree.
- `CLAUDE.md`, four skill files, three docs (this file, architecture, methodology), README.
- `pyproject.toml` with declared dependencies (not yet installed).
- `.gitignore`, `.env.example`, `.python-version`.
- CI workflow (`.github/workflows/ci.yml`), `.pre-commit-config.yaml`.

**Exit:** repo can be opened by any contributor and conventions are clear; nothing is yet implemented.

---

## Phase 1 — Cold data layer

**Entry:** phase 0 complete; `uv sync` succeeds.

**Deliverables:**
- Sackmann `tennis_atp` and `tennis_wta` added as git submodules under `data/raw/`.
- DuckDB schema created: `matches`, `players`, `rankings`, `player_aliases`, `market_implied_probabilities` (schema only), `llm_traces` (schema only), `elo_state` (schema only), `training_features` (schema only).
- Ingestion module loading Sackmann CSVs into `matches`, `players`, `rankings`.
- `tennis-data-co-uk` archive loader populating `market_implied_probabilities` (best-effort coverage).
- Player reconciliation using `rapidfuzz` with the manual-review checkpoint.
- Tests: data-loading correctness, idempotent ingestion, fuzzy reconciliation.

**Exit:**
- Reproducible: `uv run python scripts/refresh_data.py` rebuilds the DuckDB file from scratch.
- All phase-1 tests pass in CI.
- `aliases_review.csv` exists and has been reviewed at least once.
- Coverage report: number of matches ingested per tour per year; flagged gaps.

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
