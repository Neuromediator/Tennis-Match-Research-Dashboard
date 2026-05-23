# Architecture

A single-process Python application with five logical layers. No microservices. One DuckDB file is the system of record.

```
┌────────────────────────────────────────────────────────────────────┐
│ Interface layer (Streamlit)                                        │
│   - Prediction page (LLM-mediated)                                 │
│   - Evaluation dashboard (calibration plots, llm_traces browser)   │
└────────────────────┬───────────────────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────────────────┐
│ LLM agent layer                                                    │
│   - LLMClient (Anthropic) with prompt caching + llm_traces logging │
│   - Tools: get_player_stats / get_head_to_head / get_recent_form / │
│     get_model_prediction / search_tennis_news / get_player_ranking │
│   - Structured output: AgentResponse (no LLM-emitted probability)  │
└──────────┬─────────────────────────────────┬───────────────────────┘
           │                                 │
┌──────────▼──────────────┐    ┌─────────────▼──────────────────────┐
│ Modeling layer          │    │ Feature engineering layer          │
│   - Four models         │    │   - build_training_features()      │
│     (2 types × 2 tours: │    │     (chronological replay + state) │
│     Elo / LightGBM)     │    │   - compute_features(...)          │
│   - Walk-forward CV     │    │     (inference, returns            │
│   - Isotonic / Platt    │    │     FeatureVector v2 — 39 fields)  │
│   - Market benchmark    │    │                                    │
└──────────┬──────────────┘    └─────────────┬──────────────────────┘
           │                                 │
┌──────────▼─────────────────────────────────▼──────────────────────┐
│ Data layer (DuckDB, single file)                                  │
│   matches, scheduled_matches, players, rankings, player_aliases,  │
│   market_implied_probabilities, elo_state, last_match_state,      │
│   training_features, llm_traces, ingestion_runs                   │
└───────────────────────────────────────────────────────────────────┘
```

## Data sources

- **Cold:** Jeff Sackmann's `tennis_atp` and `tennis_wta` repos as git submodules. Used for both prediction targets (tour-level singles) and feature computation (Challengers/Futures contribute to ratings but not to training labels).
- **Hot:** matchstat Tennis API ("Tennis API - ATP WTA ITF" on RapidAPI), free tier 500 req/month. Two responsibilities: (a) currently-known upcoming fixtures into `scheduled_matches` — in tennis this is whatever the draw has surfaced so far (full R1 right after a draw, then today/tomorrow's matches as the bracket resolves), this is what the product lets users predict against, (b) inter-week ranking overlay between weekly Sackmann snapshots. Completed matches do NOT come from matchstat — its `calendar/{year}` is forward-only and silently drops currently-active tournaments, making the calendar-driven results path unreliable. Sackmann (cold) is the source of truth for finished matches; the trade-off is a 1–7 day lag for newly-finished events. Daily refresh logs to `ingestion_runs` so the UI can surface freshness and so we stay inside the monthly quota.
- **Benchmark:** tennis-data.co.uk archives provide closing-price implied probabilities. Loaded into `market_implied_probabilities`. **Not a feature.**

## Cross-cutting concerns

- **Provenance.** Every match row carries `source` and `match_external_id`.
- **Player ID reconciliation.** A single canonical `player_id` per player. All non-canonical names map via `player_aliases` (with `source` distinguishing how the alias got there: `sackmann` seeded, `manual_review` approved by a human, hot-API source later). Ambiguous fuzzy matches go to `data/processed/aliases_review.csv` for manual review; `scripts/apply_aliases_review.py` promotes approved decisions back to `player_aliases`.
- **Audit trail.** Two append-only CSVs under `data/processed/` survive every refresh: `aliases_review.csv` (low-confidence resolutions) and `unmatched_market_rows.csv` (resolved names that didn't join a match row). The first feeds the manual-review loop; the second is a debugging signal explored in `notebooks/explore_unmatched.ipynb`.
- **Point-in-time correctness.** Enforced by tests, not by convention. The feature layer is the only sanctioned source of feature values.
- **Observability.** Every LLM call is logged to `llm_traces` with token counts, cache stats, latency, and tool-call sequence.
- **Configuration.** All paths and env vars flow through `src/tennis_predictor/config.py`. `DATA_DIR` resolves both local dev and containerized deployment.

## Deployment shape

- Local dev: `uv run streamlit run ...`. DuckDB file at `data/processed/tennis.duckdb`.
- Containerized (phase 7): single Dockerfile. DuckDB file mounted from a volume. The container is stateless except for that volume.
- Target: Fly.io (locked in Phase 7 design — see `docs/phases.md`). Public URL, no auth gate; budget protection via per-IP rate limit + hard daily $ cap.

## What is intentionally absent

- No microservices, no message queue, no Redis.
- No model server (predictions run in-process).
- No user accounts / sessions / multi-user state.
- No realtime data path; all ingestion is batch.
- No managed Postgres; DuckDB is sufficient at this scale.
