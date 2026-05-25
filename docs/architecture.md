# Architecture

A single-process Python application with five logical layers. No microservices. One DuckDB file is the system of record.

```
┌────────────────────────────────────────────────────────────────────┐
│ Interface layer (Streamlit) — Phase 6.2: research dashboard        │
│   - Home (upcoming matches)                                        │
│   - Match dashboard: comparison row (market / model / surface-Elo) │
│     + "why model differs" panel when gap > 10pp + H2H detail +     │
│     two-column recent form + LLM-discovered news                   │
│   - Custom match (3-input form with player autocomplete)           │
│   - Model evaluation (calibration plots, llm_traces, costs,        │
│     last-20 model-vs-market gaps scoreboard)                       │
└────────────────────┬───────────────────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────────────────┐
│ LLM agent layer (Phase 6.1 scope)                                  │
│   - LLMClient (Anthropic) with prompt caching + llm_traces logging │
│   - LLM tools: get_model_prediction (mandatory) /                  │
│     get_head_to_head (detailed) / get_surface_elo / web_search /   │
│     submit_analysis                                                │
│   - View-layer helpers (NOT LLM tools): fetch_recent_n_matches,    │
│     fetch_pre_match_odds (Phase 6.2)                               │
│   - Structured output: AgentResponse = news_items + status only.   │
│     No narrative, no caveats, no confidence_band — every shown     │
│     fact carries source+date as adjacent UI metadata.              │
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
│   training_features, llm_traces, ingestion_runs,                  │
│   matchstat_player_recent_cache, matchstat_h2h_cache,             │
│   matchstat_quota, pre_match_odds (Phase 6.2)                     │
└───────────────────────────────────────────────────────────────────┘
```

## Data sources

- **Cold:** Jeff Sackmann's `tennis_atp` and `tennis_wta` repos as git submodules. Used for both prediction targets (tour-level singles) and feature computation (Challengers/Futures contribute to ratings but not to training labels).
- **Hot:** matchstat Tennis API ("Tennis API - ATP WTA ITF" on RapidAPI), free tier 500 req/month. Three responsibilities: (a) currently-known upcoming fixtures into `scheduled_matches` — in tennis this is whatever the draw has surfaced so far (full R1 right after a draw, then today/tomorrow's matches as the bracket resolves), this is what the product lets users predict against, (b) inter-week ranking overlay between weekly Sackmann snapshots, (c) **Phase 6.1: on-demand per-player past-matches + H2H** via `/atp/player/past-matches/{id}` and `/atp/h2h/matches/{a}/{b}` (the correct H2H endpoint per `tennisapidoc.matchstat.com/h2h` — Phase 6.1 wrongly used `/atp/fixtures/h2h/{a}/{b}` which returns upcoming fixtures, fix scheduled in Phase 6.2 Step 4.1), used to show the user fresh "last 8 matches" + detailed H2H on the Prediction page. Cached 24h in `matchstat_player_recent_cache` / `matchstat_h2h_cache`; quota tracked in `matchstat_quota` with a 480/500 graceful-fallback threshold. Completed matches in bulk (full-tournament `tournament/results`) still do NOT come from matchstat — its `calendar/{year}` is forward-only and silently drops currently-active tournaments. Sackmann (cold) remains the source of truth for the training matches table; the on-demand path only enriches the per-prediction view layer. Daily refresh logs to `ingestion_runs`.
- **Live market odds (Phase 6.2):** The Odds API (`the-odds-api.com`), free tier 500 credits/month. Per-tournament tennis sport keys (`tennis_atp_french_open`, `tennis_wta_madrid`, etc.) discovered daily via `GET /v4/sports/?all=false`, then odds fetched per active key with `regions=eu&markets=h2h&oddsFormat=decimal`. Persisted to `pre_match_odds` table with both median-across-books and Pinnacle-specific columns. Daily batch via `scripts/refresh_pre_match_odds.py` (~120-180 credits/month) plus lazy 24h refresh on Prediction-page load (~30 credits/month). Not used as a training feature (hard rule #3); rendered in the dashboard's comparison row alongside the model's probability.
- **Historical benchmark:** tennis-data.co.uk archives provide closing-price implied probabilities. Loaded into `market_implied_probabilities`. **Not a feature.** Used for Phase 4 calibration reports (model Brier vs market Brier on historical walk-forward folds).

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
