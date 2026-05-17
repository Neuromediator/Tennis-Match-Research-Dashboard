# Data layer

This directory holds raw data sources and processed analytical storage.

## Layout

- `raw/` — git-ignored. Holds:
  - Sackmann submodules (`tennis_atp/`, `tennis_wta/`) — pinned via `.gitmodules`.
  - Hot API dumps from phase 2 (`api-tennis/*.json` or similar).
  - Tennis-data.co.uk historical odds archives (XLSX/CSV) used as a *calibration benchmark only*.
- `processed/` — git-ignored. Holds the DuckDB file (`tennis.duckdb`) and any intermediate Parquet artifacts.

## DuckDB tables (created in phase 1; some populated later)

| Table | Phase | Purpose |
|---|---|---|
| `matches` | 1 | Canonical completed-match rows from all sources. `source` + `match_external_id` carry provenance. |
| `scheduled_matches` | 2 | Upcoming fixtures known to the hot API at refresh time. Drives the home page. Rows are dropped (or moved into `matches` with a result) once the match completes. |
| `players` | 1 | Canonical player records. |
| `rankings` | 1 | Weekly rankings, ATP and WTA. The hot ranking overlay (phase 2) sits in front of this at query time without overwriting cold snapshots. |
| `player_aliases` | 1 | Maps source-specific names/IDs to a canonical `player_id`. |
| `market_implied_probabilities` | 1 (schema), 1–2 (load) | Historical closing-price implied probabilities. **Benchmark only — never a feature.** |
| `elo_state` | 3 | Per-surface Elo snapshots at given dates. |
| `training_features` | 3 | Pre-match feature vectors (one row per match) produced by `build_training_features`. |
| `ingestion_runs` | 2 | One row per refresh run: source, started/finished_at, rows added/skipped/failed, error if any. The UI reads this for the freshness signal. |
| `llm_traces` | 1 (schema), 5 (load) | Audit log of every LLM call. |

## Hard rules

- Raw files are never modified in place. Ingestion is idempotent.
- All match rows are tagged with `source` and `match_external_id`. Cold and hot rows live in the same `matches` table.
- Player IDs are reconciled exclusively via `player_aliases`. Ambiguous fuzzy matches go to `aliases_review.csv` for manual review.
- `market_implied_probabilities` is loaded but is **forbidden as a model feature**. It is used only to compute the market-vs-model calibration plot.
