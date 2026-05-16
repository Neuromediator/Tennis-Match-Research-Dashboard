---
name: data-ingestion
description: Use when adding a new data source, modifying the DuckDB schema, or working on player reconciliation. Establishes the contracts for source provenance, canonical IDs, and the manual-review checkpoint.
---

# Data ingestion

## Sources

| Source | Tier | Phase | Notes |
|---|---|---|---|
| Sackmann `tennis_atp` / `tennis_wta` | cold | 1 | Git submodules under `data/raw/`. Pinned commits. |
| Hot tennis API (api-tennis.com or alternative) | hot | 2 | Last ~30 days of completed matches. Daily refresh. |
| tennis-data.co.uk archives | benchmark | 1 | Historical closing-price implied probabilities. **Not a feature source.** |

## Canonical schema

The `matches` table holds rows from every source, distinguished by:

- `source` — one of `"sackmann"`, `"api-tennis"`, `"tennis-data-co-uk"`, ...
- `match_external_id` — the source's own identifier; `(source, match_external_id)` is unique.
- `tour` — `"ATP"` or `"WTA"`.
- `match_date`, `tournament_name`, `tournament_level`, `surface`, `round`, `best_of`.
- `winner_player_id`, `loser_player_id` — references to canonical `players.player_id`.
- Source-specific stats columns (aces, double-faults, etc.) live as nullable columns; not every source provides them.

Players, rankings, aliases, market probabilities, llm_traces, and feature/state tables are documented in `data/README.md`.

## Hard rules

1. **Idempotent ingestion.** Re-running an ingestion script must not produce duplicates. Enforce with `INSERT ... ON CONFLICT DO NOTHING` keyed on `(source, match_external_id)`.
2. **Raw files are immutable.** Never write back into `data/raw/`. Anything derived goes to `data/processed/` or the DuckDB file.
3. **Every row knows where it came from.** No anonymous rows in `matches` — `source` is NOT NULL.

## Player reconciliation

Library: `rapidfuzz`. Process:

1. For each new player from a non-canonical source, run `rapidfuzz.process.extractOne` against the canonical roster, **restricted to the same tour**.
2. Score ≥ 0.90 → write to `player_aliases` automatically.
3. 0.75 ≤ Score < 0.90 → append to `data/processed/aliases_review.csv` for manual review. The pipeline does **not** auto-merge these.
4. Score < 0.75 → flag as `unrecognized`; investigate (likely a new young player not in cold source yet).

Same-name players exist (Coria brothers, Pliskova sisters). Auto-match never handles these correctly — that's exactly why the review checkpoint exists.

Always normalize unicode (`unicodedata.normalize("NFKD", name)`) and strip diacritics before fuzzy comparison.

## DuckDB conventions

- One DuckDB file: `data/processed/tennis.duckdb`.
- Use `read_csv_auto` on Sackmann files; cast types explicitly afterward — don't trust auto-inferred types for date columns.
- Create indexes on `(player_id, match_date)` and `(match_date, tour)`.
- Schema changes happen via `src/tennis_predictor/data/migrations.py` (TODO phase-1). No ad-hoc `ALTER TABLE` from scripts.

## Tests required for this skill

- Row count of Sackmann CSV matches `matches` row count after ingestion (for that source).
- `(source, match_external_id)` is unique.
- Re-running ingestion changes nothing.
- Fuzzy reconciliation: hand-crafted cases for diacritics, name order, ambiguous same-surname.
