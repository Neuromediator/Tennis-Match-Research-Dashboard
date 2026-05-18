---
name: data-ingestion
description: Use when adding a new data source, modifying the DuckDB schema, or working on player reconciliation. Establishes the contracts for source provenance, canonical IDs, and the manual-review checkpoint.
---

# Data ingestion

## Sources

| Source | Tier | Phase | Notes |
|---|---|---|---|
| Sackmann `tennis_atp` / `tennis_wta` | cold | 1 | Git submodules under `data/raw/`. Pinned commits. |
| matchstat Tennis API ("Tennis API - ATP WTA ITF" on RapidAPI) | hot | 2 | Free tier: 500 req/month, hard cap — refresh script must budget tightly (no naive retry loops, request count logged to `ingestion_runs`). Daily responsibilities: last ~30 days of completed matches → `matches`; currently-known fixtures → `scheduled_matches` (lookahead is naturally short — full R1 right after a draw, then today/tomorrow as the bracket resolves); inter-week ranking overlay on top of weekly Sackmann snapshots. |
| tennis-data.co.uk archives | benchmark | 1 | Historical closing-price implied probabilities. **Not a feature source.** |

## Canonical schema

The `matches` table holds rows from every source, distinguished by:

- `source` — one of `"sackmann"`, `"matchstat"`, `"tennis-data-co-uk"`, ...
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

Library: `rapidfuzz`. Pipeline (`src/tennis_predictor/data/reconcile.py`):

1. `seed_aliases_from_players(conn, tour)` seeds three alias forms per canonical Sackmann player: `"First Last"`, `"Last First"`, `"Last F"`. The abbreviated form makes tennis-data.co.uk's `Last F.` format hit the exact-match fast path with no fuzzy cost. Players whose first or last name is `'Unknown'` are excluded.
2. `AliasIndex.lookup(raw_name)` returns a `ReconciliationResult` with one of three statuses:
   - **auto** — confidence ≥ 0.90 with no different-canonical-id runner-up within 0.05.
   - **review** — confidence 0.75-0.90, OR ≥ 0.90 but ambiguous.
   - **unknown** — confidence < 0.75.
3. The market-data loader writes `review` outcomes to `data/processed/aliases_review.csv` (full match context, not just the alias). A human reviews them and runs `scripts/apply_aliases_review.py`, which dedupes (raw, tour, canonical_player_id) tuples and INSERTs them into `player_aliases` with `source='manual_review'` and `confidence=1.0`. ON CONFLICT DO NOTHING makes this idempotent.

Effect: on the next refresh, the same raw names hit the exact-match fast path via the manual_review row and no longer surface in review.

Always normalize unicode (precomposed letters like Đ/Ł/Ø/Æ/Þ/ß require explicit substitution before NFKD — see `_PRECOMPOSED_REPLACEMENTS` in `reconcile.py`).

## Audit artefacts

Each refresh writes two append-only CSVs under `data/processed/`:

- `aliases_review.csv` — fuzzy resolved at low confidence. Workflow above.
- `unmatched_market_rows.csv` — fuzzy succeeded but the JOIN against `matches` returned zero candidates. Most common cause: same-surname collision in the abbreviated-form seed. Analyzed in `notebooks/explore_unmatched.ipynb`.

Both are append-only across runs — older entries persist so the file doubles as a long-term record of edge cases.

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
