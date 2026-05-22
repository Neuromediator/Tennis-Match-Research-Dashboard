---
name: feature-engineering
description: Use when adding or modifying features, building the training table, or working with the per-match state replay. Establishes the strict point-in-time rule, the replay-and-snapshot pattern, and the FeatureVector contract.
---

# Feature engineering

## The contract

There are exactly two sanctioned entry points. Anything else is a bug.

### `build_training_features(conn) -> BuildSummary`

Chronological replay over every match in the database. Maintains in-memory **state objects**:

- `EloState` — rating per `(player_id, surface)`. K=32, default 1500. **Persisted** to `elo_state` table at end of replay.
- `RollingFormState` — per-player list of `(date, surface, won)`. Snapshots `(win_pct_last10_any, win_pct_last25_surface)`.
- `H2HState` — pairwise counters keyed by canonical (lex-sorted) player pair.
- `FatigueState` — per-player rolling counts: matches in last 7d, sets in last 14d.
- `ServeReturnState` — per-player surface-filtered last-25 serve/return raw counts (skips matches with NULL stat columns).

Plus `RankingLookup` (in-memory bisect over the `rankings` table — not a state object, but used by the orchestrator).

For each match, in chronological order (`tourney_date ASC, tourney_id ASC, match_num ASC, match_id ASC`):

1. Apply the **state-update gate**: must be `match_status='completed'` AND `surface` (after normalization) is not None. RET / W/O / DEF / NULL-surface rows are skipped from BOTH state AND labels.
2. Apply the **label-write gate** (additional, on top of state gate):
   - `match_tier == 'main'` OR tour-level main-draw qualifying (ATP `qual_chall` with `tourney_level in {G, M, A}`; WTA `qual_itf` with `tourney_level in {G, PM, P, I, T1, T2, W}`).
   - Normalized `tournament_level` is not None (excludes D, O, WTA-OOS, WTA-125).
   - `best_of in (3, 5)`.
   - Both players have ≥ 5 completed matches in history (the "history floor").
3. If label-eligible: pick `(p1, p2)` as lex-sorted player IDs. Read pre-match state from all 5 state objects + `RankingLookup` → build a `FeatureVector`. Write a row to `training_features` with `label_winner_is_p1 = 1` iff the actual winner was `p1`.
4. **Then** update all state objects with the match result.

Step ordering (snapshot → write → update) is critical. State must never be updated before the snapshot is taken.

`training_features` is fully overwritten on each run (DELETE + bulk INSERT in one transaction). `elo_state` is overwritten as a full snapshot at the end. Other state objects are NOT persisted; they rebuild from scratch each run.

### `compute_features(conn, player_id, opponent_id, surface, tour, as_of_date, tournament_level, best_of, *, elo=None, ranking_lookup=None) -> FeatureVector`

Inference path. Returns a Pydantic `FeatureVector` for a single hypothetical `(p1, p2, surface, as_of_date)` instance.

Algorithm:

1. Canonical ordering: `(p1, p2) = sorted([player_id, opponent_id])`.
2. Load `RankingLookup` from `rankings` (or accept caller-supplied cache).
3. Elo decision: if `persisted_snapshot_date < as_of_date`, load `EloState.from_db(conn)` and roll forward through matches in the gap. **Otherwise** (historical inference / equivalence test) rebuild `EloState` from scratch — the persisted snapshot reflects state AFTER every DB match and would leak the target match's update.
4. Build the other 4 state objects fresh by querying every completed, surface-resolved match involving `p1` OR `p2` with `match_date < as_of_date`. Replay in chronological order into all states.
5. Snapshot each state → assemble `FeatureVector` via `model_validate`.

The equivalence contract (Phase 3 exit criterion): `compute_features` returns identical values to the corresponding row in `training_features` for the same `(player, opponent, surface, as_of_date)`. Tested in `tests/test_compute_features.py::test_equivalence_with_training_replay`.

`elo` and `ranking_lookup` may be passed pre-loaded so a Streamlit session amortizes the load across many predictions (Phase 6 concern).

## The hard rule

**No feature computation may read data with `match_date >= as_of_date`.** Period.

Tests assert this by writing tampered future rows and confirming feature values are unchanged. If you write a feature that fails the leakage test, the feature is wrong — do not "fix" the test.

## `FeatureVector`

A Pydantic model in `src/tennis_predictor/features/schema.py`. Fields are typed and bounded where it makes sense (e.g., `elo_diff: float`, `recent_form_pct: float = Field(ge=0, le=1)`). Adding a feature means adding a field and regenerating the `training_features` table.

Why Pydantic, not `dict`: training and inference run on different code paths months apart. Schema drift between them is the single most common silent bug in ML pipelines. A typed contract catches it at construction time.

## Feature families — v2 (Phase 4.1)

`FeatureVector` has **39 fields** in v2 (28 v1 fields + 11 added in Phase 4.1) organized into eleven conceptual families. Counts in parentheses are field counts in that family. LightGBM consumes the flat vector; the families are for our mental model and feature-importance grouping.

| # | Family (count) | Fields |
|---|---|---|
| Surface-Elo (3) | `elo_p1_surface`, `elo_p2_surface`, `elo_diff_surface` |
| Recent form (4) | `win_pct_last10_p1/p2`, `win_pct_last25_surface_p1/p2` |
| Serve/return rolling (8) | `first_serve_win_pct_p1/p2`, `second_serve_win_pct_p1/p2`, `bp_saved_pct_p1/p2`, `bp_converted_pct_p1/p2` |
| H2H (3) | `h2h_p1_wins`, `h2h_p2_wins`, `h2h_recency_days` |
| Fatigue (4) | `fatigue_matches_7d_p1/p2`, `fatigue_sets_14d_p1/p2` |
| Ranking (3) | `rank_p1`, `rank_p2`, `rank_diff` |
| Tournament context (3) | `tournament_level` (cat), `best_of` (3 or 5), `surface` (cat) |
| Handedness (2) — *Phase 4.1* | `hand_p1` (cat R/L/A/U), `hand_p2` (cat R/L/A/U) |
| Age (4) — *Phase 4.1* | `age_p1/p2` (years), `age_vs_peak_p1/p2` (signed years from `PEAK_AGE[tour]`: ATP=26, WTA=24) |
| Height (3) — *Phase 4.1* | `height_p1/p2` (cm), `height_diff_cm` |
| Recovery (2) — *Phase 4.1* | `days_since_last_match_p1/p2` (capped at 365) |

35 numeric + 4 categorical fields (`tournament_level`, `surface`, `hand_p1`, `hand_p2`). LightGBM handles categoricals natively — no one-hot encoding. Hand categories are declared in `models.feature_spec.HAND_CATEGORIES = ("R", "L", "A", "U")`.

The module-level `SCHEMA_VERSION` in `features/schema.py` is `2`. The `training_features` table writes `schema_version` per row; the migration in `_migrate_training_features` detects v1→v2 via the `days_since_last_match_p1` sentinel column and drops the v1 table so the v2 DDL takes over.

### Choices in serve/return that aren't obvious

The four serve/return fields above were chosen deliberately; v1 explicitly excludes some intuitive metrics. Reasoning:

- **`first_serve_win_pct`** is included, **`first_serve_in_pct` is not**. Just landing the first serve in court is weak signal — a slice that lands in but loses the rally (Errani on WTA, many lower-ranked men) is not the same as a serve that lands AND wins the point. Win % captures pressure; in % captures only safety.
- **`second_serve_win_pct`** is the single most discriminating serve stat per tour, with a clean ~10-point gap between top-10 and rank-100+. Often a stronger signal than first-serve win %.
- **`bp_saved_pct`** is a per-opportunity rate, NOT an absolute count. A player who lost serve once but saved 14/15 break points actually has a poor-quality serve — the rate exposes this; the count hides it.
- **`bp_converted_pct`** is the mirror return-side rate.
- **`aces_per_game` is excluded** in v1. Correlates strongly with `first_serve_win_pct` (a fast first serve produces both), so it adds little new signal while consuming parameter capacity. If feature-importance after Phase 4 surprises us, we can revisit.
- **`double_faults_per_game` is excluded** for the same reason — already absorbed by `second_serve_win_pct` (DFs are the worst-case 2nd-serve outcome).

All four serve/return rates are **rolling over the last ~25 matches** and **surface-specific where applicable** — clay rallies are longer, BP-conversion rates differ by surface; surface filtering removes that as a confound.

### EloState mechanics

Standard Elo formula adapted per `(player, surface)`:

```
E_A   = 1 / (1 + 10^((R_B - R_A) / 400))
R_A_new = R_A + K * (S_A - E_A)
```

- Default rating: 1500.
- K-factor: constant 32 in v1. If calibration analysis after Phase 4 suggests the model is over-confident on freshly-rated players (low matches_played), revisit with a matches-played-aware K decay.
- Storage: one row per `(player_id, surface)` in `elo_state`. ~137k players × 4 surfaces = ~550k rows — fits in memory during replay.
- Persistence: snapshotted after each `build_training_features()` run; inference rolls forward incrementally from snapshot using any matches with `snapshot_date < match_date ≤ as_of_date`.

## Surface taxonomy

4 canonical values: `Hard`, `IHard`, `Clay`, `Grass`. Normalization happens in `features.surface.normalize_surface(raw_surface, tourney_name)`:

- `Carpet` → `IHard` (carpet was always indoor; merging keeps indoor-hard ratings continuous through the 2009 transition).
- `Hard` + tournament in `INDOOR_TOURNAMENTS` whitelist → `IHard` (Paris Bercy, Vienna, Rotterdam, Basel, Marseille, Stockholm, Memphis, ATP Finals, WTA Finals, Linz, Luxembourg, Quebec, Zurich, etc.).
- `Hard` otherwise → `Hard` (outdoor).
- `Clay` (case-insensitive: `'clay'` and `'Clay'` both map) → `Clay`.
- `Grass` → `Grass`.
- `NULL` or unrecognized → `None` (match is excluded from BOTH state AND labels).

The indoor whitelist is hand-curated in `src/tennis_predictor/features/indoor_tournaments.py`. Review by reading the file.

## Tournament-level taxonomy

7 canonical values: `Slam`, `M1000`, `ATP500`, `ATP250`, `WTA500`, `WTA250`, `Finals`. Normalization in `features.tournament_level.normalize_tournament_level(tour, raw_level, tourney_name)`:

- `G` → `Slam`; `F` → `Finals` (tour-agnostic).
- `D` (Davis Cup), `O` (Olympics) → `None` (excluded from labels by user decision).
- ATP: `M` → `M1000`; `A` + name in `ATP_500_TOURNAMENTS` → `ATP500`, else `ATP250`.
- WTA modern: `PM` → `M1000`; `P` → `WTA500`; `I` → `WTA250`.
- WTA legacy (pre-2009): `T1` → `M1000`; `T2` → `WTA500`; `T3/T4/T5` → `WTA250`.
- WTA catch-all: `W` → `WTA250` (default).
- WTA out-of-scope (`CC`, `E`, `50+H`, `35+H`, `J`) → `None`.
- WTA 125 events (Challenger-equivalent for women): caught by `'125'` substring in `tourney_name`, returns `None` regardless of declared tier.

The ATP 500 list is hand-curated in `features.tournament_level.ATP_500_TOURNAMENTS`.

## State storage

- `elo_state` is persisted to DuckDB after each `build_training_features` run; inference reads it and rolls forward (or rebuilds from scratch if `as_of_date <= snapshot_date`).
- `last_match_state` (Phase 4.1) persisted alongside `elo_state` via `LastMatchState.save_to_db`; same baseline-date discipline in inference — load + roll forward when `as_of_date > snapshot_date`, else rebuild from scratch.
- Other state objects (`RollingForm`, `H2H`, `Fatigue`, `ServeReturn`) are rebuilt in-memory each training run (cheap on our data volume); not persisted.
- `RankingLookup` is rebuilt on demand from the `rankings` table; ~5.6M rows fit in memory as parallel `{player_id: [dates], [ranks]}` arrays with O(log N) bisect.
- `PlayerMetadataLookup` (Phase 4.1) is built once via `from_db(conn)` at the start of `build_training_features` / `compute_features` — pure JOIN against `players` for `hand` / `dob` / `height`. Static data, so no leakage and no state object.

## NaN policy in the FeatureVector

Required (never None): Elo (3), H2H wins (2), fatigue (4), ranking (3, with `9999` sentinel for unranked), tournament context (3), handedness (2 — default `"U"` when missing).

Optional (None allowed):
- Recent form (4): None when window has < 3 matches.
- Serve/return (8): None when surface-filtered window has < 5 matches with non-null stat columns (~58% of pre-1990s `main` matches have NULL serve stats in Sackmann).
- `h2h_recency_days`: None when the pair has never met.
- Age (4): None when `players.dob` is missing. No Pydantic bounds — Sackmann has a few obviously wrong DOBs (e.g. player listed as 3 yo at a tour-level match); LightGBM handles outliers cleanly.
- Height (3): None when `players.height` is missing. ATP ~88% coverage on training rows, WTA ~57%. No bounds.
- Recovery (2): None when the player has no prior completed match. Capped at 365 days (`LastMatchState.CAP_DAYS`) — beyond a year the semantics flip from "recovery" to "returning from long absence".

## When you add a feature

1. Add the field to `FeatureVector` with type and bounds in `features/schema.py`. Bump `SCHEMA_VERSION`.
2. Add the computation to the appropriate state object (or create a new one if no fit). For player-static fields (no time dependency), add to `PlayerMetadataLookup` instead of a state object.
3. Add a row to the `training_features` DDL in `src/tennis_predictor/data/schema.py`. Extend `_migrate_training_features` with a sentinel-column check for the new schema version so the next `create_all_tables` call drops the stale table.
4. Update `_TRAINING_FEATURES_COLUMNS` + `_to_insert_row` in `features/build.py`.
5. Update `compute_features` to read the new state.
6. Add a leakage test in `tests/test_feature_leakage.py` that mutates a future row and asserts the new field is unchanged. For static `players` JOIN fields, also add a sanity test that they don't drift under future-match mutations.
7. Update `CATEGORICAL_COLUMNS` + `CATEGORY_VALUES` in `models/feature_spec.py` if the new field is categorical.
8. Extend `tests/test_train_models_smoke.py` synthetic features + `players` seed so the smoke test still produces a deterministic round-trip fixture.
9. Run `build_training_features()` end-to-end on the real DB to validate distribution.
