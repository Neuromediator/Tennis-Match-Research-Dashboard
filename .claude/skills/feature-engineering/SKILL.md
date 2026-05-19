---
name: feature-engineering
description: Use when adding or modifying features, building the training table, or working with the per-match state replay. Establishes the strict point-in-time rule, the replay-and-snapshot pattern, and the FeatureVector contract.
---

# Feature engineering

## The contract

There are exactly two sanctioned entry points. Anything else is a bug.

### `build_training_features()`

Chronological replay over every match in the database. Maintains in-memory **state objects**:

- `EloState` — rating per `(player_id, surface)`.
- `RollingFormState` — last-N match results per player.
- `H2HState` — pairwise counters.
- `FatigueState` — matches and sets in last 7/14 days.

For each match, in chronological order:

1. Read pre-match state for both players → produce a `FeatureVector`.
2. Write a row to the `training_features` table: `(match_id, p1_features..., p2_features..., label)`.
3. **Then** update state with the match result.

Step ordering is critical. State must never be updated before the snapshot is taken.

### `compute_features(player_id, opponent_id, surface, tour, as_of_date) -> FeatureVector`

Inference path. Reads the most recent state snapshot with `snapshot_date ≤ as_of_date`, rolls forward through any matches in `(snapshot_date, as_of_date)`, then computes the vector. Returns a Pydantic `FeatureVector`.

## The hard rule

**No feature computation may read data with `match_date >= as_of_date`.** Period.

Tests assert this by writing tampered future rows and confirming feature values are unchanged. If you write a feature that fails the leakage test, the feature is wrong — do not "fix" the test.

## `FeatureVector`

A Pydantic model in `src/tennis_predictor/features/schema.py`. Fields are typed and bounded where it makes sense (e.g., `elo_diff: float`, `recent_form_pct: float = Field(ge=0, le=1)`). Adding a feature means adding a field and regenerating the `training_features` table.

Why Pydantic, not `dict`: training and inference run on different code paths months apart. Schema drift between them is the single most common silent bug in ML pipelines. A typed contract catches it at construction time.

## Feature families (v1) — final list

`FeatureVector` has **28 fields** organized into seven conceptual families. Counts in parentheses are field counts in that family. LightGBM consumes the flat vector; the families are for our mental model and feature-importance grouping.

| # | Family (count) | Fields |
|---|---|---|
| Surface-Elo (3) | `elo_p1_surface`, `elo_p2_surface`, `elo_diff_surface` |
| Recent form (4) | `win_pct_last10_p1/p2`, `win_pct_last25_surface_p1/p2` |
| Serve/return rolling (8) | `first_serve_win_pct_p1/p2`, `second_serve_win_pct_p1/p2`, `bp_saved_pct_p1/p2`, `bp_converted_pct_p1/p2` |
| H2H (3) | `h2h_p1_wins`, `h2h_p2_wins`, `h2h_recency_days` |
| Fatigue (4) | `fatigue_matches_7d_p1/p2`, `fatigue_sets_14d_p1/p2` |
| Ranking (3) | `rank_p1`, `rank_p2`, `rank_diff` |
| Tournament context (3) | `tournament_level` (cat), `best_of` (3 or 5), `surface` (cat) |

26 numeric + 2 categorical fields (`tournament_level`, `surface`). LightGBM handles categoricals natively — no one-hot encoding.

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

## State storage

- `elo_state` is persisted to DuckDB after each `build_training_features` run; inference reads it and rolls forward.
- Other state objects are rebuilt in-memory each training run (cheap on our data volume); not persisted.

## When you add a feature

1. Add the field to `FeatureVector` with type and bounds.
2. Add the computation to the appropriate state object.
3. Add a leakage test that proves it can't see the future.
4. Run `build_training_features()` end-to-end and commit the regenerated metadata.
