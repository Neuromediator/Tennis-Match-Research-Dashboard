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

## Feature families (v1)

- Surface-Elo and Elo difference.
- Recent form: win % over last 10 / 25 matches, surface-specific.
- Serve/return rolling stats (aces/game, 1st-serve win %, break-points-saved %).
- H2H: total record, recent-meeting recency.
- Fatigue: matches and sets in last 7 days, last 14 days.
- Ranking and ranking delta.
- Tournament level (Grand Slam / Masters / 500 / 250 / other).

## State storage

- `elo_state` is persisted to DuckDB after each `build_training_features` run; inference reads it and rolls forward.
- Other state objects are rebuilt in-memory each training run (cheap on our data volume); not persisted.

## When you add a feature

1. Add the field to `FeatureVector` with type and bounds.
2. Add the computation to the appropriate state object.
3. Add a leakage test that proves it can't see the future.
4. Run `build_training_features()` end-to-end and commit the regenerated metadata.
