"""Feature column specification used by every model.

Single source of truth for which columns of `training_features` are inputs.
Excludes label, match_id, tour, dates, player ids, and schema_version.

Phase 4.1 v2: 39 feature columns total (4 categorical, 35 numeric). The
two handedness columns are categorical (R / L / A / U) — LightGBM consumes
them natively via the category levels declared below.
"""

from __future__ import annotations

from tennis_predictor.features.schema import FEATURE_FIELD_NAMES

# All 39 v2 FeatureVector fields (in the canonical FeatureVector ordering).
FEATURE_COLUMNS: tuple[str, ...] = FEATURE_FIELD_NAMES

# Four columns are categorical; rest are numeric and may contain NaN.
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "tournament_level",
    "surface",
    "hand_p1",
    "hand_p2",
)

# Categories must be declared explicitly so train/inference share index codes.
TOURNAMENT_LEVEL_CATEGORIES: tuple[str, ...] = (
    "Slam",
    "M1000",
    "ATP500",
    "ATP250",
    "WTA500",
    "WTA250",
    "Finals",
)
SURFACE_CATEGORIES: tuple[str, ...] = ("Hard", "IHard", "Clay", "Grass")
HAND_CATEGORIES: tuple[str, ...] = ("R", "L", "A", "U")
"""Handedness categories. Order matches `features.schema.Hand`; the 'U'
sentinel covers both unknown values and Sackmann roster misses."""

CATEGORY_VALUES: dict[str, tuple[str, ...]] = {
    "tournament_level": TOURNAMENT_LEVEL_CATEGORIES,
    "surface": SURFACE_CATEGORIES,
    "hand_p1": HAND_CATEGORIES,
    "hand_p2": HAND_CATEGORIES,
}

LABEL_COLUMN: str = "label_winner_is_p1"
