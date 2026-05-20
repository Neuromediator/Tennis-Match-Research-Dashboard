"""Feature engineering: replay-and-snapshot, point-in-time correctness. Phase 3.

The two sanctioned entry points (see CLAUDE.md hard rule #2) are exposed
here for convenience:

- `build_training_features(conn)` — chronological replay → `training_features`.
- `compute_features(...) -> FeatureVector` — inference path (Task #10).

`FeatureVector` is the Pydantic contract returned by `compute_features` and
the row layout written by `build_training_features`.
"""

from tennis_predictor.features.build import BuildSummary, build_training_features
from tennis_predictor.features.inference import compute_features
from tennis_predictor.features.schema import (
    FEATURE_FIELD_NAMES,
    FeatureVector,
    Surface,
    TournamentLevel,
)

__all__ = [
    "FEATURE_FIELD_NAMES",
    "BuildSummary",
    "FeatureVector",
    "Surface",
    "TournamentLevel",
    "build_training_features",
    "compute_features",
]
