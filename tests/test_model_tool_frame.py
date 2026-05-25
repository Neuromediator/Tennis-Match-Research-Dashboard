"""Unit tests for `_feature_vector_to_frame`.

The agent's inference path packs a `FeatureVector` into a one-row pandas
DataFrame before calling LightGBM. When the vector contains `None` in
nullable int columns (e.g. `h2h_recency_days=None` for never-met pairs),
pandas falls back to `object` dtype, which LightGBM rejects with
`pandas dtypes must be int, float or bool`. The cast in
`_feature_vector_to_frame` forces every non-categorical column to a
numeric dtype; these tests pin that contract.
"""

from __future__ import annotations

import numpy as np

from tennis_predictor.features.schema import FeatureVector
from tennis_predictor.llm.tools.model_tool import _feature_vector_to_frame
from tennis_predictor.models.feature_spec import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def _build_fv_with_nulls() -> FeatureVector:
    """A `FeatureVector` that exercises every nullable field, matching
    the "first-time meeting between two debutants on an unknown surface"
    shape that triggered the production bug."""
    return FeatureVector(
        elo_p1_surface=1500.0,
        elo_p2_surface=1500.0,
        elo_diff_surface=0.0,
        # All recent-form / serve-return columns left at None default.
        h2h_p1_wins=0,
        h2h_p2_wins=0,
        h2h_recency_days=None,  # <- the column that triggered the bug
        fatigue_matches_7d_p1=0,
        fatigue_matches_7d_p2=0,
        fatigue_sets_14d_p1=0,
        fatigue_sets_14d_p2=0,
        rank_p1=9999,
        rank_p2=9999,
        rank_diff=0,
        tournament_level="Slam",
        best_of=5,
        surface="Clay",
        # All age / height / recovery left at None.
    )


def test_feature_vector_to_frame_non_categoricals_are_numeric() -> None:
    df = _feature_vector_to_frame(_build_fv_with_nulls())
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            assert df[col].dtype.name == "category", f"{col} should be category"
            continue
        kind = df[col].dtype.kind
        assert kind in ("i", "f", "b"), (
            f"{col} dtype is {df[col].dtype} (kind={kind!r}); "
            "LightGBM only accepts int / float / bool"
        )


def test_feature_vector_to_frame_h2h_recency_days_none_becomes_nan() -> None:
    """The original failure: `h2h_recency_days=None` → pandas object dtype.
    After the fix it must end up as NaN in a float column."""
    df = _feature_vector_to_frame(_build_fv_with_nulls())
    col = df["h2h_recency_days"]
    assert col.dtype.kind == "f"
    assert np.isnan(col.iloc[0])


def test_feature_vector_to_frame_preserves_int_when_no_nulls() -> None:
    fv = _build_fv_with_nulls().model_copy(update={"h2h_recency_days": 30})
    df = _feature_vector_to_frame(fv)
    assert df["h2h_recency_days"].iloc[0] == 30
    # Cast to numeric is idempotent on already-numeric data.
    assert df["h2h_recency_days"].dtype.kind in ("i", "f")
