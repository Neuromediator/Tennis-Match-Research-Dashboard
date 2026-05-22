"""Walk-forward data loader for training_features.

Slices `training_features` chronologically into (train, calibrate, validate)
splits per fold. Default schedule for a validation year ``V``:

    train     = matches with year(match_date) <= V - 2
    calibrate = matches with year(match_date) == V - 1
    validate  = matches with year(match_date) == V

So the most recent year inside the training cone always lives in the
calibration set, and validation is the year after that. No future leakage
in any direction.

Categorical columns (`tournament_level`, `surface`) are returned as pandas
Categoricals with explicit category levels — train and inference must
share the same level ordering, otherwise LightGBM's category codes drift
silently.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import cast

import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd

from tennis_predictor.models.feature_spec import (
    CATEGORICAL_COLUMNS,
    CATEGORY_VALUES,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
)


@dataclass(frozen=True)
class FoldSlice:
    """One of the three partitions inside a walk-forward fold."""

    features: pd.DataFrame
    labels: npt.NDArray[np.int_]
    match_ids: npt.NDArray[np.str_]
    match_dates: npt.NDArray[np.datetime64]
    rank_gap: npt.NDArray[np.int_]
    p1_player_ids: npt.NDArray[np.str_]

    @property
    def n(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class WalkForwardFold:
    """A single walk-forward fold for one tour."""

    fold_index: int
    validate_year: int
    train: FoldSlice
    calibrate: FoldSlice
    validate: FoldSlice

    @property
    def data_range(self) -> tuple[date, date]:
        all_dates = np.concatenate(
            [self.train.match_dates, self.calibrate.match_dates, self.validate.match_dates]
        )
        lo = cast(date, pd.Timestamp(all_dates.min()).date())
        hi = cast(date, pd.Timestamp(all_dates.max()).date())
        return lo, hi


def _attach_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to pandas Categorical with declared levels."""
    out = df.copy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = pd.Categorical(out[col], categories=CATEGORY_VALUES[col])
    return out


def load_training_frame(conn: duckdb.DuckDBPyConnection, tour: str) -> pd.DataFrame:
    """Load all `training_features` rows for one tour, ordered by date.

    The returned DataFrame contains feature columns, label, match_id, and
    match_date. Categorical columns are typed.
    """
    select_cols = [
        "match_id",
        "match_date",
        "p1_player_id",
        LABEL_COLUMN,
        *FEATURE_COLUMNS,
    ]
    query = f"""
        SELECT {", ".join(select_cols)}
        FROM training_features
        WHERE tour = ?
        ORDER BY match_date, match_id
    """
    df = conn.execute(query, [tour]).fetchdf()
    df["match_date"] = pd.to_datetime(df["match_date"])
    return _attach_categoricals(df)


def _slice(df: pd.DataFrame, mask: pd.Series) -> FoldSlice:
    sub = df.loc[mask]
    features = sub[list(FEATURE_COLUMNS)].copy()
    labels = sub[LABEL_COLUMN].to_numpy(dtype=int)
    match_ids = sub["match_id"].to_numpy(dtype=str)
    match_dates = sub["match_date"].to_numpy(dtype="datetime64[ns]")
    rank_gap = (sub["rank_p1"] - sub["rank_p2"]).to_numpy(dtype=int)
    p1_player_ids = sub["p1_player_id"].to_numpy(dtype=str)
    return FoldSlice(
        features=features,
        labels=labels,
        match_ids=match_ids,
        match_dates=match_dates,
        rank_gap=rank_gap,
        p1_player_ids=p1_player_ids,
    )


def build_folds(
    df: pd.DataFrame,
    validate_years: list[int],
) -> Iterator[WalkForwardFold]:
    """Yield one WalkForwardFold per validation year.

    The DataFrame must come from `load_training_frame` (sorted, categoricals
    attached). Folds that have zero rows in any partition are skipped with
    no error — caller will simply see fewer folds than requested.
    """
    years = df["match_date"].dt.year.astype(int)
    for i, v in enumerate(validate_years):
        train_mask = years <= (v - 2)
        cal_mask = years == (v - 1)
        val_mask = years == v
        if train_mask.sum() == 0 or cal_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        yield WalkForwardFold(
            fold_index=i,
            validate_year=v,
            train=_slice(df, train_mask),
            calibrate=_slice(df, cal_mask),
            validate=_slice(df, val_mask),
        )


def build_production_split(
    df: pd.DataFrame,
    last_full_year: int,
) -> tuple[FoldSlice, FoldSlice]:
    """Split for the shipped model: train on ``<= last_full_year - 1``,
    calibrate on ``last_full_year``. Validation set is implicit and lives
    in the walk-forward report — the shipped model has already proven
    itself there.
    """
    years = df["match_date"].dt.year.astype(int)
    train_mask = years <= (last_full_year - 1)
    cal_mask = years == last_full_year
    return _slice(df, train_mask), _slice(df, cal_mask)
