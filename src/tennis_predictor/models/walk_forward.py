"""Walk-forward harness.

Orchestrates the per-fold loop for one (tour, model_type) combination:
trains a base estimator, calibrates it on the fold's calibration set,
scores both stages on the held-out validation set, and gathers per-fold
predictions for the market benchmark overlay.

The harness is model-agnostic: it accepts a ``train_fn`` callable that
returns a pre-fit base estimator with ``predict_proba``. Both
``SurfaceEloBaseline`` and the LightGBM trainer plug in here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import duckdb
import numpy as np
import numpy.typing as npt

from tennis_predictor.models.calibration import (
    CalibratedPredictor,
    CalibrationMethod,
    choose_calibration_method,
)
from tennis_predictor.models.data import FoldSlice, WalkForwardFold
from tennis_predictor.models.market_benchmark import (
    MIN_MARKET_OVERLAY_ROWS,
    MarketProbabilities,
    fetch_market_probabilities,
    market_metrics,
)
from tennis_predictor.models.metrics import ClassificationMetrics, compute_metrics

logger = logging.getLogger(__name__)


TrainFn = Callable[[FoldSlice, FoldSlice], object]
"""Signature for a base-estimator trainer.

Args:
    train: training partition (years <= V-2).
    calibrate: calibration partition (year V-1) — may also be used for
        early stopping inside the trainer.

Returns:
    Any object exposing ``predict_proba(X) -> ndarray of shape (n, 2)``
    where column 1 is ``P(label = 1)``.
"""


@dataclass(frozen=True)
class FoldResult:
    fold_index: int
    validate_year: int
    n_train: int
    n_calibrate: int
    n_validate: int
    calibration_method: CalibrationMethod
    metrics_pre: ClassificationMetrics
    metrics_post: ClassificationMetrics
    market: ClassificationMetrics | None
    market_n: int
    # Per-fold validation predictions (for downstream aggregated plotting).
    y_true: npt.NDArray[np.int_]
    y_prob_post: npt.NDArray[np.float64]
    market_probs: MarketProbabilities | None


@dataclass(frozen=True)
class WalkForwardResult:
    tour: str
    model_type: str
    folds: list[FoldResult] = field(default_factory=list)

    def recent(self, last_n: int) -> list[FoldResult]:
        return self.folds[-last_n:]

    def aggregate_brier_post(self, last_n: int | None = None) -> float:
        scope = self.folds if last_n is None else self.recent(last_n)
        if not scope:
            return float("nan")
        weights = np.array([f.n_validate for f in scope], dtype=float)
        briers = np.array([f.metrics_post.brier for f in scope], dtype=float)
        return float(np.average(briers, weights=weights))


def run_walk_forward(
    folds: list[WalkForwardFold],
    train_fn: TrainFn,
    tour: str,
    model_type: str,
    conn: duckdb.DuckDBPyConnection | None = None,
    calibration_method: CalibrationMethod | None = None,
) -> WalkForwardResult:
    """Run walk-forward for one (tour, model_type) combo and gather results."""
    fold_results: list[FoldResult] = []
    for fold in folds:
        logger.info(
            "[%s/%s] fold %d (validate=%d): train=%d cal=%d val=%d",
            tour,
            model_type,
            fold.fold_index,
            fold.validate_year,
            fold.train.n,
            fold.calibrate.n,
            fold.validate.n,
        )
        base = train_fn(fold.train, fold.calibrate)
        method = calibration_method or choose_calibration_method(fold.calibrate.n)
        calibrated = CalibratedPredictor.fit(
            base=base,  # type: ignore[arg-type]
            X_cal=fold.calibrate.features,
            y_cal=fold.calibrate.labels,
            method=method,
        )
        y_prob_pre = base.predict_proba(fold.validate.features)[:, 1]  # type: ignore[attr-defined]
        y_prob_post = calibrated.predict_proba(fold.validate.features)[:, 1]
        m_pre = compute_metrics(fold.validate.labels, y_prob_pre, fold.validate.rank_gap)
        m_post = compute_metrics(fold.validate.labels, y_prob_post, fold.validate.rank_gap)

        market_probs: MarketProbabilities | None = None
        market = None
        market_n = 0
        if conn is not None:
            probs = fetch_market_probabilities(
                conn=conn,
                match_ids=fold.validate.match_ids,
                labels=fold.validate.labels,
                p1_player_ids=fold.validate.p1_player_ids,
            )
            market_n = probs.n
            if market_n >= MIN_MARKET_OVERLAY_ROWS:
                market_probs = probs
                market = market_metrics(probs)

        fold_results.append(
            FoldResult(
                fold_index=fold.fold_index,
                validate_year=fold.validate_year,
                n_train=fold.train.n,
                n_calibrate=fold.calibrate.n,
                n_validate=fold.validate.n,
                calibration_method=method,
                metrics_pre=m_pre,
                metrics_post=m_post,
                market=market,
                market_n=market_n,
                y_true=fold.validate.labels,
                y_prob_post=y_prob_post,
                market_probs=market_probs,
            )
        )
    return WalkForwardResult(tour=tour, model_type=model_type, folds=fold_results)
