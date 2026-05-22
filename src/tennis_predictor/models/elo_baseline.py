"""Surface-Elo baseline predictor.

No learning. Pure formula:

    P(p1 wins) = 1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400))

Reads only ``elo_p1_surface`` and ``elo_p2_surface`` and ignores the other
26 features. The model-training skill keeps this around as the honest floor:
a shipped LightGBM model must beat it on Brier score, or we ship the
baseline instead.

Implements a minimal sklearn-style interface so the calibration wrapper
and walk-forward harness can treat it identically to LightGBM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class SurfaceEloBaseline:
    """Hardcoded Elo-difference predictor. Joblib-serializable."""

    classes_: npt.NDArray[np.int_] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.classes_ is None:
            self.classes_ = np.array([0, 1], dtype=int)

    # The sklearn calibration & harness conventions both call .fit; for
    # this estimator the call is a no-op that just confirms the label set.
    def fit(
        self,
        X: pd.DataFrame,
        y: npt.NDArray[np.int_],
    ) -> SurfaceEloBaseline:
        return self

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        diff = X["elo_p1_surface"].to_numpy(dtype=float) - X["elo_p2_surface"].to_numpy(dtype=float)
        p1_win = 1.0 / (1.0 + np.power(10.0, -diff / 400.0))
        return np.column_stack([1.0 - p1_win, p1_win])

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int_]:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
