"""Post-hoc probability calibration (isotonic / Platt).

Wraps a pre-fit base estimator and learns a 1-D mapping from
``base.predict_proba(X_cal)[:, 1]`` to empirical win-rate. The method is
picked from sample size per the methodology rule:

    >= 1000 calibration samples → isotonic regression
    <  1000                     → Platt (sigmoid)

Isotonic is non-parametric and shape-flexible but needs samples; Platt is
biased but stable on small calibration sets.

Implemented manually rather than via ``sklearn.calibration.CalibratedClassifierCV``
to avoid the ``cv='prefit'`` deprecation churn and to make joblib round-trip
predictably across sklearn/lightgbm minor versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["isotonic", "platt"]
ISOTONIC_THRESHOLD: int = 1000


class _BaseEstimator(Protocol):
    classes_: npt.NDArray[np.int_]

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]: ...


def choose_calibration_method(n_samples: int) -> CalibrationMethod:
    return "isotonic" if n_samples >= ISOTONIC_THRESHOLD else "platt"


@dataclass
class CalibratedPredictor:
    """Pre-fit base + 1-D calibrator. Joblib-serializable."""

    base: _BaseEstimator
    method: CalibrationMethod
    calibrator: IsotonicRegression | LogisticRegression

    @classmethod
    def fit(
        cls,
        base: _BaseEstimator,
        X_cal: pd.DataFrame,
        y_cal: npt.NDArray[np.int_],
        method: CalibrationMethod | None = None,
    ) -> CalibratedPredictor:
        chosen = method or choose_calibration_method(len(y_cal))
        base_proba = base.predict_proba(X_cal)[:, 1]
        cal: IsotonicRegression | LogisticRegression
        if chosen == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(base_proba, y_cal)
        else:
            cal = LogisticRegression()
            cal.fit(base_proba.reshape(-1, 1), y_cal)
        return cls(base=base, method=chosen, calibrator=cal)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        base_proba = self.base.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            assert isinstance(self.calibrator, IsotonicRegression)
            p1 = self.calibrator.transform(base_proba)
        else:
            assert isinstance(self.calibrator, LogisticRegression)
            p1 = self.calibrator.predict_proba(base_proba.reshape(-1, 1))[:, 1]
        p1 = np.clip(np.asarray(p1, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int_]:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
