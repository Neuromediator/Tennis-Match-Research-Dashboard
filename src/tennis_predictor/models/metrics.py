"""Metrics for binary probabilistic forecasts.

Headline metric is Brier score; log loss and accuracy are reported too.
Calibration curve binning follows the 10-bin convention in the
model-training skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class CalibrationBin:
    bin_lower: float
    bin_upper: float
    n: int
    mean_predicted: float
    mean_actual: float


@dataclass(frozen=True)
class ClassificationMetrics:
    """Probabilistic metrics for a binary classifier."""

    n: int
    brier: float
    log_loss: float
    accuracy: float
    # Accuracy bucketed by abs(rank_p1 - rank_p2); buckets [0,50), [50,200), [200,inf).
    accuracy_by_rank_gap: dict[str, float]
    calibration_bins: list[CalibrationBin]

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
            "accuracy_by_rank_gap": self.accuracy_by_rank_gap,
            "calibration_bins": [
                {
                    "bin_lower": b.bin_lower,
                    "bin_upper": b.bin_upper,
                    "n": b.n,
                    "mean_predicted": b.mean_predicted,
                    "mean_actual": b.mean_actual,
                }
                for b in self.calibration_bins
            ],
        }


_RANK_GAP_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0_50", 0, 50),
    ("50_200", 50, 200),
    ("200_inf", 200, 10_000),
)


def _calibration_bins(
    y_true: npt.NDArray[np.int_],
    y_prob: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Equal-width bins in [0, 1] on predicted probability."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Last bin is right-inclusive.
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append(
                CalibrationBin(
                    bin_lower=lo,
                    bin_upper=hi,
                    n=0,
                    mean_predicted=float("nan"),
                    mean_actual=float("nan"),
                )
            )
            continue
        bins.append(
            CalibrationBin(
                bin_lower=lo,
                bin_upper=hi,
                n=n,
                mean_predicted=float(y_prob[mask].mean()),
                mean_actual=float(y_true[mask].mean()),
            )
        )
    return bins


def compute_metrics(
    y_true: npt.NDArray[np.int_],
    y_prob: npt.NDArray[np.float64],
    rank_gap: npt.NDArray[np.int_] | None = None,
) -> ClassificationMetrics:
    """Compute the full metric bundle for one set of predictions."""
    y_true_clipped = y_true.astype(int)
    y_prob_clipped = np.clip(y_prob.astype(float), 1e-15, 1 - 1e-15)

    brier = float(brier_score_loss(y_true_clipped, y_prob_clipped))
    ll = float(log_loss(y_true_clipped, y_prob_clipped, labels=[0, 1]))
    preds = (y_prob_clipped >= 0.5).astype(int)
    accuracy = float((preds == y_true_clipped).mean())

    acc_by_gap: dict[str, float] = {}
    if rank_gap is not None:
        gaps = np.abs(rank_gap.astype(int))
        for label, lo, hi in _RANK_GAP_BUCKETS:
            mask = (gaps >= lo) & (gaps < hi)
            if mask.sum() == 0:
                acc_by_gap[label] = float("nan")
            else:
                acc_by_gap[label] = float((preds[mask] == y_true_clipped[mask]).mean())

    return ClassificationMetrics(
        n=len(y_true_clipped),
        brier=brier,
        log_loss=ll,
        accuracy=accuracy,
        accuracy_by_rank_gap=acc_by_gap,
        calibration_bins=_calibration_bins(y_true_clipped, y_prob_clipped),
    )
