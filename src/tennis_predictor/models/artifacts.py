"""Model artifact persistence.

For each (tour, model_type) we produce one directory under
``models/<tour>/<model_type>/<YYYYMMDD-HHMM>/`` containing:

- ``model.joblib`` — the production-fit ``CalibratedPredictor``.
- ``metadata.json`` — training date, data range, feature list, metrics
  (pre/post calibration, aggregated and per-fold), calibration method,
  git commit, walk-forward fold count.
- ``report.md`` — Markdown summary.
- ``calibration_plot.png`` — model vs market overlay (10 bins).
- ``roundtrip_fixture.json`` — 16 fixture rows + expected probabilities
  for the round-trip serialization test.

A ``latest`` symlink in the parent dir is updated atomically to point at
the new directory.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: required for CI / scripts / pytest
import matplotlib.pyplot as plt
import numpy as np

from tennis_predictor.models.calibration import CalibratedPredictor
from tennis_predictor.models.data import FoldSlice
from tennis_predictor.models.feature_spec import FEATURE_COLUMNS
from tennis_predictor.models.walk_forward import FoldResult, WalkForwardResult

logger = logging.getLogger(__name__)

ROUNDTRIP_FIXTURE_SIZE: int = 16


@dataclass(frozen=True)
class ArtifactBundle:
    tour: str
    model_type: str
    walk_forward: WalkForwardResult
    production_predictor: CalibratedPredictor
    production_train: FoldSlice
    production_calibrate: FoldSlice


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _data_range(bundle: ArtifactBundle) -> tuple[str, str]:
    all_dates = np.concatenate(
        [bundle.production_train.match_dates, bundle.production_calibrate.match_dates]
    )
    import pandas as pd

    lo = pd.Timestamp(all_dates.min()).date().isoformat()
    hi = pd.Timestamp(all_dates.max()).date().isoformat()
    return lo, hi


def _aggregate_metrics(folds: list[FoldResult], stage: str) -> dict[str, object]:
    """Sample-weighted aggregate of per-fold metrics."""
    if not folds:
        return {}
    weights = np.array([f.n_validate for f in folds], dtype=float)
    get = (lambda f: f.metrics_post) if stage == "post" else (lambda f: f.metrics_pre)
    brier = float(np.average([get(f).brier for f in folds], weights=weights))
    ll = float(np.average([get(f).log_loss for f in folds], weights=weights))
    acc = float(np.average([get(f).accuracy for f in folds], weights=weights))
    return {
        "n_validate_total": int(weights.sum()),
        "brier": brier,
        "log_loss": ll,
        "accuracy": acc,
    }


def _build_metadata(bundle: ArtifactBundle, timestamp: str) -> dict[str, object]:
    folds = bundle.walk_forward.folds
    data_lo, data_hi = _data_range(bundle)
    per_fold: list[dict[str, object]] = []
    for f in folds:
        per_fold.append(
            {
                "fold_index": f.fold_index,
                "validate_year": f.validate_year,
                "n_train": f.n_train,
                "n_calibrate": f.n_calibrate,
                "n_validate": f.n_validate,
                "calibration_method": f.calibration_method,
                "metrics_pre_calibration": f.metrics_pre.to_dict(),
                "metrics_post_calibration": f.metrics_post.to_dict(),
                "market_n": f.market_n,
                "market_metrics": f.market.to_dict() if f.market is not None else None,
            }
        )
    return {
        "tour": bundle.tour,
        "model_type": bundle.model_type,
        "training_date": timestamp,
        "data_range": [data_lo, data_hi],
        "features": list(FEATURE_COLUMNS),
        "calibration_method": bundle.production_predictor.method,
        "git_commit": _git_commit(),
        "walk_forward": {
            "n_folds": len(folds),
            "metrics_pre_calibration_aggregate": _aggregate_metrics(folds, "pre"),
            "metrics_post_calibration_aggregate": _aggregate_metrics(folds, "post"),
            "per_fold": per_fold,
        },
    }


def _plot_calibration(
    folds: list[FoldResult],
    out_path: Path,
    tour: str,
    model_type: str,
) -> None:
    """10-bin reliability diagram, model vs market, overlaid on y=x."""
    if not folds:
        return
    all_y = np.concatenate([f.y_true for f in folds])
    all_p = np.concatenate([f.y_prob_post for f in folds])
    edges = np.linspace(0.0, 1.0, 11)
    centers = (edges[:-1] + edges[1:]) / 2.0
    model_mean: list[float] = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        mask = (all_p >= lo) & (all_p <= hi) if i == 9 else (all_p >= lo) & (all_p < hi)
        model_mean.append(float(all_y[mask].mean()) if mask.sum() > 0 else float("nan"))

    market_y_list = [f.market_probs.y_true for f in folds if f.market_probs is not None]
    market_p_list = [f.market_probs.y_prob for f in folds if f.market_probs is not None]
    market_mean: list[float] | None = None
    if market_y_list:
        all_my = np.concatenate(market_y_list)
        all_mp = np.concatenate(market_p_list)
        market_mean = []
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            mask = (all_mp >= lo) & (all_mp <= hi) if i == 9 else (all_mp >= lo) & (all_mp < hi)
            market_mean.append(float(all_my[mask].mean()) if mask.sum() > 0 else float("nan"))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.plot(centers, model_mean, marker="o", label=f"{model_type} (post-cal)")
    if market_mean is not None:
        ax.plot(centers, market_mean, marker="s", label="market (closing odds)")
    ax.set_xlabel("predicted P(p1 wins)")
    ax.set_ylabel("empirical P(p1 wins)")
    ax.set_title(f"{tour} — {model_type} calibration (10 bins, walk-forward)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _render_report(
    bundle: ArtifactBundle,
    metadata: dict[str, object],
    timestamp: str,
) -> str:
    folds = bundle.walk_forward.folds
    lines: list[str] = []
    lines.append(f"# {bundle.tour} — {bundle.model_type}")
    lines.append("")
    lines.append(f"Trained: {timestamp}")
    lines.append(
        f"Data range: {metadata['data_range'][0]} → {metadata['data_range'][1]}"  # type: ignore[index]
    )
    lines.append(f"Calibration method (production): {bundle.production_predictor.method}")
    lines.append(f"Git commit: {metadata['git_commit']}")
    lines.append("")
    lines.append("## Walk-forward aggregate")
    lines.append("")
    pre = metadata["walk_forward"]["metrics_pre_calibration_aggregate"]  # type: ignore[index]
    post = metadata["walk_forward"]["metrics_post_calibration_aggregate"]  # type: ignore[index]
    lines.append("| stage | Brier | log loss | accuracy | n |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| pre-calibration  | {pre['brier']:.4f} | {pre['log_loss']:.4f} | "  # type: ignore[index]
        f"{pre['accuracy']:.4f} | {pre['n_validate_total']} |"  # type: ignore[index]
    )
    lines.append(
        f"| post-calibration | {post['brier']:.4f} | {post['log_loss']:.4f} | "  # type: ignore[index]
        f"{post['accuracy']:.4f} | {post['n_validate_total']} |"  # type: ignore[index]
    )
    lines.append("")
    lines.append("## Per-fold (post-calibration)")
    lines.append("")
    lines.append(
        "| fold | validate year | n_val | Brier | log loss | accuracy | cal method | market n | market Brier |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for f in folds:
        market_brier = f"{f.market.brier:.4f}" if f.market is not None else "—"
        lines.append(
            f"| {f.fold_index} | {f.validate_year} | {f.n_validate} | "
            f"{f.metrics_post.brier:.4f} | {f.metrics_post.log_loss:.4f} | "
            f"{f.metrics_post.accuracy:.4f} | {f.calibration_method} | "
            f"{f.market_n} | {market_brier} |"
        )
    lines.append("")
    lines.append("## Calibration plot")
    lines.append("")
    lines.append("![calibration](calibration_plot.png)")
    lines.append("")
    lines.append("## Methodology notes")
    lines.append("")
    lines.append("- Walk-forward split per fold: train <= V-2, calibrate = V-1, validate = V.")
    lines.append("- Market overlay shown when fold has ≥ 1000 matches with closing odds.")
    lines.append("- Calibration method picked per held-out size: isotonic (≥1000) else Platt.")
    return "\n".join(lines) + "\n"


def _build_roundtrip_fixture(
    predictor: CalibratedPredictor,
    calibrate_slice: FoldSlice,
) -> dict[str, object]:
    """Sample N rows from the calibration set, store features + expected probas."""
    n = min(ROUNDTRIP_FIXTURE_SIZE, calibrate_slice.n)
    rng = np.random.default_rng(seed=0)
    idx = rng.choice(calibrate_slice.n, size=n, replace=False)
    feats = calibrate_slice.features.iloc[idx].copy()
    expected = predictor.predict_proba(feats)[:, 1]
    return {
        "match_ids": calibrate_slice.match_ids[idx].tolist(),
        "features": feats.to_dict(orient="records"),
        "expected_p1_proba": expected.tolist(),
    }


def write_artifacts(
    bundle: ArtifactBundle,
    models_root: Path,
    timestamp: datetime | None = None,
) -> Path:
    """Write the full artifact directory and return its path."""
    ts = timestamp or datetime.now()
    ts_str = ts.strftime("%Y%m%d-%H%M")
    parent = models_root / bundle.tour / bundle.model_type
    out_dir = parent / ts_str
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle.production_predictor, out_dir / "model.joblib")

    metadata = _build_metadata(bundle, timestamp=ts.isoformat(timespec="seconds"))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    _plot_calibration(
        folds=bundle.walk_forward.folds,
        out_path=out_dir / "calibration_plot.png",
        tour=bundle.tour,
        model_type=bundle.model_type,
    )

    report = _render_report(bundle, metadata, timestamp=ts.isoformat(timespec="seconds"))
    (out_dir / "report.md").write_text(report)

    fixture = _build_roundtrip_fixture(
        predictor=bundle.production_predictor,
        calibrate_slice=bundle.production_calibrate,
    )
    (out_dir / "roundtrip_fixture.json").write_text(json.dumps(fixture, indent=2, default=str))

    _update_latest_symlink(parent=parent, target_name=ts_str)
    logger.info("wrote artifact bundle: %s", out_dir)
    return out_dir


def _update_latest_symlink(parent: Path, target_name: str) -> None:
    latest = parent / "latest"
    tmp = parent / f".latest.{target_name}.tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target_name)
    tmp.replace(latest)


def comparison_summary(
    elo_result: WalkForwardResult,
    lgbm_result: WalkForwardResult,
    last_n: int,
) -> dict[str, object]:
    """Summarise whether LightGBM beats Elo on recent folds (per Default #8)."""
    elo_brier = elo_result.aggregate_brier_post(last_n=last_n)
    lgbm_brier = lgbm_result.aggregate_brier_post(last_n=last_n)
    return {
        "tour": elo_result.tour,
        "last_n_folds": last_n,
        "elo_brier_post": elo_brier,
        "lightgbm_brier_post": lgbm_brier,
        "lightgbm_beats_elo": (
            lgbm_brier < elo_brier if not (np.isnan(elo_brier) or np.isnan(lgbm_brier)) else False
        ),
        "delta": (elo_brier - lgbm_brier)
        if not (np.isnan(elo_brier) or np.isnan(lgbm_brier))
        else float("nan"),
    }
