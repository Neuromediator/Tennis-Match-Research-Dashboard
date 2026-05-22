"""Train the four production models (Elo baseline + LightGBM, per tour).

Usage:
    uv run python scripts/train_models.py
    uv run python scripts/train_models.py --tour ATP --model-type lightgbm
    uv run python scripts/train_models.py --validate-years 2023 2024 2025

For each (tour, model_type) pair the script:

1. Runs walk-forward validation across the requested validation years
   (default: 2018..2025 — eight folds).
2. Fits the production model on years ≤ ``last_full_year - 1``, calibrated
   on ``last_full_year`` (default: 2024 train / 2025 calibrate).
3. Writes an artifact bundle to
   ``models/<tour>/<model_type>/<YYYYMMDD-HHMM>/`` and updates the
   ``latest`` symlink.

The Elo baseline is also calibrated post-hoc — the formula alone is well
ordered but not necessarily well centred on empirical win rate; isotonic
nudges it onto the diagonal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb

from tennis_predictor import config
from tennis_predictor.data.db import open_connection
from tennis_predictor.models.artifacts import (
    ArtifactBundle,
    comparison_summary,
    write_artifacts,
)
from tennis_predictor.models.calibration import CalibratedPredictor
from tennis_predictor.models.data import (
    FoldSlice,
    build_folds,
    build_production_split,
    load_training_frame,
)
from tennis_predictor.models.elo_baseline import SurfaceEloBaseline
from tennis_predictor.models.lightgbm_trainer import train_lightgbm
from tennis_predictor.models.walk_forward import run_walk_forward

ModelType = Literal["elo", "lightgbm"]

DEFAULT_VALIDATE_YEARS: list[int] = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
DEFAULT_LAST_FULL_YEAR: int = 2025
DEFAULT_TOURS: list[str] = ["ATP", "WTA"]
DEFAULT_MODEL_TYPES: list[ModelType] = ["elo", "lightgbm"]
DEFAULT_COMPARISON_LAST_N: int = 5

logger = logging.getLogger("train_models")


def _train_elo(train: FoldSlice, _calibrate: FoldSlice) -> SurfaceEloBaseline:
    return SurfaceEloBaseline().fit(train.features, train.labels)


def _train_lgbm(train: FoldSlice, calibrate: FoldSlice):
    return train_lightgbm(
        X_train=train.features,
        y_train=train.labels,
        X_eval=calibrate.features,
        y_eval=calibrate.labels,
    )


def _make_train_fn(model_type: ModelType):
    if model_type == "elo":
        return _train_elo
    return _train_lgbm


def _fit_production(
    model_type: ModelType,
    train: FoldSlice,
    calibrate: FoldSlice,
) -> CalibratedPredictor:
    base = _make_train_fn(model_type)(train, calibrate)
    return CalibratedPredictor.fit(base=base, X_cal=calibrate.features, y_cal=calibrate.labels)  # type: ignore[arg-type]


def _train_one(
    conn: duckdb.DuckDBPyConnection,
    tour: str,
    model_type: ModelType,
    validate_years: list[int],
    last_full_year: int,
    models_root: Path,
    ts: datetime,
) -> tuple[Path, object]:
    logger.info("loading training_features for %s", tour)
    df = load_training_frame(conn, tour=tour)
    folds = list(build_folds(df, validate_years=validate_years))
    logger.info("%s: %d walk-forward folds resolved", tour, len(folds))

    wf_result = run_walk_forward(
        folds=folds,
        train_fn=_make_train_fn(model_type),
        tour=tour,
        model_type=model_type,
        conn=conn,
    )

    prod_train, prod_calibrate = build_production_split(df, last_full_year=last_full_year)
    logger.info(
        "%s/%s: fitting production model (train n=%d, cal n=%d)",
        tour,
        model_type,
        prod_train.n,
        prod_calibrate.n,
    )
    predictor = _fit_production(model_type, prod_train, prod_calibrate)

    bundle = ArtifactBundle(
        tour=tour,
        model_type=model_type,
        walk_forward=wf_result,
        production_predictor=predictor,
        production_train=prod_train,
        production_calibrate=prod_calibrate,
    )
    out_dir = write_artifacts(bundle=bundle, models_root=models_root, timestamp=ts)
    return out_dir, wf_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tour", choices=["ATP", "WTA"], action="append")
    parser.add_argument("--model-type", choices=["elo", "lightgbm"], action="append")
    parser.add_argument(
        "--validate-years",
        type=int,
        nargs="+",
        default=DEFAULT_VALIDATE_YEARS,
        help="Validation years for walk-forward (default: 2018-2025).",
    )
    parser.add_argument(
        "--last-full-year",
        type=int,
        default=DEFAULT_LAST_FULL_YEAR,
        help="Year used for production calibration; train uses <= last_full_year-1.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=config.MODELS_DIR,
        help="Output root directory for artifacts.",
    )
    parser.add_argument(
        "--comparison-last-n",
        type=int,
        default=DEFAULT_COMPARISON_LAST_N,
        help="Number of most-recent folds used for the LightGBM-vs-Elo Brier comparison.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    tours: list[str] = args.tour or DEFAULT_TOURS
    model_types: list[ModelType] = args.model_type or DEFAULT_MODEL_TYPES
    ts = datetime.now()
    args.models_root.mkdir(parents=True, exist_ok=True)

    summary_by_tour: dict[str, dict[ModelType, object]] = {}
    with open_connection(read_only=True) as conn:
        for tour in tours:
            summary_by_tour[tour] = {}
            for model_type in model_types:
                logger.info("=== %s / %s ===", tour, model_type)
                _, wf_result = _train_one(
                    conn=conn,
                    tour=tour,
                    model_type=model_type,
                    validate_years=args.validate_years,
                    last_full_year=args.last_full_year,
                    models_root=args.models_root,
                    ts=ts,
                )
                summary_by_tour[tour][model_type] = wf_result

    print("\n=== Walk-forward summary (post-calibration, sample-weighted Brier) ===")
    for tour, by_type in summary_by_tour.items():
        line_parts = [f"{tour}:"]
        for model_type, wf in by_type.items():
            if hasattr(wf, "aggregate_brier_post"):
                line_parts.append(
                    f"{model_type}={wf.aggregate_brier_post(args.comparison_last_n):.4f}"  # type: ignore[union-attr]
                )
        print("  " + " ".join(line_parts))
        if "elo" in by_type and "lightgbm" in by_type:
            cmp = comparison_summary(
                elo_result=by_type["elo"],  # type: ignore[arg-type]
                lgbm_result=by_type["lightgbm"],  # type: ignore[arg-type]
                last_n=args.comparison_last_n,
            )
            verdict = "✅ LightGBM beats Elo" if cmp["lightgbm_beats_elo"] else "⚠ baseline wins"
            print(
                f"    last {cmp['last_n_folds']} folds: "
                f"Elo Brier {cmp['elo_brier_post']:.4f} vs LightGBM {cmp['lightgbm_brier_post']:.4f}"
                f" (Δ={cmp['delta']:+.4f}) — {verdict}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
