---
name: model-training
description: Use when training a model, modifying validation strategy, choosing a calibration method, or producing a model report. Establishes walk-forward discipline and required artifacts.
---

# Model training

## Models trained (v1)

Six total, two per type, one per tour:

| Tour | Type | Notes |
|---|---|---|
| ATP | Surface-Elo baseline | No learning — pure rating-based prediction. |
| ATP | Logistic regression | Linear baseline. |
| ATP | LightGBM | Gradient boosting. |
| WTA | (same three) | Trained independently. |

ATP and WTA are **never** mixed in training. Surfaces play differently across tours.

## Walk-forward validation

Train on years up to `Y`. Validate on year `Y+1`. Advance `Y` by one, repeat. No random shuffling. No future leakage in cross-validation folds.

Reported metrics per fold and averaged:

- **Brier score** (lower is better) — the headline metric.
- **Log loss** — secondary calibration-aware metric.
- **Accuracy** — reported but not the headline.
- **Accuracy by ranking gap** — buckets such as `[0, 50)`, `[50, 200)`, `[200, +∞)` — sanity check for overfitting to favorites.
- **Calibration plot** with 10 bins.

## Calibration

Post-processing step using `CalibratedClassifierCV` or equivalent on a **held-out** calibration set (separate from train and validation).

**Decision rule:**

- Held-out calibration set size ≥ 1000 → **isotonic regression**.
- Less than 1000 → **Platt (sigmoid)**. Isotonic overfits the calibration curve on small samples.

Report both pre- and post-calibration Brier score in the model report.

## Market benchmark

For every validation fold, also compute the Brier score and calibration of the market's closing-price implied probabilities for the same matches (where available). The model report includes an overlaid calibration plot: our model vs the market.

Honest framing: "our model is approaching market calibration." Not "we beat the market."

## Required artifacts per training run

Under `models/<tour>/<model_type>/<YYYYMMDD-HHMM>/`:

- `model.joblib` — serialized sklearn/lightgbm pipeline.
- `metadata.json`:
  ```
  {
    "training_date": "...",
    "data_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
    "features": [...],
    "metrics_pre_calibration": {...},
    "metrics_post_calibration": {...},
    "calibration_method": "isotonic" | "platt",
    "git_commit": "..."
  }
  ```
- `report.md` — terse Markdown summary.
- `calibration_plot.png` — model and market overlaid.

`models/<tour>/<model_type>/latest` is a symlink to the most recent run.

## Round-trip serialization test

A test loads each saved model and predicts on a fixed fixture, asserting the predictions match the values saved at training time. Catches scikit-learn / lightgbm version drift.

## What NOT to do

- Do not optimize hyperparameters on the validation set used for metrics reporting. Use a separate inner CV split.
- Do not use accuracy as the loss objective for tuning.
- Do not omit the calibration plot from the report.
- Do not train without a market-benchmark comparison (when market data is available for that period).
