"""LightGBM trainer.

Sensible hardcoded hyperparameters + early stopping on the calibration
set's log loss (Brier and log loss are both proper scoring rules and pick
near-identical iteration counts in practice; log loss is LightGBM's
native objective).

Why calibration set for early stopping:
- Train set is used to fit trees.
- The calibration set is later used to fit the post-hoc isotonic / Platt
  calibrator (one-dimensional). Using it twice — once for an integer
  early-stopping choice, once for a monotone 1-D calibrator — is the
  standard pragmatic pattern. Validation set stays fully held out for the
  reported metrics.

Categorical features are passed to LightGBM via the ``categorical_feature``
parameter so that ``tournament_level`` and ``surface`` are treated natively
(no one-hot expansion).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from tennis_predictor.models.feature_spec import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


@dataclass(frozen=True)
class LightGBMHyperparams:
    n_estimators: int = 1500
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 75
    random_state: int = 42


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: npt.NDArray[np.int_],
    X_eval: pd.DataFrame,
    y_eval: npt.NDArray[np.int_],
    hyperparams: LightGBMHyperparams | None = None,
) -> lgb.LGBMClassifier:
    """Fit a LightGBM binary classifier with early stopping on ``X_eval``.

    Both ``X_train`` and ``X_eval`` must have the same column ordering as
    ``FEATURE_COLUMNS`` and matching pandas ``Categorical`` dtypes on the
    categorical columns.
    """
    hp = hyperparams or LightGBMHyperparams()
    model = lgb.LGBMClassifier(
        n_estimators=hp.n_estimators,
        learning_rate=hp.learning_rate,
        num_leaves=hp.num_leaves,
        min_child_samples=hp.min_child_samples,
        feature_fraction=hp.feature_fraction,
        bagging_fraction=hp.bagging_fraction,
        bagging_freq=hp.bagging_freq,
        reg_alpha=hp.reg_alpha,
        reg_lambda=hp.reg_lambda,
        random_state=hp.random_state,
        objective="binary",
        verbose=-1,
        n_jobs=-1,
    )
    callbacks = [lgb.early_stopping(hp.early_stopping_rounds, verbose=False)]
    model.fit(
        X_train[list(FEATURE_COLUMNS)],
        y_train,
        eval_set=[(X_eval[list(FEATURE_COLUMNS)], y_eval)],
        eval_metric="binary_logloss",
        categorical_feature=list(CATEGORICAL_COLUMNS),
        callbacks=callbacks,
    )
    return model
