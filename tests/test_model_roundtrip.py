"""Round-trip serialization test for saved model artifacts.

For every ``models/<tour>/<model_type>/latest`` directory, load
``model.joblib`` and the 16-row ``roundtrip_fixture.json`` saved alongside
it, run ``predict_proba`` on the fixture rows, and assert the predictions
match the reference values within a tight tolerance.

Mandated by ``.claude/skills/model-training/SKILL.md`` (LightGBM /
scikit-learn version drift can silently change predictions; this test
catches it).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from tennis_predictor import config
from tennis_predictor.models.calibration import CalibratedPredictor
from tennis_predictor.models.feature_spec import CATEGORICAL_COLUMNS, CATEGORY_VALUES


def _iter_latest_dirs(models_root: Path):
    if not models_root.exists():
        return
    for tour_dir in sorted(models_root.iterdir()):
        if not tour_dir.is_dir():
            continue
        for model_dir in sorted(tour_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            latest = model_dir / "latest"
            if latest.exists():
                yield tour_dir.name, model_dir.name, latest.resolve()


def _restore_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast each categorical column to its declared levels.

    Skips columns not present in `df` so the test tolerates feature-schema
    bumps: a Phase 4.1 v2 fixture has `hand_p1` / `hand_p2`, a Phase 4 v1
    fixture does not. Either way the predictor only sees columns it was
    trained on."""
    out = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = pd.Categorical(out[col], categories=CATEGORY_VALUES[col])
    return out


@pytest.mark.parametrize("artifact", list(_iter_latest_dirs(config.MODELS_DIR)))
def test_roundtrip_predictions_match(artifact: tuple[str, str, Path]) -> None:
    tour, model_type, latest = artifact
    model_path = latest / "model.joblib"
    fixture_path = latest / "roundtrip_fixture.json"
    if not (model_path.exists() and fixture_path.exists()):
        pytest.skip(f"missing artifact files for {tour}/{model_type}")

    predictor: CalibratedPredictor = joblib.load(model_path)
    fixture = json.loads(fixture_path.read_text())
    feats = _restore_categoricals(pd.DataFrame(fixture["features"]))
    expected = np.array(fixture["expected_p1_proba"], dtype=float)
    actual = predictor.predict_proba(feats)[:, 1]
    np.testing.assert_allclose(actual, expected, atol=1e-9)
