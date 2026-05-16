"""Smoke test: package imports and exposes version.

Exists primarily so CI's pytest step does not exit 5 ("no tests collected")
during the bootstrap phase before real tests land.
"""

import tennis_predictor


def test_package_imports() -> None:
    assert tennis_predictor.__version__
