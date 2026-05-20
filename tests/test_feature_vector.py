"""FeatureVector contract tests (Pydantic-level).

Validates field-level invariants: required vs nullable, bounded ranges,
literal categorical sets, immutability, no extras allowed.

Negative tests use `model_validate(dict)` so pyright doesn't complain about
intentionally-invalid keyword arguments — the validation happens at runtime
and is the whole point of the test.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tennis_predictor.features.schema import FeatureVector


def _valid_kwargs() -> dict[str, Any]:
    """Minimal valid FeatureVector — every required field set, all
    optional fields default to None."""
    return {
        "elo_p1_surface": 1612.4,
        "elo_p2_surface": 1488.9,
        "elo_diff_surface": 123.5,
        "h2h_p1_wins": 2,
        "h2h_p2_wins": 1,
        "fatigue_matches_7d_p1": 1,
        "fatigue_matches_7d_p2": 2,
        "fatigue_sets_14d_p1": 5,
        "fatigue_sets_14d_p2": 7,
        "rank_p1": 12,
        "rank_p2": 47,
        "rank_diff": -35,
        "tournament_level": "M1000",
        "best_of": 3,
        "surface": "IHard",
    }


def test_minimal_valid_vector_constructs() -> None:
    fv = FeatureVector.model_validate(_valid_kwargs())
    assert fv.tournament_level == "M1000"
    assert fv.win_pct_last10_p1 is None  # nullable default
    assert fv.h2h_recency_days is None


def test_frozen_model_rejects_mutation() -> None:
    fv = FeatureVector.model_validate(_valid_kwargs())
    with pytest.raises(ValidationError):
        fv.rank_p1 = 99  # type: ignore[misc]


def test_extra_fields_rejected() -> None:
    kwargs = _valid_kwargs() | {"some_new_feature": 0.5}
    with pytest.raises(ValidationError):
        FeatureVector.model_validate(kwargs)


def test_win_pct_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        FeatureVector.model_validate(_valid_kwargs() | {"win_pct_last10_p1": 1.1})

    with pytest.raises(ValidationError):
        FeatureVector.model_validate(_valid_kwargs() | {"win_pct_last10_p1": -0.01})


def test_rank_sentinel_upper_bound_enforced() -> None:
    """rank_p1 above the 9999 sentinel signals a bug in the ranking lookup."""
    with pytest.raises(ValidationError):
        FeatureVector.model_validate(_valid_kwargs() | {"rank_p1": 10_000})


def test_rank_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        FeatureVector.model_validate(_valid_kwargs() | {"rank_p1": 0})


def test_surface_literal_set_locked() -> None:
    """Carpet must NOT be accepted as a standalone surface — Phase 3
    canonical set is {Hard, IHard, Clay, Grass}. Carpet maps to IHard
    upstream in the normalizer."""
    with pytest.raises(ValidationError):
        FeatureVector.model_validate(_valid_kwargs() | {"surface": "Carpet"})


def test_tournament_level_literal_set_locked() -> None:
    """Davis Cup (D) and Olympics (O) are excluded eligibility-wise, so
    they must not appear as valid tournament_level values either."""
    for bad in ("DavisCup", "Olympics", "D", "O", "Challenger"):
        with pytest.raises(ValidationError):
            FeatureVector.model_validate(_valid_kwargs() | {"tournament_level": bad})


def test_best_of_only_3_or_5() -> None:
    with pytest.raises(ValidationError):
        FeatureVector.model_validate(_valid_kwargs() | {"best_of": 4})


def test_h2h_recency_days_non_negative_when_set() -> None:
    with pytest.raises(ValidationError):
        FeatureVector.model_validate(_valid_kwargs() | {"h2h_recency_days": -1})


def test_h2h_wins_non_negative() -> None:
    with pytest.raises(ValidationError):
        FeatureVector.model_validate(_valid_kwargs() | {"h2h_p1_wins": -1})
