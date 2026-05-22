"""Unit tests for the Phase 4.1 player-metadata helpers.

Three pure functions — all behavior is captured here. No DuckDB, no
fixtures beyond inline values.
"""

from __future__ import annotations

from datetime import date

import pytest

from tennis_predictor.features.player_metadata import (
    PEAK_AGE,
    compute_age,
    compute_age_vs_peak,
    compute_height_diff,
)


class TestComputeAge:
    def test_returns_none_when_dob_missing(self) -> None:
        assert compute_age(None, date(2024, 6, 1)) is None

    def test_round_birthday(self) -> None:
        # 30 years to the day, allowing for Julian-year leap-day drift.
        age = compute_age(date(1990, 6, 1), date(2020, 6, 1))
        assert age is not None
        assert abs(age - 30.0) < 0.05  # within ~18 days of 30.0

    def test_subyear_resolution_is_kept(self) -> None:
        """LightGBM benefits from the decimal — verify we don't int-truncate."""
        age_at_midyear = compute_age(date(2000, 1, 1), date(2024, 7, 1))
        assert age_at_midyear is not None
        # ~24.5 years, not 24.
        assert 24.3 < age_at_midyear < 24.7

    def test_negative_age_when_match_before_dob(self) -> None:
        """Defensive: the function shouldn't crash if a bad input slips
        through; downstream Pydantic bounds (age >= 10) will reject it."""
        age = compute_age(date(2020, 1, 1), date(2019, 1, 1))
        assert age is not None
        assert age < 0


class TestComputeAgeVsPeak:
    def test_returns_none_when_age_none(self) -> None:
        assert compute_age_vs_peak(None, "ATP") is None

    def test_atp_at_peak_is_zero(self) -> None:
        assert compute_age_vs_peak(26.0, "ATP") == 0.0

    def test_wta_at_peak_is_zero(self) -> None:
        assert compute_age_vs_peak(24.0, "WTA") == 0.0

    def test_signed_negative_for_young_player(self) -> None:
        """22-year-old ATP player -> -4 (rising)."""
        assert compute_age_vs_peak(22.0, "ATP") == pytest.approx(-4.0)

    def test_signed_positive_for_aging_player(self) -> None:
        """34-year-old ATP player -> +8 (declining)."""
        assert compute_age_vs_peak(34.0, "ATP") == pytest.approx(8.0)

    def test_different_peaks_per_tour(self) -> None:
        """The same raw age maps differently across tours — verifying we
        actually consult `PEAK_AGE[tour]`."""
        atp = compute_age_vs_peak(25.0, "ATP")
        wta = compute_age_vs_peak(25.0, "WTA")
        assert atp == pytest.approx(-1.0)
        assert wta == pytest.approx(1.0)

    def test_unknown_tour_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tour"):
            compute_age_vs_peak(25.0, "ITF")

    def test_peak_age_constants_match_design(self) -> None:
        """If someone tunes the peak constants, the family expectation in
        the FeatureVector range checks needs to be re-examined too. Pin
        the published defaults explicitly."""
        assert PEAK_AGE == {"ATP": 26.0, "WTA": 24.0}


class TestComputeHeightDiff:
    def test_none_when_either_missing(self) -> None:
        assert compute_height_diff(None, 180) is None
        assert compute_height_diff(190, None) is None
        assert compute_height_diff(None, None) is None

    def test_signed_diff(self) -> None:
        assert compute_height_diff(198, 175) == 23
        assert compute_height_diff(175, 198) == -23

    def test_zero_diff_for_same_height(self) -> None:
        assert compute_height_diff(185, 185) == 0
