"""Tests for the tournament_level normalizer.

Locks in:
- Davis Cup (D) and Olympics (O) return None (excluded by user decision).
- ATP `A` is disambiguated into ATP500 vs ATP250 via the whitelist.
- WTA legacy Tier codes (T1..T5) map correctly across the 2009 rebrand.
- Out-of-scope WTA codes (CC/E/seniors/juniors) return None.
"""

from __future__ import annotations

from tennis_predictor.features.tournament_level import (
    ATP_500_TOURNAMENTS,
    normalize_tournament_level,
)


def test_grand_slam_both_tours() -> None:
    assert normalize_tournament_level("ATP", "G", "Wimbledon") == "Slam"
    assert normalize_tournament_level("WTA", "G", "US Open") == "Slam"


def test_year_end_finals_both_tours() -> None:
    assert normalize_tournament_level("ATP", "F", "Tour Finals") == "Finals"
    assert normalize_tournament_level("WTA", "F", "WTA Finals") == "Finals"


def test_davis_cup_excluded() -> None:
    """User decision: Davis Cup excluded from training_features."""
    assert normalize_tournament_level("ATP", "D", "Davis Cup") is None


def test_olympics_excluded() -> None:
    """User decision: Olympics excluded from training_features."""
    assert normalize_tournament_level("ATP", "O", "Tokyo Olympics") is None
    assert normalize_tournament_level("WTA", "O", "Rio Olympics") is None


def test_atp_masters_1000() -> None:
    assert normalize_tournament_level("ATP", "M", "Indian Wells Masters") == "M1000"
    assert normalize_tournament_level("ATP", "M", "Paris Masters") == "M1000"


def test_atp_500_tournaments_classified_as_500() -> None:
    for name in ("Rotterdam", "Dubai", "Barcelona", "Halle", "Vienna", "Basel"):
        assert normalize_tournament_level("ATP", "A", name) == "ATP500", name


def test_atp_a_unknown_falls_through_to_250() -> None:
    for name in ("Auckland", "Delray Beach", "Doha", "Marseille", "Stockholm"):
        assert normalize_tournament_level("ATP", "A", name) == "ATP250", name


def test_atp_a_case_insensitive_for_500_whitelist() -> None:
    assert normalize_tournament_level("ATP", "A", "ROTTERDAM") == "ATP500"
    assert normalize_tournament_level("ATP", "A", "  rotterdam  ") == "ATP500"


def test_wta_premier_mandatory_is_m1000() -> None:
    assert normalize_tournament_level("WTA", "PM", "Indian Wells") == "M1000"
    assert normalize_tournament_level("WTA", "PM", "Madrid") == "M1000"


def test_wta_premier_is_500() -> None:
    assert normalize_tournament_level("WTA", "P", "Doha") == "WTA500"


def test_wta_international_is_250() -> None:
    assert normalize_tournament_level("WTA", "I", "Hobart") == "WTA250"


def test_wta_legacy_tier_system_maps_correctly() -> None:
    """Pre-2009 Tier I = M1000 equivalent; T2 = WTA500; T3-T5 = WTA250."""
    assert normalize_tournament_level("WTA", "T1", "Berlin") == "M1000"
    assert normalize_tournament_level("WTA", "T2", "Charleston") == "WTA500"
    assert normalize_tournament_level("WTA", "T3", "Estoril") == "WTA250"
    assert normalize_tournament_level("WTA", "T4", "Auckland") == "WTA250"
    assert normalize_tournament_level("WTA", "T5", "Hobart") == "WTA250"


def test_wta_catchall_w_defaults_to_250() -> None:
    assert normalize_tournament_level("WTA", "W", "Strasbourg") == "WTA250"


def test_wta_125_events_excluded() -> None:
    """WTA 125 events are women's Challenger-equivalent — never tour-level."""
    # Real event from our DB that Sackmann mis-classified into main.
    assert normalize_tournament_level("WTA", "W", "Buenos Aires 125") is None
    # Other common naming conventions.
    assert normalize_tournament_level("WTA", "W", "Charleston 2 125") is None
    assert normalize_tournament_level("WTA", "I", "Tampico 125") is None
    # Doesn't affect events without 125 in the name.
    assert normalize_tournament_level("WTA", "W", "Buenos Aires") == "WTA250"


def test_wta_out_of_scope_codes_return_none() -> None:
    for code in ("CC", "E", "50+H", "35+H", "J"):
        assert normalize_tournament_level("WTA", code, "anything") is None, code


def test_unknown_code_returns_none() -> None:
    """G (Slam) and F (Finals) are tour-agnostic; everything else needs a
    known (tour, code) pair."""
    assert normalize_tournament_level("ATP", "ZZ", "anything") is None
    assert normalize_tournament_level("ATP", None, "anything") is None
    assert normalize_tournament_level("ITF", "M", "anything") is None


def test_slam_and_finals_are_tour_agnostic() -> None:
    """Grand Slams and year-end Finals are categorically the same event
    across the men's and women's draws — no per-tour split needed."""
    assert normalize_tournament_level("ATP", "G", "Wimbledon") == "Slam"
    assert normalize_tournament_level("WTA", "G", "Wimbledon") == "Slam"
    assert normalize_tournament_level("ATP", "F", "Tour Finals") == "Finals"
    assert normalize_tournament_level("WTA", "F", "WTA Finals") == "Finals"


def test_atp_500_whitelist_lowercase_invariant() -> None:
    """Sanity: every entry is lowercased & stripped — case-insensitive lookup
    relies on this invariant."""
    for name in ATP_500_TOURNAMENTS:
        assert name == name.lower(), f"ATP 500 entry not lowercased: {name!r}"
        assert name == name.strip(), f"ATP 500 entry has whitespace: {name!r}"
