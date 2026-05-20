"""Tests for surface normalizer and indoor whitelist."""

from __future__ import annotations

from tennis_predictor.features.indoor_tournaments import INDOOR_TOURNAMENTS, is_indoor
from tennis_predictor.features.surface import normalize_surface


def test_clay_normalizes_regardless_of_case() -> None:
    assert normalize_surface("clay", "Roland Garros") == "Clay"
    assert normalize_surface("Clay", "Roland Garros") == "Clay"
    assert normalize_surface("  CLAY  ", None) == "Clay"


def test_grass_normalizes() -> None:
    assert normalize_surface("Grass", "Wimbledon") == "Grass"


def test_carpet_maps_to_indoor_hard() -> None:
    """Carpet was always indoor — merges into IHard for rating continuity."""
    assert normalize_surface("Carpet", "Some old indoor event") == "IHard"


def test_outdoor_hard_stays_hard() -> None:
    """A hard-court event NOT in the indoor whitelist remains outdoor."""
    assert normalize_surface("Hard", "US Open") == "Hard"
    assert normalize_surface("Hard", "Cincinnati Masters") == "Hard"
    assert normalize_surface("Hard", "Australian Open") == "Hard"


def test_indoor_hard_promotes_to_ihard() -> None:
    """Known indoor hard tournaments lift Hard → IHard."""
    assert normalize_surface("Hard", "Paris Masters") == "IHard"
    assert normalize_surface("Hard", "Tour Finals") == "IHard"
    assert normalize_surface("Hard", "Rotterdam") == "IHard"
    assert normalize_surface("Hard", "Vienna") == "IHard"
    assert normalize_surface("Hard", "Linz") == "IHard"


def test_indoor_whitelist_only_applies_to_hard() -> None:
    """Stuttgart's WTA event is indoor RED CLAY — must stay 'Clay' even though
    'Porsche Grand Prix' is in the indoor whitelist for non-Stuttgart years.
    The surface field is authoritative; whitelist only upgrades Hard."""
    assert normalize_surface("Clay", "Porsche Grand Prix") == "Clay"


def test_null_surface_returns_none() -> None:
    assert normalize_surface(None, "anything") is None


def test_unrecognized_surface_returns_none() -> None:
    assert normalize_surface("Sand", "Beach Tennis") is None


def test_is_indoor_handles_none_and_unknown() -> None:
    assert is_indoor(None) is False
    assert is_indoor("Random Outdoor Open") is False


def test_indoor_whitelist_is_lowercased_consistently() -> None:
    """Sanity: every entry is already lowercased — case-insensitive lookup
    relies on this invariant."""
    for name in INDOOR_TOURNAMENTS:
        assert name == name.lower(), f"Indoor whitelist entry not lowercased: {name!r}"
        assert name == name.strip(), f"Indoor whitelist entry has whitespace: {name!r}"
