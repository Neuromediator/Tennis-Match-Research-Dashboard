"""Surface normalization for the feature layer.

Maps Sackmann's raw `surface` field (case-mixed `{Hard, Clay, clay, Grass,
Carpet}`, plus NULL) to the canonical 4-category taxonomy used by
`FeatureVector`:

    {Hard, IHard, Clay, Grass}

Indoor hard is split out from outdoor hard via the
`indoor_tournaments.is_indoor` whitelist. Carpet (always indoor in tour
history) merges into `IHard` — keeps the IHard rating continuous through
the 2009 transition when carpet events switched to indoor hard.

Returns `None` for unmappable inputs (NULL surface, unrecognized value).
Callers filter those out at the eligibility check — they do not feed
state or labels.
"""

from __future__ import annotations

from tennis_predictor.features.indoor_tournaments import is_indoor
from tennis_predictor.features.schema import Surface


def normalize_surface(
    raw_surface: str | None,
    tourney_name: str | None,
) -> Surface | None:
    """Map raw Sackmann surface + tourney_name to canonical Surface.

    Returns None if the raw surface is NULL or unrecognized — those matches
    must be excluded from state AND from training_features.
    """
    if raw_surface is None:
        return None
    s = raw_surface.strip().lower()
    if s == "clay":
        return "Clay"
    if s == "grass":
        return "Grass"
    if s == "carpet":
        return "IHard"
    if s == "hard":
        return "IHard" if is_indoor(tourney_name) else "Hard"
    return None
