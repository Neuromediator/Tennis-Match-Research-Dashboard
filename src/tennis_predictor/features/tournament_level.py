"""Tournament-level normalization for the feature layer.

Maps Sackmann's raw `tourney_level` field to the canonical 7-category
taxonomy used by `FeatureVector`:

    {Slam, M1000, ATP500, ATP250, WTA500, WTA250, Finals}

Davis Cup (`D`) and Olympics (`O`) are intentionally NOT in the canonical
set — they are excluded from training eligibility (different competitive
dynamics: team format, sporadic scheduling, non-standard surfaces).
`normalize_tournament_level` returns `None` for them, and the orchestrator
treats `None` as "ineligible — skip training_features row, do not feed
state for label purposes (the match still feeds Elo/H2H/etc as a regular
match in chronological replay)."

ATP's `tourney_level='A'` collapses both 500-level and 250-level events.
We disambiguate via a hand-curated `tourney_name` whitelist (modern ATP 500
plus pre-2009 International Series Gold equivalents). WTA's coding is
cleaner: `PM`/`T1` → M1000, `P`/`T2` → WTA500, `I`/`T3..T5` → WTA250.

The post-2021 WTA `W` bucket is a catch-all (modern smaller events plus
historical pre-Tier-system rows). We default it to `WTA250`. Refinement
is possible later if feature importance suggests we are losing signal.
"""

from __future__ import annotations

from tennis_predictor.features.schema import TournamentLevel

# Tour-level events historically classified as ATP 500 (modern ATP 500
# series + pre-2009 ATP International Series Gold equivalents). Names match
# the raw Sackmann `tourney_name` (case-insensitive after lowering+strip).
# Anything not in this set with tourney_level='A' falls through to ATP250.
ATP_500_TOURNAMENTS: frozenset[str] = frozenset(
    {
        "rotterdam",
        "rio de janeiro",
        "rio",
        "acapulco",
        "dubai",
        "barcelona",
        "queen's club",
        "queens club",
        "queens",
        "halle",
        "hamburg",
        "washington",
        "beijing",
        "tokyo",
        "vienna",
        "basel",
        "memphis",
        "indianapolis",
        "milan",
    }
)


def _norm_name(name: str | None) -> str:
    return "" if name is None else name.strip().lower()


def _is_wta_125_event(tourney_name: str | None) -> bool:
    """WTA 125 events are the women's Challenger-equivalent — 125 ranking
    points, smaller draws, mostly developmental. Sackmann normally stores
    them in qual_chall files but occasional rows leak into main; we filter
    by name as a safety net.

    Pattern: tourney_name contains ' 125' as a separate token (avoids false
    positives like 'W125 Charleston' — those are also WTA 125 by design, so
    we accept that match too).
    """
    if tourney_name is None:
        return False
    return "125" in tourney_name


def normalize_tournament_level(
    tour: str,
    raw_level: str | None,
    tourney_name: str | None = None,
) -> TournamentLevel | None:
    """Map (tour, raw Sackmann tourney_level, tourney_name) to canonical level.

    Returns None for:
      - Davis Cup (`D`), Olympics (`O`) — explicitly excluded eligibility-wise.
      - Out-of-scope WTA codes (`CC`, `E`, `50+H`, `35+H`, `J`).
      - WTA 125 events (Challenger-equivalent) — detected by tourney_name
        even if Sackmann mis-classifies the tier.
      - Unrecognized levels — defensive, caller filters them out.

    Tour string is `"ATP"` or `"WTA"` (case-sensitive — matches the value
    stored in `matches.tour`).
    """
    if raw_level is None:
        return None

    # WTA 125 events are women's Challenger-equivalent — out of scope
    # regardless of how Sackmann tagged the tier.
    if tour == "WTA" and _is_wta_125_event(tourney_name):
        return None

    code = raw_level.strip()

    # Grand Slams + year-end Finals are common to both tours.
    if code == "G":
        return "Slam"
    if code == "F":
        return "Finals"
    if code in ("D", "O"):
        return None  # Davis Cup / Olympics — excluded by user decision.

    if tour == "ATP":
        if code == "M":
            return "M1000"
        if code == "A":
            return "ATP500" if _norm_name(tourney_name) in ATP_500_TOURNAMENTS else "ATP250"
        return None

    if tour == "WTA":
        # Modern coding (post-2009)
        if code == "PM":
            return "M1000"
        if code == "P":
            return "WTA500"
        if code == "I":
            return "WTA250"
        # Legacy Tier system (pre-2009)
        if code == "T1":
            return "M1000"
        if code == "T2":
            return "WTA500"
        if code in ("T3", "T4", "T5"):
            return "WTA250"
        # Post-2021 catch-all bucket
        if code == "W":
            return "WTA250"
        # Out-of-scope WTA codes
        return None

    return None
