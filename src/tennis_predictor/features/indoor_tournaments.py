"""Whitelist of tour-level tournaments played on indoor hard courts.

Used by `surface.normalize_surface` to upgrade `surface='Hard'` rows to
`IHard` when the tournament is known to be played indoors. Carpet rows
are also `IHard` but are handled upstream — carpet was always indoor.

Matching is case-insensitive against the raw Sackmann `tourney_name`.
The set is curated by hand from public tour scheduling history; please
review when adding events.

Notes on inclusion criteria:
- ATP Finals (rotates venue but always indoor since the Masters Cup era).
- Paris Masters (Bercy) — only ATP Masters 1000 played indoors.
- Madrid Masters was indoor hard 2002-2008, then moved to outdoor clay.
  Listing "madrid masters" only triggers when `surface='Hard'` (which holds
  for those early years), so it does not mis-tag the modern Madrid event.
- Stuttgart Porsche Tennis Grand Prix (WTA) is indoor red CLAY, so it
  intentionally is NOT in this set — surface stays 'Clay'.
"""

from __future__ import annotations

INDOOR_TOURNAMENTS: frozenset[str] = frozenset(
    {
        # --- ATP indoor hard ---
        # Year-end / season finals
        "atp finals",
        "tour finals",
        "masters cup",
        "nextgen finals",
        "next gen finals",
        # Masters 1000 indoor
        "paris masters",
        "paris",
        "bercy",
        "madrid masters",
        "stuttgart masters",
        # 500-level indoor
        "rotterdam",
        "vienna",
        "basel",
        "memphis",
        # 250-level indoor
        "marseille",
        "stockholm",
        "sofia",
        "antwerp",
        "metz",
        "san jose",
        "moscow",
        "milan",
        "zagreb",
        "st. petersburg",
        "st petersburg",
        "copenhagen",
        "lyon",
        "atp paris",
        "tashkent",
        "long island",
        "las vegas",
        "cologne",
        # --- WTA indoor hard ---
        "wta finals",
        "wta tour championships",
        "wta championships",
        "tour championships",
        "tournament of champions",
        "linz",
        "luxembourg",
        "quebec city",
        "quebec",
        "zurich",
        "filderstadt",
        "ostrava",
        "moscow kremlin cup",
        "kremlin cup",
        "porsche grand prix",  # indoor hard when held under that name in non-Stuttgart years
    }
)


def is_indoor(tourney_name: str | None) -> bool:
    """Return True iff `tourney_name` matches a known indoor hard tournament.

    Match is case-insensitive on the trimmed name. Returns False for None
    or unrecognized tournaments — the default is outdoor.
    """
    if tourney_name is None:
        return False
    return tourney_name.strip().lower() in INDOOR_TOURNAMENTS
