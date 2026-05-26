"""Pydantic schema for the FeatureVector contract.

Single source of truth for what the training table writes and what the
inference path produces.

**v1 (Phase 3) — 28 fields**: Surface-Elo (3), recent form (4), serve/return
rolling (8), H2H (3), fatigue (4), ranking (3), tournament context (3).

**v2 (Phase 4.1) — 39 fields**: v1 + player metadata (9) + recovery (2).
The 11 new fields are:

- Handedness match-up: `hand_p1`, `hand_p2` (categorical R/L/A/U).
- Age: `age_p1`, `age_p2`, `age_vs_peak_p1`, `age_vs_peak_p2`.
- Height: `height_p1`, `height_p2`, `height_diff_cm`.
- Recovery: `days_since_last_match_p1`, `days_since_last_match_p2`
  (capped at 365 days — beyond that the feature is "returning from a long
  absence", not "recovery", per the Phase 4.1 design doc).

**v3 (Phase 4.2) — 41 fields**: v2 + surface-specific recovery (2):

- `days_since_last_match_surface_p1`, `days_since_last_match_surface_p2`
  — gap since the player's most recent completed match ON THIS SURFACE,
  capped at 365 days. None on cold start (no prior match on this
  surface). Lets LightGBM learn the "stale surface-Elo" pattern from
  data instead of restructuring Elo itself. See
  `docs/tutorials/phase_4_2_notes.md`.

Per-field rationale lives in `.claude/skills/feature-engineering/SKILL.md`
and `docs/tutorials/phase_4_1_notes.md` / `phase_4_2_notes.md`.

Canonical pair ordering: `p1` is the lex-smaller `player_id`. Callers of
`compute_features` may pass players in any order; the function normalizes
internally before building the vector.

Nullability:
- Required (no None): Elo (3), H2H wins (2), fatigue (4), ranking (3),
  tournament context (3), `hand_p1`/`hand_p2` (default `"U"` when unknown).
- Optional (None allowed): recent form (4) when window < 3 matches;
  serve/return (8) when window has < 5 matches with non-null stat columns;
  `h2h_recency_days` (NULL when the pair has never met); the 7 numeric
  player-metadata / recovery fields when source data is missing (LightGBM
  consumes NaN cleanly).

`rank_*` uses 9999 as sentinel for unranked — Pydantic bound `le=9999`
keeps that discipline explicit. Sackmann rankings top out around 2500 in
practice, so any observed value above the sentinel is a bug.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: int = 3
"""FeatureVector schema generation. Bumped whenever the field set or
contract changes — used by the `training_features` DDL migration and by
`metadata.json` in trained-model artifacts.

History:
- v1 (Phase 3): 28 fields, no player metadata, no recovery signal.
- v2 (Phase 4.1): 39 fields = v1 + 9 player-metadata + 2 recovery.
- v3 (Phase 4.2): 41 fields = v2 + 2 surface-specific recovery.
"""

TournamentLevel = Literal[
    "Slam",
    "M1000",
    "ATP500",
    "ATP250",
    "WTA500",
    "WTA250",
    "Finals",
]
Surface = Literal["Hard", "IHard", "Clay", "Grass"]
Hand = Literal["R", "L", "A", "U"]
"""Handedness category as stored in Sackmann's `players.hand`.

- `R` = right-handed
- `L` = left-handed
- `A` = ambidextrous (rare)
- `U` = unknown — assigned when the JOIN against `players` finds no row
  for the player_id, or when `players.hand` is NULL / unknown literal.
  Coverage on active tour-level players is ~100%, so `U` is rare for the
  matches the model actually scores."""


class FeatureVector(BaseModel):
    """v3 FeatureVector — 41 fields for one (p1, p2, surface, as_of_date) instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- Surface-Elo (3) ---
    elo_p1_surface: float
    elo_p2_surface: float
    elo_diff_surface: float

    # --- Recent form (4) — None if window has < 3 matches ---
    win_pct_last10_p1: float | None = Field(default=None, ge=0.0, le=1.0)
    win_pct_last10_p2: float | None = Field(default=None, ge=0.0, le=1.0)
    win_pct_last25_surface_p1: float | None = Field(default=None, ge=0.0, le=1.0)
    win_pct_last25_surface_p2: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- Serve/return rolling (8) — last 25 surface-filtered; None if < 5 stat-rich ---
    first_serve_win_pct_p1: float | None = Field(default=None, ge=0.0, le=1.0)
    first_serve_win_pct_p2: float | None = Field(default=None, ge=0.0, le=1.0)
    second_serve_win_pct_p1: float | None = Field(default=None, ge=0.0, le=1.0)
    second_serve_win_pct_p2: float | None = Field(default=None, ge=0.0, le=1.0)
    bp_saved_pct_p1: float | None = Field(default=None, ge=0.0, le=1.0)
    bp_saved_pct_p2: float | None = Field(default=None, ge=0.0, le=1.0)
    bp_converted_pct_p1: float | None = Field(default=None, ge=0.0, le=1.0)
    bp_converted_pct_p2: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- H2H (3) — recency_days None if never met ---
    h2h_p1_wins: int = Field(ge=0)
    h2h_p2_wins: int = Field(ge=0)
    h2h_recency_days: int | None = Field(default=None, ge=0)

    # --- Fatigue (4) ---
    fatigue_matches_7d_p1: int = Field(ge=0)
    fatigue_matches_7d_p2: int = Field(ge=0)
    fatigue_sets_14d_p1: int = Field(ge=0)
    fatigue_sets_14d_p2: int = Field(ge=0)

    # --- Ranking (3) — 9999 sentinel for unranked ---
    rank_p1: int = Field(ge=1, le=9999)
    rank_p2: int = Field(ge=1, le=9999)
    rank_diff: int

    # --- Tournament context (3) ---
    tournament_level: TournamentLevel
    best_of: Literal[3, 5]
    surface: Surface

    # --- Player metadata: handedness (2) ---
    # Default "U" so a missing `players` row doesn't crash construction.
    hand_p1: Hand = "U"
    hand_p2: Hand = "U"

    # --- Player metadata: age (4) — None when `players.dob` is missing ---
    # No hard bounds: Sackmann's DOB column has a handful of obviously
    # wrong values (e.g. a player listed as 3 years old at a tour-level
    # match — clearly a typo upstream). LightGBM handles outliers cleanly
    # and we'd rather pass them through than crash the rebuild. Surface
    # them via post-build data-quality checks instead of constructor bounds.
    age_p1: float | None = None
    age_p2: float | None = None
    age_vs_peak_p1: float | None = None
    age_vs_peak_p2: float | None = None

    # --- Player metadata: height (3) — None when `players.height` is missing ---
    # ATP coverage ~57%, WTA ~25% on active players; LightGBM uses NaN as signal.
    # Same defensive-bounds-removal rationale as age above.
    height_p1: int | None = None
    height_p2: int | None = None
    height_diff_cm: int | None = None

    # --- Recovery (2) — None when the player has no prior completed match ---
    # Capped at 365 by the LastMatchState (beyond that the semantic flips
    # from "recovery" to "returning from long absence" — different effect).
    days_since_last_match_p1: int | None = Field(default=None, ge=0, le=365)
    days_since_last_match_p2: int | None = Field(default=None, ge=0, le=365)

    # --- Phase 4.2: surface-specific recovery (2) ---
    # None when the player has no prior completed match ON THIS SURFACE
    # (cold start). Same 365-day cap and semantic flip as the global
    # recovery features. Surface normalisation matches the existing
    # taxonomy (Hard / IHard / Clay / Grass; Carpet→IHard).
    days_since_last_match_surface_p1: int | None = Field(default=None, ge=0, le=365)
    days_since_last_match_surface_p2: int | None = Field(default=None, ge=0, le=365)


FEATURE_FIELD_NAMES: tuple[str, ...] = tuple(FeatureVector.model_fields.keys())
"""Canonical ordering of FeatureVector fields. Used to build the SELECT/INSERT
column list for the `training_features` table — keeping `data/schema.py` (DDL)
and `features/schema.py` (Pydantic) in lockstep is a Phase 3 contract."""
