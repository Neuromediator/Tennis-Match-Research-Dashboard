"""Pydantic schema for the FeatureVector contract.

Single source of truth for what the training table writes and what the
inference path produces. 28 fields across seven families — see
`.claude/skills/feature-engineering/SKILL.md` for per-field rationale.

Canonical pair ordering: `p1` is the lex-smaller `player_id`. Callers of
`compute_features` may pass players in any order; the function normalizes
internally before building the vector.

Nullability:
- Required (no None): Elo (3), H2H wins (2), fatigue (4), ranking (3),
  tournament context (3).
- Optional (None allowed): recent form (4) when window < 3 matches;
  serve/return (8) when window has < 5 matches with non-null stat columns;
  `h2h_recency_days` (NULL when the pair has never met).

`rank_*` uses 9999 as sentinel for unranked — Pydantic bound `le=9999`
keeps that discipline explicit. Sackmann rankings top out around 2500 in
practice, so any observed value above the sentinel is a bug.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class FeatureVector(BaseModel):
    """28-field feature vector for one (p1, p2, surface, as_of_date) instance."""

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


FEATURE_FIELD_NAMES: tuple[str, ...] = tuple(FeatureVector.model_fields.keys())
"""Canonical ordering of FeatureVector fields. Used to build the SELECT/INSERT
column list for the `training_features` table — keeping schema.py (DDL) and
schema.py (Pydantic) in lockstep is a phase-3 contract."""
