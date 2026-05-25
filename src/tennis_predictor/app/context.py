"""Build a `MatchContext` for the LLM agent.

Two entry points:

  * `load_context_from_match_id(conn, scheduled_match_id)` — for the Home
    page click-through, where the row in `scheduled_matches` carries
    everything needed (players, surface, tier, scheduled start).
  * `load_context_from_freeform(...)` — for the Custom Prediction page,
    where the user types every field by hand.

The logic originally lived in `scripts/predict_match.py`; both Streamlit
pages and the CLI now share it from here. Anything that needs to translate
external strings (matchstat tier names) onto our canonical `TournamentLevel`
goes through `_MATCHSTAT_TIER_TO_LEVEL`, and best-of inference uses
`_LEVEL_BEST_OF_DEFAULT` — both kept module-private since callers should
go through the public builders.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, cast, get_args

import duckdb

from tennis_predictor.features.schema import Surface, TournamentLevel
from tennis_predictor.llm.tools.schemas import MatchContext, Tour


class ContextBuildError(ValueError):
    """Raised when an input row / form payload cannot be turned into a
    valid `MatchContext`. Callers (CLI, Streamlit pages) translate this
    into a user-facing message; the underlying cause stays in the message."""


# matchstat's tier strings → our canonical `TournamentLevel`. Mirrors the
# whitelist in `data/matchstat.py` and the canonical levels in
# `features/schema.py`. Anything not in this dict aborts with a clean
# error: we will not silently coerce a Challenger row into "ATP250".
_MATCHSTAT_TIER_TO_LEVEL: dict[str, TournamentLevel] = {
    "Grand Slam": "Slam",
    "ATP Masters 1000": "M1000",
    "ATP 500": "ATP500",
    "ATP 250": "ATP250",
    "WTA Masters 1000": "M1000",
    "WTA 1000": "M1000",
    "WTA 500": "WTA500",
    "WTA 250": "WTA250",
    "Finals": "Finals",
}

# Tournament-name fallback for the matchstat-calendar gap: per Phase 2
# notes, `tournament/calendar/{year}` is forward-only — active tournaments
# disappear from the listing once they start, so their `tournament_tier`
# in `scheduled_matches` ends up NULL. For the four Grand Slams the name
# is unambiguous enough to recover the level offline.
_SLAM_NAME_PATTERNS: tuple[tuple[str, TournamentLevel], ...] = (
    ("roland garros", "Slam"),
    ("french open", "Slam"),
    ("wimbledon", "Slam"),
    ("australian open", "Slam"),
    ("us open", "Slam"),
)


def infer_tournament_level(tier: str | None, tournament_name: str | None) -> TournamentLevel | None:
    """Map a `scheduled_matches` row's (tier, name) to a model
    `TournamentLevel`, or None for out-of-scope events.

    Resolution order:
    1. Exact tier string lookup (the happy path — matchstat calendar
       provides the tier).
    2. Slam-name fallback (Grand Slams currently in progress whose tier
       was dropped from the calendar).
    3. Otherwise unmatched → None.
    """
    if tier and tier in _MATCHSTAT_TIER_TO_LEVEL:
        return _MATCHSTAT_TIER_TO_LEVEL[tier]
    if tournament_name:
        lowered = tournament_name.lower()
        for needle, level in _SLAM_NAME_PATTERNS:
            if needle in lowered:
                return level
    return None


# Best-of inferred from the tournament level when running in --match-id
# mode: ATP Slams are best-of-5 and so are no other current tour-level
# events on the men's side; WTA is best-of-3 everywhere. Free-form mode
# accepts an explicit `best_of`.
_LEVEL_BEST_OF_DEFAULT: dict[tuple[str, TournamentLevel], int] = {
    ("ATP", "Slam"): 5,
    ("ATP", "M1000"): 3,
    ("ATP", "ATP500"): 3,
    ("ATP", "ATP250"): 3,
    ("ATP", "Finals"): 3,
    ("WTA", "Slam"): 3,
    ("WTA", "M1000"): 3,
    ("WTA", "WTA500"): 3,
    ("WTA", "WTA250"): 3,
    ("WTA", "Finals"): 3,
}


def load_context_from_match_id(
    conn: duckdb.DuckDBPyConnection, scheduled_match_id: str
) -> MatchContext:
    """Look up `scheduled_matches` by id and map its fields onto a
    `MatchContext`. Raises `ContextBuildError` on lookup miss, unsupported
    tour, unsupported surface, or out-of-scope tournament tier."""
    row = conn.execute(
        """
        SELECT tour, player1_name, player2_name, surface, tournament_tier,
               tournament_name, scheduled_start_utc
        FROM scheduled_matches
        WHERE scheduled_match_id = ?
        """,
        [scheduled_match_id],
    ).fetchone()
    if row is None:
        raise ContextBuildError(f"no scheduled match found with id={scheduled_match_id!r}")
    (
        tour,
        player1_name,
        player2_name,
        surface,
        tier,
        tournament_name,
        scheduled_start_utc,
    ) = row

    if tour not in ("ATP", "WTA"):
        raise ContextBuildError(f"unsupported tour {tour!r} in scheduled_matches row")
    if surface not in get_args(Surface):
        raise ContextBuildError(
            f"surface {surface!r} not in supported set {get_args(Surface)}; "
            "row may pre-date Phase-2 surface normalisation."
        )
    level = infer_tournament_level(tier, tournament_name)
    if level is None:
        raise ContextBuildError(
            f"tournament_tier {tier!r} (tournament {tournament_name!r}) "
            "does not map to a model tournament_level. Out-of-scope events "
            "(Challengers, ITF) cannot be predicted."
        )

    best_of = cast(Literal[3, 5], _LEVEL_BEST_OF_DEFAULT[(tour, level)])
    match_date = (
        scheduled_start_utc.date()
        if isinstance(scheduled_start_utc, datetime)
        else (scheduled_start_utc or date.today())
    )
    return MatchContext(
        tour=cast(Tour, tour),
        player_a_name=player1_name,
        player_b_name=player2_name,
        surface=cast(Surface, surface),
        tournament_level=level,
        tournament_name=tournament_name,
        best_of=best_of,
        match_date=match_date,
        scheduled_match_id=scheduled_match_id,
    )


def load_context_from_freeform(
    *,
    tour: Tour,
    player_a_name: str,
    player_b_name: str,
    surface: Surface,
    tournament_level: TournamentLevel,
    match_date: date,
    best_of: Literal[3, 5] | None = None,
    tournament_name: str | None = None,
) -> MatchContext:
    """Build a `MatchContext` from manually-entered fields. `best_of` is
    inferred from `(tour, tournament_level)` when omitted; if no default
    exists for that combination, raises `ContextBuildError`."""
    if best_of is None:
        inferred = _LEVEL_BEST_OF_DEFAULT.get((tour, tournament_level))
        if inferred is None:
            raise ContextBuildError(
                f"could not infer best_of for ({tour}, {tournament_level}); pass it explicitly"
            )
        best_of = cast(Literal[3, 5], inferred)

    return MatchContext(
        tour=tour,
        player_a_name=player_a_name,
        player_b_name=player_b_name,
        surface=surface,
        tournament_level=tournament_level,
        tournament_name=tournament_name,
        best_of=best_of,
        match_date=match_date,
        scheduled_match_id=None,
    )


__all__ = [
    "ContextBuildError",
    "infer_tournament_level",
    "load_context_from_freeform",
    "load_context_from_match_id",
]
