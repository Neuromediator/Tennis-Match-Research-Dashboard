"""Serve/return rolling state.

Produces 8 FeatureVector fields per player, all rolling over the last
`WINDOW=25` matches **on the same surface**:

| field                       | numerator           | denominator           |
|-----------------------------|---------------------|-----------------------|
| `first_serve_win_pct`       | first_won           | first_in              |
| `second_serve_win_pct`      | second_won          | svpt - first_in       |
| `bp_saved_pct`              | bp_saved            | bp_faced              |
| `bp_converted_pct` (return) | opp_bp_faced - opp_bp_saved | opp_bp_faced  |

All four are `None` when fewer than `MIN_STAT_MATCHES=5` stat-rich matches
sit in the surface-filtered window. That threshold reflects the v1 NaN
policy: serve/return rates need a non-trivial sample to be meaningful.

# Rate aggregation

Rates are computed by summing numerators and denominators across the
window — NOT by averaging per-match rates. Per-match averaging would
bias toward short matches; sum-then-divide weights each point equally,
which is the statistically honest aggregation.

# Stats availability

~58% of historical main matches have NULL serve/return columns in
Sackmann (pre-mid-1990s recorded the score and little else). The
orchestrator passes `MatchStats=None` for those rows; this state
silently skips them. They still feed other state objects (Elo, form,
fatigue, H2H) — they just don't sharpen the serve/return window.

# Contract (snapshot-before-update)

The orchestrator calls `snapshot(player_id, surface)` BEFORE applying
the current match via `update(...)`. Leakage tests assert this.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class MatchStats:
    """Per-match raw serve/return counts for both players.

    Built by the orchestrator from a `matches` row. All fields must be
    non-NULL — if any one is NULL, the orchestrator passes None to
    `ServeReturnState.update` instead.

    `w_*` are the winner's stats (serving), `l_*` are the loser's.
    """

    # Winner's serve side
    w_first_in: int
    w_first_won: int
    w_second_won: int
    w_svpt: int
    w_bp_saved: int
    w_bp_faced: int
    # Loser's serve side
    l_first_in: int
    l_first_won: int
    l_second_won: int
    l_svpt: int
    l_bp_saved: int
    l_bp_faced: int


@dataclass(frozen=True, slots=True)
class ServeReturnSample:
    """One player-match entry: own serve counts + opponent's BP stats
    (needed for the return-side bp_converted rate)."""

    match_date: date
    surface: str
    # Own serve side
    first_in: int
    first_won: int
    second_won: int
    svpt: int
    bp_saved: int
    bp_faced: int
    # Opponent's serve side (for our return-side rate)
    opp_bp_faced: int
    opp_bp_saved: int


class ServeReturnState:
    """Per-player rolling serve/return stats, surface-filtered."""

    WINDOW: ClassVar[int] = 25
    MIN_STAT_MATCHES: ClassVar[int] = 5

    def __init__(self) -> None:
        self._history: dict[str, list[ServeReturnSample]] = {}

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(
        self,
        player_id: str,
        surface: str,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Return `(first_serve_win_pct, second_serve_win_pct, bp_saved_pct,
        bp_converted_pct)`.

        All four components are `None` when the surface-filtered window
        contains fewer than `MIN_STAT_MATCHES` entries.
        """
        entries = self._history.get(player_id)
        if entries is None:
            return (None, None, None, None)

        window = [e for e in entries if e.surface == surface][-self.WINDOW :]
        if len(window) < self.MIN_STAT_MATCHES:
            return (None, None, None, None)

        first_in = sum(e.first_in for e in window)
        first_won = sum(e.first_won for e in window)
        svpt = sum(e.svpt for e in window)
        second_won = sum(e.second_won for e in window)
        bp_saved = sum(e.bp_saved for e in window)
        bp_faced = sum(e.bp_faced for e in window)
        opp_bp_faced = sum(e.opp_bp_faced for e in window)
        opp_bp_saved = sum(e.opp_bp_saved for e in window)

        first_pct = first_won / first_in if first_in > 0 else None
        second_attempts = svpt - first_in
        second_pct = second_won / second_attempts if second_attempts > 0 else None
        bp_saved_pct = bp_saved / bp_faced if bp_faced > 0 else None
        bp_converted_pct = (
            (opp_bp_faced - opp_bp_saved) / opp_bp_faced if opp_bp_faced > 0 else None
        )

        return (first_pct, second_pct, bp_saved_pct, bp_converted_pct)

    def stat_match_count(self, player_id: str, surface: str) -> int:
        """How many surface-filtered stat-rich matches the window currently
        sees for this player. Useful for diagnostics + tests."""
        entries = self._history.get(player_id)
        if entries is None:
            return 0
        return len([e for e in entries if e.surface == surface][-self.WINDOW :])

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(
        self,
        winner_id: str,
        loser_id: str,
        surface: str,
        match_date: date,
        stats: MatchStats | None,
    ) -> None:
        """Append the match to BOTH players' histories.

        Pass `stats=None` for older matches without recorded serve/return
        counts — this state silently skips them. Other state objects
        (Elo, form, fatigue, H2H) still receive the update via their own
        `update` calls in the orchestrator.
        """
        if stats is None:
            return

        winner_sample = ServeReturnSample(
            match_date=match_date,
            surface=surface,
            first_in=stats.w_first_in,
            first_won=stats.w_first_won,
            second_won=stats.w_second_won,
            svpt=stats.w_svpt,
            bp_saved=stats.w_bp_saved,
            bp_faced=stats.w_bp_faced,
            opp_bp_faced=stats.l_bp_faced,
            opp_bp_saved=stats.l_bp_saved,
        )
        loser_sample = ServeReturnSample(
            match_date=match_date,
            surface=surface,
            first_in=stats.l_first_in,
            first_won=stats.l_first_won,
            second_won=stats.l_second_won,
            svpt=stats.l_svpt,
            bp_saved=stats.l_bp_saved,
            bp_faced=stats.l_bp_faced,
            opp_bp_faced=stats.w_bp_faced,
            opp_bp_saved=stats.w_bp_saved,
        )
        self._history.setdefault(winner_id, []).append(winner_sample)
        self._history.setdefault(loser_id, []).append(loser_sample)
