"""Rolling recent-form state.

Tracks per-player match outcomes chronologically. Produces two windowed
win-rate features for the FeatureVector:

- `win_pct_last10_p1/p2`   — last 10 matches, any surface.
- `win_pct_last25_surface_p1/p2` — last 25 matches on the *given* surface.

Returns `None` for either rate when the window contains fewer than
`MIN_MATCHES_FOR_RATE` (3) entries — sparse data is noise, not signal.

# Contract (snapshot-before-update)

The orchestrator calls `snapshot(player_id, surface)` BEFORE applying the
match outcome via `update`. Leakage tests assert this invariant.

# State sizing

Per-player history is unbounded (list, not deque) — replay is rebuilt
fresh each training run (~1.7M matches → ~3.4M player-match rows, well
within memory at our scale). Skill: "Other state objects are rebuilt
in-memory each training run; not persisted."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class FormEntry:
    """One player-match row in chronological order."""

    match_date: date
    surface: str  # normalized: one of {Hard, IHard, Clay, Grass}
    won: bool


class RollingFormState:
    """Per-player chronological history → windowed win rates."""

    LAST_N_ANY: ClassVar[int] = 10
    LAST_N_SURFACE: ClassVar[int] = 25
    MIN_MATCHES_FOR_RATE: ClassVar[int] = 3

    def __init__(self) -> None:
        self._history: dict[str, list[FormEntry]] = {}

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(
        self,
        player_id: str,
        surface: str,
    ) -> tuple[float | None, float | None]:
        """Return `(win_pct_last10_any, win_pct_last25_surface)`.

        Either component is `None` if the window has fewer than
        `MIN_MATCHES_FOR_RATE` matches.
        """
        entries = self._history.get(player_id)
        if entries is None:
            return (None, None)

        last_any = entries[-self.LAST_N_ANY :]
        last_surface = [e for e in entries if e.surface == surface][-self.LAST_N_SURFACE :]
        return (self._rate(last_any), self._rate(last_surface))

    def matches_played(self, player_id: str) -> int:
        return len(self._history.get(player_id, []))

    @classmethod
    def _rate(cls, entries: list[FormEntry]) -> float | None:
        if len(entries) < cls.MIN_MATCHES_FOR_RATE:
            return None
        wins = sum(1 for e in entries if e.won)
        return wins / len(entries)

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(
        self,
        winner_id: str,
        loser_id: str,
        surface: str,
        match_date: date,
    ) -> None:
        """Append the outcome to BOTH players' histories. Caller MUST call
        `snapshot` for the relevant players BEFORE invoking this."""
        self._history.setdefault(winner_id, []).append(
            FormEntry(match_date=match_date, surface=surface, won=True)
        )
        self._history.setdefault(loser_id, []).append(
            FormEntry(match_date=match_date, surface=surface, won=False)
        )
