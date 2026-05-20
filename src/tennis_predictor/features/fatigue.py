"""Fatigue state.

Produces two FeatureVector fields per player:

- `fatigue_matches_7d`: number of matches played in the 7 calendar days
  leading up to (and including) `as_of_date`.
- `fatigue_sets_14d`: total sets played in the 14 calendar days leading
  up to (and including) `as_of_date`.

Both default to 0 when there is no recent activity — no `None`, since a
player with no recent matches is genuinely "well-rested", not "unknown".

# Set counting

Set count is parsed from Sackmann's `score` string. Examples:

| score                  | sets |
|------------------------|------|
| `"6-4 6-3"`            | 2    |
| `"6-4 7-6(5) 6-3"`     | 3    |
| `"6-4 2-1 RET"`        | 2    |  ← partial second set still counts
| `"5-5 DEF"`            | 1    |  ← partial first set still counts
| `"W/O"`                | 0    |  ← walkover, no play
| `None` or `""`         | 0    |

A "set" is any whitespace-delimited token matching `\\d+-\\d+(\\(\\d+\\))?`.
Tokens like `"RET"`, `"DEF"`, `"W/O"` are not sets and are ignored.

# Contract (snapshot-before-update)

The orchestrator calls `snapshot(player_id, as_of_date)` BEFORE applying
the current match's outcome via `update`. The current match itself is
NOT in state at snapshot time — that's the central anti-leakage rule.

# State sizing

Per-player chronological list, rebuilt each replay (not persisted). The
snapshot walks the history from the most-recent end and stops once it
crosses the 14-day boundary — O(window_size) per call regardless of how
many matches the player has played overall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

_SET_TOKEN_PATTERN = re.compile(r"^\d+-\d+(\(\d+\))?$")


def count_sets(score: str | None) -> int:
    """Parse Sackmann's `score` string into a number of sets played.

    Tokens that look like a set score (`6-4`, `7-6(5)`) count;
    `"RET"`, `"DEF"`, `"W/O"` and the like are ignored. NULL / empty
    inputs return 0.
    """
    if not score:
        return 0
    return sum(1 for token in score.split() if _SET_TOKEN_PATTERN.match(token))


@dataclass(frozen=True, slots=True)
class FatigueEntry:
    match_date: date
    sets: int


class FatigueState:
    """Per-player chronological match log → rolling fatigue counters."""

    WINDOW_MATCHES_DAYS: ClassVar[int] = 7
    WINDOW_SETS_DAYS: ClassVar[int] = 14

    def __init__(self) -> None:
        self._history: dict[str, list[FatigueEntry]] = {}

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(
        self,
        player_id: str,
        as_of_date: date,
    ) -> tuple[int, int]:
        """Return `(fatigue_matches_7d, fatigue_sets_14d)`.

        Walks the player's history backwards from the most recent entry,
        stopping once an entry is older than 14 days — so per-snapshot
        cost is O(matches_within_window), not O(total_history).
        """
        entries = self._history.get(player_id)
        if entries is None:
            return (0, 0)

        matches_7d = 0
        sets_14d = 0
        for entry in reversed(entries):
            delta = (as_of_date - entry.match_date).days
            if delta < 0:
                # Defensive: future entry. Chronological replay should never
                # produce one — skip rather than crash.
                continue
            if delta > self.WINDOW_SETS_DAYS:
                break  # everything further back is even older
            sets_14d += entry.sets
            if delta <= self.WINDOW_MATCHES_DAYS:
                matches_7d += 1

        return (matches_7d, sets_14d)

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(
        self,
        winner_id: str,
        loser_id: str,
        sets_played: int,
        match_date: date,
    ) -> None:
        """Append the match to BOTH players' histories. Both players paid
        the same physical cost (same number of sets contested)."""
        entry = FatigueEntry(match_date=match_date, sets=sets_played)
        self._history.setdefault(winner_id, []).append(entry)
        self._history.setdefault(loser_id, []).append(entry)
