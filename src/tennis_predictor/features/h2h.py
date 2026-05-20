"""Head-to-head state.

Tracks per-pair win counters and the date of the last meeting. Produces
three FeatureVector fields:

- `h2h_p1_wins: int`        — required, defaults to 0 if pair never met
- `h2h_p2_wins: int`        — required, defaults to 0 if pair never met
- `h2h_recency_days: int|None` — None if pair never met; else
  `(as_of_date - last_meeting_date).days`

# Key canonicalization

The pair (A, B) is the same matchup as (B, A). We canonicalize internally
by sorting the two `player_id`s lexicographically, store one entry per
unordered pair, and mirror the counts back into the caller's argument
order at snapshot time.

# Contract (snapshot-before-update)

The orchestrator calls `snapshot(p1, p2, as_of_date)` BEFORE applying the
current match's outcome via `update`. Leakage tests assert this.

# State sizing

Per the feature-engineering skill, H2H state is not persisted — rebuilt
each replay. With ~1.7M matches the dict holds well under 1M unique
pairs (most player pairs meet rarely or never).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class H2HEntry:
    """Counters for one canonical pair (player_lo, player_hi).

    `wins_lo` = matches won by the lex-smaller player_id; `wins_hi` =
    matches won by the lex-larger. `last_date` is the date of the most
    recent meeting (used to compute `h2h_recency_days`).
    """

    wins_lo: int
    wins_hi: int
    last_date: date


class H2HState:
    """Per-pair win counters and last-meeting date."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], H2HEntry] = {}

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(
        self,
        p1: str,
        p2: str,
        as_of_date: date,
    ) -> tuple[int, int, int | None]:
        """Return `(h2h_p1_wins, h2h_p2_wins, h2h_recency_days)` from `p1`'s
        perspective.

        If the pair has never met, returns `(0, 0, None)`. The caller may
        pass `p1` and `p2` in any order — counts are mirrored to match.
        """
        key = self._key(p1, p2)
        entry = self._store.get(key)
        if entry is None:
            return (0, 0, None)
        recency_days = (as_of_date - entry.last_date).days
        if p1 == key[0]:
            return (entry.wins_lo, entry.wins_hi, recency_days)
        return (entry.wins_hi, entry.wins_lo, recency_days)

    def has_met(self, p1: str, p2: str) -> bool:
        return self._key(p1, p2) in self._store

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(
        self,
        winner_id: str,
        loser_id: str,
        match_date: date,
    ) -> None:
        """Increment the winner's count and refresh the last-meeting date.

        Caller MUST hand matches in chronological order — the state takes
        `match_date` at face value and stores it as `last_date`.
        """
        key = self._key(winner_id, loser_id)
        entry = self._store.get(key)
        if entry is None:
            wins_lo = 1 if winner_id == key[0] else 0
            self._store[key] = H2HEntry(
                wins_lo=wins_lo,
                wins_hi=1 - wins_lo,
                last_date=match_date,
            )
            return
        if winner_id == key[0]:
            self._store[key] = H2HEntry(
                wins_lo=entry.wins_lo + 1,
                wins_hi=entry.wins_hi,
                last_date=match_date,
            )
        else:
            self._store[key] = H2HEntry(
                wins_lo=entry.wins_lo,
                wins_hi=entry.wins_hi + 1,
                last_date=match_date,
            )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def __len__(self) -> int:
        return len(self._store)
