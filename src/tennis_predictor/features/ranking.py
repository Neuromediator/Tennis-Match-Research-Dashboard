"""Ranking lookup for the feature layer.

Sackmann's `rankings` table holds weekly snapshots: one row per
`(ranking_date, player_id)` pair. The feature `rank_p1` / `rank_p2`
fields need the **most recent rank with `ranking_date <= as_of_date`**
for each player.

Unlike Elo / form / fatigue / H2H, ranking is NOT a state we maintain
during replay — the rankings table is itself the canonical history,
populated once by ingestion. The orchestrator gets a `RankingLookup`,
loaded once from DuckDB, and calls `get(player_id, as_of_date)` for
each snapshot.

# Sentinel

When no ranking row exists with `ranking_date <= as_of_date` (debutants,
extended absentees, juniors), `get` returns `SENTINEL_UNRANKED = 9999`.
This matches the Pydantic `le=9999` bound on `FeatureVector.rank_p1/p2`
— see `tests/test_feature_vector.py::test_rank_sentinel_upper_bound_enforced`.

# Performance

In-memory lookup with O(log N) bisect per call. Used during chronological
replay across ~720k training matches x 2 lookups = ~1.4M calls; naive
per-call SQL would dominate. The full rankings table (~5.6M rows) loads
in ~1 GB-sec; bisect-based lookup is sub-microsecond per call.
"""

from __future__ import annotations

import bisect
from datetime import date
from typing import ClassVar

import duckdb


class RankingLookup:
    """In-memory rank lookup keyed on `(player_id, as_of_date)`."""

    SENTINEL_UNRANKED: ClassVar[int] = 9999

    def __init__(self) -> None:
        # Parallel arrays per player: sorted dates + corresponding ranks.
        # Parallel-array layout keeps bisect cheap (no list comprehension per
        # call) and is more memory-efficient than list-of-tuples for ~5.6M rows.
        self._dates: dict[str, list[date]] = {}
        self._ranks: dict[str, list[int]] = {}

    @classmethod
    def from_db(cls, conn: duckdb.DuckDBPyConnection) -> RankingLookup:
        """Load every (player_id, ranking_date, rank) row into memory.

        Rows are pre-sorted by `(player_id, ranking_date)` so the per-player
        lists are already in date order — no extra sort needed.
        """
        lookup = cls()
        rows = conn.execute(
            "SELECT player_id, ranking_date, rank FROM rankings ORDER BY player_id, ranking_date"
        ).fetchall()
        for pid, dt, rank in rows:
            lookup._dates.setdefault(pid, []).append(dt)
            lookup._ranks.setdefault(pid, []).append(int(rank))
        return lookup

    def get(self, player_id: str, as_of_date: date) -> int:
        """Most recent rank with `ranking_date <= as_of_date`.

        Returns `SENTINEL_UNRANKED` when no such row exists (unseen player,
        or `as_of_date` precedes the player's earliest ranking).
        """
        dates = self._dates.get(player_id)
        if not dates:
            return self.SENTINEL_UNRANKED
        # bisect_right returns the insertion index AFTER all entries with
        # date == as_of_date, so the most-recent qualifying entry is at
        # index (bisect_right - 1).
        idx = bisect.bisect_right(dates, as_of_date) - 1
        if idx < 0:
            return self.SENTINEL_UNRANKED
        return self._ranks[player_id][idx]

    def has_ranking_history(self, player_id: str) -> bool:
        return bool(self._dates.get(player_id))

    def __len__(self) -> int:
        return sum(len(v) for v in self._dates.values())
