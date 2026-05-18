"""matchstat-source player resolver wrapping `AliasIndex`.

For each `(name, tour)` lookup against the player_aliases table:

- `auto` (confidence ≥ 0.90, no ambiguous runner-up) → returns the
  canonical player_id, cached in memory for the rest of the session.
- `review` (confidence 0.75-0.90, or >= 0.90 but ambiguous) → returns
  None, appends to `review_buffer` so the orchestrator can flush it
  to `aliases_review_matchstat.csv` at end-of-run.
- `unknown` (confidence < 0.75) → returns None, NOT recorded.

The instance is callable with the `PlayerResolver` signature used by
`load_hot.py`, so it can be wired in directly:

    resolver = MatchstatResolver(conn)
    insert_completed_matches(conn, matches, ..., resolve_player=resolver)

`AliasIndex` is built lazily per tour — building both ATP and WTA up
front is ~one fast SQL plus an in-memory dict construction, but if a
given refresh only touches one tour we don't want to pay for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import duckdb

from tennis_predictor.data.reconcile import AliasIndex, Tour

TourArg = Literal["ATP", "WTA"]


@dataclass(frozen=True)
class ReviewCandidate:
    """One low-confidence resolution surfaced for human verification.

    Mirrors the fields a reviewer needs to make the call: the raw name
    from matchstat, the candidate alias_text we'd auto-link it to, the
    confidence score, and the runner-up score to flag ambiguity.
    """

    raw_name: str
    tour: TourArg
    candidate_name: str | None
    confidence: float
    runner_up_confidence: float


class MatchstatResolver:
    """Cached, tour-aware fuzzy resolver from matchstat names → canonical IDs."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self._indexes: dict[Tour, AliasIndex] = {}
        # Cache outcome (canonical_id or None) per (name, tour). Hot path for
        # daily refreshes since top players show up across many fixtures /
        # results in the same run.
        self._cache: dict[tuple[str, TourArg], str | None] = {}
        self.review_buffer: list[ReviewCandidate] = []

    def _index_for(self, tour: Tour) -> AliasIndex:
        if tour not in self._indexes:
            self._indexes[tour] = AliasIndex(self._conn, tour)
        return self._indexes[tour]

    def __call__(self, name: str, tour: str) -> str | None:
        """PlayerResolver-compatible: returns canonical player_id or None.

        Unknown tour strings (anything other than 'ATP'/'WTA') return None
        without consulting the index — defensive against typos in caller
        code; doesn't surface as a review row.
        """
        if tour not in get_args(TourArg):
            return None
        tour_typed: TourArg = tour  # type: ignore[assignment]
        cache_key = (name, tour_typed)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._index_for(tour_typed).lookup(name)

        if result.status == "auto":
            canonical = result.canonical_player_id
            self._cache[cache_key] = canonical
            return canonical

        if result.status == "review":
            self.review_buffer.append(
                ReviewCandidate(
                    raw_name=name,
                    tour=tour_typed,
                    candidate_name=result.candidate_name,
                    confidence=result.confidence,
                    runner_up_confidence=result.runner_up_confidence,
                )
            )

        self._cache[cache_key] = None
        return None

    def stats(self) -> dict[str, int]:
        """Quick rollup the orchestrator can include in the run summary."""
        resolved = sum(1 for v in self._cache.values() if v is not None)
        unresolved = sum(1 for v in self._cache.values() if v is None)
        return {
            "unique_names_seen": len(self._cache),
            "resolved_auto": resolved,
            "unresolved": unresolved,
            "review_candidates": len(self.review_buffer),
        }
