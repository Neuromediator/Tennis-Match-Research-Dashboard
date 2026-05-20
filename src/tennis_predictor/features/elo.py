"""Surface-Elo rating state.

Maintains one rating per `(player_id, surface)` pair. Standard Elo update
with K=32 and default rating 1500. The 4-surface canonical taxonomy is
defined in `features.schema.Surface`; `EloState` stores `surface` as `str`
internally — the orchestrator passes already-normalized values, and the
DuckDB column is VARCHAR.

# Contract

Two operations and their ordering matters:

1. `get(player_id, surface)` — snapshot the rating BEFORE the match.
2. `update(winner_id, loser_id, surface, match_date)` — apply the result.

If `update` runs before `get`, the snapshot leaks the post-match rating —
that's a leakage bug and is asserted by `tests/test_feature_leakage.py`.

# Persistence

`save_to_db` writes a full snapshot to the `elo_state` table (DELETE + bulk
INSERT in one transaction). `from_db` reconstructs an `EloState` from the
same table. The pair (`save_to_db`, `from_db`) is a round-trip; this is
exercised by tests.

For inference, the typical pattern is:
    state = EloState.from_db(conn)
    state.roll_forward(matches_in_window)   # snapshot-then-update per match
    snapshot = (state.get(p1, surf), state.get(p2, surf))
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import duckdb


@dataclass(frozen=True, slots=True)
class EloEntry:
    """One per `(player_id, surface)` pair."""

    rating: float
    matches_played: int
    last_updated_date: date


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """Compact tuple used by `roll_forward`. Caller builds these from the
    `matches` table — the state object stays decoupled from SQL."""

    winner_id: str
    loser_id: str
    surface: str
    match_date: date


class EloState:
    """In-memory Elo state per `(player_id, surface)` pair."""

    K_FACTOR: ClassVar[float] = 32.0
    DEFAULT_RATING: ClassVar[float] = 1500.0

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], EloEntry] = {}

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def get(self, player_id: str, surface: str) -> float:
        """Current rating for the pair, or `DEFAULT_RATING` if unseen.

        Called BEFORE `update` for a given match — this is the snapshot
        that goes into the FeatureVector.
        """
        entry = self._entries.get((player_id, surface))
        return entry.rating if entry is not None else self.DEFAULT_RATING

    def matches_played(self, player_id: str, surface: str) -> int:
        entry = self._entries.get((player_id, surface))
        return entry.matches_played if entry is not None else 0

    def last_updated(self, player_id: str, surface: str) -> date | None:
        entry = self._entries.get((player_id, surface))
        return entry.last_updated_date if entry is not None else None

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Standard Elo expected score for player A vs B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        winner_id: str,
        loser_id: str,
        surface: str,
        match_date: date,
    ) -> None:
        """Apply Elo update for one match. Zero-sum: winner gains exactly
        what loser loses."""
        r_w = self.get(winner_id, surface)
        r_l = self.get(loser_id, surface)
        e_w = self.expected_score(r_w, r_l)
        delta = self.K_FACTOR * (1.0 - e_w)  # winner's gain == loser's loss
        self._set(winner_id, surface, r_w + delta, match_date)
        self._set(loser_id, surface, r_l - delta, match_date)

    def roll_forward(self, matches: list[MatchOutcome]) -> None:
        """Apply a sequence of updates in the given order.

        Caller must hand matches in chronological order (typically
        `ORDER BY tourney_date, tourney_id, match_num, match_id`). The
        state does not validate ordering — that's the orchestrator's job.
        """
        for m in matches:
            self.update(m.winner_id, m.loser_id, m.surface, m.match_date)

    def _set(
        self,
        player_id: str,
        surface: str,
        rating: float,
        match_date: date,
    ) -> None:
        existing = self._entries.get((player_id, surface))
        n = (existing.matches_played if existing else 0) + 1
        self._entries[(player_id, surface)] = EloEntry(
            rating=rating,
            matches_played=n,
            last_updated_date=match_date,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @classmethod
    def from_db(cls, conn: duckdb.DuckDBPyConnection) -> EloState:
        """Reconstruct state from the `elo_state` table."""
        state = cls()
        rows = conn.execute(
            "SELECT player_id, surface, rating, matches_played, last_updated_date FROM elo_state"
        ).fetchall()
        for player_id, surface, rating, n, dt in rows:
            state._entries[(player_id, surface)] = EloEntry(
                rating=float(rating),
                matches_played=int(n),
                last_updated_date=dt,
            )
        return state

    def save_to_db(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Atomically replace the contents of `elo_state` with the current
        in-memory state. Uses one transaction so partial writes can't
        corrupt the table.
        """
        rows = [
            (pid, surf, e.rating, e.matches_played, e.last_updated_date)
            for (pid, surf), e in self._entries.items()
        ]
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DELETE FROM elo_state")
            if rows:
                conn.executemany(
                    "INSERT INTO elo_state "
                    "(player_id, surface, rating, matches_played, last_updated_date) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._entries
