"""LastMatchState — Phase 4.1 recovery signal.

Tracks one date per player: the date of their most-recent completed match
(any surface, any tier). Used to compute the
`days_since_last_match_p1` / `_p2` features. Capped at 365 days because
beyond that the feature changes semantics from "recovery" to "returning
from a long absence" — a different effect we'd want to model explicitly
if we cared about it. See Phase 4.1 design doc default #4.

# Contract

Mirrors the same snapshot-then-update discipline as `EloState`:

1. `days_since(player_id, current_date)` — snapshot BEFORE the match.
2. `update(player_id, match_date)` — applied AFTER the snapshot.

Reversing the order leaks the target match's date into the snapshot and
the leakage tests will catch it.

# Persistence

`save_to_db` writes a full snapshot to the `last_match_state` table in one
transaction. `from_db` reconstructs the state. Same pattern as
`EloState.save_to_db` / `from_db`. The pair is a round-trip.

For inference, the typical pattern is:

    state = LastMatchState.from_db(conn)
    for h in history_after_snapshot:
        state.update(h.player_id, h.match_date)
    days = state.days_since(player_id, as_of_date)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import duckdb


@dataclass(frozen=True, slots=True)
class LastMatchEntry:
    """One per player: the last date they completed a match on."""

    last_match_date: date


class LastMatchState:
    """In-memory map `player_id -> last completed match date`.

    State-update gate at the call site mirrors Elo's: only matches with
    `match_status == 'completed'` and a normalized surface feed this
    state. The state object itself is decoupled from those concerns —
    the orchestrator filters before calling `.update`.
    """

    CAP_DAYS: ClassVar[int] = 365
    """Maximum value returned by `days_since`. Anything longer than a year
    falls back to the cap; the feature ceases to be a recovery signal
    beyond that horizon. Per Phase 4.1 design doc default #4."""

    def __init__(self) -> None:
        self._entries: dict[str, LastMatchEntry] = {}

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def days_since(self, player_id: str, current_date: date) -> int | None:
        """Days from the player's last completed match to `current_date`,
        capped at `CAP_DAYS`. Returns None if the player has no recorded
        prior match (cold start).

        Negative gaps (last_match_date > current_date) clamp to 0 — they
        indicate a same-day query against a state already updated for
        today's match, which is a caller bug, not something to crash on.
        """
        entry = self._entries.get(player_id)
        if entry is None:
            return None
        gap = (current_date - entry.last_match_date).days
        if gap < 0:
            return 0
        if gap > self.CAP_DAYS:
            return self.CAP_DAYS
        return gap

    def last_date(self, player_id: str) -> date | None:
        entry = self._entries.get(player_id)
        return entry.last_match_date if entry is not None else None

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, player_id: str, match_date: date) -> None:
        """Record that `player_id` completed a match on `match_date`.

        Idempotent on chronological replay: if a later match for the same
        player overwrites an earlier one, the stored date moves forward.
        Out-of-order updates also do the right thing — we keep the
        maximum date seen, not the latest write — because the
        chronological-replay contract isn't strictly enforceable here
        (the snapshot call doesn't know about ordering).
        """
        existing = self._entries.get(player_id)
        if existing is None or match_date > existing.last_match_date:
            self._entries[player_id] = LastMatchEntry(last_match_date=match_date)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @classmethod
    def from_db(cls, conn: duckdb.DuckDBPyConnection) -> LastMatchState:
        """Reconstruct state from the `last_match_state` table."""
        state = cls()
        rows = conn.execute("SELECT player_id, last_match_date FROM last_match_state").fetchall()
        for player_id, last_date in rows:
            state._entries[player_id] = LastMatchEntry(last_match_date=last_date)
        return state

    def save_to_db(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Atomically replace the contents of `last_match_state` with the
        current in-memory state. One transaction so partial writes can't
        corrupt the table.
        """
        rows = [(pid, e.last_match_date) for pid, e in self._entries.items()]
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DELETE FROM last_match_state")
            if rows:
                conn.executemany(
                    "INSERT INTO last_match_state (player_id, last_match_date) VALUES (?, ?)",
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

    def __contains__(self, player_id: object) -> bool:
        return player_id in self._entries
