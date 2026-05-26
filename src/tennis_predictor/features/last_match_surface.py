"""LastMatchPerSurfaceState — Phase 4.2 surface-specific recovery signal.

Tracks one date per `(player_id, surface)` pair: the date of the player's
most-recent completed match on that surface. Used to compute the
`days_since_last_match_surface_p1` / `_p2` features added in Phase 4.2.

Why surface-specific? The Phase 4.1 global recovery signal averages over
all surfaces, so a player who hasn't touched clay in a year but played
hard last week shows `days_since_last_match` ≈ 7 with no signal that
their clay rating is stale. Surface-specific isolates the cases where a
player's surface-Elo is effectively frozen (Djokovic clay 2026, Opelka
post-injury hard, Kasatkina spring-clay returns). See
`docs/tutorials/phase_4_2_notes.md` for the design rationale.

# Contract

Mirrors the snapshot-then-update discipline of `LastMatchState` and
`EloState`:

1. `days_since(player_id, surface, current_date)` — snapshot BEFORE the match.
2. `update(player_id, surface, match_date)` — applied AFTER the snapshot.

Surface normalisation (`Carpet -> IHard`, indoor-whitelist split for
hard) and `match_status == 'completed'` filtering happen at the
orchestrator BEFORE calling `.update`, exactly like Elo.

# Persistence

Persistent snapshot lives in `last_match_per_surface_state`. `save_to_db`
writes a full replace in one transaction; `from_db` reconstructs the
state. Same pattern as `LastMatchState.save_to_db` / `from_db`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import duckdb


@dataclass(frozen=True, slots=True)
class LastMatchPerSurfaceEntry:
    """One per `(player_id, surface)`: the last date that pair completed a match."""

    last_match_date: date


class LastMatchPerSurfaceState:
    """In-memory map `(player_id, surface) -> last completed match date`.

    State-update gate at the call site mirrors Elo's: only matches with
    `match_status == 'completed'` and a normalised surface feed this
    state. The state object itself is decoupled from those concerns —
    the orchestrator filters before calling `.update`.
    """

    CAP_DAYS: ClassVar[int] = 365
    """Maximum value returned by `days_since`. Beyond a year the feature
    flips semantics from "recovery" to "career return" — a different
    effect we are not modelling here. Same cap as `LastMatchState`."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], LastMatchPerSurfaceEntry] = {}

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def days_since(self, player_id: str, surface: str, current_date: date) -> int | None:
        """Days from the player's last completed match on `surface` to
        `current_date`, capped at `CAP_DAYS`. Returns None when the player
        has no recorded prior match on this surface (cold start).

        Negative gaps clamp to 0 — same defensive behaviour as
        `LastMatchState.days_since`.
        """
        entry = self._entries.get((player_id, surface))
        if entry is None:
            return None
        gap = (current_date - entry.last_match_date).days
        if gap < 0:
            return 0
        if gap > self.CAP_DAYS:
            return self.CAP_DAYS
        return gap

    def last_date(self, player_id: str, surface: str) -> date | None:
        entry = self._entries.get((player_id, surface))
        return entry.last_match_date if entry is not None else None

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, player_id: str, surface: str, match_date: date) -> None:
        """Record that `player_id` completed a match on `surface` at
        `match_date`. Idempotent on chronological replay; out-of-order
        updates keep the maximum date — same contract as `LastMatchState.update`.
        """
        key = (player_id, surface)
        existing = self._entries.get(key)
        if existing is None or match_date > existing.last_match_date:
            self._entries[key] = LastMatchPerSurfaceEntry(last_match_date=match_date)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @classmethod
    def from_db(cls, conn: duckdb.DuckDBPyConnection) -> LastMatchPerSurfaceState:
        """Reconstruct state from `last_match_per_surface_state`."""
        state = cls()
        rows = conn.execute(
            "SELECT player_id, surface, last_match_date FROM last_match_per_surface_state"
        ).fetchall()
        for player_id, surface, last_date in rows:
            state._entries[(player_id, surface)] = LastMatchPerSurfaceEntry(
                last_match_date=last_date
            )
        return state

    def save_to_db(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Atomically replace the contents of `last_match_per_surface_state`
        with the current in-memory state. One transaction so partial writes
        cannot corrupt the table.
        """
        rows = [
            (player_id, surface, e.last_match_date)
            for (player_id, surface), e in self._entries.items()
        ]
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DELETE FROM last_match_per_surface_state")
            if rows:
                conn.executemany(
                    "INSERT INTO last_match_per_surface_state "
                    "(player_id, surface, last_match_date) VALUES (?, ?, ?)",
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

    def __contains__(self, key: object) -> bool:
        return key in self._entries
