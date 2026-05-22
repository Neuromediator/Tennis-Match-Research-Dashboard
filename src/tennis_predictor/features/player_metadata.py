"""Player metadata: pure helpers + bulk lookup against the `players` table.

Phase 4.1 added nine player-metadata FeatureVector fields. They are sourced
from `players.hand`, `players.dob`, `players.height` via a JOIN — the
metadata is static so there's no leakage risk and no state object. This
module exposes:

- Three pure helpers — `compute_age`, `compute_age_vs_peak`,
  `compute_height_diff` — unit-tested in
  `tests/test_player_metadata.py`. They turn raw `players` values plus a
  `match_date` / `tour` into the four derived numeric features
  (`age_*`, `age_vs_peak_*`, `height_diff_cm`).
- `PlayerMetadataLookup` — pre-loaded dict keyed by `player_id`. Mirrors
  the `RankingLookup` pattern: read once from DB, query in-memory many
  times. Cheaper than a per-row SQL JOIN inside the chronological replay.

Peak ages per tour are the Phase 4.1 defaults: modern-literature consensus
of ATP=26, WTA=24. Empirical re-fit is a Phase 5+ concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb

DAYS_PER_YEAR: float = 365.25
"""Julian year used to convert (match_date - dob) days into a float age.
Matches the convention in Sackmann's own age computation and keeps leap-
year wobble below 0.003 yr (~1 day)."""

PEAK_AGE: dict[str, float] = {"ATP": 26.0, "WTA": 24.0}
"""Per-tour modal peak age, default for the `age_vs_peak_*` feature. From
the modern-literature consensus cited in `docs/tutorials/phase_4_1_notes.md`.
Refine empirically (from our own data) after Phase 4.1 lands."""

_VALID_HANDS: frozenset[str] = frozenset({"R", "L", "A", "U"})
"""Sackmann's `players.hand` is mostly clean but a handful of legacy rows
carry empty strings or junk. We coerce anything not in this set to 'U' so
the categorical column stays well-defined for LightGBM."""


def compute_age(dob: date | None, match_date: date) -> float | None:
    """Player age in years on `match_date`, or None if `dob` is missing.

    Returns a float; LightGBM benefits from sub-year resolution so we keep
    the decimal part rather than truncating to int years.
    """
    if dob is None:
        return None
    delta_days = (match_date - dob).days
    return delta_days / DAYS_PER_YEAR


def compute_age_vs_peak(age: float | None, tour: str) -> float | None:
    """Signed distance from the tour's peak age, or None if `age` is None.

    Positive = past peak (decline phase), negative = before peak (rise
    phase). The signed-linear form lets LightGBM learn the asymmetric
    rise-vs-decline curve as it sees fit; a squared/absolute encoding
    would force symmetry. See Phase 4.1 design doc default #3.

    Unknown tours raise ValueError — silently returning None would
    mask a wiring bug in the caller. The orchestrator filters to ATP/WTA
    upstream so this only fires on test/script misuse.
    """
    if age is None:
        return None
    if tour not in PEAK_AGE:
        raise ValueError(f"Unknown tour {tour!r}; expected one of {sorted(PEAK_AGE)}")
    return age - PEAK_AGE[tour]


def compute_height_diff(h1: int | None, h2: int | None) -> int | None:
    """Signed height difference in cm (p1 minus p2), or None when either
    height is unknown.

    Redundant with `height_p1` / `height_p2` individually but useful for
    shallow-tree LightGBM — exposing the pairwise diff directly saves
    the model from rediscovering it through axis-aligned splits.
    """
    if h1 is None or h2 is None:
        return None
    return h1 - h2


def normalize_hand(raw: str | None) -> str:
    """Coerce a raw `players.hand` value to one of {R, L, A, U}.

    Anything outside the canonical set (None, empty string, junk) becomes
    'U'. The FeatureVector schema enforces the same set via Literal, so
    pushing the coercion here keeps construction site clean.
    """
    if raw is None:
        return "U"
    cleaned = raw.strip().upper()
    return cleaned if cleaned in _VALID_HANDS else "U"


@dataclass(frozen=True, slots=True)
class PlayerMetadataEntry:
    """One row from `players`, narrowed to the columns Phase 4.1 needs.

    `hand` is already normalized to {R, L, A, U}. `dob` / `height` keep
    their raw nullability."""

    hand: str
    dob: date | None
    height: int | None


class PlayerMetadataLookup:
    """In-memory `player_id -> PlayerMetadataEntry`.

    Built once via `from_db(conn)` at the start of `build_training_features`
    (and inside `compute_features` per call — the table is ~137k rows, the
    read is fast). Misses (players not in the table — e.g., a brand-new
    Sackmann ID that never made it into the players file) return a
    default-unknown entry rather than raising, so the chronological replay
    never aborts on a stale roster.
    """

    _DEFAULT: PlayerMetadataEntry = PlayerMetadataEntry(hand="U", dob=None, height=None)

    def __init__(self, entries: dict[str, PlayerMetadataEntry]) -> None:
        self._entries = entries

    @classmethod
    def from_db(cls, conn: duckdb.DuckDBPyConnection) -> PlayerMetadataLookup:
        rows = conn.execute("SELECT player_id, hand, dob, height FROM players").fetchall()
        entries: dict[str, PlayerMetadataEntry] = {}
        for player_id, hand, dob, height in rows:
            entries[player_id] = PlayerMetadataEntry(
                hand=normalize_hand(hand),
                dob=dob,
                height=int(height) if height is not None else None,
            )
        return cls(entries)

    def get(self, player_id: str) -> PlayerMetadataEntry:
        """Always returns an entry — missing players get the U/None default."""
        return self._entries.get(player_id, self._DEFAULT)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, player_id: object) -> bool:
        return player_id in self._entries
