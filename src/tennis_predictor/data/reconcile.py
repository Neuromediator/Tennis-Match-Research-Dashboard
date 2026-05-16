"""Player name reconciliation.

Sackmann's player_id is canonical. Every other source — tennis-data.co.uk
historical odds, future hot APIs — emits player names in its own format
("Federer R.", "Đoković", "Hsieh Su-wei"). Reconciliation maps these raw
strings to the canonical player_id.

Two pieces:

1. `seed_aliases_from_players(conn, tour)` — populate `player_aliases` with
   identity mappings from the canonical roster. After this runs, exact
   Sackmann names match at confidence 1.0.

2. `AliasIndex` — build once per tour, look up many times. Uses rapidfuzz
   to fuzzy-match against the seeded aliases. Returns a `ReconciliationResult`
   that the caller turns into either an auto-insert, a row in
   `aliases_review.csv`, or a row in `unmatched_*.csv`.

Threshold rules (codified in CLAUDE.md):
- confidence >= 0.90 with no close competitor → "auto"
- confidence in [0.75, 0.90) OR a close competitor near 0.90 → "review"
- confidence < 0.75 → "unknown"
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

import duckdb
from rapidfuzz import fuzz, process

from tennis_predictor.data.ingest_sackmann import Tour, _validate_tour

AUTO_THRESHOLD: float = 0.90
REVIEW_THRESHOLD: float = 0.75
AMBIGUITY_GAP: float = 0.05  # if top two candidates are within this, flag as review

ReconciliationStatus = Literal["auto", "review", "unknown"]


@dataclass(frozen=True)
class ReconciliationResult:
    """Outcome of a fuzzy lookup against the alias index.

    - `canonical_player_id` is None when status == 'unknown'.
    - `confidence` is on a 0..1 scale (rapidfuzz score / 100).
    - `candidate_name` is the alias_text we matched against (or None
      if the index was empty).
    - `runner_up_confidence` is the second-best score, useful for
      diagnosing ambiguity.
    """

    canonical_player_id: str | None
    confidence: float
    status: ReconciliationStatus
    candidate_name: str | None
    runner_up_confidence: float = 0.0


# Some letters used in tennis-player names are *precomposed* and NFKD does
# not split them into a base letter + combining mark — we need to substitute
# them explicitly before normalization. The list covers common ATP/WTA cases:
# Croatian/Serbian (Đ), Polish (Ł), Norwegian/Danish (Ø, Æ), Icelandic (Þ),
# German (ß). Add to this map if a future name doesn't normalize cleanly.
_PRECOMPOSED_REPLACEMENTS = str.maketrans(
    {
        "Đ": "D",
        "đ": "d",
        "Ł": "L",
        "ł": "l",
        "Ø": "O",
        "ø": "o",
        "Æ": "AE",
        "æ": "ae",
        "Þ": "Th",
        "þ": "th",
        "ß": "ss",
    }
)


def normalize_name(raw: str | None) -> str:
    """Strip diacritics, lower-case, remove punctuation, collapse whitespace.

    "Đoković, N." -> "dokovic n"
    "Federer R."  -> "federer r"
    """
    if not raw:
        return ""
    substituted = raw.translate(_PRECOMPOSED_REPLACEMENTS)
    decomposed = unicodedata.normalize("NFKD", substituted)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = without_marks.lower()
    for ch in (".", ",", "'", '"', "-"):
        cleaned = cleaned.replace(ch, " ")
    return " ".join(cleaned.split())


# ---------------------------------------------------------------------------
# Seeding


def seed_aliases_from_players(conn: duckdb.DuckDBPyConnection, tour: Tour) -> int:
    """Identity-seed player_aliases from the canonical players roster.

    For each canonical Sackmann player, inserts up to three alias rows:
    - "First Last"  — Sackmann's canonical full name
    - "Last First"  — reversed order (some sources use this)
    - "Last F"      — abbreviated form (tennis-data.co.uk uses this everywhere)

    All seeded with confidence 1.0 and source='sackmann'. Idempotent via the
    PRIMARY KEY on (alias_text, tour, source).

    Players with name_first or name_last equal to 'Unknown' are deliberately
    excluded — these are Sackmann placeholders for partially-known historical
    players, and seeding them creates wide false-positive fuzzy matches.

    Returns the number of new rows inserted across all alias forms.
    """
    _validate_tour(tour)
    before = _alias_count(conn, tour)
    conn.execute(
        """
        INSERT INTO player_aliases BY NAME
        WITH eligible AS (
            SELECT player_id, name_first, name_last, full_name, tour
            FROM players
            WHERE tour = ?
              AND full_name IS NOT NULL AND full_name <> ''
              AND name_first IS NOT NULL AND name_first <> '' AND name_first <> 'Unknown'
              AND name_last  IS NOT NULL AND name_last  <> '' AND name_last  <> 'Unknown'
        ),
        forms AS (
            SELECT player_id, tour, full_name AS alias_text FROM eligible
            UNION ALL
            SELECT player_id, tour, name_last || ' ' || name_first AS alias_text FROM eligible
            UNION ALL
            SELECT player_id, tour,
                   name_last || ' ' || SUBSTRING(name_first, 1, 1) AS alias_text
            FROM eligible
        )
        SELECT
            alias_text,
            tour,
            'sackmann' AS source,
            player_id AS canonical_player_id,
            1.0 AS confidence
        FROM forms
        ON CONFLICT (alias_text, tour, source) DO NOTHING
        """,
        [tour],
    )
    return _alias_count(conn, tour) - before


def find_namesakes(conn: duckdb.DuckDBPyConnection, tour: Tour) -> list[tuple[str, list[str]]]:
    """Return (full_name, [player_id, ...]) for Sackmann players sharing a name."""
    _validate_tour(tour)
    rows = conn.execute(
        """
        SELECT full_name, list(player_id) AS ids
        FROM players
        WHERE tour = ? AND full_name IS NOT NULL AND full_name <> ''
        GROUP BY full_name
        HAVING COUNT(*) > 1
        ORDER BY full_name
        """,
        [tour],
    ).fetchall()
    return list(rows)


# ---------------------------------------------------------------------------
# Lookup


class AliasIndex:
    """In-memory fuzzy index over `player_aliases` for one tour.

    Build once, look up many times. Cheap enough to rebuild whenever the
    underlying table changes (memory only).
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, tour: Tour) -> None:
        _validate_tour(tour)
        self.tour: Tour = tour
        rows = conn.execute(
            "SELECT alias_text, canonical_player_id FROM player_aliases WHERE tour = ?",
            [tour],
        ).fetchall()
        self._choices: list[str] = []  # normalized
        self._raw_for_norm: dict[str, str] = {}  # normalized -> first raw seen
        self._canonical_for_norm: dict[str, str] = {}
        for raw, canonical in rows:
            norm = normalize_name(raw)
            if not norm or norm in self._canonical_for_norm:
                continue
            self._choices.append(norm)
            self._raw_for_norm[norm] = raw
            self._canonical_for_norm[norm] = canonical

    def __len__(self) -> int:
        return len(self._choices)

    def lookup(self, raw_name: str) -> ReconciliationResult:
        if not self._choices:
            return ReconciliationResult(None, 0.0, "unknown", None)

        query = normalize_name(raw_name)
        if not query:
            return ReconciliationResult(None, 0.0, "unknown", None)

        # extract top-2 so we can detect tight-cluster ambiguity
        top = process.extract(query, self._choices, scorer=fuzz.WRatio, limit=2)
        if not top:
            return ReconciliationResult(None, 0.0, "unknown", None)

        best_choice, best_score, _ = top[0]
        confidence = best_score / 100.0
        canonical = self._canonical_for_norm[best_choice]
        candidate_raw = self._raw_for_norm[best_choice]

        # Ambiguity is only real when the runner-up points to a *different*
        # player. Since each player has multiple seeded alias forms (canonical,
        # reversed, abbreviated), close hits on the same canonical_player_id
        # are reinforcement, not ambiguity.
        runner_up_score = 0.0
        ambiguous = False
        if len(top) > 1:
            runner_choice, runner_raw_score, _ = top[1]
            runner_up_score = runner_raw_score / 100.0
            runner_canonical = self._canonical_for_norm[runner_choice]
            if runner_canonical != canonical:
                ambiguous = (
                    confidence >= AUTO_THRESHOLD
                    and runner_up_score >= AUTO_THRESHOLD - AMBIGUITY_GAP
                    and (confidence - runner_up_score) < AMBIGUITY_GAP
                )

        if confidence >= AUTO_THRESHOLD and not ambiguous:
            status: ReconciliationStatus = "auto"
        elif confidence >= REVIEW_THRESHOLD or ambiguous:
            status = "review"
        else:
            return ReconciliationResult(None, confidence, "unknown", candidate_raw, runner_up_score)

        return ReconciliationResult(canonical, confidence, status, candidate_raw, runner_up_score)


# ---------------------------------------------------------------------------
# Helpers


def _alias_count(conn: duckdb.DuckDBPyConnection, tour: Tour) -> int:
    row = conn.execute("SELECT COUNT(*) FROM player_aliases WHERE tour = ?", [tour]).fetchone()
    return int(row[0]) if row is not None else 0
