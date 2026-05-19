"""Promote reviewed aliases into `player_aliases` as manually confirmed.

Two CSV schemas are supported and auto-detected by their column set:

- **Paired format** (Phase 1, written by `load_market.py`): one row per
  match with both winner and loser names already linked to canonical IDs.
  Confirms each unique (raw, canonical) pair from both sides.
- **Per-player format** (Phase 2, written by `refresh_hot.py` for the
  matchstat hot source): one row per low-confidence resolution, the
  reviewer fills a `verdict` column with 'y' to confirm. Rows without
  'y' are dropped.

In both cases, confirmed pairs INSERT into `player_aliases` with
`source='manual_review'` and `confidence=1.0`. Idempotent via
`ON CONFLICT DO NOTHING`. After running, the same raw names hit the
exact-match fast path in AliasIndex on future refreshes.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

MANUAL_SOURCE = "manual_review"

# Each tuple = required columns of one supported schema. The first tuple
# whose columns are all present in the CSV wins; the other branch is skipped.
_PAIRED_REQUIRED = frozenset(
    {"tour", "winner_raw", "winner_player_id", "loser_raw", "loser_player_id"}
)
_MATCHSTAT_REQUIRED = frozenset({"raw_name", "tour", "candidate_canonical_id", "verdict"})


def apply_review(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> dict[str, int]:
    """Read a reviewed CSV and insert manually-confirmed aliases.

    Auto-detects format (paired vs per-player). Returns a summary dict
    with keys csv_rows, unique_pairs, newly_inserted, already_present.
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    columns = set(df.columns)

    if _PAIRED_REQUIRED.issubset(columns):
        return _apply_paired(conn, df)
    if _MATCHSTAT_REQUIRED.issubset(columns):
        return _apply_matchstat(conn, df)
    raise ValueError(
        f"{csv_path} columns {sorted(columns)} match neither the paired "
        f"format ({sorted(_PAIRED_REQUIRED)}) nor the matchstat format "
        f"({sorted(_MATCHSTAT_REQUIRED)})."
    )


def _apply_paired(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> dict[str, int]:
    """Paired (Phase 1) schema: winner+loser both pre-linked to canonicals."""
    winner_pairs = df[["winner_raw", "tour", "winner_player_id"]].rename(
        columns={  # pyright: ignore[reportCallIssue]
            "winner_raw": "alias_text",
            "winner_player_id": "canonical_player_id",
        }
    )
    loser_pairs = df[["loser_raw", "tour", "loser_player_id"]].rename(
        columns={  # pyright: ignore[reportCallIssue]
            "loser_raw": "alias_text",
            "loser_player_id": "canonical_player_id",
        }
    )
    all_pairs = pd.concat([winner_pairs, loser_pairs], ignore_index=True)
    return _insert_aliases(conn, df_rows=len(df), pairs=all_pairs)


def _apply_matchstat(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> dict[str, int]:
    """Per-player (Phase 2) schema: reviewer marks `verdict='y'` to confirm
    a `raw_name -> candidate_canonical_id` link.

    Anything other than literal 'y' / 'Y' (case-insensitive, trimmed) is a
    reject — including blank, 'n', 'no', 'skip'.
    """
    verdicts = df["verdict"].astype(str).str.strip().str.lower()
    confirmed: pd.DataFrame = df[verdicts == "y"].copy()  # pyright: ignore[reportAssignmentType]
    pairs: pd.DataFrame = confirmed[["raw_name", "tour", "candidate_canonical_id"]].rename(
        columns={  # pyright: ignore[reportCallIssue]
            "raw_name": "alias_text",
            "candidate_canonical_id": "canonical_player_id",
        }
    )
    return _insert_aliases(conn, df_rows=len(df), pairs=pairs)


def _insert_aliases(
    conn: duckdb.DuckDBPyConnection, *, df_rows: int, pairs: pd.DataFrame
) -> dict[str, int]:
    pairs = pairs.dropna(subset=["alias_text", "canonical_player_id"])
    pairs = pairs[pairs["alias_text"].astype(str).str.strip() != ""]  # pyright: ignore[reportAssignmentType]
    pairs = pairs[pairs["canonical_player_id"].astype(str).str.strip() != ""]  # pyright: ignore[reportAssignmentType]

    unique = pairs.drop_duplicates(
        subset=["alias_text", "tour", "canonical_player_id"]  # pyright: ignore[reportCallIssue]
    ).copy()

    before = _count_manual(conn)

    if not unique.empty:
        unique["source"] = MANUAL_SOURCE
        unique["confidence"] = 1.0
        conn.register("manual_aliases", unique)
        try:
            conn.execute(
                """
                INSERT INTO player_aliases BY NAME
                SELECT alias_text, tour, source, canonical_player_id, confidence
                FROM manual_aliases
                ON CONFLICT (alias_text, tour, source) DO NOTHING
                """
            )
        finally:
            conn.unregister("manual_aliases")

    after = _count_manual(conn)
    return {
        "csv_rows": df_rows,
        "unique_pairs": len(unique),
        "newly_inserted": after - before,
        "already_present": len(unique) - (after - before),
    }


def _count_manual(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM player_aliases WHERE source = ?", [MANUAL_SOURCE]
    ).fetchone()
    return int(row[0]) if row else 0
