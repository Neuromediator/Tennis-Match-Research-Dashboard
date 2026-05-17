"""Promote reviewed aliases into `player_aliases` as manually confirmed.

Given a CSV file with the same schema that `load_market.py` writes for
review (tour, winner_raw, winner_player_id, loser_raw, loser_player_id, ...),
extract each unique (raw_name, tour, canonical_player_id) tuple from both
winner and loser sides and insert into `player_aliases` with
`source='manual_review'` and `confidence=1.0`. Idempotent via
`ON CONFLICT DO NOTHING`.

After running, the same raw names hit the exact-match fast path in
AliasIndex on future refreshes — no more review entries for them.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

MANUAL_SOURCE = "manual_review"

_REQUIRED_COLUMNS = frozenset(
    {"tour", "winner_raw", "winner_player_id", "loser_raw", "loser_player_id"}
)


def apply_review(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> dict[str, int]:
    """Read a reviewed CSV and insert manually-confirmed aliases.

    Returns a summary dict with keys csv_rows, unique_pairs, newly_inserted,
    already_present.
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

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
    all_pairs = all_pairs.dropna(subset=["alias_text", "canonical_player_id"])
    all_pairs = all_pairs[all_pairs["alias_text"].astype(str).str.strip() != ""]

    unique = all_pairs.drop_duplicates(
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
        "csv_rows": len(df),
        "unique_pairs": len(unique),
        "newly_inserted": after - before,
        "already_present": len(unique) - (after - before),
    }


def _count_manual(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM player_aliases WHERE source = ?", [MANUAL_SOURCE]
    ).fetchone()
    return int(row[0]) if row else 0
