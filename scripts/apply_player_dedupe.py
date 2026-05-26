"""Apply approved duplicate-player merges — Phase 4.2 follow-up.

Reads `data/processed/duplicate_players_review.csv` and, for each row
where `verdict == 'y'`, repoints every reference from the stale
player_id to the canonical player_id across:

- `player_aliases`           — alias rows pointing at the stale ID
- `scheduled_matches`        — player1_canonical_id / player2_canonical_id
- `matches`                  — winner_player_id / loser_player_id
- `rankings`                 — player_id
- `elo_state`                — entries for the stale ID (re-keyed onto
                               canonical; on collision the canonical row
                               wins because it has more matches behind it)
- `last_match_state`         — same
- `last_match_per_surface_state` — same
- `training_features`        — DELETED for the stale player; the next
                               `scripts/build_features.py` run rebuilds
                               them under the canonical key.

After all repoints, the stale row in `players` is DELETED.

Workflow:
  1. Stop Streamlit (this script needs a write lock on the DB).
  2. uv run python scripts/find_duplicate_players.py
  3. Open data/processed/duplicate_players_review.csv, write 'y' next
     to every row you want to apply. Save.
  4. uv run python scripts/apply_player_dedupe.py
  5. uv run python scripts/build_features.py
     (rebuilds elo_state, last_match_state, training_features so they
     reflect the new canonical IDs and the migrated match counts.)

Use --dry-run to preview row counts without writing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb

from tennis_predictor import config

REVIEW_CSV_DEFAULT: Path = config.PROCESSED_DIR / "duplicate_players_review.csv"


@dataclass(frozen=True)
class MergePair:
    """One approved (canonical, stale) pair."""

    canonical_id: str
    stale_id: str
    full_name: str


def _read_approved(csv_path: Path) -> list[MergePair]:
    """Read review CSV; keep rows where verdict (case-insensitive) is one
    of {'y', 'yes', 'approved'}."""
    if not csv_path.exists():
        raise FileNotFoundError(f"review CSV not found: {csv_path}")
    out: list[MergePair] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict not in {"y", "yes", "approved"}:
                continue
            canonical = (row.get("canonical_player_id") or "").strip()
            stale = (row.get("stale_player_id") or "").strip()
            full_name = (row.get("full_name") or "").strip()
            if not canonical or not stale or canonical == stale:
                continue
            out.append(MergePair(canonical_id=canonical, stale_id=stale, full_name=full_name))
    return out


def _apply_pair(
    conn: duckdb.DuckDBPyConnection,
    pair: MergePair,
    dry_run: bool,
) -> dict[str, int]:
    """Repoint everything referencing `pair.stale_id` to `pair.canonical_id`.
    Returns row counts touched per table."""
    counts: dict[str, int] = defaultdict(int)

    # Helper: count then act
    def count_then(query: str, params: list) -> int:
        n = conn.execute(query, params).fetchone()
        return int(n[0]) if n is not None else 0

    # player_aliases — repoint
    n = count_then(
        "SELECT COUNT(*) FROM player_aliases WHERE canonical_player_id = ?",
        [pair.stale_id],
    )
    counts["aliases"] = n
    if n and not dry_run:
        conn.execute(
            "UPDATE player_aliases SET canonical_player_id = ? WHERE canonical_player_id = ?",
            [pair.canonical_id, pair.stale_id],
        )

    # scheduled_matches — both player slots
    for col in ("player1_canonical_id", "player2_canonical_id"):
        n = count_then(
            f"SELECT COUNT(*) FROM scheduled_matches WHERE {col} = ?",
            [pair.stale_id],
        )
        counts[f"scheduled.{col}"] = n
        if n and not dry_run:
            conn.execute(
                f"UPDATE scheduled_matches SET {col} = ? WHERE {col} = ?",
                [pair.canonical_id, pair.stale_id],
            )

    # matches — winner / loser
    for col in ("winner_player_id", "loser_player_id"):
        n = count_then(
            f"SELECT COUNT(*) FROM matches WHERE {col} = ?",
            [pair.stale_id],
        )
        counts[f"matches.{col}"] = n
        if n and not dry_run:
            conn.execute(
                f"UPDATE matches SET {col} = ? WHERE {col} = ?",
                [pair.canonical_id, pair.stale_id],
            )

    # rankings — player_id (may collide on (ranking_date, player_id) PK;
    # in that case DELETE the stale row and trust the canonical).
    n = count_then(
        "SELECT COUNT(*) FROM rankings WHERE player_id = ?",
        [pair.stale_id],
    )
    counts["rankings"] = n
    if n and not dry_run:
        # First delete rows where canonical already has that date.
        conn.execute(
            """
            DELETE FROM rankings
            WHERE player_id = ?
              AND ranking_date IN (SELECT ranking_date FROM rankings WHERE player_id = ?)
            """,
            [pair.stale_id, pair.canonical_id],
        )
        # Then repoint what's left.
        conn.execute(
            "UPDATE rankings SET player_id = ? WHERE player_id = ?",
            [pair.canonical_id, pair.stale_id],
        )

    # State tables — DROP stale rows. They'll be rebuilt by the next
    # `build_features.py` run anyway, and we can't safely merge two Elo
    # ratings into one (the math doesn't average linearly).
    for table in ("elo_state", "last_match_state", "last_match_per_surface_state"):
        n = count_then(f"SELECT COUNT(*) FROM {table} WHERE player_id = ?", [pair.stale_id])
        counts[table] = n
        if n and not dry_run:
            conn.execute(f"DELETE FROM {table} WHERE player_id = ?", [pair.stale_id])
        # Also drop canonical rows so they get rebuilt cleanly with the
        # migrated matches. Same reasoning.
        n_c = count_then(f"SELECT COUNT(*) FROM {table} WHERE player_id = ?", [pair.canonical_id])
        counts[f"{table}.canonical_dropped"] = n_c
        if n_c and not dry_run:
            conn.execute(f"DELETE FROM {table} WHERE player_id = ?", [pair.canonical_id])

    # training_features — drop rows referencing either ID; build_features
    # rebuilds them under the canonical key.
    for col in ("p1_player_id", "p2_player_id"):
        n = count_then(
            f"SELECT COUNT(*) FROM training_features WHERE {col} IN (?, ?)",
            [pair.stale_id, pair.canonical_id],
        )
        counts[f"training_features.{col}"] = n
        if n and not dry_run:
            conn.execute(
                f"DELETE FROM training_features WHERE {col} IN (?, ?)",
                [pair.stale_id, pair.canonical_id],
            )

    # Finally — drop the stale player row.
    n = count_then("SELECT COUNT(*) FROM players WHERE player_id = ?", [pair.stale_id])
    counts["players"] = n
    if n and not dry_run:
        conn.execute("DELETE FROM players WHERE player_id = ?", [pair.stale_id])

    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=config.DUCKDB_PATH)
    parser.add_argument("--csv", type=Path, default=REVIEW_CSV_DEFAULT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print row counts that WOULD be touched; do not write.",
    )
    args = parser.parse_args()

    pairs = _read_approved(args.csv)
    if not pairs:
        print("No approved rows in CSV (verdict column empty / not 'y'). Nothing to do.")
        return 0

    print(f"Approved merges: {len(pairs)}")
    if args.dry_run:
        print("DRY RUN — no writes.")
    print()

    conn = duckdb.connect(str(args.db))

    # Show a few examples before action
    sample = pairs[:5]
    print("First few approved pairs:")
    for p in sample:
        print(f"  {p.full_name}: {p.stale_id} -> {p.canonical_id}")
    if len(pairs) > len(sample):
        print(f"  ... and {len(pairs) - len(sample)} more")
    print()

    grand_total: dict[str, int] = defaultdict(int)
    for pair in pairs:
        counts = _apply_pair(conn, pair, dry_run=args.dry_run)
        for k, v in counts.items():
            grand_total[k] += v

    print("Grand total rows touched:")
    for k, v in sorted(grand_total.items()):
        print(f"  {k:<48} {v:>8}")
    print()

    if not args.dry_run:
        print("Dedupe applied. Next step: re-run `scripts/build_features.py` to")
        print("rebuild elo_state, last_match_state, last_match_per_surface_state,")
        print("and training_features under the merged canonical IDs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
