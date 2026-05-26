"""Detect duplicate players in Sackmann roster — Phase 4.2 follow-up.

Sackmann's `players` table occasionally carries two (or more) rows for
the same physical player: identical `full_name` + `dob` + `tour` +
`ioc`, different `sackmann_id`. Live case: Martin Landaluce (b.
2006-01-08, ESP) had `ATP_212021` (114 hard / 79 clay matches) AND
`ATP_211776` (2 hard matches). `AliasIndex` automerged "Martin
Landaluce" → `ATP_211776` (the duplicate), so the Match-dashboard
surface-Elo was the duplicate's 1500 default instead of the real
player's 1801 clay rating.

This script surfaces every duplicate group in the roster, picks the
canonical row (most matches in `matches`), writes the rest to
`data/processed/duplicate_players_review.csv` for manual approval.

After review, run `scripts/apply_player_dedupe.py` to repoint
aliases / scheduled_matches / matches to the canonical IDs and drop
the stale rows.

Detection criteria:
- Group on `(tour, full_name, dob, ioc)` — all four must match.
- Only groups with at least one matched row AND dob non-null are
  surfaced. Without DOB the "namesake vs duplicate" question is
  ambiguous (Phase 1 noted ~840 ATP namesake pairs); we don't auto-
  merge those.

Output columns mirror the existing `aliases_review.csv` pattern — one
row per stale player, with `verdict` blank for the reviewer.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from tennis_predictor import config

REVIEW_CSV_DEFAULT: Path = config.PROCESSED_DIR / "duplicate_players_review.csv"

REVIEW_FIELDNAMES = [
    "group_id",
    "tour",
    "full_name",
    "dob",
    "ioc",
    "canonical_player_id",
    "canonical_matches",
    "stale_player_id",
    "stale_matches",
    "verdict",  # reviewer writes 'y' to approve repoint, blank to skip
]


@dataclass(frozen=True)
class PlayerRow:
    player_id: str
    match_count: int


def _detect_groups(
    conn: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str, date, str | None, list[PlayerRow]]]:
    """Return list of `(tour, full_name, dob, ioc, sorted_rows)` for every
    multi-row group, sorted by match_count desc within the group."""
    rows = conn.execute(
        """
        WITH match_counts AS (
            SELECT player_id, COUNT(*) AS n
            FROM (
                SELECT winner_player_id AS player_id FROM matches WHERE match_status = 'completed'
                UNION ALL
                SELECT loser_player_id AS player_id FROM matches WHERE match_status = 'completed'
            )
            GROUP BY player_id
        ),
        candidates AS (
            SELECT tour, full_name, dob, ioc, COUNT(*) AS n_ids
            FROM players
            WHERE full_name IS NOT NULL
              AND dob IS NOT NULL
            GROUP BY tour, full_name, dob, ioc
            HAVING COUNT(*) > 1
        )
        SELECT p.tour, p.full_name, p.dob, p.ioc, p.player_id,
               COALESCE(mc.n, 0) AS match_count
        FROM players p
        JOIN candidates c
          ON p.tour = c.tour
         AND p.full_name = c.full_name
         AND p.dob = c.dob
         AND ((p.ioc IS NULL AND c.ioc IS NULL) OR p.ioc = c.ioc)
        LEFT JOIN match_counts mc ON mc.player_id = p.player_id
        ORDER BY p.full_name, p.dob, p.tour, match_count DESC, p.player_id
        """
    ).fetchall()

    groups: dict[tuple[str, str, date, str | None], list[PlayerRow]] = defaultdict(list)
    for tour, full_name, dob, ioc, player_id, match_count in rows:
        groups[(tour, full_name, dob, ioc)].append(PlayerRow(player_id, int(match_count)))

    return [(*key, rows) for key, rows in groups.items()]


def _write_csv(
    groups: list[tuple[str, str, date, str | None, list[PlayerRow]]],
    out_path: Path,
) -> int:
    """Write one CSV row per (canonical, stale) pair. Returns count.

    For a group of N members the canonical is the highest-match-count row;
    the remaining N-1 are written as stale candidates.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        for idx, (tour, full_name, dob, ioc, rows) in enumerate(groups, start=1):
            if len(rows) < 2:
                continue
            canonical = rows[0]
            for stale in rows[1:]:
                writer.writerow(
                    {
                        "group_id": idx,
                        "tour": tour,
                        "full_name": full_name,
                        "dob": dob.isoformat() if dob is not None else "",
                        "ioc": ioc or "",
                        "canonical_player_id": canonical.player_id,
                        "canonical_matches": canonical.match_count,
                        "stale_player_id": stale.player_id,
                        "stale_matches": stale.match_count,
                        "verdict": "",
                    }
                )
                written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=config.DUCKDB_PATH,
        help=f"DuckDB path. Default: {config.DUCKDB_PATH}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REVIEW_CSV_DEFAULT,
        help=f"Output CSV path. Default: {REVIEW_CSV_DEFAULT}",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Print this many sample groups to stdout (default 10).",
    )
    args = parser.parse_args()

    conn = duckdb.connect(str(args.db), read_only=True)
    groups = _detect_groups(conn)

    if not groups:
        print("No duplicate player groups found.")
        return 0

    multi = [g for g in groups if len(g[4]) > 1]
    total_stale = sum(len(g[4]) - 1 for g in multi)
    canonical_with_matches = sum(1 for g in multi if g[4][0].match_count > 0)

    print(f"Duplicate groups found: {len(multi)}")
    print(f"  → {total_stale} stale player rows to consider repointing")
    print(f"  → {canonical_with_matches} groups have a canonical row with ≥1 match")
    print()

    n = _write_csv(multi, args.out)
    print(f"Wrote {n} stale-candidate rows to {args.out}")
    print()

    # Print sample
    print(f"Sample groups (top {args.sample} by canonical match count):")
    for tour, full_name, dob, ioc, rows in sorted(multi, key=lambda g: -g[4][0].match_count)[
        : args.sample
    ]:
        ids_str = ", ".join(f"{r.player_id}({r.match_count}m)" for r in rows)
        print(f"  [{tour}] {full_name}  dob={dob}  ioc={ioc or '-'}")
        print(f"    candidates: {ids_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
