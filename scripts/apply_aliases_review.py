"""CLI wrapper that promotes reviewed aliases into `player_aliases`.

Defaults to `data/processed/aliases_review.csv`. Re-runnable — already-present
pairs are skipped via ON CONFLICT.

    uv run python scripts/apply_aliases_review.py
    uv run python scripts/apply_aliases_review.py --csv path/to/file.csv

The work survives `refresh_data.py --clean` only if the CSV is still on
disk; consider archiving it after applying.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tennis_predictor import config
from tennis_predictor.data import db, manual_review, schema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=config.PROCESSED_DIR / "aliases_review.csv",
        help="Path to the reviewed CSV file",
    )
    args = parser.parse_args()

    print(f"Reading {args.csv}")
    conn = db.open_connection()
    schema.create_all_tables(conn)
    stats = manual_review.apply_review(conn, args.csv)
    conn.close()

    print(f"  CSV rows:         {stats['csv_rows']:>6d}")
    print(f"  Unique pairs:     {stats['unique_pairs']:>6d}")
    print(f"  Newly inserted:   {stats['newly_inserted']:>6d}")
    print(f"  Already present:  {stats['already_present']:>6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
