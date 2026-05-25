"""One-shot helper to clear `scheduled_matches` ahead of a fresh ingest.

Phase 6.1 changed the default `MATCHSTAT_SOURCE_TZ` from `UTC` to
`Europe/Moscow` to undo matchstat's empirical Moscow-time-labelled-as-Z
bug. Rows ingested before the default flip are off by ~3 hours; the
on-conflict UPSERT in `load_hot.py` will fix the timestamp on the next
refresh for any fixture matchstat is still serving, but completed /
dropped fixtures stay with the wrong stored time.

Easiest fix: truncate `scheduled_matches` once, then re-run the daily
hot refresh, which repopulates it cleanly under the new default.

Usage:
    uv run python scripts/clear_scheduled_matches.py
    uv run python scripts/refresh_hot.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from tennis_predictor import config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=config.DUCKDB_PATH,
        help=f"DuckDB path. Default: {config.DUCKDB_PATH}",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DuckDB file not found at {args.db}; nothing to clear.", file=sys.stderr)
        return 1

    conn = duckdb.connect(str(args.db))
    try:
        before_row = conn.execute("SELECT COUNT(*) FROM scheduled_matches").fetchone()
        before = int(before_row[0]) if before_row else 0
        conn.execute("DELETE FROM scheduled_matches")
        print(f"Cleared {before} rows from scheduled_matches in {args.db}.")
        print("Now run: uv run python scripts/refresh_hot.py")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
