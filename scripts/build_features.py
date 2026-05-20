"""Build the `training_features` table by replaying every match.

One of the two sanctioned feature entry points (CLAUDE.md hard rule #2).
Wraps `tennis_predictor.features.build_training_features` with the
default DuckDB path and basic logging.

Usage:
    uv run python scripts/build_features.py
    uv run python scripts/build_features.py --db data/processed/tennis.duckdb

Idempotent: each run DELETEs all `training_features` rows and re-INSERTs.
`elo_state` is overwritten as a full snapshot at the end.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import duckdb

from tennis_predictor.config import DATA_DIR
from tennis_predictor.features import build_training_features

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=DATA_DIR / "processed" / "tennis.duckdb",
        help="Path to the DuckDB file (default: data/processed/tennis.duckdb).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.db.exists():
        raise SystemExit(f"DuckDB not found at {args.db} — run `scripts/refresh_data.py` first.")

    logger.info("Building training_features from %s", args.db)
    conn = duckdb.connect(str(args.db))
    try:
        t0 = time.time()
        summary = build_training_features(conn)
        elapsed = time.time() - t0
        logger.info("Total wall time: %.1fs", elapsed)
        print(
            f"\nDone. {summary.training_rows_written:,} training_features rows written "
            f"(from {summary.matches_scanned:,} matches scanned).\n"
            f"Elo persisted to elo_state."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
