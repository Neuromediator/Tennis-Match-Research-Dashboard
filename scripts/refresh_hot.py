"""Daily hot-data refresh from matchstat.

Examples:
    uv run python scripts/refresh_hot.py
    uv run python scripts/refresh_hot.py --tours ATP
    uv run python scripts/refresh_hot.py --date 2026-05-15
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import duckdb

from tennis_predictor import config
from tennis_predictor.data import schema
from tennis_predictor.data.matchstat import MatchstatClient
from tennis_predictor.data.refresh_hot import RefreshSummary, refresh_hot

DEFAULT_REVIEW_CSV = config.PROCESSED_DIR / "aliases_review_matchstat.csv"


def _print_summary(s: RefreshSummary) -> None:
    duration_s = (s.finished_at - s.started_at).total_seconds()
    print(f"\nrefresh_hot {s.run_id[:8]}  status={s.status}  duration={duration_s:.1f}s")
    print(f"  requests used: {s.requests_used}")
    for tour, ts in s.per_tour.items():
        print(f"  [{tour}]")
        for label, counts in (
            ("matches main ", ts.matches),
            ("matches quali", ts.qualifying),
            ("market odds  ", ts.market_odds),
            ("fixtures     ", ts.fixtures),
            ("rankings     ", ts.rankings),
        ):
            print(
                f"    {label}: added={counts.added:>4d}  skipped={counts.skipped:>4d}  "
                f"failed={counts.failed:>4d}"
            )
    print(f"  promoted fixtures (now completed): {s.promoted_fixtures}")
    print(f"  review candidates written:        {s.review_candidates_written}")
    if s.error_message:
        print(f"  error: {s.error_message}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tours",
        nargs="+",
        default=["ATP", "WTA"],
        choices=["ATP", "WTA"],
        help="Tours to refresh (default: both).",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help='Treat this date as "today" (YYYY-MM-DD). Defaults to system today UTC.',
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=config.DUCKDB_PATH,
        help=f"DuckDB path. Default: {config.DUCKDB_PATH}",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=DEFAULT_REVIEW_CSV,
        help=f"Where to append review-band resolutions. Default: {DEFAULT_REVIEW_CSV}",
    )
    parser.add_argument(
        "--fixture-lookahead-days",
        type=int,
        default=1,
        help="How many days past 'today' to pull fixtures for (default 1 — today + tomorrow).",
    )
    args = parser.parse_args()

    if config.X_RAPIDAPI_KEY is None:
        print(
            "ERROR: X_RAPIDAPI_KEY not set in environment / .env",
            file=sys.stderr,
        )
        return 1

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(args.db))
    schema.create_all_tables(conn)

    with MatchstatClient(config.X_RAPIDAPI_KEY) as client:
        summary = refresh_hot(
            conn,
            client,
            tours=args.tours,
            today=args.date,
            review_csv_path=args.review_csv,
            fixture_lookahead_days=args.fixture_lookahead_days,
        )

    _print_summary(summary)
    return 0 if summary.status in ("success", "partial") else 2


if __name__ == "__main__":
    raise SystemExit(main())
