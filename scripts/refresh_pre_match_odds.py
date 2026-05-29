"""Daily refresh of `pre_match_odds` from The Odds API.

Discovers currently-active tennis sport keys via `/v4/sports/?all=false`,
then iterates each with one `/v4/sports/{key}/odds` call (1 credit per
call against the 500/month free-tier cap). Upserts aggregated rows into
`pre_match_odds`; logs one `ingestion_runs` row per execution.

Examples:
    uv run python scripts/refresh_pre_match_odds.py
    uv run python scripts/refresh_pre_match_odds.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from tennis_predictor import config
from tennis_predictor.data import schema
from tennis_predictor.data.odds_api import OddsApiQuotaExceeded
from tennis_predictor.data.odds_refresh import log_ingestion_run, refresh

logger = logging.getLogger(__name__)


def _open_db(path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(path))
    schema.create_all_tables(conn)
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=config.DUCKDB_PATH,
        help=f"DuckDB path. Default: {config.DUCKDB_PATH}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hit the API but do not write to the DB (useful for cost-only probing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    api_key = config.THE_ODDS_API_KEY
    if not api_key:
        print("THE_ODDS_API_KEY is not set; aborting.", file=sys.stderr)
        return 2

    conn = _open_db(args.db)
    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC)
    status = "succeeded"
    error_message: str | None = None
    rows_added = 0
    requests_used = 0
    try:
        rows_added, requests_used = refresh(conn, api_key, dry_run=args.dry_run)
    except OddsApiQuotaExceeded as exc:
        status = "partial"
        error_message = str(exc)
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
    finished_at = datetime.now(UTC)

    if not args.dry_run:
        log_ingestion_run(
            conn,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            rows_added=rows_added,
            rows_failed=0,
            requests_used=requests_used,
            error_message=error_message,
        )

    duration = (finished_at - started_at).total_seconds()
    print(f"\nrefresh_pre_match_odds {run_id[:8]}  status={status}  duration={duration:.1f}s")
    print(f"  rows upserted: {rows_added}")
    print(f"  requests used (this run): {requests_used}")
    if error_message:
        print(f"  error: {error_message}", file=sys.stderr)
    return 0 if status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
