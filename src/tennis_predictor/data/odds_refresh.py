"""Library entry point for The Odds API pre-match refresh.

Pulled out of `scripts/refresh_pre_match_odds.py` so both the CLI and
the in-process APScheduler job in `src/tennis_predictor/data/refresh_jobs.py`
share the same orchestrator. The CLI still wraps argparse / logging
setup; this module is pure orchestration with a conn + api_key.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import duckdb

from tennis_predictor.data.odds_api import (
    OddsApiClient,
    OddsApiError,
    OddsApiQuotaExceeded,
    aggregate_events,
)
from tennis_predictor.data.pre_match_odds import (
    check_quota_or_raise,
    increment_quota,
    upsert_aggregated,
)

logger = logging.getLogger(__name__)


def refresh(
    conn: duckdb.DuckDBPyConnection,
    api_key: str,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> tuple[int, int]:
    """Run the refresh end-to-end. Returns (rows_upserted, requests_used).

    On quota exhaustion, bails after the last successful sport_key and
    surfaces `(rows_so_far, requests_so_far)` so the caller can log a
    partial run rather than a total failure."""
    moment = now or datetime.now(UTC)
    rows_upserted = 0
    requests_used = 0

    with OddsApiClient(api_key) as client:
        check_quota_or_raise(conn, moment)
        sports = client.list_active_tennis_sports()
        requests_used += 1
        # Discovery is documented as free, but we still increment by
        # 0 for parity — bookkeeping clarity matters more than a
        # hypothetical 1-credit difference.

        logger.info("active tennis sport keys: %d", len(sports))
        for sport in sports:
            try:
                check_quota_or_raise(conn, moment)
            except OddsApiQuotaExceeded:
                logger.warning("quota exhausted mid-run; stopping after %s", sport.key)
                break
            try:
                events = client.fetch_odds(sport.key)
            except OddsApiError as exc:
                logger.warning("fetch_odds(%s) failed: %s", sport.key, exc)
                continue
            requests_used += 1
            if not dry_run:
                increment_quota(conn, 1, moment)
            aggregated = aggregate_events(events)
            logger.info(
                "%s: %d events → %d aggregated rows", sport.key, len(events), len(aggregated)
            )
            if not dry_run and aggregated:
                rows_upserted += upsert_aggregated(conn, aggregated, now=moment)

    return rows_upserted, requests_used


def log_ingestion_run(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    rows_added: int,
    rows_failed: int,
    requests_used: int,
    error_message: str | None,
) -> None:
    """Record one `ingestion_runs` row for an odds-refresh execution."""
    conn.execute(
        """
        INSERT INTO ingestion_runs (
            run_id, source, tour, started_at, finished_at, status,
            rows_added, rows_skipped, rows_failed, requests_used,
            error_message, notes
        ) VALUES (?, 'the_odds_api', NULL, ?, ?, ?, ?, NULL, ?, ?, ?, NULL)
        """,
        [
            run_id,
            started_at.replace(tzinfo=None),
            finished_at.replace(tzinfo=None),
            status,
            rows_added,
            rows_failed,
            requests_used,
            error_message,
        ],
    )
