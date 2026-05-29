"""In-process refresh jobs called by `app/scheduler.py`.

DuckDB does not support multi-process write access — the file lock is
held by the Streamlit process for its full lifetime. A separate cron
Machine would deadlock on the first open. Instead, the daily refresh
runs in a background thread of the Streamlit process via APScheduler,
opening a fresh DuckDB connection per run.

DuckDB allows multiple connections to the same file *within one process*
(they share the in-memory database instance internally, no lock fight).
Opening a per-job connection avoids cross-thread sharing of the
Streamlit-owned `@st.cache_resource` connection, which is bound to the
main thread.

Weekly Sackmann cold ingest is **out of scope for the scheduler**:
refactoring `scripts/refresh_data.py` into a library function is
meaningful work and weekly Sackmann updates are not user-facing
critical (last 7 days of fixtures come from matchstat, not Sackmann).
Run manually when needed:

    fly machine stop <id>
    fly ssh console
    python scripts/refresh_data.py --skip-submodules
    fly machine start <id>
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb

from tennis_predictor import config
from tennis_predictor.data import schema

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyRefreshResult:
    hot_ok: bool
    hot_error: str | None
    odds_ok: bool
    odds_error: str | None
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.hot_ok and self.odds_ok


def _run_hot(conn: duckdb.DuckDBPyConnection) -> tuple[bool, str | None]:
    """Run matchstat hot refresh. Returns (ok, error_message)."""
    from tennis_predictor.data.matchstat import MatchstatClient
    from tennis_predictor.data.refresh_hot import refresh_hot

    api_key = config.X_RAPIDAPI_KEY
    if not api_key:
        return False, "X_RAPIDAPI_KEY not set"

    try:
        with MatchstatClient(api_key) as client:
            summary = refresh_hot(
                conn,
                client,
                tours=["ATP", "WTA"],
                review_csv_path=config.PROCESSED_DIR / "aliases_review_matchstat.csv",
            )
        log.info(
            "[refresh_jobs] hot done: status=%s requests=%d",
            summary.status,
            summary.requests_used,
        )
        if summary.status == "failed":
            return False, f"refresh_hot failed: {summary.error_message or 'unknown'}"
        return True, None
    except Exception as exc:
        log.exception("[refresh_jobs] hot raised")
        return False, str(exc)


def _run_odds(conn: duckdb.DuckDBPyConnection) -> tuple[bool, str | None]:
    """Run The Odds API pre-match refresh. Returns (ok, error_message)."""
    from tennis_predictor.data.odds_api import OddsApiQuotaExceeded
    from tennis_predictor.data.odds_refresh import log_ingestion_run, refresh

    api_key = config.THE_ODDS_API_KEY
    if not api_key:
        return False, "THE_ODDS_API_KEY not set"

    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC)
    status = "succeeded"
    error_message: str | None = None
    rows_added = 0
    requests_used = 0

    try:
        rows_added, requests_used = refresh(conn, api_key)
    except OddsApiQuotaExceeded as exc:
        status = "partial"
        error_message = str(exc)
    except Exception as exc:
        log.exception("[refresh_jobs] odds raised")
        status = "failed"
        error_message = str(exc)

    finished_at = datetime.now(UTC)
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
    log.info(
        "[refresh_jobs] odds done: status=%s rows=%d requests=%d",
        status,
        rows_added,
        requests_used,
    )
    return status != "failed", error_message


def run_daily_refreshes() -> DailyRefreshResult:
    """Open a fresh DuckDB connection, run hot + odds refresh, close.

    Designed to be called by `app/scheduler.py` from a background
    thread. Each refresh step is isolated — a failure in one does not
    short-circuit the other, mirroring `scripts/refresh_all.py`."""
    started = datetime.now(UTC)
    log.info("[refresh_jobs] starting daily refresh")
    conn = duckdb.connect(str(config.DUCKDB_PATH))
    try:
        schema.create_all_tables(conn)
        hot_ok, hot_err = _run_hot(conn)
        odds_ok, odds_err = _run_odds(conn)
    finally:
        conn.close()
    duration = (datetime.now(UTC) - started).total_seconds()
    result = DailyRefreshResult(
        hot_ok=hot_ok,
        hot_error=hot_err,
        odds_ok=odds_ok,
        odds_error=odds_err,
        duration_s=duration,
    )
    log.info(
        "[refresh_jobs] daily refresh finished in %.1fs hot=%s odds=%s",
        duration,
        hot_ok,
        odds_ok,
    )
    return result
