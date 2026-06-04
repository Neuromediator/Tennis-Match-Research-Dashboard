"""APScheduler integration for the daily refresh.

The scheduler is a singleton per Streamlit process, created lazily by
`get_scheduler()` and cached via `@st.cache_resource` (one instance
across script reruns). It is started exactly once when the first user
visits any page.

Gating:
- `ENABLE_SCHEDULER=true` is required to actually start the scheduler.
  Local `streamlit run` leaves it off so devs don't accidentally hit
  matchstat / Odds API quotas. Production sets it via `fly secrets set`.
- `REFRESH_HOUR_UTC` / `REFRESH_MINUTE_UTC` override the default 21:00 UTC.

Thread model:
- APScheduler runs jobs in a worker thread pool (default size 10).
- Our job opens its own DuckDB connection (see `data.refresh_jobs`),
  separate from the Streamlit-cached connection on the main thread.
- DuckDB serialises writes within one process, so the background
  refresh can run alongside user-facing reads without lock errors.
"""

from __future__ import annotations

import logging
import os
import threading

import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from tennis_predictor.data.refresh_jobs import run_daily_refreshes

log = logging.getLogger(__name__)

_DEFAULT_HOUR_UTC: int = 21
_DEFAULT_MINUTE_UTC: int = 0
_MISFIRE_GRACE_S: int = 3600  # 1h — catch a fire missed due to brief Machine restart

# Serialises refresh runs within the process. The daily cron and the
# catch-up-on-wake job can both call `run_daily_refreshes`; concurrent
# writes to the same DuckDB tables would race, so a second invocation
# while one is in flight is skipped (the in-flight run already covers it).
_refresh_lock = threading.Lock()

# Ensures the catch-up check is attempted at most once per process. On a
# scale-to-zero / sleepy host (HF Spaces sleeps after 48h idle) the cron
# cannot fire while the container is asleep, so the first visit after a
# wake is what triggers the catch-up.
_catch_up_lock = threading.Lock()
_catch_up_attempted = False


def _scheduler_enabled() -> bool:
    return os.environ.get("ENABLE_SCHEDULER", "").lower() in ("1", "true", "yes")


def _refresh_hour() -> int:
    raw = os.environ.get("REFRESH_HOUR_UTC")
    if raw is None:
        return _DEFAULT_HOUR_UTC
    try:
        return int(raw)
    except ValueError:
        log.warning("REFRESH_HOUR_UTC=%r is not an int; falling back to %d", raw, _DEFAULT_HOUR_UTC)
        return _DEFAULT_HOUR_UTC


def _refresh_minute() -> int:
    raw = os.environ.get("REFRESH_MINUTE_UTC")
    if raw is None:
        return _DEFAULT_MINUTE_UTC
    try:
        return int(raw)
    except ValueError:
        log.warning(
            "REFRESH_MINUTE_UTC=%r is not an int; falling back to %d", raw, _DEFAULT_MINUTE_UTC
        )
        return _DEFAULT_MINUTE_UTC


def _daily_job() -> None:
    """APScheduler entry point. Swallows exceptions so the scheduler
    keeps running on failure (a one-day refresh hiccup is not fatal).

    Guarded by `_refresh_lock`: if a refresh is already running (e.g. the
    daily cron fired while a catch-up-on-wake run is still in flight), this
    invocation is skipped rather than racing on the DuckDB writes."""
    if not _refresh_lock.acquire(blocking=False):
        log.info("[scheduler] refresh already in progress — skipping this fire")
        return
    try:
        run_daily_refreshes()
    except Exception:
        log.exception("[scheduler] daily refresh raised — scheduler continues")
    finally:
        _refresh_lock.release()


@st.cache_resource(show_spinner=False)
def get_scheduler() -> BackgroundScheduler | None:
    """Build and start the BackgroundScheduler if `ENABLE_SCHEDULER` is on.

    Returns the scheduler instance (None if gated off). Idempotent —
    `@st.cache_resource` ensures one scheduler per process across script
    reruns. Streamlit closes the scheduler on app shutdown via the
    cache-resource cleanup hook (the scheduler exposes `shutdown()` but
    we rely on the process exit to tear it down)."""
    if not _scheduler_enabled():
        log.info("[scheduler] ENABLE_SCHEDULER not set — scheduler disabled")
        return None

    hour = _refresh_hour()
    minute = _refresh_minute()
    scheduler = BackgroundScheduler(timezone="UTC", daemon=True)
    scheduler.add_job(
        _daily_job,
        trigger=CronTrigger(hour=hour, minute=minute, timezone="UTC"),
        id="daily_refresh",
        coalesce=True,
        misfire_grace_time=_MISFIRE_GRACE_S,
        max_instances=1,
        replace_existing=True,
    )
    scheduler.start()
    log.info("[scheduler] started — daily refresh at %02d:%02d UTC", hour, minute)
    return scheduler


def _hot_data_is_stale() -> bool:
    """Read-only check: is the most recent successful hot refresh older than
    the staleness threshold (or missing entirely)?

    Opens a short-lived read-only DuckDB connection so it never contends
    with the Streamlit-owned write connection. Returns False (no catch-up)
    if the database file does not exist yet — e.g. a fresh host before the
    one-shot bootstrap has populated the volume."""
    from datetime import UTC, datetime

    import duckdb

    from tennis_predictor import config
    from tennis_predictor.app.widgets import is_data_stale, query_last_hot_refresh

    if not config.DUCKDB_PATH.exists():
        return False
    try:
        conn = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    except Exception:
        log.exception("[scheduler] catch-up staleness check could not open DB")
        return False
    try:
        last = query_last_hot_refresh(conn)
    finally:
        conn.close()
    return is_data_stale(last, now=datetime.now(UTC))


def maybe_catch_up_refresh(scheduler: BackgroundScheduler | None) -> None:
    """If hot data is stale, schedule an immediate background refresh.

    This is the primary freshness mechanism on a host that sleeps when
    idle (HF Spaces): the in-process cron cannot fire while the container
    is asleep, so the first page load after a wake triggers the catch-up
    here. Runs at most once per process and never blocks the page render —
    the refresh executes on an APScheduler worker thread, so the current
    request returns immediately (showing the still-stale data); the next
    rerun picks up the freshly-written rows.

    No-op when the scheduler is gated off (`ENABLE_SCHEDULER` unset)."""
    global _catch_up_attempted
    if scheduler is None:
        return
    if not _catch_up_lock.acquire(blocking=False):
        return
    try:
        if _catch_up_attempted:
            return
        _catch_up_attempted = True
        if not _hot_data_is_stale():
            log.info("[scheduler] hot data fresh — no catch-up needed")
            return
        log.info("[scheduler] hot data stale on wake — scheduling catch-up refresh")
        scheduler.add_job(
            _daily_job,
            id="catch_up_refresh",
            max_instances=1,
            replace_existing=True,
            misfire_grace_time=_MISFIRE_GRACE_S,
        )
    finally:
        _catch_up_lock.release()
