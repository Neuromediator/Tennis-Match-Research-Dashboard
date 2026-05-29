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

import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from tennis_predictor.data.refresh_jobs import run_daily_refreshes

log = logging.getLogger(__name__)

_DEFAULT_HOUR_UTC: int = 21
_DEFAULT_MINUTE_UTC: int = 0
_MISFIRE_GRACE_S: int = 3600  # 1h — catch a fire missed due to brief Machine restart


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
    keeps running on failure (a one-day refresh hiccup is not fatal)."""
    try:
        run_daily_refreshes()
    except Exception:
        log.exception("[scheduler] daily refresh raised — scheduler continues")


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
