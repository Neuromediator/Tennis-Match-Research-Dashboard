"""Daily LLM call budget cap (Phase 7).

A simple soft cap on how many `LLMClient.acall` traces happen per UTC
day. Counts existing `llm_traces` rows since today UTC midnight; when
the count meets or exceeds `DAILY_LLM_BUDGET`, future `agent.predict`
invocations skip the LLM entirely and fall back to a direct
`get_model_prediction` call. The user-visible page still renders the
model + market + surface-Elo blocks; only the LLM news block is paused.

The cap is **soft** (not transactional): a prediction that starts at
59/60 will run to completion and push the count to ~62/60 — the next
prediction is then blocked. This keeps the failure mode boring (no
mid-render aborts) at the cost of a few extra calls past the cap.

Default 60 traces/day ≈ 15-20 unique predictions/day ≈ $1-2/day max
Anthropic spend (Sonnet 4.6 with prompt cache). Override via
`DAILY_LLM_BUDGET` env var.

Resets implicitly at 00:00 UTC because the count window is
`ts >= today_utc_midnight`.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import duckdb

log = logging.getLogger(__name__)

DAILY_LLM_BUDGET_DEFAULT: int = 60


def get_budget() -> int:
    """Resolve the cap from `DAILY_LLM_BUDGET` env, falling back to the
    default. Logs a warning and returns the default on invalid values."""
    raw = os.environ.get("DAILY_LLM_BUDGET")
    if raw is None:
        return DAILY_LLM_BUDGET_DEFAULT
    try:
        n = int(raw)
    except ValueError:
        log.warning(
            "DAILY_LLM_BUDGET=%r is not an int; using default %d",
            raw,
            DAILY_LLM_BUDGET_DEFAULT,
        )
        return DAILY_LLM_BUDGET_DEFAULT
    if n < 1:
        log.warning("DAILY_LLM_BUDGET=%r < 1; using default %d", raw, DAILY_LLM_BUDGET_DEFAULT)
        return DAILY_LLM_BUDGET_DEFAULT
    return n


def today_trace_count(conn: duckdb.DuckDBPyConnection, *, now: datetime | None = None) -> int:
    """Number of `llm_traces` rows logged since UTC midnight of `now`.

    `now` is a test seam — production callers omit it and pick up
    `datetime.now(UTC)`."""
    moment = now or datetime.now(UTC)
    today_start = moment.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    row = conn.execute(
        "SELECT COUNT(*) FROM llm_traces WHERE ts >= ?",
        [today_start],
    ).fetchone()
    return int(row[0]) if row else 0


def is_budget_exhausted(conn: duckdb.DuckDBPyConnection, *, now: datetime | None = None) -> bool:
    """True when today's trace count meets or exceeds `get_budget()`."""
    return today_trace_count(conn, now=now) >= get_budget()
