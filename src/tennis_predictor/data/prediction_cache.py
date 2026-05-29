"""24h persistent cache for `AgentResponse` keyed by `scheduled_match_id`.

Phase 7 cost defense: repeat clicks on a Match dashboard within 24h
read the cached response from DuckDB instead of re-invoking the LLM
agent (~$0.10 per call). Only scheduled fixtures are cached — free-form
Custom predictions stay on Streamlit's in-memory `@st.cache_data`
(5 min TTL) and are operator-gated by basic-auth.

This sits BELOW the Streamlit `@st.cache_data` layer:
  - Streamlit cache: per-process, ~5 min, fast path inside a single
    process visit.
  - DuckDB cache:    cross-session, persistent, 24h, survives process
    restarts and serves all visitors equally.

Eviction: stale rows (older than 24h) are silently ignored on read.
They accumulate in the table but each row is small (~5-50KB of JSON);
no scheduled cleanup is wired in v1.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import duckdb

from tennis_predictor.llm.tools.submit import AgentResponse

log = logging.getLogger(__name__)

CACHE_TTL: timedelta = timedelta(hours=24)


def get_cached(
    conn: duckdb.DuckDBPyConnection,
    scheduled_match_id: str,
    *,
    now: datetime | None = None,
) -> AgentResponse | None:
    """Return the cached `AgentResponse` if it exists and is younger
    than `CACHE_TTL`. None on cache miss, stale entry, or deserialization
    error (treats schema drift as a miss so the next predict overwrites)."""
    moment = (now or datetime.now(UTC)).replace(tzinfo=None)
    cutoff = moment - CACHE_TTL
    row = conn.execute(
        "SELECT agent_response_json FROM prediction_cache "
        "WHERE scheduled_match_id = ? AND cached_at >= ?",
        [scheduled_match_id, cutoff],
    ).fetchone()
    if row is None:
        return None
    try:
        return AgentResponse.model_validate_json(row[0])
    except Exception as exc:
        log.warning(
            "prediction_cache: deserialize failed for %s (%s) — treating as miss",
            scheduled_match_id,
            exc,
        )
        return None


def store(
    conn: duckdb.DuckDBPyConnection,
    scheduled_match_id: str,
    response: AgentResponse,
    *,
    now: datetime | None = None,
) -> None:
    """Upsert one cache row for the given scheduled fixture. Subsequent
    `get_cached` calls within `CACHE_TTL` will return the same response
    without invoking the LLM agent."""
    moment = (now or datetime.now(UTC)).replace(tzinfo=None)
    payload = response.model_dump_json()
    conn.execute(
        """
        INSERT INTO prediction_cache (scheduled_match_id, cached_at, agent_response_json)
        VALUES (?, ?, ?)
        ON CONFLICT (scheduled_match_id) DO UPDATE SET
            cached_at = excluded.cached_at,
            agent_response_json = excluded.agent_response_json
        """,
        [scheduled_match_id, moment, payload],
    )
