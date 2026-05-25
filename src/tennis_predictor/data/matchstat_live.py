"""On-demand matchstat fetcher with 24h DuckDB cache and quota tracking.

Phase 6.1 introduces two new per-prediction matchstat calls:
`player/past-matches/{id}` (x2, one per player) and `h2h/matches/{a}/{b}`
(URL corrected in Phase 6.2 — was `fixtures/h2h/{a}/{b}` before).
At 3 fresh calls per prediction the free tier (500 req/month, ~250 of
which are already consumed by the daily hot refresh) only sustains a
handful of unique predictions per day — so every fetch is wrapped here
with a 24h cache and a per-month quota counter that fails closed before
the API hard-caps us with a 429.

Caller contract:

- `fetch_player_past_matches(conn, tour, player_id)` and
  `fetch_h2h(conn, tour, player_a_id, player_b_id)` return a
  `RichMatchesPage`. On quota exhaustion or 429, they raise
  `MatchstatBudgetExceeded` — callers (`db_tools.get_head_to_head`, the
  view-layer `fetch_recent_n_matches` helper) catch this and fall back
  to Sackmann cold data.
- Cache writes go through DuckDB JSON columns; we serialise the raw
  matchstat payload (not the Pydantic model) so a future field addition
  on matchstat's side doesn't require a cache invalidation.
- H2H rows are stored with the lex-smaller player_id as `p1_id`; the
  H2H endpoint is orientation-agnostic, so this single row serves both
  `(A,B)` and `(B,A)` lookups.

Why DuckDB cache (not in-process / Streamlit `st.cache_data`):

- Predictions cross process boundaries (CLI + Streamlit + Phase 7
  scheduled refresh). A shared on-disk cache is the only one that
  survives them all.
- The 24h TTL is intentionally long: a player's last-8-matches list
  changes at most once per day (when a new completed match lands),
  and the catchup window for a match completed today is "tomorrow's
  fetch" — well inside the value of saving a quota slot.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import duckdb

from tennis_predictor.data.matchstat import (
    MatchstatClient,
    MatchstatError,
    RichMatchesPage,
    TourCode,
)

if TYPE_CHECKING:
    pass


# 24h cache TTL — see module docstring.
CACHE_TTL = timedelta(hours=24)

# Buffer below the matchstat hard cap. The daily hot refresh consumes
# ~10 reqs/day = ~300/month; leaving 20 reqs of headroom for an
# in-flight prediction batch when the counter approaches the cap.
QUOTA_BUFFER = 20


class MatchstatBudgetExceeded(RuntimeError):
    """Raised when the month-to-date quota counter has reached the buffered
    cap, OR when matchstat itself returns 429. Caller responsibility:
    catch and fall back to Sackmann."""

    def __init__(self, requests_used: int, cap: int) -> None:
        super().__init__(f"matchstat quota exhausted: {requests_used}/{cap} used this month")
        self.requests_used = requests_used
        self.cap = cap


def _current_month_bucket(now: datetime | None = None) -> str:
    """Return the month bucket for `now` (default: utcnow) in YYYY-MM form."""
    moment = now or datetime.now(UTC)
    return f"{moment.year:04d}-{moment.month:02d}"


def _ensure_quota_row(conn: duckdb.DuckDBPyConnection, month: str) -> tuple[int, int]:
    """Make sure a `matchstat_quota` row exists for `month`; return
    (requests_used, cap). Idempotent — runs an INSERT ... ON CONFLICT
    so concurrent callers don't fight over the row."""
    conn.execute(
        "INSERT INTO matchstat_quota (month, requests_used, cap) "
        "VALUES (?, 0, 500) ON CONFLICT (month) DO NOTHING",
        [month],
    )
    row = conn.execute(
        "SELECT requests_used, cap FROM matchstat_quota WHERE month = ?",
        [month],
    ).fetchone()
    assert row is not None  # we just inserted it
    return int(row[0]), int(row[1])


def _check_quota_or_raise(conn: duckdb.DuckDBPyConnection, now: datetime | None = None) -> None:
    """Pre-flight: raise `MatchstatBudgetExceeded` if the current-month
    bucket has already hit the buffered cap. Called BEFORE issuing any
    fetch so we don't burn an API call to then-discover we're over."""
    month = _current_month_bucket(now)
    requests_used, cap = _ensure_quota_row(conn, month)
    if requests_used >= cap - QUOTA_BUFFER:
        raise MatchstatBudgetExceeded(requests_used, cap)


def _increment_quota(conn: duckdb.DuckDBPyConnection, now: datetime | None = None) -> None:
    """Increment the month-to-date counter by 1. Called after a successful
    live fetch (not on cache hits)."""
    month = _current_month_bucket(now)
    _ensure_quota_row(conn, month)
    conn.execute(
        "UPDATE matchstat_quota SET requests_used = requests_used + 1 WHERE month = ?",
        [month],
    )


def _is_cache_fresh(fetched_at: datetime, now: datetime | None = None) -> bool:
    """True iff `fetched_at` is within `CACHE_TTL` of `now`."""
    moment = now or datetime.now(UTC)
    # DuckDB returns naive timestamps; coerce both to naive UTC for the diff.
    fetched_naive = fetched_at.replace(tzinfo=None) if fetched_at.tzinfo is not None else fetched_at
    moment_naive = moment.replace(tzinfo=None) if moment.tzinfo is not None else moment
    return (moment_naive - fetched_naive) <= CACHE_TTL


def _read_player_cache(
    conn: duckdb.DuckDBPyConnection,
    tour: TourCode,
    player_id: int,
    now: datetime | None = None,
) -> RichMatchesPage | None:
    row = conn.execute(
        "SELECT fetched_at, payload FROM matchstat_player_recent_cache "
        "WHERE tour = ? AND player_id = ?",
        [tour, player_id],
    ).fetchone()
    if row is None:
        return None
    fetched_at, payload_json = row
    if not _is_cache_fresh(fetched_at, now):
        return None
    payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
    return RichMatchesPage.model_validate(payload)


def _write_player_cache(
    conn: duckdb.DuckDBPyConnection,
    tour: TourCode,
    player_id: int,
    payload: dict[str, object],
    now: datetime | None = None,
) -> None:
    moment = now or datetime.now(UTC)
    conn.execute(
        "INSERT INTO matchstat_player_recent_cache (tour, player_id, fetched_at, payload) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT (tour, player_id) DO UPDATE SET "
        "  fetched_at = excluded.fetched_at, payload = excluded.payload",
        [tour, player_id, moment.replace(tzinfo=None), json.dumps(payload)],
    )


def _canonical_h2h_pair(a: int, b: int) -> tuple[int, int]:
    """Return (smaller_id, larger_id). H2H is orientation-agnostic — we
    enforce lex order so a single cache row serves both directions."""
    return (a, b) if a < b else (b, a)


def _read_h2h_cache(
    conn: duckdb.DuckDBPyConnection,
    tour: TourCode,
    a: int,
    b: int,
    now: datetime | None = None,
) -> RichMatchesPage | None:
    p1, p2 = _canonical_h2h_pair(a, b)
    row = conn.execute(
        "SELECT fetched_at, payload FROM matchstat_h2h_cache "
        "WHERE tour = ? AND p1_id = ? AND p2_id = ?",
        [tour, p1, p2],
    ).fetchone()
    if row is None:
        return None
    fetched_at, payload_json = row
    if not _is_cache_fresh(fetched_at, now):
        return None
    payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
    return RichMatchesPage.model_validate(payload)


def _write_h2h_cache(
    conn: duckdb.DuckDBPyConnection,
    tour: TourCode,
    a: int,
    b: int,
    payload: dict[str, object],
    now: datetime | None = None,
) -> None:
    p1, p2 = _canonical_h2h_pair(a, b)
    moment = now or datetime.now(UTC)
    conn.execute(
        "INSERT INTO matchstat_h2h_cache (tour, p1_id, p2_id, fetched_at, payload) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT (tour, p1_id, p2_id) DO UPDATE SET "
        "  fetched_at = excluded.fetched_at, payload = excluded.payload",
        [tour, p1, p2, moment.replace(tzinfo=None), json.dumps(payload)],
    )


def fetch_player_past_matches(
    conn: duckdb.DuckDBPyConnection,
    tour: TourCode,
    player_id: int,
    *,
    client: MatchstatClient | None = None,
    api_key: str | None = None,
    now: datetime | None = None,
    page_size: int = 10,
) -> RichMatchesPage:
    """Cache-aware fetch of `/{tour}/player/past-matches/{player_id}`.

    Lookup order:
    1. DuckDB cache — return if fresh (<24h old).
    2. Pre-flight quota check — raise `MatchstatBudgetExceeded` if over.
    3. Live API call. 429 → also raise `MatchstatBudgetExceeded`.
    4. Write payload to cache and increment quota counter.

    `client` lets tests inject a `MatchstatClient` with a `MockTransport`;
    in production it's None and we build one from `api_key` (or
    `config.X_RAPIDAPI_KEY` if also None).
    """
    cached = _read_player_cache(conn, tour, player_id, now)
    if cached is not None:
        return cached

    _check_quota_or_raise(conn, now)

    owns_client = client is None
    if client is None:
        # Local import to avoid pulling config at module-import time
        # in tests that don't need it.
        from tennis_predictor.config import X_RAPIDAPI_KEY

        resolved_key = api_key or X_RAPIDAPI_KEY
        if not resolved_key:
            raise MatchstatBudgetExceeded(0, 500)  # treat missing key as "no budget"
        client = MatchstatClient(api_key=resolved_key)

    try:
        page = client.player_past_matches(tour, player_id, page_size=page_size)
    except MatchstatError as exc:
        if exc.status_code == 429:
            month = _current_month_bucket(now)
            used, cap = _ensure_quota_row(conn, month)
            raise MatchstatBudgetExceeded(used, cap) from exc
        raise
    finally:
        if owns_client:
            client.close()

    _write_player_cache(conn, tour, player_id, page.model_dump(by_alias=True, mode="json"), now)
    _increment_quota(conn, now)
    return page


def fetch_h2h(
    conn: duckdb.DuckDBPyConnection,
    tour: TourCode,
    player_a_id: int,
    player_b_id: int,
    *,
    client: MatchstatClient | None = None,
    api_key: str | None = None,
    now: datetime | None = None,
    page_size: int = 50,
) -> RichMatchesPage:
    """Cache-aware fetch of `/{tour}/h2h/matches/{a}/{b}`. Same
    semantics as `fetch_player_past_matches`; cache key is canonicalised
    to (smaller_id, larger_id)."""
    cached = _read_h2h_cache(conn, tour, player_a_id, player_b_id, now)
    if cached is not None:
        return cached

    _check_quota_or_raise(conn, now)

    owns_client = client is None
    if client is None:
        from tennis_predictor.config import X_RAPIDAPI_KEY

        resolved_key = api_key or X_RAPIDAPI_KEY
        if not resolved_key:
            raise MatchstatBudgetExceeded(0, 500)
        client = MatchstatClient(api_key=resolved_key)

    try:
        page = client.h2h(tour, player_a_id, player_b_id, page_size=page_size)
    except MatchstatError as exc:
        if exc.status_code == 429:
            month = _current_month_bucket(now)
            used, cap = _ensure_quota_row(conn, month)
            raise MatchstatBudgetExceeded(used, cap) from exc
        raise
    finally:
        if owns_client:
            client.close()

    _write_h2h_cache(
        conn,
        tour,
        player_a_id,
        player_b_id,
        page.model_dump(by_alias=True, mode="json"),
        now,
    )
    _increment_quota(conn, now)
    return page


def quota_status(conn: duckdb.DuckDBPyConnection, now: datetime | None = None) -> tuple[int, int]:
    """Return (requests_used, cap) for the current month bucket. Used by
    the Dashboard page's "matchstat M/500" indicator."""
    month = _current_month_bucket(now)
    return _ensure_quota_row(conn, month)


__all__ = [
    "CACHE_TTL",
    "QUOTA_BUFFER",
    "MatchstatBudgetExceeded",
    "fetch_h2h",
    "fetch_player_past_matches",
    "quota_status",
]
