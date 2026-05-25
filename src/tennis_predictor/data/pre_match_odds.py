"""Reconcile The Odds API events with `scheduled_matches` and persist.

Two-tier name matcher:

1. Set-membership on lowercased canonical names within the same tour
   and UTC date. The Odds API and matchstat both publish canonical
   full names ("Jannik Sinner") for tour-level players, so an
   ordering-independent set match catches the vast majority.
2. If no scheduled-matches row exists yet (Odds API may publish lines
   for fixtures matchstat hasn't surfaced as a `scheduled_matches`
   row yet — different draw-publication schedules) the row is still
   persisted with `fixture_match_key` derived directly from the
   normalised pair + date, so it is ready to JOIN onto whatever
   scheduled row eventually lands.

The 24h freshness check (`is_row_fresh`) drives the lazy refresh on
Prediction-page load: a row older than 24h triggers a single odds call
for the relevant sport_key.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import duckdb

from tennis_predictor.data.odds_api import (
    AggregatedOdds,
    OddsApiQuotaExceeded,
)

logger = logging.getLogger(__name__)

# 24h freshness window — matches the cache TTL elsewhere in the project.
FRESHNESS_TTL: timedelta = timedelta(hours=24)


def _normalise_name(name: str) -> str:
    """Lowercase, strip whitespace, collapse internal spaces, drop
    diacritics-style punctuation. Conservative: the canonical-name
    contract is strong enough that we don't try fancy fuzzy matching."""
    text = name.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def fixture_match_key(
    tour: str,
    player_a_name: str,
    player_b_name: str,
    match_date_utc: datetime,
) -> str:
    """Deterministic primary key for `pre_match_odds`. Order-independent
    on (player_a, player_b) so the same fixture, regardless of which
    side The Odds API listed as `home_team`, upserts the same row."""
    a = _normalise_name(player_a_name)
    b = _normalise_name(player_b_name)
    pair = "::".join(sorted([a, b]))
    date_part = match_date_utc.date().isoformat()
    raw = f"{tour.upper()}::{pair}::{date_part}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def check_quota_or_raise(conn: duckdb.DuckDBPyConnection, now: datetime | None = None) -> None:
    """Pre-flight: raise `OddsApiQuotaExceeded` if the current-month
    bucket has already hit the buffered cap. Called by the daily
    refresh script before issuing any odds-billable call so we don't
    burn a credit just to discover we're over."""
    from tennis_predictor.data.odds_api import QUOTA_BUFFER

    moment = now or datetime.now(UTC)
    month = f"{moment.year:04d}-{moment.month:02d}"
    conn.execute(
        "INSERT INTO odds_api_quota (month, requests_used, cap) "
        "VALUES (?, 0, 500) ON CONFLICT (month) DO NOTHING",
        [month],
    )
    row = conn.execute(
        "SELECT requests_used, cap FROM odds_api_quota WHERE month = ?", [month]
    ).fetchone()
    assert row is not None
    used, cap = int(row[0]), int(row[1])
    if used >= cap - QUOTA_BUFFER:
        raise OddsApiQuotaExceeded(used, cap)


def increment_quota(
    conn: duckdb.DuckDBPyConnection, delta: int = 1, now: datetime | None = None
) -> None:
    """Increment the month-to-date credit counter by `delta`. The Odds
    API charges 1 credit per `regions=eu` odds call; discovery
    (`/v4/sports/`) is free per docs but callers count it for parity."""
    moment = now or datetime.now(UTC)
    month = f"{moment.year:04d}-{moment.month:02d}"
    conn.execute(
        "INSERT INTO odds_api_quota (month, requests_used, cap) "
        "VALUES (?, ?, 500) ON CONFLICT (month) DO UPDATE SET "
        "  requests_used = odds_api_quota.requests_used + excluded.requests_used",
        [month, delta],
    )


def quota_status(conn: duckdb.DuckDBPyConnection, now: datetime | None = None) -> tuple[int, int]:
    """Return (requests_used, cap) for the current month bucket. Used by
    the Dashboard widget."""
    from tennis_predictor.data.odds_api import QUOTA_CAP

    moment = now or datetime.now(UTC)
    month = f"{moment.year:04d}-{moment.month:02d}"
    row = conn.execute(
        "SELECT requests_used, cap FROM odds_api_quota WHERE month = ?", [month]
    ).fetchone()
    if row is None:
        return 0, QUOTA_CAP
    return int(row[0]), int(row[1])


def upsert_aggregated(
    conn: duckdb.DuckDBPyConnection,
    rows: Iterable[AggregatedOdds],
    *,
    source: str = "the_odds_api",
    now: datetime | None = None,
) -> int:
    """Upsert a batch of aggregated rows into `pre_match_odds`. Returns
    the number of rows touched (insert + update combined). `source` is
    `the_odds_api` for the daily-batch path and `tavily` for the
    regex-extract fallback."""
    fetched_at = (now or datetime.now(UTC)).replace(tzinfo=None)
    touched = 0
    for r in rows:
        key = fixture_match_key(r.tour, r.player_a_name, r.player_b_name, r.commence_time_utc)
        # DuckDB datetimes are naive — strip the tz before binding.
        commence_naive = (
            r.commence_time_utc.replace(tzinfo=None)
            if r.commence_time_utc.tzinfo is not None
            else r.commence_time_utc
        )
        conn.execute(
            """
            INSERT INTO pre_match_odds (
                fixture_match_key, tour, sport_key, event_id,
                player_a_name, player_b_name, commence_time_utc,
                median_odds_a, median_odds_b, best_odds_a, best_odds_b,
                median_implied_prob_a, median_implied_prob_b, books_count,
                pinnacle_odds_a, pinnacle_odds_b,
                pinnacle_implied_prob_a, pinnacle_implied_prob_b,
                fetched_at, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (fixture_match_key) DO UPDATE SET
                sport_key               = excluded.sport_key,
                event_id                = excluded.event_id,
                commence_time_utc       = excluded.commence_time_utc,
                median_odds_a           = excluded.median_odds_a,
                median_odds_b           = excluded.median_odds_b,
                best_odds_a             = excluded.best_odds_a,
                best_odds_b             = excluded.best_odds_b,
                median_implied_prob_a   = excluded.median_implied_prob_a,
                median_implied_prob_b   = excluded.median_implied_prob_b,
                books_count             = excluded.books_count,
                pinnacle_odds_a         = excluded.pinnacle_odds_a,
                pinnacle_odds_b         = excluded.pinnacle_odds_b,
                pinnacle_implied_prob_a = excluded.pinnacle_implied_prob_a,
                pinnacle_implied_prob_b = excluded.pinnacle_implied_prob_b,
                fetched_at              = excluded.fetched_at,
                source                  = excluded.source
            """,
            [
                key,
                r.tour,
                r.sport_key,
                r.event_id,
                r.player_a_name,
                r.player_b_name,
                commence_naive,
                r.median_odds_a,
                r.median_odds_b,
                r.best_odds_a,
                r.best_odds_b,
                r.median_implied_prob_a,
                r.median_implied_prob_b,
                r.books_count,
                r.pinnacle_odds_a,
                r.pinnacle_odds_b,
                r.pinnacle_implied_prob_a,
                r.pinnacle_implied_prob_b,
                fetched_at,
                source,
            ],
        )
        touched += 1
    return touched


def lookup_for_fixture(
    conn: duckdb.DuckDBPyConnection,
    tour: str,
    player_a_name: str,
    player_b_name: str,
    scheduled_start_utc: datetime,
) -> dict[str, Any] | None:
    """Return the `pre_match_odds` row that matches the given fixture, or
    None if no row exists yet. Match is on the same key derivation as
    `upsert_aggregated` (order-independent over the players + UTC date)."""
    key = fixture_match_key(tour, player_a_name, player_b_name, scheduled_start_utc)
    row = conn.execute(
        """
        SELECT fixture_match_key, tour, sport_key, event_id,
               player_a_name, player_b_name, commence_time_utc,
               median_odds_a, median_odds_b, best_odds_a, best_odds_b,
               median_implied_prob_a, median_implied_prob_b, books_count,
               pinnacle_odds_a, pinnacle_odds_b,
               pinnacle_implied_prob_a, pinnacle_implied_prob_b,
               fetched_at, source
        FROM pre_match_odds
        WHERE fixture_match_key = ?
        """,
        [key],
    ).fetchone()
    if row is None:
        return None
    return {
        "fixture_match_key": row[0],
        "tour": row[1],
        "sport_key": row[2],
        "event_id": row[3],
        "player_a_name": row[4],
        "player_b_name": row[5],
        "commence_time_utc": row[6],
        "median_odds_a": row[7],
        "median_odds_b": row[8],
        "best_odds_a": row[9],
        "best_odds_b": row[10],
        "median_implied_prob_a": row[11],
        "median_implied_prob_b": row[12],
        "books_count": row[13],
        "pinnacle_odds_a": row[14],
        "pinnacle_odds_b": row[15],
        "pinnacle_implied_prob_a": row[16],
        "pinnacle_implied_prob_b": row[17],
        "fetched_at": row[18],
        "source": row[19],
    }


def is_row_fresh(row: dict[str, Any] | None, now: datetime | None = None) -> bool:
    """A row counts as fresh when `fetched_at` is within `FRESHNESS_TTL`
    of `now`. None rows are never fresh."""
    if row is None:
        return False
    fetched_at = row.get("fetched_at")
    if fetched_at is None:
        return False
    moment = (now or datetime.now(UTC)).replace(tzinfo=None)
    fetched_naive = fetched_at.replace(tzinfo=None) if fetched_at.tzinfo is not None else fetched_at
    return (moment - fetched_naive) <= FRESHNESS_TTL


__all__ = [
    "FRESHNESS_TTL",
    "check_quota_or_raise",
    "fixture_match_key",
    "increment_quota",
    "is_row_fresh",
    "lookup_for_fixture",
    "quota_status",
    "upsert_aggregated",
]
