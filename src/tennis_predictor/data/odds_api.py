"""The Odds API client (Phase 6.2 pre-match odds source).

Thin typed wrapper around `the-odds-api.com` v4. Two endpoints used:

- `GET /v4/sports/?all=false` — list currently-active sport keys; we
  filter to those starting with `tennis_` and feed each into the next
  endpoint.
- `GET /v4/sports/{sport_key}/odds?regions=eu&markets=h2h&oddsFormat=decimal`
  — h2h decimal odds for every upcoming fixture in a tournament. One
  credit per call (regions=eu only).

The aggregator (`aggregate_event`) collapses each event's `bookmakers[]`
into the database row shape (median + best across books + Pinnacle when
present). Margin-stripped implied probabilities are computed per book
and then medianised so the displayed two-way market sums to 1.0.

CLAUDE.md hard rule #3 applies: odds returned here are display-only,
never a training feature.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Base URL is hard-coded — there is no staging endpoint, and the v4 prefix
# is part of the published contract (docs.the-odds-api.com/liveapi/guides/v4/).
BASE_URL: str = "https://api.the-odds-api.com/v4"

# Each odds call against `regions=eu` costs 1 credit; sport discovery is
# free per the docs. The buffered cap leaves headroom for the daily
# refresh batch when the counter is near the limit.
QUOTA_CAP: int = 500
QUOTA_BUFFER: int = 20

# Pinnacle is one of the bookmakers The Odds API surfaces for EU. We
# extract its price into a dedicated subtitle row on the Prediction
# page (sharp-line preference per the design doc).
PINNACLE_BOOK_KEY: str = "pinnacle"

# Default HTTP timeout. The endpoint typically responds in <2s; 15s
# leaves room for tail latency without letting a stuck request eat the
# refresh budget.
DEFAULT_TIMEOUT: float = 15.0


class _Permissive(BaseModel):
    """Drop unexpected fields silently. The Odds API may add new
    fields without notice (the docs explicitly reserve that right)."""

    model_config = ConfigDict(extra="ignore")


class OddsApiOutcome(_Permissive):
    name: str  # canonical player name (matches `home_team` / `away_team`)
    price: float  # decimal odds


class OddsApiMarket(_Permissive):
    key: str  # "h2h", "totals", etc. We only request h2h.
    outcomes: list[OddsApiOutcome] = Field(default_factory=list)


class OddsApiBookmaker(_Permissive):
    key: str  # "pinnacle", "betfair_ex_eu", "draftkings", ...
    title: str | None = None
    last_update: datetime | None = None
    markets: list[OddsApiMarket] = Field(default_factory=list)


class OddsApiEvent(_Permissive):
    """One upcoming fixture from `/v4/sports/{key}/odds`."""

    id: str  # stable event id
    sport_key: str
    commence_time: datetime
    home_team: str  # canonical full name
    away_team: str
    bookmakers: list[OddsApiBookmaker] = Field(default_factory=list)


class OddsApiSport(_Permissive):
    key: str  # e.g. "tennis_atp_french_open"
    group: str | None = None
    title: str | None = None
    active: bool = True


class OddsApiError(RuntimeError):
    """Non-2xx response or transport failure. The refresh script catches
    this and writes an `ingestion_runs` row with `status='failed'` so
    the next run picks up where we left off."""

    def __init__(self, status_code: int, body: str, url: str) -> None:
        super().__init__(f"odds-api {status_code} at {url}: {body[:200]}")
        self.status_code = status_code
        self.body = body
        self.url = url


class OddsApiQuotaExceeded(RuntimeError):
    """Raised when the month-to-date credit counter has hit the buffered
    cap. Callers catch and fall back to Tavily-regex extraction (or
    surface "odds unavailable" in the UI)."""

    def __init__(self, requests_used: int, cap: int) -> None:
        super().__init__(f"odds-api quota exhausted: {requests_used}/{cap} used this month")
        self.requests_used = requests_used
        self.cap = cap


class OddsApiClient:
    """Typed wrapper over The Odds API v4. Tracks `requests_used` so the
    caller can log it into `ingestion_runs` / `odds_api_quota`."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(base_url=BASE_URL, timeout=timeout)
        self._owns_client = client is None
        self.requests_used: int = 0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OddsApiClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        merged = {"apiKey": self._api_key, **(params or {})}
        self.requests_used += 1
        response = self._client.get(path, params=merged)
        if response.status_code != 200:
            raise OddsApiError(response.status_code, response.text, str(response.url))
        return response.json()

    def list_active_tennis_sports(self) -> list[OddsApiSport]:
        """`GET /v4/sports/?all=false` filtered to `tennis_*` keys.

        Per the docs the discovery endpoint does not consume credits,
        but we count the HTTP call into `requests_used` for parity with
        other clients; the caller may ignore the discovery delta when
        writing to `odds_api_quota` if precision matters."""
        payload = self._get_json("/sports", {"all": "false"})
        if not isinstance(payload, list):
            return []
        sports = [OddsApiSport.model_validate(item) for item in payload]
        return [s for s in sports if s.key.startswith("tennis_")]

    def fetch_odds(self, sport_key: str) -> list[OddsApiEvent]:
        """`GET /v4/sports/{key}/odds?regions=eu&markets=h2h&oddsFormat=decimal`.

        1 credit per call. Returns one `OddsApiEvent` per upcoming
        fixture (in-play events excluded by the regions=eu filter on
        the API side)."""
        payload = self._get_json(
            f"/sports/{sport_key}/odds",
            {
                "regions": "eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        if not isinstance(payload, list):
            return []
        return [OddsApiEvent.model_validate(item) for item in payload]


@contextmanager
def odds_api_client(api_key: str) -> Iterator[OddsApiClient]:
    """`with odds_api_client(key) as c: ...` — symmetric with
    `matchstat_client` so refresh scripts feel uniform."""
    client = OddsApiClient(api_key)
    try:
        yield client
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Aggregation: events → DB row shape
# ---------------------------------------------------------------------------


class AggregatedOdds(BaseModel):
    """One pre_match_odds row, sans key derivation. The refresh script
    fills `fixture_match_key` after deriving it from the matched
    `scheduled_matches` row (or directly from the event when no match
    is found)."""

    sport_key: str
    event_id: str
    tour: str  # "ATP" | "WTA"
    player_a_name: str
    player_b_name: str
    commence_time_utc: datetime
    median_odds_a: float | None
    median_odds_b: float | None
    best_odds_a: float | None
    best_odds_b: float | None
    median_implied_prob_a: float | None
    median_implied_prob_b: float | None
    books_count: int
    pinnacle_odds_a: float | None
    pinnacle_odds_b: float | None
    pinnacle_implied_prob_a: float | None
    pinnacle_implied_prob_b: float | None


def tour_from_sport_key(sport_key: str) -> str | None:
    """Map a `tennis_atp_*` / `tennis_wta_*` sport key to the tour code.
    Returns None for anything else (e.g. ITF, mixed doubles) so callers
    can skip the row silently."""
    if sport_key.startswith("tennis_atp"):
        return "ATP"
    if sport_key.startswith("tennis_wta"):
        return "WTA"
    return None


def decimal_odds_to_implied_probs(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Margin-stripped implied probabilities from a two-way market.
    `1/odds_a / (1/odds_a + 1/odds_b)` removes the bookmaker overround.

    Public helper so the Tavily-fallback path (`odds_fallback.py`) and
    the Streamlit comparison-row widget can reuse the same math the
    aggregator applies."""
    inv_a = 1.0 / odds_a
    inv_b = 1.0 / odds_b
    total = inv_a + inv_b
    return inv_a / total, inv_b / total


def aggregate_event(event: OddsApiEvent) -> AggregatedOdds | None:
    """Collapse one `OddsApiEvent` into the `pre_match_odds` row shape.

    Drops rows with no usable h2h market across any bookmaker. The
    fixture's player ordering (home/away) defines the (a, b) axis;
    name reconciliation against `scheduled_matches` happens at the
    matcher layer."""
    tour = tour_from_sport_key(event.sport_key)
    if tour is None:
        return None

    odds_a_samples: list[float] = []
    odds_b_samples: list[float] = []
    implied_a_samples: list[float] = []
    implied_b_samples: list[float] = []
    pinnacle_a: float | None = None
    pinnacle_b: float | None = None

    for book in event.bookmakers:
        h2h_market = next((m for m in book.markets if m.key == "h2h"), None)
        if h2h_market is None or len(h2h_market.outcomes) < 2:
            continue
        outcome_map = {o.name: o.price for o in h2h_market.outcomes}
        odds_a = outcome_map.get(event.home_team)
        odds_b = outcome_map.get(event.away_team)
        if odds_a is None or odds_b is None or odds_a <= 1 or odds_b <= 1:
            continue
        odds_a_samples.append(odds_a)
        odds_b_samples.append(odds_b)
        prob_a, prob_b = decimal_odds_to_implied_probs(odds_a, odds_b)
        implied_a_samples.append(prob_a)
        implied_b_samples.append(prob_b)
        if book.key == PINNACLE_BOOK_KEY:
            pinnacle_a = odds_a
            pinnacle_b = odds_b

    if not odds_a_samples:
        return None

    # "Best odds" = largest decimal per side = the sharpest price the
    # user could shop. UI shows median as the headline.
    best_odds_a = max(odds_a_samples)
    best_odds_b = max(odds_b_samples)

    pinnacle_prob_a: float | None
    pinnacle_prob_b: float | None
    if pinnacle_a is not None and pinnacle_b is not None:
        pinnacle_prob_a, pinnacle_prob_b = decimal_odds_to_implied_probs(pinnacle_a, pinnacle_b)
    else:
        pinnacle_prob_a, pinnacle_prob_b = None, None

    return AggregatedOdds(
        sport_key=event.sport_key,
        event_id=event.id,
        tour=tour,
        player_a_name=event.home_team,
        player_b_name=event.away_team,
        commence_time_utc=event.commence_time,
        median_odds_a=statistics.median(odds_a_samples),
        median_odds_b=statistics.median(odds_b_samples),
        best_odds_a=best_odds_a,
        best_odds_b=best_odds_b,
        median_implied_prob_a=statistics.median(implied_a_samples),
        median_implied_prob_b=statistics.median(implied_b_samples),
        books_count=len(odds_a_samples),
        pinnacle_odds_a=pinnacle_a,
        pinnacle_odds_b=pinnacle_b,
        pinnacle_implied_prob_a=pinnacle_prob_a,
        pinnacle_implied_prob_b=pinnacle_prob_b,
    )


def aggregate_events(events: Iterable[OddsApiEvent]) -> list[AggregatedOdds]:
    """Batch aggregation helper — drops `None` results from
    `aggregate_event`."""
    out: list[AggregatedOdds] = []
    for ev in events:
        agg = aggregate_event(ev)
        if agg is not None:
            out.append(agg)
    return out


__all__ = [
    "BASE_URL",
    "PINNACLE_BOOK_KEY",
    "QUOTA_BUFFER",
    "QUOTA_CAP",
    "AggregatedOdds",
    "OddsApiBookmaker",
    "OddsApiClient",
    "OddsApiError",
    "OddsApiEvent",
    "OddsApiMarket",
    "OddsApiOutcome",
    "OddsApiQuotaExceeded",
    "OddsApiSport",
    "aggregate_event",
    "aggregate_events",
    "decimal_odds_to_implied_probs",
    "odds_api_client",
    "tour_from_sport_key",
]
