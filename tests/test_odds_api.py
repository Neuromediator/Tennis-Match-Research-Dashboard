"""Unit tests for The Odds API client + aggregator (Phase 6.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from tennis_predictor.data.odds_api import (
    BASE_URL,
    OddsApiClient,
    OddsApiError,
    aggregate_event,
    aggregate_events,
    tour_from_sport_key,
)


def _make_client(handler: Any) -> OddsApiClient:
    transport = httpx.MockTransport(handler)
    inner = httpx.Client(base_url=BASE_URL, transport=transport)
    return OddsApiClient(api_key="test-key", client=inner)


def test_list_active_tennis_sports_filters_to_tennis_keys() -> None:
    payload = [
        {"key": "tennis_atp_french_open", "group": "Tennis", "title": "ATP RG", "active": True},
        {"key": "tennis_wta_madrid", "group": "Tennis", "title": "WTA Madrid", "active": True},
        {"key": "soccer_epl", "group": "Soccer", "title": "EPL", "active": True},
    ]

    seen_params: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/sports")
        for k, v in request.url.params.items():
            seen_params[k] = v
        return httpx.Response(200, json=payload)

    with _make_client(handler) as client:
        sports = client.list_active_tennis_sports()

    assert seen_params["all"] == "false"
    assert seen_params["apiKey"] == "test-key"
    assert [s.key for s in sports] == ["tennis_atp_french_open", "tennis_wta_madrid"]


def test_fetch_odds_passes_regions_and_market() -> None:
    payload = [
        {
            "id": "evt-1",
            "sport_key": "tennis_atp_french_open",
            "commence_time": "2026-05-26T18:00:00Z",
            "home_team": "Jannik Sinner",
            "away_team": "Novak Djokovic",
            "bookmakers": [],
        }
    ]

    seen_params: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/sports/tennis_atp_french_open/odds")
        for k, v in request.url.params.items():
            seen_params[k] = v
        return httpx.Response(200, json=payload)

    with _make_client(handler) as client:
        events = client.fetch_odds("tennis_atp_french_open")

    assert seen_params["regions"] == "eu"
    assert seen_params["markets"] == "h2h"
    assert seen_params["oddsFormat"] == "decimal"
    assert events[0].home_team == "Jannik Sinner"


def test_fetch_odds_raises_on_non_200() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="missing or invalid api key")

    with _make_client(handler) as client, pytest.raises(OddsApiError) as excinfo:
        client.fetch_odds("tennis_atp_french_open")

    assert excinfo.value.status_code == 401


def test_tour_from_sport_key() -> None:
    assert tour_from_sport_key("tennis_atp_french_open") == "ATP"
    assert tour_from_sport_key("tennis_atp") == "ATP"
    assert tour_from_sport_key("tennis_wta_madrid") == "WTA"
    assert tour_from_sport_key("tennis_itf_w15") is None
    assert tour_from_sport_key("soccer_epl") is None


def _event_with_two_books(*, pinnacle: bool = True) -> dict[str, Any]:
    books = [
        {
            "key": "betfair_ex_eu",
            "title": "Betfair Exchange",
            "last_update": "2026-05-26T14:00:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Jannik Sinner", "price": 1.10},
                        {"name": "Novak Djokovic", "price": 9.50},
                    ],
                }
            ],
        }
    ]
    if pinnacle:
        books.insert(
            0,
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": "2026-05-26T14:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Jannik Sinner", "price": 1.07},
                            {"name": "Novak Djokovic", "price": 11.00},
                        ],
                    }
                ],
            },
        )
    return {
        "id": "evt-sd",
        "sport_key": "tennis_atp_french_open",
        "commence_time": "2026-05-26T18:00:00Z",
        "home_team": "Jannik Sinner",
        "away_team": "Novak Djokovic",
        "bookmakers": books,
    }


def test_aggregate_event_extracts_median_best_and_pinnacle() -> None:
    from tennis_predictor.data.odds_api import OddsApiEvent

    event = OddsApiEvent.model_validate(_event_with_two_books(pinnacle=True))
    agg = aggregate_event(event)
    assert agg is not None
    assert agg.tour == "ATP"
    assert agg.player_a_name == "Jannik Sinner"
    assert agg.player_b_name == "Novak Djokovic"
    assert agg.books_count == 2
    # Best decimal odds = max across books.
    assert agg.best_odds_a == 1.10
    assert agg.best_odds_b == 11.00
    # Pinnacle row preserved separately.
    assert agg.pinnacle_odds_a == 1.07
    assert agg.pinnacle_odds_b == 11.00
    # Implied probabilities are margin-stripped and sum to ~1.
    assert agg.median_implied_prob_a is not None
    assert agg.median_implied_prob_b is not None
    assert abs(agg.median_implied_prob_a + agg.median_implied_prob_b - 1.0) < 1e-6


def test_aggregate_event_without_pinnacle_leaves_columns_null() -> None:
    from tennis_predictor.data.odds_api import OddsApiEvent

    event = OddsApiEvent.model_validate(_event_with_two_books(pinnacle=False))
    agg = aggregate_event(event)
    assert agg is not None
    assert agg.pinnacle_odds_a is None
    assert agg.pinnacle_odds_b is None
    assert agg.pinnacle_implied_prob_a is None
    assert agg.pinnacle_implied_prob_b is None


def test_aggregate_event_drops_event_with_no_usable_market() -> None:
    from tennis_predictor.data.odds_api import OddsApiEvent

    payload: dict[str, Any] = {
        "id": "evt-empty",
        "sport_key": "tennis_atp_french_open",
        "commence_time": "2026-05-26T18:00:00Z",
        "home_team": "Jannik Sinner",
        "away_team": "Novak Djokovic",
        "bookmakers": [],
    }
    event = OddsApiEvent.model_validate(payload)
    assert aggregate_event(event) is None


def test_aggregate_events_skips_non_tennis_keys() -> None:
    from tennis_predictor.data.odds_api import OddsApiEvent

    tennis = OddsApiEvent.model_validate(_event_with_two_books(pinnacle=False))
    soccer = tennis.model_copy(update={"sport_key": "soccer_epl"})
    aggregated = aggregate_events([tennis, soccer])
    assert len(aggregated) == 1
    assert aggregated[0].tour == "ATP"


def test_aggregate_event_parses_commence_time_as_utc() -> None:
    from tennis_predictor.data.odds_api import OddsApiEvent

    event = OddsApiEvent.model_validate(_event_with_two_books(pinnacle=False))
    agg = aggregate_event(event)
    assert agg is not None
    assert agg.commence_time_utc == datetime(2026, 5, 26, 18, 0, tzinfo=UTC)
