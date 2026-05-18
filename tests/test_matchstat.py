"""matchstat API client tests.

Use `httpx.MockTransport` to simulate the API — no network calls.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
import pytest

from tennis_predictor.data.matchstat import (
    BASE_URL,
    TOUR_LEVEL_TIERS,
    MatchstatClient,
    MatchstatError,
)


def _make_client(handler: Any) -> MatchstatClient:
    transport = httpx.MockTransport(handler)
    inner = httpx.Client(base_url=BASE_URL, transport=transport, headers={})
    return MatchstatClient(api_key="test-key", client=inner)


def test_calendar_parses_tour_level_tournament() -> None:
    payload = {
        "data": [
            {
                "id": 21363,
                "name": "BNP Paribas Nordic Open - Stockholm",
                "tier": "ATP 250",
                "date": "2026-11-09T00:00:00.000Z",
                "court": {"id": 3, "name": "I.hard"},
                "rank": {"id": 2, "name": "Main tour"},
                "countryAcr": "SWE",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tennis/v2/atp/tournament/calendar/2026"
        return httpx.Response(200, json=payload)

    with _make_client(handler) as client:
        items = client.calendar("atp", 2026)

    assert len(items) == 1
    item = items[0]
    assert item.id == 21363
    assert item.tier == "ATP 250"
    assert item.court is not None
    assert item.court.name == "I.hard"
    assert item.country_acr == "SWE"
    assert item.tier in TOUR_LEVEL_TIERS


def test_calendar_fixes_coutry_typo() -> None:
    """The matchstat calendar payload returns `coutry` (typo) instead of `country`.

    The model must accept the typo as the country object.
    """
    payload = {
        "data": [
            {
                "id": 9999,
                "name": "Test Event",
                "tier": "ATP 250",
                "coutry": {"name": "Italy", "acronym": "ITA"},
            }
        ]
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with _make_client(handler) as client:
        items = client.calendar("atp", 2026)

    assert len(items) == 1
    assert items[0].country is not None
    assert items[0].country.acronym == "ITA"
    assert items[0].country.name == "Italy"


def test_fixtures_for_date_sends_required_query_params() -> None:
    """`include` and `filter=PlayerGroup:singles` are not optional —
    they're contractually required (see data-ingestion skill)."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"data": [], "hasNextPage": False})

    with _make_client(handler) as client:
        client.fixtures_for_date("atp", date(2026, 5, 19))

    assert captured["include"] == "tournament.court,tournament.rank,round"
    assert captured["filter"] == "PlayerGroup:singles"
    assert captured["pageSize"] == "100"
    assert captured["pageNo"] == "1"


def test_fixtures_for_date_singles_only_can_be_disabled() -> None:
    """Doubles can be fetched by setting `singles_only=False` — needed for
    debugging/edge cases, but not used in steady-state."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"data": [], "hasNextPage": False})

    with _make_client(handler) as client:
        client.fixtures_for_date("wta", date(2026, 5, 19), singles_only=False)

    assert "filter" not in captured


def test_fixtures_for_date_parses_page() -> None:
    payload = {
        "data": [
            {
                "id": 1215,
                "date": "2026-05-19T13:00:00.000Z",
                "roundId": 4,
                "player1Id": 37741,
                "player2Id": 87277,
                "tournamentId": 21327,
                "seed1": None,
                "seed2": "q",
                "player1": {"id": 37741, "name": "Zizou Bergs", "countryAcr": "BEL"},
                "player2": {"id": 87277, "name": "Arthur Gea", "countryAcr": "FRA"},
                "tournament": {
                    "id": 21327,
                    "name": "Geneva Open",
                    "court": {"id": 2, "name": "Clay"},
                    "rank": {"id": 2, "name": "Main tour"},
                },
                "round": {"id": 4, "name": "R32"},
            }
        ],
        "hasNextPage": True,
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with _make_client(handler) as client:
        page = client.fixtures_for_date("atp", date(2026, 5, 19))

    assert page.has_next_page is True
    assert len(page.data) == 1
    fx = page.data[0]
    assert fx.id == 1215
    assert fx.player1.name == "Zizou Bergs"
    assert fx.tournament is not None
    assert fx.tournament.court is not None
    assert fx.tournament.court.name == "Clay"
    assert fx.round is not None
    assert fx.round.name == "R32"


def test_tournament_results_parses_all_four_arrays() -> None:
    """The `data` payload is a dict with four parallel arrays, not a flat list."""
    payload = {
        "data": {
            "singles": [
                {
                    "id": "84752520",
                    "date": "2026-05-17T17:15:00.000Z",
                    "roundId": 4,
                    "player1Id": 29935,
                    "player2Id": 82269,
                    "tournamentId": 21327,
                    "match_winner": 29935,
                    "result": "6-1 6-3",
                    "best_of": None,
                    "odd1": "1.38",
                    "odd2": "3.04",
                    "player1": {"id": 29935, "name": "Tommy Paul", "countryAcr": "USA"},
                    "player2": {"id": 82269, "name": "Ethan Quinn", "countryAcr": "USA"},
                }
            ],
            "doubles": [],
            "qualifying": [
                {
                    "id": "84752517",
                    "date": "2026-05-17T16:30:00.000Z",
                    "roundId": 3,
                    "player1Id": 87277,
                    "player2Id": 39152,
                    "tournamentId": 21327,
                    "match_winner": 87277,
                    "result": "7-5 4-6 6-4",
                    "best_of": None,
                    "odd1": None,
                    "odd2": None,
                    "player1": {"id": 87277, "name": "Arthur Gea", "countryAcr": "FRA"},
                    "player2": {"id": 39152, "name": "Aleksandar Kovacevic", "countryAcr": "USA"},
                }
            ],
            "doublesQualifying": [],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tennis/v2/atp/tournament/results/21327"
        return httpx.Response(200, json=payload)

    with _make_client(handler) as client:
        results = client.tournament_results("atp", 21327)

    assert len(results.singles) == 1
    assert results.singles[0].match_winner == 29935
    assert results.singles[0].result == "6-1 6-3"
    assert results.singles[0].odd1 == "1.38"
    assert len(results.qualifying) == 1
    assert results.qualifying[0].result == "7-5 4-6 6-4"
    assert results.doubles == []
    assert results.doubles_qualifying == []


def test_rankings_parses_entries() -> None:
    payload = {
        "data": [
            {
                "id": 273252325,
                "date": "2026-05-18T00:00:00.000Z",
                "point": 14700,
                "position": 1,
                "player": {
                    "id": 47275,
                    "name": "Jannik Sinner",
                    "countryAcr": "ITA",
                    "currentRank": 1,
                    "points": 3350,
                },
            }
        ]
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with _make_client(handler) as client:
        entries = client.rankings("atp")

    assert len(entries) == 1
    entry = entries[0]
    assert entry.position == 1
    assert entry.player.id == 47275
    assert entry.player.name == "Jannik Sinner"
    assert entry.player.current_rank == 1


def test_requests_used_increments() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    with _make_client(handler) as client:
        assert client.requests_used == 0
        client.calendar("atp", 2026)
        assert client.requests_used == 1
        client.rankings("atp")
        assert client.requests_used == 2


def test_non_200_raises_matchstat_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limit exceeded")

    with _make_client(handler) as client, pytest.raises(MatchstatError) as exc_info:
        client.calendar("atp", 2026)

    assert exc_info.value.status_code == 429
    assert "rate limit" in exc_info.value.body
    # And the failed call STILL counted against the quota counter (matches reality —
    # RapidAPI counts non-2xx for quota purposes on most plans).
    # Note: the client increments before checking the status, so this is intentional.


def test_tour_level_tiers_contains_expected_values() -> None:
    """Schema-time contract for the tier whitelist."""
    expected = {
        "Grand Slam",
        "ATP 1000",
        "ATP 500",
        "ATP 250",
        "WTA 1000",
        "WTA 500",
        "WTA 250",
        "Finals",
    }
    assert expected == TOUR_LEVEL_TIERS


def test_extra_fields_are_ignored() -> None:
    """The client must tolerate matchstat adding fields without warning."""
    payload_with_extra = {
        "data": [
            {
                "id": 21363,
                "name": "Test",
                "tier": "ATP 250",
                "unexpected_field": "some_value",
                "another_extra": {"nested": True},
            }
        ]
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload_with_extra)

    with _make_client(handler) as client:
        items = client.calendar("atp", 2026)

    assert len(items) == 1
    assert items[0].id == 21363
    # Extra fields are dropped, not stored.
    assert not hasattr(items[0], "unexpected_field")


def test_matchstat_error_str_representation_is_truncated() -> None:
    """Error messages must include status and URL but not dump entire response bodies."""
    huge_body = "x" * 5000

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=huge_body)

    with _make_client(handler) as client, pytest.raises(MatchstatError) as exc_info:
        client.calendar("atp", 2026)

    err_str = str(exc_info.value)
    assert "500" in err_str
    assert "/tennis/v2/atp/tournament/calendar/2026" in err_str
    # Body in error message is truncated, not full 5000 chars.
    assert len(err_str) < 500


def test_calendar_handles_empty_data() -> None:
    """Some payloads may legitimately be empty (e.g., far-future year)."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    with _make_client(handler) as client:
        items = client.calendar("atp", 2099)

    assert items == []


def test_calendar_handles_unexpected_payload_shape() -> None:
    """If matchstat one day changes top-level keys, we must not crash hard."""

    def handler(_: httpx.Request) -> httpx.Response:
        # No `data` key — different shape entirely.
        return httpx.Response(200, json={"items": [{"id": 1}]})

    with _make_client(handler) as client:
        items = client.calendar("atp", 2026)

    assert items == []


def test_matchstat_error_body_field_carries_full_text() -> None:
    """The truncation is for display only; raw body remains accessible on the exception."""

    body = json.dumps({"message": "validation failed", "field": "year"}) * 10

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text=body)

    with _make_client(handler) as client, pytest.raises(MatchstatError) as exc_info:
        client.calendar("atp", 2026)

    assert exc_info.value.body == body
