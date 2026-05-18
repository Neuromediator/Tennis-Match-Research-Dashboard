"""matchstat Tennis API client (Phase 2 hot data source).

Typed HTTP wrapper for the four endpoint families we use:

- `/{tour}/tournament/calendar/{year}` — season inventory.
- `/{tour}/fixtures/{date}` — upcoming fixtures for a date (with mandatory
  include + singles filter).
- `/{tour}/tournament/results/{seasonid}` — completed matches with scores
  and pre-match odds.
- `/{tour}/ranking/singles` — current rankings.

Each method returns parsed Pydantic objects, NOT raw dicts. Pagination is
not handled internally — caller passes `page_no` / `page_size` and loops
on `has_next_page`. This keeps the client thin; the orchestrator decides
when to stop paging based on its own budget.

Responses are validated; unexpected fields are silently dropped (we don't
want a schema drift on matchstat's side to break refresh outright — a
later test asserts that the contract fields are still present).

Does NOT touch DuckDB. See `load_hot.py` (TODO phase-2) for the transform
into table rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

API_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}/tennis/v2"

TourCode = Literal["atp", "wta"]

TOUR_LEVEL_TIERS: frozenset[str] = frozenset(
    {
        "Grand Slam",
        "ATP 1000",
        "ATP 500",
        "ATP 250",
        "WTA 1000",
        "WTA 500",
        "WTA 250",
        "Finals",
    }
)


class _Permissive(BaseModel):
    """Drop unexpected fields silently. matchstat may add fields without notice."""

    model_config = ConfigDict(extra="ignore")


class Court(_Permissive):
    id: int | None = None
    name: str | None = None  # "Clay" | "Hard" | "I.hard" | "Grass" | ...


class Rank(_Permissive):
    id: int | None = None
    name: str | None = None


class Country(_Permissive):
    name: str | None = None
    acronym: str | None = None


class CalendarTournament(_Permissive):
    """One tournament from `/tournament/calendar/{year}`.

    `id` here is the seasonid used to look up results. `tier` drives the
    tour-level filter (see `TOUR_LEVEL_TIERS`).
    """

    id: int
    name: str | None = None
    tier: str | None = None
    date: datetime | None = None
    court: Court | None = None
    rank: Rank | None = None
    country: Country | None = None
    country_acr: str | None = Field(default=None, alias="countryAcr")

    @model_validator(mode="before")
    @classmethod
    def _fix_typo(cls, data: Any) -> Any:
        # API returns `coutry` (missing the n) in calendar payloads.
        if isinstance(data, dict) and "coutry" in data and "country" not in data:
            data["country"] = data.pop("coutry")
        return data


class FixturePlayer(_Permissive):
    id: int
    name: str
    country_acr: str | None = Field(default=None, alias="countryAcr")


class FixtureTournament(_Permissive):
    id: int
    name: str | None = None
    court: Court | None = None
    rank: Rank | None = None
    country_acr: str | None = Field(default=None, alias="countryAcr")


class FixtureRound(_Permissive):
    id: int | None = None
    name: str | None = None


class Fixture(_Permissive):
    """One upcoming-match row from `/fixtures/{date}`.

    `id` is the fixture-row id (small integer). Note this is NOT the same
    identifier as `Match.id` from tournament/results.
    """

    id: int
    date: datetime | None = None
    round_id: int | None = Field(default=None, alias="roundId")
    player1_id: int = Field(alias="player1Id")
    player2_id: int = Field(alias="player2Id")
    tournament_id: int = Field(alias="tournamentId")
    seed1: str | None = None
    seed2: str | None = None
    player1: FixturePlayer
    player2: FixturePlayer
    tournament: FixtureTournament | None = None
    round: FixtureRound | None = None


class FixturesPage(_Permissive):
    data: list[Fixture]
    has_next_page: bool = Field(default=False, alias="hasNextPage")


class Match(_Permissive):
    """One completed-match row from `/tournament/results/{seasonid}`.

    `id` is the 8-digit match-record id (string). `result` is a
    space-separated score like `"6-4 6-3"`. `odd1` / `odd2` are pre-match
    decimal odds for player1 / player2, stored as strings by the API.
    """

    id: str
    date: datetime | None = None
    round_id: int | None = Field(default=None, alias="roundId")
    player1_id: int = Field(alias="player1Id")
    player2_id: int = Field(alias="player2Id")
    tournament_id: int = Field(alias="tournamentId")
    match_winner: int | None = None
    result: str | None = None
    best_of: int | None = None
    odd1: str | None = None
    odd2: str | None = None
    player1: FixturePlayer
    player2: FixturePlayer


class TournamentResults(_Permissive):
    """`/tournament/results/{seasonid}` payload.

    Four arrays — we consume `singles` only for the predictor's `matches`
    table; the other three are exposed in case a future use needs them.
    """

    singles: list[Match] = Field(default_factory=list)
    doubles: list[Match] = Field(default_factory=list)
    qualifying: list[Match] = Field(default_factory=list)
    doubles_qualifying: list[Match] = Field(default_factory=list, alias="doublesQualifying")


class RankingPlayer(_Permissive):
    id: int
    name: str
    country_acr: str | None = Field(default=None, alias="countryAcr")
    current_rank: int | None = Field(default=None, alias="currentRank")
    points: int | None = None


class RankingEntry(_Permissive):
    position: int
    point: int | None = None
    date: datetime | None = None
    player: RankingPlayer


class MatchstatError(RuntimeError):
    """Raised on non-2xx responses or invalid JSON."""

    def __init__(self, status_code: int, body: str, url: str) -> None:
        super().__init__(f"matchstat {status_code} at {url}: {body[:200]}")
        self.status_code = status_code
        self.body = body
        self.url = url


class MatchstatClient:
    """Thin typed wrapper over the matchstat REST API.

    Tracks `requests_used` for quota accounting — caller writes it into
    `ingestion_runs` after each refresh.
    """

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": API_HOST,
        }
        self._client = client or httpx.Client(
            base_url=BASE_URL,
            headers=self._headers,
            timeout=timeout,
        )
        self._owns_client = client is None
        self.requests_used: int = 0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> MatchstatClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        self.requests_used += 1
        response = self._client.get(path, params=params)
        if response.status_code != 200:
            raise MatchstatError(response.status_code, response.text, str(response.url))
        return response.json()

    def calendar(self, tour: TourCode, year: int) -> list[CalendarTournament]:
        payload = self._get_json(f"/{tour}/tournament/calendar/{year}", {"pageSize": "100"})
        items = payload.get("data", []) if isinstance(payload, dict) else []
        return [CalendarTournament.model_validate(item) for item in items]

    def fixtures_for_date(
        self,
        tour: TourCode,
        match_date: date,
        *,
        singles_only: bool = True,
        page_size: int = 100,
        page_no: int = 1,
    ) -> FixturesPage:
        params: dict[str, str] = {
            "include": "tournament.court,tournament.rank,round",
            "pageSize": str(page_size),
            "pageNo": str(page_no),
        }
        if singles_only:
            params["filter"] = "PlayerGroup:singles"
        payload = self._get_json(f"/{tour}/fixtures/{match_date.isoformat()}", params)
        return FixturesPage.model_validate(payload)

    def tournament_results(self, tour: TourCode, season_id: int) -> TournamentResults:
        payload = self._get_json(f"/{tour}/tournament/results/{season_id}")
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        return TournamentResults.model_validate(data)

    def rankings(
        self,
        tour: TourCode,
        *,
        page_size: int = 100,
        page_no: int = 1,
    ) -> list[RankingEntry]:
        params = {"pageSize": str(page_size), "pageNo": str(page_no)}
        payload = self._get_json(f"/{tour}/ranking/singles", params)
        items = payload.get("data", []) if isinstance(payload, dict) else []
        return [RankingEntry.model_validate(item) for item in items]


@contextmanager
def matchstat_client(api_key: str) -> Iterator[MatchstatClient]:
    """Convenience context manager: `with matchstat_client(key) as c: ...`."""
    client = MatchstatClient(api_key)
    try:
        yield client
    finally:
        client.close()
