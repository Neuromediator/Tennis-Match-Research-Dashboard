"""Probe matchstat Tennis API to inspect endpoint response shapes.

Hits a small set of read-only endpoints and prints abbreviated JSON to
stdout. Used to design the schema for `scheduled_matches`, `matches`
enrichment, and the hot refresh script; NOT part of the daily refresh
pipeline. The `probes` list in main() is the only thing that changes
between probe rounds.

Cost: ~4-5 requests per run against the 500/month free-tier quota.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

import httpx

from tennis_predictor.config import X_RAPIDAPI_KEY

API_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}/tennis/v2"
LIST_PREVIEW_LIMIT = 2


def abbreviate(obj: Any) -> Any:
    if isinstance(obj, list):
        if len(obj) <= LIST_PREVIEW_LIMIT:
            return [abbreviate(x) for x in obj]
        head = [abbreviate(x) for x in obj[:LIST_PREVIEW_LIMIT]]
        return [*head, f"...and {len(obj) - LIST_PREVIEW_LIMIT} more items omitted"]
    if isinstance(obj, dict):
        return {k: abbreviate(v) for k, v in obj.items()}
    return obj


def probe(
    client: httpx.Client,
    path: str,
    params: dict[str, str] | None,
    label: str,
) -> Any | None:
    """Hit one endpoint, print abbreviated JSON, return the parsed data (or None on failure)."""
    url = f"{BASE_URL}{path}"
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)

    try:
        response = client.get(url, params=params, timeout=15.0)
    except httpx.RequestError as e:
        print(f"GET {url}\nREQUEST ERROR: {e}")
        return None

    print(f"GET {response.url}")
    remaining = response.headers.get("x-ratelimit-requests-remaining", "n/a")
    print(f"status: {response.status_code}   quota remaining: {remaining}")

    if response.status_code != 200:
        print(f"body: {response.text[:500]}")
        return None

    try:
        data = response.json()
    except ValueError:
        print(f"non-json body: {response.text[:500]}")
        return None

    print(json.dumps(abbreviate(data), indent=2, ensure_ascii=False, default=str))
    return data


def pick_recent_tour_level_seasonid(calendar_data: Any) -> int | None:
    """From a calendar response, find a seasonid of a recently-finished tour-level tournament."""
    if not isinstance(calendar_data, dict):
        return None
    items = calendar_data.get("data") if "data" in calendar_data else calendar_data
    if not isinstance(items, list):
        return None
    today = datetime.now(UTC).date()
    candidates: list[tuple[int, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rank = item.get("rank")
        if rank is None:
            continue
        season_id = item.get("id")
        end_date_str = item.get("endDate") or item.get("date")
        if not isinstance(season_id, int):
            continue
        candidates.append((season_id, end_date_str))

    for season_id, end_date_str in candidates:
        if isinstance(end_date_str, str):
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).date()
            except ValueError:
                continue
            if end_date < today:
                return season_id
    return candidates[0][0] if candidates else None


def main() -> int:
    if X_RAPIDAPI_KEY is None:
        print("X_RAPIDAPI_KEY not set in environment (.env)", file=sys.stderr)
        return 1

    today = datetime.now(UTC).date()
    year = today.year
    sinner_id = 47275  # ATP #1 from round-1 probe
    alcaraz_id = 68074  # ATP #2 from round-1 probe

    headers = {
        "X-RapidAPI-Key": X_RAPIDAPI_KEY,
        "X-RapidAPI-Host": API_HOST,
    }
    with httpx.Client(headers=headers) as client:
        # 1. Calendar of tournaments for the current year — we need a seasonid to query results.
        calendar_data = probe(
            client,
            f"/atp/tournament/calendar/{year}",
            {"pageSize": "30"},
            f"1. ATP tournament calendar {year} (find a seasonid for tour-level event)",
        )

        # 2. Tournament results — try a recently-finished tour-level event from the calendar,
        #    or fall back to the season-id we observed in round-1 probe (21327).
        season_id = pick_recent_tour_level_seasonid(calendar_data) or 21327
        probe(
            client,
            f"/atp/tournament/results/{season_id}",
            None,
            f"2. ATP tournament/results/{season_id} — does `result` field carry the score?",
        )

        # 3. Player past-matches — the per-player history doc'd as having scores.
        probe(
            client,
            f"/atp/player/past-matches/{sinner_id}",
            {"pageSize": "10"},
            f"3. ATP player/past-matches/{sinner_id} (Sinner) — score format and shape",
        )

        # 4. H2H — confirm scores per the original fixtures-doc claim.
        probe(
            client,
            f"/atp/fixtures/h2h/{sinner_id}/{alcaraz_id}",
            {"pageSize": "10"},
            f"4. ATP fixtures/h2h/{sinner_id}/{alcaraz_id} (Sinner vs Alcaraz) — H2H scores",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
