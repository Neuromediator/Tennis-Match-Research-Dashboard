"""Tavily-snippet fallback for pre-match odds (Phase 6.2).

Used when The Odds API has no row for a fixture — small qualifier
events, late-published draws, or rare gaps in tour-level coverage.
The fallback issues a single Tavily search and regex-extracts decimal
odds from the snippets. Confidence is lower than the API path, so the
resulting `pre_match_odds` row is flagged `source='tavily'` and the
Prediction page surfaces it as "estimated from web search."

Contract:
- Single async call (`tavily_extract_odds`) that returns at most one
  `AggregatedOdds`-shaped object. Caller upserts to the table or skips.
- Never raises on missing odds in the result — returns None and the UI
  shows "Market: odds unavailable." Only raises `TavilyError` on a
  transport/HTTP failure (same contract as `search_web`).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from tennis_predictor.data.odds_api import AggregatedOdds, decimal_odds_to_implied_probs
from tennis_predictor.llm.tools.schemas import TavilyError, WebSearchInput
from tennis_predictor.llm.tools.web_search import search_web

logger = logging.getLogger(__name__)

# Decimal-odds pattern: one digit `1-9` then optional second digit, dot,
# two-digit fraction. Anchored on word boundaries so prices embedded in
# sentences ("Pinnacle 1.07 on Sinner") are extracted but version
# numbers like "v1.05" are not. Range covers 1.01 (heavy favourite) to
# 99.99 (heavy underdog).
_DECIMAL_ODDS_RE = re.compile(r"\b([1-9]\d?\.\d{2})\b")


def _extract_pair(text: str) -> tuple[float, float] | None:
    """Return the first plausible (favourite_odds, underdog_odds) pair
    from `text`. "Plausible" means one of the two decimals is below 2.0
    (a real fixture has at least one side priced as favourite) AND the
    two prices together imply an overround under 25% (loose guard
    against picking unrelated decimals from the same snippet)."""
    matches = [float(m) for m in _DECIMAL_ODDS_RE.findall(text)]
    if len(matches) < 2:
        return None
    for i, a in enumerate(matches):
        for b in matches[i + 1 :]:
            if 1.01 <= a <= 99.0 and 1.01 <= b <= 99.0:
                overround = 1 / a + 1 / b
                if 0.95 <= overround <= 1.25:
                    if a <= b:
                        return a, b
                    return b, a
    return None


async def tavily_extract_odds(
    *,
    tour: str,
    player_a_name: str,
    player_b_name: str,
    tournament_name: str | None,
    commence_time_utc: datetime,
    max_results: int = 5,
) -> AggregatedOdds | None:
    """Run a single Tavily search for the matchup and regex-extract odds
    from the snippets. Returns None when no plausible pair is found.

    The pair returned is oriented (player_a = favourite). Callers store
    the row keyed by the deterministic `fixture_match_key` (which is
    order-independent), so a swap between favourite/underdog vs the
    fixture's `home_team`/`away_team` is invisible at JOIN time."""
    query_parts = [player_a_name, "vs", player_b_name]
    if tournament_name:
        query_parts.append(tournament_name)
    query_parts.extend(["odds", "pinnacle"])
    query = " ".join(query_parts)

    try:
        result = await search_web(WebSearchInput(query=query, max_results=max_results))
    except TavilyError as exc:
        logger.warning(
            "tavily odds fallback failed for %s vs %s: %s", player_a_name, player_b_name, exc
        )
        return None

    for hit in result.results:
        pair = _extract_pair(hit.snippet) or _extract_pair(hit.title)
        if pair is None:
            continue
        odds_a, odds_b = pair
        prob_a, prob_b = decimal_odds_to_implied_probs(odds_a, odds_b)
        return AggregatedOdds(
            sport_key=f"tavily_{tour.lower()}",
            event_id="tavily-extract",
            tour=tour,
            player_a_name=player_a_name,
            player_b_name=player_b_name,
            commence_time_utc=(
                commence_time_utc
                if commence_time_utc.tzinfo is not None
                else commence_time_utc.replace(tzinfo=UTC)
            ),
            median_odds_a=odds_a,
            median_odds_b=odds_b,
            best_odds_a=odds_a,
            best_odds_b=odds_b,
            median_implied_prob_a=prob_a,
            median_implied_prob_b=prob_b,
            books_count=1,
            pinnacle_odds_a=None,
            pinnacle_odds_b=None,
            pinnacle_implied_prob_a=None,
            pinnacle_implied_prob_b=None,
        )

    return None


__all__ = [
    "tavily_extract_odds",
]
