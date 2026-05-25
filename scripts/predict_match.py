"""CLI entry point — Phase 5 deliverable.

Two invocation modes:

  uv run python scripts/predict_match.py --match-id matchstat::42
      Load a fixture from `scheduled_matches`. Tournament tier / surface /
      players come from the row.

  uv run python scripts/predict_match.py \
      --tour ATP --player-a "Carlos Alcaraz" --player-b "Jannik Sinner" \
      --surface Clay --tournament-level Slam --best-of 5 \
      --date 2026-06-08 --tournament "Roland Garros"
      Free-form: type the match yourself.

The script:

1. Builds a `MatchContext`.
2. Constructs a `TennisAgent` (and through it, an `AnthropicLLMClient`).
3. Runs `TennisAgent.predict()`.
4. Prints the `AgentResponse` and a small footer with cost + cache-hit
   rate sourced from the `llm_traces` row(s) written during this call.

Exit code: 0 on success, 1 on any handled failure (model unavailable,
agent error, validation error). Crash with stack trace on anything else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date
from typing import Literal, cast, get_args

import duckdb

from tennis_predictor.app.context import (
    ContextBuildError,
    load_context_from_freeform,
    load_context_from_match_id,
)
from tennis_predictor.config import DUCKDB_PATH
from tennis_predictor.data.schema import create_all_tables
from tennis_predictor.features.schema import Surface, TournamentLevel
from tennis_predictor.llm.agent import (
    AgentError,
    TennisAgent,
)
from tennis_predictor.llm.cost import cache_hit_rate
from tennis_predictor.llm.tools.schemas import (
    MatchContext,
    ModelUnavailableError,
    PlayerResolutionError,
    Tour,
)
from tennis_predictor.llm.tools.submit import AgentResponse

logger = logging.getLogger("predict_match")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict_match",
        description=(
            "Run the LLM agent on a single upcoming match. Either pass "
            "--match-id from scheduled_matches, or specify the match manually."
        ),
    )
    p.add_argument(
        "--match-id",
        type=str,
        help="scheduled_matches.scheduled_match_id to look up (e.g. 'matchstat::42').",
    )
    p.add_argument("--tour", choices=get_args(Tour))
    p.add_argument("--player-a", type=str, help="Free-form mode: player A full name.")
    p.add_argument("--player-b", type=str, help="Free-form mode: player B full name.")
    p.add_argument(
        "--surface",
        choices=get_args(Surface),
        help="Free-form mode: Hard / IHard / Clay / Grass.",
    )
    p.add_argument(
        "--tournament-level",
        choices=get_args(TournamentLevel),
        help="Free-form mode: Slam / M1000 / ATP500 / ATP250 / WTA500 / WTA250 / Finals.",
    )
    p.add_argument(
        "--tournament",
        type=str,
        default=None,
        help="Optional tournament name; surfaced in the user message for the LLM.",
    )
    p.add_argument(
        "--best-of",
        type=int,
        choices=(3, 5),
        help="Free-form mode: 3 or 5 (defaulted from --tour and --tournament-level).",
    )
    p.add_argument(
        "--date",
        type=str,
        help="Free-form mode: match date YYYY-MM-DD.",
    )
    p.add_argument(
        "--db",
        type=str,
        default=str(DUCKDB_PATH),
        help="Path to the DuckDB file (defaults to config.DUCKDB_PATH).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print the AgentResponse as JSON instead of human-readable text.",
    )
    return p


def _build_freeform_context(args: argparse.Namespace) -> MatchContext:
    """Validate the CLI free-form flags and call the shared builder."""
    missing = [
        flag
        for flag, val in (
            ("--tour", args.tour),
            ("--player-a", args.player_a),
            ("--player-b", args.player_b),
            ("--surface", args.surface),
            ("--tournament-level", args.tournament_level),
            ("--date", args.date),
        )
        if val is None
    ]
    if missing:
        raise SystemExit("free-form mode requires: " + ", ".join(missing) + " (or use --match-id)")
    try:
        match_date = date.fromisoformat(args.date)
    except ValueError as exc:
        raise SystemExit(f"invalid --date {args.date!r}: {exc}") from exc

    best_of_arg = cast(Literal[3, 5] | None, args.best_of)
    try:
        return load_context_from_freeform(
            tour=args.tour,
            player_a_name=args.player_a,
            player_b_name=args.player_b,
            surface=args.surface,
            tournament_level=args.tournament_level,
            match_date=match_date,
            best_of=best_of_arg,
            tournament_name=args.tournament,
        )
    except ContextBuildError as exc:
        raise SystemExit(str(exc)) from exc


def _print_human_readable(ctx: MatchContext, resp: AgentResponse) -> None:
    print()
    print(f"=== {ctx.player_a_name} vs {ctx.player_b_name} ({ctx.tour}) ===")
    if ctx.tournament_name:
        print(
            f"    {ctx.tournament_name}, {ctx.surface}, {ctx.tournament_level}, best of {ctx.best_of}"
        )
    else:
        print(f"    {ctx.surface}, {ctx.tournament_level}, best of {ctx.best_of}")
    print(f"    Match date: {ctx.match_date.isoformat()}")
    print()
    print("Model probability:")
    print(f"  P({ctx.player_a_name} wins) = {resp.model_probability_player_a:.3f}")
    print(f"  P({ctx.player_b_name} wins) = {resp.model_probability_player_b:.3f}")
    print()
    print(f"News lookup status: {resp.news_lookup_status}")
    print()
    if resp.news_items:
        print(f"News items ({len(resp.news_items)}):")
        for item in resp.news_items:
            date_part = item.published_date or "date unknown"
            print(f"  - [{item.source_domain}, {date_part}] ({item.category}) {item.title}")
            print(f"      {item.url}")
            if item.snippet:
                print(f"      {item.snippet}")
    else:
        if resp.news_lookup_status == "no_results":
            print("No notable news in the last 32 days for either player.")
        elif resp.news_lookup_status == "failed":
            print("News lookup unavailable.")
    if resp.tools_used:
        print()
        print("Tools used: " + ", ".join(resp.tools_used))


def _print_cost_footer(conn: duckdb.DuckDBPyConnection, since_trace_id: int) -> None:
    """Sum cost + token counters from llm_traces rows >= since_trace_id
    and print one line. `since_trace_id` is captured before the agent
    runs so only this prediction's rows are summed."""
    row = conn.execute(
        """
        SELECT
            COALESCE(SUM(estimated_cost_usd), 0.0),
            COALESCE(SUM(tokens_in), 0),
            COALESCE(SUM(cache_read_tokens), 0),
            COALESCE(SUM(cache_creation_tokens), 0),
            COALESCE(SUM(web_search_count), 0),
            COUNT(*)
        FROM llm_traces
        WHERE trace_id > ?
        """,
        [since_trace_id],
    ).fetchone()
    if row is None:
        return
    cost, tokens_in, cache_read, cache_creation, web_searches, n = row
    rate = cache_hit_rate(
        tokens_in=int(tokens_in),
        cache_read_tokens=int(cache_read),
        cache_creation_tokens=int(cache_creation),
    )
    print()
    print(
        f"Cost: ${float(cost):.4f}  "
        f"cache hit rate: {rate * 100:.0f}%  "
        f"iterations: {int(n)}  "
        f"web searches: {int(web_searches)}"
    )


def _current_trace_id(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute("SELECT COALESCE(max(trace_id), 0) FROM llm_traces").fetchone()
    return int(row[0]) if row is not None else 0


async def _run(ctx: MatchContext, conn: duckdb.DuckDBPyConnection) -> AgentResponse:
    agent = TennisAgent(conn)
    return await agent.predict(ctx)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    conn = duckdb.connect(args.db)
    create_all_tables(conn)

    if args.match_id:
        try:
            ctx = load_context_from_match_id(conn, args.match_id)
        except ContextBuildError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        ctx = _build_freeform_context(args)

    since_trace_id = _current_trace_id(conn)
    try:
        response = asyncio.run(_run(ctx, conn))
    except ModelUnavailableError as exc:
        logger.error("model artifact unavailable: %s", exc)
        return 1
    except PlayerResolutionError as exc:
        logger.error("player not resolved: %s", exc)
        return 1
    except AgentError as exc:
        logger.error("agent loop failed: %s", exc)
        return 1

    if args.json:
        print(json.dumps(response.model_dump(), indent=2, default=str))
    else:
        _print_human_readable(ctx, response)
    _print_cost_footer(conn, since_trace_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
