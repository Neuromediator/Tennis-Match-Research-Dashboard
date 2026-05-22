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
from datetime import date, datetime
from typing import Literal, cast, get_args

import duckdb

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


# matchstat's tier strings → our canonical `TournamentLevel`. Mirrors the
# whitelist in `data/matchstat.py` and the canonical levels in
# `features/schema.py`. Anything not in this dict aborts with a clean
# error: we will not silently coerce a Challenger row into "ATP250".
_MATCHSTAT_TIER_TO_LEVEL: dict[str, TournamentLevel] = {
    "Grand Slam": "Slam",
    "ATP Masters 1000": "M1000",
    "ATP 500": "ATP500",
    "ATP 250": "ATP250",
    "WTA Masters 1000": "M1000",
    "WTA 1000": "M1000",
    "WTA 500": "WTA500",
    "WTA 250": "WTA250",
    "Finals": "Finals",
}

# Best-of inferred from the tournament level when running in --match-id
# mode: ATP Slams are best-of-5 and so are no other current tour-level
# events on the men's side; WTA is best-of-3 everywhere. Free-form mode
# accepts an explicit --best-of.
_LEVEL_BEST_OF_DEFAULT: dict[tuple[str, TournamentLevel], int] = {
    ("ATP", "Slam"): 5,
    ("ATP", "M1000"): 3,
    ("ATP", "ATP500"): 3,
    ("ATP", "ATP250"): 3,
    ("ATP", "Finals"): 3,
    ("WTA", "Slam"): 3,
    ("WTA", "M1000"): 3,
    ("WTA", "WTA500"): 3,
    ("WTA", "WTA250"): 3,
    ("WTA", "Finals"): 3,
}


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


def _load_context_from_match_id(
    conn: duckdb.DuckDBPyConnection, scheduled_match_id: str
) -> MatchContext:
    """Look up `scheduled_matches` by scheduled_match_id and map its
    fields onto a `MatchContext`."""
    row = conn.execute(
        """
        SELECT tour, player1_name, player2_name, surface, tournament_tier,
               tournament_name, scheduled_start_utc
        FROM scheduled_matches
        WHERE scheduled_match_id = ?
        """,
        [scheduled_match_id],
    ).fetchone()
    if row is None:
        raise SystemExit(f"no scheduled match found with id={scheduled_match_id!r}")
    (
        tour,
        player1_name,
        player2_name,
        surface,
        tier,
        tournament_name,
        scheduled_start_utc,
    ) = row

    if tour not in ("ATP", "WTA"):
        raise SystemExit(f"unsupported tour {tour!r} in scheduled_matches row")
    if surface not in get_args(Surface):
        raise SystemExit(
            f"surface {surface!r} not in supported set {get_args(Surface)}; "
            "row may pre-date Phase-2 surface normalisation."
        )
    level = _MATCHSTAT_TIER_TO_LEVEL.get(tier or "")
    if level is None:
        raise SystemExit(
            f"tournament_tier {tier!r} does not map to a model tournament_level. "
            "Out-of-scope events (Challengers, ITF) cannot be predicted."
        )

    best_of = cast(Literal[3, 5], _LEVEL_BEST_OF_DEFAULT[(tour, level)])
    match_date = (
        scheduled_start_utc.date()
        if isinstance(scheduled_start_utc, datetime)
        else (scheduled_start_utc or date.today())
    )
    return MatchContext(
        tour=tour,
        player_a_name=player1_name,
        player_b_name=player2_name,
        surface=surface,
        tournament_level=level,
        tournament_name=tournament_name,
        best_of=best_of,
        match_date=match_date,
        scheduled_match_id=scheduled_match_id,
    )


def _load_context_from_freeform(args: argparse.Namespace) -> MatchContext:
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

    best_of_raw = args.best_of or _LEVEL_BEST_OF_DEFAULT.get((args.tour, args.tournament_level))
    if best_of_raw is None:
        raise SystemExit(
            "could not infer --best-of for "
            f"({args.tour}, {args.tournament_level}); please pass it explicitly"
        )
    best_of = cast(Literal[3, 5], best_of_raw)
    return MatchContext(
        tour=args.tour,
        player_a_name=args.player_a,
        player_b_name=args.player_b,
        surface=args.surface,
        tournament_level=args.tournament_level,
        tournament_name=args.tournament,
        best_of=best_of,
        match_date=match_date,
        scheduled_match_id=None,
    )


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
    print(f"Confidence band: {resp.confidence_band}")
    print()
    print("Key factors:")
    for factor in resp.key_factors:
        print(f"  - {factor}")
    print()
    print("Narrative:")
    print("  " + resp.narrative.replace("\n", "\n  "))
    if resp.caveats:
        print()
        print("Caveats:")
        for caveat in resp.caveats:
            print(f"  - {caveat}")
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
        ctx = _load_context_from_match_id(conn, args.match_id)
    else:
        ctx = _load_context_from_freeform(args)

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
