"""Single-match prediction page (Phase 6.1 layout).

Renders a stack of deterministic blocks for every match: model
probability + tools-used header, H2H detail, surface Elo, two-column
recent form, then the LLM-discovered news block.

Entry route: clicked from `home.py` (which stashes the id in
`st.session_state[PENDING_MATCH_KEY]`) or opened directly with
`?match_id=matchstat::42` in the URL. The page loads the row from
`scheduled_matches`, builds a `MatchContext`, runs the cached
`TennisAgent.predict` flow for the news block, and renders everything
deterministically alongside.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import streamlit as st

from tennis_predictor.app.context import (
    ContextBuildError,
    load_context_from_match_id,
)
from tennis_predictor.app.db import DuckDBLockError, get_connection
from tennis_predictor.app.why_differs import compute_reasons
from tennis_predictor.app.widgets import (
    COMPARISON_DIFF_THRESHOLD_PP,
    ComparisonRow,
    back_to_home_button,
    format_match_time_for_display,
    news_block,
    prediction_card,
    render_h2h_for_context,
    render_recent_form_for_context,
    render_surface_elo_for_context,
    signal_comparison_block,
    stale_data_banner,
    why_model_differs_block,
)
from tennis_predictor.data.pre_match_odds import (
    is_row_fresh,
    lookup_for_fixture,
    upsert_aggregated,
)
from tennis_predictor.data.reconcile import AliasIndex
from tennis_predictor.llm.agent import (
    AgentError,
    BudgetExceededError,
    TennisAgent,
)
from tennis_predictor.llm.tools.db_tools import get_surface_elo
from tennis_predictor.llm.tools.schemas import (
    GetSurfaceEloInput,
    MatchContext,
    ModelUnavailableError,
    PlayerResolutionError,
    TavilyError,
)

PENDING_MATCH_KEY = "pending_match_id"
# Phase 6.2: per-fixture cache key in `st.session_state`. Holds the
# AgentResponse the first run produced so revisiting the page (sidebar
# nav round-trip, browser back) does not re-invoke the LLM agent.
PREDICTION_SESSION_PREFIX = "prediction::"

_MIN_RESOLUTION_CONFIDENCE: float = 0.85


def _session_key_for_ctx(ctx: MatchContext) -> str:
    """Stable session-state key for a given MatchContext. Scheduled
    fixtures key on `scheduled_match_id`; custom-prediction inputs key
    on the (tour, players, surface, level, date) tuple so each distinct
    custom matchup caches independently."""
    if ctx.scheduled_match_id is not None:
        return f"{PREDICTION_SESSION_PREFIX}{ctx.scheduled_match_id}"
    return (
        f"{PREDICTION_SESSION_PREFIX}custom::{ctx.tour}::"
        f"{ctx.player_a_name}::{ctx.player_b_name}::{ctx.surface}::"
        f"{ctx.tournament_level}::{ctx.match_date.isoformat()}"
    )


def _resolve_match_id() -> str | None:
    pending = st.session_state.pop(PENDING_MATCH_KEY, None)
    if pending:
        st.query_params["match_id"] = pending
        return pending
    return st.query_params.get("match_id")


def _resolve_player_ids(conn, ctx: MatchContext) -> tuple[str, str]:
    """Resolve both player names to canonical IDs using the same
    AliasIndex the DB tools use. Raises `PlayerResolutionError` if
    either side is below the confidence floor."""
    index = AliasIndex(conn, ctx.tour)
    a_res = index.lookup(ctx.player_a_name)
    b_res = index.lookup(ctx.player_b_name)
    if a_res.canonical_player_id is None or a_res.confidence < _MIN_RESOLUTION_CONFIDENCE:
        raise PlayerResolutionError(
            f"could not resolve {ctx.player_a_name!r} on {ctx.tour} tour "
            f"(best candidate {a_res.candidate_name!r}, confidence {a_res.confidence:.2f})"
        )
    if b_res.canonical_player_id is None or b_res.confidence < _MIN_RESOLUTION_CONFIDENCE:
        raise PlayerResolutionError(
            f"could not resolve {ctx.player_b_name!r} on {ctx.tour} tour "
            f"(best candidate {b_res.candidate_name!r}, confidence {b_res.confidence:.2f})"
        )
    return a_res.canonical_player_id, b_res.canonical_player_id


def _render_match_time(conn, ctx: MatchContext) -> None:
    """If this MatchContext came from a scheduled fixture, render the
    start time with CEST + UTC labels. Otherwise (custom prediction
    flow) skip the header line."""
    if ctx.scheduled_match_id is None:
        return
    row = conn.execute(
        "SELECT scheduled_start_utc FROM scheduled_matches WHERE scheduled_match_id = ?",
        [ctx.scheduled_match_id],
    ).fetchone()
    if row is None:
        return
    st.caption(format_match_time_for_display(row[0]))


def _run_agent(conn, ctx: MatchContext):
    """Run the agent loop once and return the AgentResponse. Maps the
    documented failure modes to user-visible warnings.

    The `st.spinner` is critical UX: the agent loop takes 15-40 seconds
    (model inference + 2-3 LLM iterations + Tavily fan-out), and without
    a spinner the user sees a frozen page and assumes the click was
    lost. Streamlit's spinner stays visible until the wrapped block
    returns, which is exactly what we need."""
    try:
        with st.spinner("Running model + LLM news lookup… (~20-40s)"):
            agent = TennisAgent(conn)
            return asyncio.run(agent.predict(ctx))
    except ModelUnavailableError as exc:
        st.error(f"Model artifact not loaded — prediction cannot run. ({exc})")
        return None
    except PlayerResolutionError as exc:
        st.error(f"Player resolution failed: {exc}")
        return None
    except BudgetExceededError as exc:
        st.warning(f"Today's prediction budget reached. ({exc})")
        return None
    except TavilyError as exc:
        st.warning(f"News lookup unavailable: {exc}.")
        return None
    except AgentError as exc:
        st.error(f"Prediction service temporarily unavailable: {exc}")
        return None


def _market_commence_time(conn, ctx: MatchContext) -> datetime | None:
    """Return the UTC start time used to key into `pre_match_odds`.

    For a scheduled fixture we use `scheduled_start_utc`. For a custom
    prediction we use midnight UTC on `match_date` — the daily refresh
    keys odds rows on the calendar-day part of `commence_time_utc` so
    midnight UTC always matches when an Odds API event exists for that
    day."""
    if ctx.scheduled_match_id is None:
        return datetime(ctx.match_date.year, ctx.match_date.month, ctx.match_date.day, tzinfo=UTC)
    row = conn.execute(
        "SELECT scheduled_start_utc FROM scheduled_matches WHERE scheduled_match_id = ?",
        [ctx.scheduled_match_id],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    raw = row[0]
    if raw.tzinfo is None:
        return raw.replace(tzinfo=UTC)
    return raw.astimezone(UTC)


def _maybe_tavily_fallback_for_odds(conn, ctx: MatchContext, commence: datetime) -> None:
    """Best-effort Tavily snippet-search fallback when no
    `pre_match_odds` row exists yet for this fixture. Swallows all
    errors — the UI just shows "Market: odds unavailable" if the
    extract finds nothing."""
    try:
        # Local import — keeps the heavyweight asyncio + httpx graph out
        # of the page-load critical path when the row already exists.
        import asyncio as _asyncio

        from tennis_predictor.data.odds_fallback import tavily_extract_odds

        agg = _asyncio.run(
            tavily_extract_odds(
                tour=ctx.tour,
                player_a_name=ctx.player_a_name,
                player_b_name=ctx.player_b_name,
                tournament_name=ctx.tournament_name,
                commence_time_utc=commence,
            )
        )
        if agg is not None:
            upsert_aggregated(conn, [agg], source="tavily")
    except Exception:
        # Lazy fallback is best-effort: a network glitch or rate-limit
        # must never block the rest of the Prediction page.
        return


def _load_market_row(conn, ctx: MatchContext) -> dict | None:
    """Resolve the `pre_match_odds` row for this fixture; trigger a
    Tavily fallback on cache miss; return the (possibly newly written)
    row or None when no data could be sourced."""
    commence = _market_commence_time(conn, ctx)
    if commence is None:
        return None
    row = lookup_for_fixture(conn, ctx.tour, ctx.player_a_name, ctx.player_b_name, commence)
    if is_row_fresh(row):
        return row
    if row is None:
        _maybe_tavily_fallback_for_odds(conn, ctx, commence)
        row = lookup_for_fixture(conn, ctx.tour, ctx.player_a_name, ctx.player_b_name, commence)
    return row


def _surface_elo_prob_a(conn, ctx: MatchContext) -> float | None:
    """Cheap independent call to `get_surface_elo` so the comparison
    row shows the Elo baseline without recomputing it inside the
    Streamlit surface-Elo block below."""
    try:
        summary = get_surface_elo(
            conn,
            GetSurfaceEloInput(
                player_a_name=ctx.player_a_name,
                player_b_name=ctx.player_b_name,
                tour=ctx.tour,
                surface=ctx.surface,
                as_of_date=ctx.match_date,
            ),
        )
    except Exception:
        return None
    return summary.baseline_prob_a


def _log_prediction(conn, ctx: MatchContext, model_probability_player_a: float) -> None:
    """Append one row to `prediction_log` so the Dashboard scoreboard
    can show recent model-vs-market gaps. Silently swallows write
    failures — the user-visible result must not break because the
    audit log hiccupped."""
    try:
        conn.execute(
            """
            INSERT INTO prediction_log (
                ts, scheduled_match_id, tour, player_a_name, player_b_name,
                surface, match_date, model_probability_player_a
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                datetime.now(UTC).replace(tzinfo=None),
                ctx.scheduled_match_id,
                ctx.tour,
                ctx.player_a_name,
                ctx.player_b_name,
                ctx.surface,
                ctx.match_date,
                float(model_probability_player_a),
            ],
        )
    except Exception:
        return


def _build_comparison_row(
    conn,
    ctx: MatchContext,
    response,
) -> ComparisonRow:
    """Assemble the four-line comparison header. The model probability
    always exists when this is called (we have an AgentResponse);
    surface-Elo and market are best-effort and rendered as '—' when
    missing."""
    market_row = _load_market_row(conn, ctx)
    elo_prob_a = _surface_elo_prob_a(conn, ctx)
    return ComparisonRow(
        player_a_name=ctx.player_a_name,
        player_b_name=ctx.player_b_name,
        model_prob_a=response.model_probability_player_a,
        surface_elo_prob_a=elo_prob_a,
        market_prob_a=market_row.get("median_implied_prob_a") if market_row else None,
        market_books_count=market_row.get("books_count") if market_row else None,
        market_source=market_row.get("source") if market_row else None,
        market_fetched_at=market_row.get("fetched_at") if market_row else None,
        pinnacle_prob_a=market_row.get("pinnacle_implied_prob_a") if market_row else None,
        median_odds_a=market_row.get("median_odds_a") if market_row else None,
        median_odds_b=market_row.get("median_odds_b") if market_row else None,
        pinnacle_odds_a=market_row.get("pinnacle_odds_a") if market_row else None,
        pinnacle_odds_b=market_row.get("pinnacle_odds_b") if market_row else None,
    )


def render_prediction_page(conn, ctx: MatchContext) -> None:
    """Compose all blocks for a given match context. Phase 6.2 reorders
    the stack so the signal comparison + news context appear together
    at the top (the most analyst-useful framing), with deterministic
    detail blocks below. Called by both the Prediction page (scheduled
    fixture) and the Custom prediction page (free-form entry)."""
    _render_match_time(conn, ctx)
    try:
        player_a_id, player_b_id = _resolve_player_ids(conn, ctx)
    except PlayerResolutionError as exc:
        st.error(str(exc))
        return

    # Phase 6.2: if a previous run for this fixture already produced an
    # AgentResponse this session, reuse it. Returning to the page (back
    # button, sidebar navigation) costs zero LLM tokens and renders
    # instantly. A fresh run is only forced by clicking "Re-run agent".
    session_key = _session_key_for_ctx(ctx)
    response = st.session_state.get(session_key)
    if response is None:
        response = _run_agent(conn, ctx)
        if response is not None:
            st.session_state[session_key] = response
            _log_prediction(conn, ctx, response.model_probability_player_a)

    if response is not None:
        prediction_card(response, ctx)
        st.divider()
        comparison = _build_comparison_row(conn, ctx, response)
        signal_comparison_block(comparison)
        if comparison.market_prob_a is not None:
            diff_pp = abs(comparison.model_prob_a - comparison.market_prob_a) * 100
            if diff_pp > COMPARISON_DIFF_THRESHOLD_PP:
                reasons = compute_reasons(
                    conn,
                    player_a_id=player_a_id,
                    player_b_id=player_b_id,
                    player_a_name=ctx.player_a_name,
                    player_b_name=ctx.player_b_name,
                    surface=ctx.surface,
                    as_of_date=ctx.match_date,
                    model_prob_a=comparison.model_prob_a,
                    market_prob_a=comparison.market_prob_a,
                    surface_elo_prob_a=comparison.surface_elo_prob_a,
                )
                why_model_differs_block(reasons)
        st.divider()
        news_block(response)
        st.divider()

    with st.spinner("Loading head-to-head…"):
        render_h2h_for_context(conn, ctx, player_a_id, player_b_id)
    st.divider()
    with st.spinner("Loading surface Elo…"):
        render_surface_elo_for_context(conn, ctx)
    st.divider()
    with st.spinner("Loading recent form for both players…"):
        render_recent_form_for_context(conn, ctx, player_a_id, player_b_id)


back_to_home_button(key="prediction_back_to_home")
st.title("Match dashboard")
try:
    conn = get_connection()
except DuckDBLockError as exc:
    st.error(str(exc))
    st.stop()
stale_data_banner(conn)

match_id = _resolve_match_id()
if not match_id:
    st.info(
        "Open this page from the **Home** tab by clicking a fixture, "
        "or use the **Custom prediction** tab to enter a match by hand."
    )
else:
    try:
        ctx = load_context_from_match_id(conn, match_id)
    except ContextBuildError as exc:
        st.error(str(exc))
    else:
        render_prediction_page(conn, ctx)
