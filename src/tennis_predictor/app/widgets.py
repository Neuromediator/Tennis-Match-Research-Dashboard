"""Shared Streamlit widgets — cost monitor, freshness banner, prediction card.

Phase 6.1 layout shift: the prediction page renders a stack of
deterministic blocks (model probability / H2H / surface Elo / recent
form) plus a final LLM-discovered news block. Most of those blocks live
as discrete widgets in this file so both `views/prediction.py` (named
fixture) and `views/custom.py` (free-form input) can share them.

The pure-Python SQL helpers (`query_cost_summary`, `query_last_hot_refresh`,
`is_data_stale`, `query_player_autocomplete_options`,
`format_match_time_for_display`) are unit-tested directly. The
Streamlit-rendering functions are thin shells that compose `st.metric` /
`st.warning` / `st.markdown` / `st.selectbox` calls; covering those in
tests would require the experimental Streamlit `AppTest` API and is not
worth the brittleness — same call we made in Phase 6.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

import duckdb
import streamlit as st

from tennis_predictor.data.matchstat_live import MatchstatBudgetExceeded
from tennis_predictor.data.recent_form_live import (
    fetch_h2h_summary,
    fetch_recent_n_matches,
)
from tennis_predictor.features.schema import Surface, TournamentLevel
from tennis_predictor.llm.agent import (
    AgentError,
    BudgetExceededError,
    TennisAgent,
)
from tennis_predictor.llm.cost import cache_hit_rate
from tennis_predictor.llm.tools.db_tools import get_surface_elo
from tennis_predictor.llm.tools.schemas import (
    GetSurfaceEloInput,
    H2HSummary,
    MatchContext,
    ModelUnavailableError,
    PlayerResolutionError,
    RecentFormPayload,
    SurfaceEloSummary,
    TavilyError,
    Tour,
)
from tennis_predictor.llm.tools.submit import AgentResponse

STALE_THRESHOLD_HOURS: float = 24.0


# ---------------------------------------------------------------------------
# Cost monitor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostSummary:
    """One snapshot of LLM spend, aggregated from `llm_traces`. Counters are
    raw trace rows (one per `LLMClient.acall`) — a single user-visible
    prediction usually spans 2-3 rows, so we label this consistently as
    "calls" in the dashboard rather than pretending it's a prediction count."""

    today_usd: float
    today_calls: int
    today_web_searches: int
    today_fetch_urls: int
    month_usd: float
    month_calls: int
    cache_hit_rate_24h: float


def query_cost_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime | None = None,
) -> CostSummary:
    """Aggregate `llm_traces` into a `CostSummary`. `now` is a seam for
    tests — production callers leave it None and pick up `datetime.now(UTC)`.

    Bucketing rules:
    * Today: `ts >= start of today (UTC)`.
    * Month: `ts >= start of the current month (UTC)`.
    * Cache hit rate: last 24h, using the same formula as `cache_hit_rate`.
    """
    now = now or datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = today_start.replace(day=1)
    day_window_start = now - timedelta(hours=24)

    today_row = conn.execute(
        """
        SELECT
            COALESCE(SUM(estimated_cost_usd), 0.0),
            COUNT(*),
            COALESCE(SUM(web_search_count), 0),
            COALESCE(SUM(fetch_url_count), 0)
        FROM llm_traces
        WHERE ts >= ?
        """,
        [today_start],
    ).fetchone()
    month_row = conn.execute(
        """
        SELECT
            COALESCE(SUM(estimated_cost_usd), 0.0),
            COUNT(*)
        FROM llm_traces
        WHERE ts >= ?
        """,
        [month_start],
    ).fetchone()
    cache_row = conn.execute(
        """
        SELECT
            COALESCE(SUM(tokens_in), 0),
            COALESCE(SUM(cache_read_tokens), 0),
            COALESCE(SUM(cache_creation_tokens), 0)
        FROM llm_traces
        WHERE ts >= ?
        """,
        [day_window_start],
    ).fetchone()

    if today_row is None or month_row is None or cache_row is None:
        return CostSummary(0.0, 0, 0, 0, 0.0, 0, 0.0)

    rate = cache_hit_rate(
        tokens_in=int(cache_row[0]),
        cache_read_tokens=int(cache_row[1]),
        cache_creation_tokens=int(cache_row[2]),
    )
    return CostSummary(
        today_usd=float(today_row[0]),
        today_calls=int(today_row[1]),
        today_web_searches=int(today_row[2]),
        today_fetch_urls=int(today_row[3]),
        month_usd=float(month_row[0]),
        month_calls=int(month_row[1]),
        cache_hit_rate_24h=rate,
    )


def cost_monitor_block(conn: duckdb.DuckDBPyConnection) -> None:
    """Render four `st.metric` cards summarising today's + month's spend."""
    summary = query_cost_summary(conn)
    cols = st.columns(4)
    cols[0].metric(
        "Today (UTC)",
        f"${summary.today_usd:.2f}",
        help=(
            f"{summary.today_calls} LLM calls · "
            f"{summary.today_web_searches} web searches · "
            f"{summary.today_fetch_urls} fetches"
        ),
    )
    cols[1].metric(
        "Month to date",
        f"${summary.month_usd:.2f}",
        help=f"{summary.month_calls} LLM calls",
    )
    cols[2].metric(
        "Cache hit rate (24h)",
        f"{summary.cache_hit_rate_24h * 100:.0f}%",
        help=(
            "Fraction of INPUT TOKENS (not requests) served from Anthropic's "
            "prompt cache over the last 24h. Formula: cache_read / "
            "(tokens_in + cache_read + cache_creation). First call in a "
            "5-min TTL window scores 0 (cache miss + write); subsequent "
            "calls score high. Higher = lower input cost (cached tokens "
            "are billed at 10% of fresh-input rate)."
        ),
    )
    cols[3].metric(
        "Web searches today",
        f"{summary.today_web_searches}",
        help=f"plus {summary.today_fetch_urls} fetch_url calls",
    )


def cost_footer(conn: duckdb.DuckDBPyConnection) -> None:
    """One-line cost summary for the bottom of every page.

    Dollar signs are escaped (`\\$`) because `st.caption` renders markdown
    and Streamlit interprets `$...$` as a LaTeX math block — without the
    escape, `$0.49 · Month: $0.51` becomes a math block that swallows
    both `$` characters and the user sees a bare `0.49 · Month: 0.51`.
    """
    summary = query_cost_summary(conn)
    st.caption(
        f"Today (UTC): \\${summary.today_usd:.2f} · "
        f"Month: \\${summary.month_usd:.2f} · "
        f"Cache hit rate: {summary.cache_hit_rate_24h * 100:.0f}%"
    )


# ---------------------------------------------------------------------------
# Freshness / staleness
# ---------------------------------------------------------------------------


def query_last_hot_refresh(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str = "matchstat",
) -> datetime | None:
    """Most recent `finished_at` for a completed `ingestion_runs` row of
    the given source, or None if no completed run is on record.

    `status='partial'` counts as a successful refresh: the orchestrator
    marks a run "partial" when some sub-step failed but rows landed; the
    UI's "data is stale" signal should NOT fire just because a single
    page of fixtures errored. Only `failed` / `running` are excluded."""
    row = conn.execute(
        """
        SELECT max(finished_at)
        FROM ingestion_runs
        WHERE source = ? AND status IN ('success', 'partial')
        """,
        [source],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    last = row[0]
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return last


def is_data_stale(
    last_refresh: datetime | None,
    *,
    now: datetime,
    threshold_hours: float = STALE_THRESHOLD_HOURS,
) -> bool:
    """True when `last_refresh` is missing or older than `threshold_hours`.

    Missing data IS stale: if `ingestion_runs` has no successful row we
    treat that as never-refreshed, which deserves the same visible warning."""
    if last_refresh is None:
        return True
    if last_refresh.tzinfo is None:
        last_refresh = last_refresh.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return (now - last_refresh) > timedelta(hours=threshold_hours)


def stale_data_banner(conn: duckdb.DuckDBPyConnection) -> None:
    """Show an `st.warning` banner when hot-API data is >24h stale."""
    last = query_last_hot_refresh(conn)
    now = datetime.now(UTC)
    if not is_data_stale(last, now=now):
        return
    if last is None:
        st.warning(
            "Hot-API data has never been refreshed in this database. "
            "Run `uv run python scripts/refresh_hot.py` before predicting "
            "live fixtures."
        )
        return
    age = now - last
    hours = age.total_seconds() / 3600
    st.warning(
        f"Data is {hours:.0f} hours stale — last successful refresh: "
        f"{last.strftime('%Y-%m-%d %H:%M UTC')}."
    )


def back_to_home_button(*, key: str = "back_to_home") -> None:
    """Render "← Back to home" at the top of a non-Home page.

    Phase 6.2: the browser back button on Streamlit re-runs the previous
    page from scratch, which re-triggers `agent.predict()` for free.
    `st.session_state[f"prediction::{match_id}"]` already neutralises
    the cost on revisit, but the user's stated mental model is "back
    from Dashboard → Home, not Prediction." This button is the
    in-app navigation primitive that behaves exactly that way."""
    if st.button("← Back to home", key=key):
        st.switch_page("views/home.py")


def freshness_indicator(conn: duckdb.DuckDBPyConnection) -> None:
    """Sidebar label: `Hot data: 4h ago` / `Hot data: never refreshed`."""
    last = query_last_hot_refresh(conn)
    if last is None:
        st.markdown("**Hot data:** never refreshed")
        return
    now = datetime.now(UTC)
    delta = now - last
    hours = delta.total_seconds() / 3600
    if hours < 1:
        label = f"{int(delta.total_seconds() // 60)}m ago"
    elif hours < 48:
        label = f"{hours:.0f}h ago"
    else:
        label = f"{int(hours // 24)}d ago"
    st.markdown(f"**Hot data:** {label}")


# ---------------------------------------------------------------------------
# Prediction card
# ---------------------------------------------------------------------------


def prediction_card(response: AgentResponse, ctx: MatchContext) -> None:
    """Phase 6.1: render the model-probability + tools-used header.

    The header alone — H2H, surface Elo, recent form, news now have
    their own widgets (`h2h_block`, `surface_elo_block`,
    `recent_form_table_two_column`, `news_block`) that callers compose
    explicitly in the order they want."""
    header = f"{ctx.player_a_name} vs {ctx.player_b_name} · {ctx.tour}"
    st.subheader(header)
    sub_parts = [ctx.surface, ctx.tournament_level, f"best of {ctx.best_of}"]
    if ctx.tournament_name:
        sub_parts.insert(0, ctx.tournament_name)
    st.caption(" · ".join(sub_parts) + f" · {ctx.match_date.isoformat()}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            f"P({ctx.player_a_name})",
            f"{response.model_probability_player_a:.1%}",
        )
        st.progress(min(max(response.model_probability_player_a, 0.0), 1.0))
    with col_b:
        st.metric(
            f"P({ctx.player_b_name})",
            f"{response.model_probability_player_b:.1%}",
        )
        st.progress(min(max(response.model_probability_player_b, 0.0), 1.0))

    if response.tools_used:
        st.caption("Tools used: " + ", ".join(response.tools_used))


# ---------------------------------------------------------------------------
# Phase 6.1: time-zone display
# ---------------------------------------------------------------------------

_DISPLAY_TZ_NAME: str = "Europe/Berlin"  # CEST / CET — common European broadcast TZ.
_DISPLAY_TZ: ZoneInfo = ZoneInfo(_DISPLAY_TZ_NAME)


def format_match_time_for_display(utc_dt: datetime | None) -> str:
    """Render a match start time honestly: local European time + UTC in
    parentheses. Returns "TBD" for None inputs (matchstat occasionally
    serves a fixture without a confirmed start time).

    The TZ is correctly labelled with the abbreviation `datetime.strftime`
    derives at the actual moment (CEST in summer, CET in winter), so the
    label is never wrong about itself — the Phase 6 bug was about
    labelling a non-UTC time as 'UTC', which this helper structurally
    avoids by never typing the string 'UTC' next to a CEST-converted
    value."""
    if utc_dt is None:
        return "TBD"
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=UTC)
    local = utc_dt.astimezone(_DISPLAY_TZ)
    utc_render = utc_dt.astimezone(UTC)
    tz_abbrev = local.strftime("%Z") or _DISPLAY_TZ_NAME
    weekday = local.strftime("%a")
    month_day = local.strftime("%b %d")
    local_time = local.strftime("%H:%M")
    utc_time = utc_render.strftime("%H:%M")
    return f"{weekday}, {month_day} — {local_time} {tz_abbrev} ({utc_time} UTC)"


# ---------------------------------------------------------------------------
# Phase 6.1: H2H block
# ---------------------------------------------------------------------------


def _format_odds(odds: float | None) -> str:
    return f"{odds:.2f}" if odds is not None else "—"


def h2h_block(h2h: H2HSummary) -> None:
    """Render the H2H detail card. Top line: overall record + per-surface
    breakdown. Below: expandable detailed list of every meeting with
    score, round, surface, and odds when present.

    `data_source` drives a small footnote so the user knows whether
    they're seeing live matchstat data or Sackmann cold-layer fallback."""
    st.markdown("### Head-to-head")
    if not h2h.matches:
        st.caption(
            f"{h2h.player_a_name} and {h2h.player_b_name} have not met before "
            f"(per {h2h.data_source})."
        )
        return

    surface_bits = " · ".join(f"{surf} {a}-{b}" for surf, (a, b) in sorted(h2h.by_surface.items()))
    st.markdown(
        f"**{h2h.player_a_name} {h2h.player_a_wins}-{h2h.player_b_wins} "
        f"{h2h.player_b_name}** overall ({surface_bits or 'surface unknown'})"
    )

    with st.expander(f"All {len(h2h.matches)} meeting(s)"):
        # Newest first for the human reader.
        for m in sorted(h2h.matches, key=lambda x: x.match_date, reverse=True):
            badge = f" [{m.completion_status}]" if m.completion_status != "W" else ""
            tournament = m.tournament_name or "(tournament unknown)"
            round_part = f" {m.round_name}" if m.round_name else ""
            surface_part = f" {m.surface}" if m.surface else ""
            score = m.score or "(score unknown)"
            winner = m.winner_name or m.winner_player_id or "?"
            odds_part = ""
            if m.odds_winner is not None or m.odds_loser is not None:
                odds_part = (
                    f"  · odds W {_format_odds(m.odds_winner)} / L {_format_odds(m.odds_loser)}"
                )
            st.markdown(
                f"- **{m.match_date.isoformat()}** · {tournament}{round_part}"
                f"{surface_part} · {winner} def. · {score}{badge}{odds_part}"
            )

    age = datetime.now(UTC) - h2h.fetched_at.replace(tzinfo=UTC)
    age_minutes = int(age.total_seconds() // 60)
    age_label = f"{age_minutes}m" if age_minutes < 60 else f"{age_minutes // 60}h"
    st.caption(f"Data source: {h2h.data_source} (fetched {age_label} ago)")


# ---------------------------------------------------------------------------
# Phase 6.2: signal comparison row (market / model / surface Elo)
# ---------------------------------------------------------------------------

# Threshold (probability points) above which the "why model differs"
# panel is rendered. 10pp matches the design doc; lower thresholds
# trigger too often on routine top-vs-middling matchups where the
# explanation isn't useful.
COMPARISON_DIFF_THRESHOLD_PP: float = 10.0


@dataclass(frozen=True)
class ComparisonRow:
    """Headline row for the Prediction page header.

    Each field is the win probability for `player_a` from one source;
    None means "unavailable" (no market row yet / surface Elo missing).
    `market_books_count` and `market_source` annotate the market line."""

    player_a_name: str
    player_b_name: str
    model_prob_a: float
    surface_elo_prob_a: float | None
    market_prob_a: float | None
    market_books_count: int | None
    market_source: str | None  # "the_odds_api" | "tavily"
    market_fetched_at: datetime | None
    pinnacle_prob_a: float | None
    median_odds_a: float | None
    median_odds_b: float | None
    pinnacle_odds_a: float | None
    pinnacle_odds_b: float | None


def _pp(a: float, b: float | None) -> str:
    """Format `(a - b) * 100` as a signed pp delta, or '—' when b is None."""
    if b is None:
        return "—"
    return f"{(a - b) * 100:+.1f}pp"


def _fmt_pct(p: float | None) -> str:
    return f"{p:.1%}" if p is not None else "—"


def signal_comparison_block(row: ComparisonRow) -> None:
    """Render the four-line comparison header. Phase 6.2 reframes the
    Prediction page around this widget: model probability is one signal
    among several, not the answer."""
    st.markdown("### Signal comparison")

    market_label = "Market"
    if row.market_books_count is not None and row.market_books_count > 0:
        market_label += f" ({row.market_books_count} books)"
    if row.market_source == "tavily":
        market_label += " · estimated from web search"

    market_pct = _fmt_pct(row.market_prob_a)
    market_b_pct = f"{(1 - row.market_prob_a):.1%}" if row.market_prob_a is not None else "—"
    model_pct = _fmt_pct(row.model_prob_a)
    model_b_pct = f"{1 - row.model_prob_a:.1%}"
    elo_pct = _fmt_pct(row.surface_elo_prob_a)
    elo_b_pct = f"{(1 - row.surface_elo_prob_a):.1%}" if row.surface_elo_prob_a is not None else "—"

    table_md = (
        f"| Source | P({row.player_a_name}) | P({row.player_b_name}) | Diff to market |\n"
        f"|---|---|---|---|\n"
        f"| {market_label} | **{market_pct}** | {market_b_pct} | — |\n"
        f"| Our model | **{model_pct}** | {model_b_pct} | {_pp(row.model_prob_a, row.market_prob_a)} |\n"
        f"| Surface Elo | {elo_pct} | {elo_b_pct} | "
        f"{_pp(row.surface_elo_prob_a, row.market_prob_a) if row.surface_elo_prob_a is not None else '—'} |\n"
    )
    st.markdown(table_md)

    sub_parts: list[str] = []
    if row.median_odds_a is not None and row.median_odds_b is not None:
        sub_parts.append(f"Median odds: A {row.median_odds_a:.2f} / B {row.median_odds_b:.2f}")
    if row.pinnacle_odds_a is not None and row.pinnacle_odds_b is not None:
        sub_parts.append(f"Pinnacle: A {row.pinnacle_odds_a:.2f} / B {row.pinnacle_odds_b:.2f}")
    if row.market_fetched_at is not None:
        age = datetime.now(UTC) - row.market_fetched_at.replace(tzinfo=UTC)
        age_h = age.total_seconds() / 3600
        age_label = f"{int(age.total_seconds() // 60)}m ago" if age_h < 1 else f"{age_h:.0f}h ago"
        sub_parts.append(f"fetched {age_label} via The Odds API")
    if sub_parts:
        st.caption(" · ".join(sub_parts))
    if row.market_prob_a is None:
        st.caption("Market: odds unavailable.")


def why_model_differs_block(reasons: list) -> None:
    """Render the deterministic "why model differs" explanation when the
    market/model gap exceeds the threshold. `reasons` is a list of
    `WhyDifferReason` from `tennis_predictor.app.why_differs`."""
    if not reasons:
        return
    st.markdown("#### Why model differs from market")
    for r in reasons:
        st.markdown(f"- **{r.headline}**")
        st.caption(r.detail)


# ---------------------------------------------------------------------------
# Phase 6.1: surface Elo block
# ---------------------------------------------------------------------------


def surface_elo_block(elo: SurfaceEloSummary) -> None:
    """Three-number compact card: each player's Elo, the diff, and the
    baseline win probability for player_a. Used right next to the model
    card so the user can see the gap between Elo-only and the LightGBM
    output at a glance."""
    st.markdown(f"### Surface Elo ({elo.surface})")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(elo.player_a_name, f"{elo.player_a_elo:.0f}")
    col_b.metric(elo.player_b_name, f"{elo.player_b_elo:.0f}")
    arrow = "→" if elo.diff_a_minus_b >= 0 else "←"
    col_c.metric(
        "Diff (A - B)",
        f"{elo.diff_a_minus_b:+.0f}",
        delta=f"baseline P(A) = {elo.baseline_prob_a:.1%}",
        delta_color="off",
    )
    if elo.elo_state_snapshot_date is not None:
        st.caption(
            f"Snapshot through {elo.elo_state_snapshot_date.isoformat()} · "
            f"{arrow} {elo.player_a_name if elo.diff_a_minus_b >= 0 else elo.player_b_name}"
        )


# ---------------------------------------------------------------------------
# Phase 6.1: recent form (two-column table)
# ---------------------------------------------------------------------------


def _render_recent_form_column(payload: RecentFormPayload) -> None:
    """Render one player's last-N matches into the current Streamlit column."""
    st.markdown(f"**{payload.player_name}**")
    if not payload.matches:
        st.caption("No completed matches found.")
        return
    for m in payload.matches:
        result_emoji = "✓" if m.result == "W" else "✗"
        badge = f" [{m.completion_status}]" if m.completion_status != "W" else ""
        score = m.score or "(score unknown)"
        round_part = f" {m.round_name}" if m.round_name else ""
        tournament = m.tournament_name or "(tournament unknown)"
        surface = m.surface or "?"
        st.markdown(
            f"{result_emoji} **{m.match_date.isoformat()}** · {tournament}{round_part}"
            f" · {surface} · {m.result} {score}{badge} · vs {m.opponent_name}"
        )


def recent_form_table_two_column(
    payload_a: RecentFormPayload,
    payload_b: RecentFormPayload,
) -> None:
    """Two side-by-side columns rendering both players' last N matches.

    Same content as `_render_recent_form_column` x 2; the wrapper exists
    so the prediction page can call one function instead of laying out
    columns inline."""
    st.markdown("### Recent form")
    col_a, col_b = st.columns(2)
    with col_a:
        _render_recent_form_column(payload_a)
    with col_b:
        _render_recent_form_column(payload_b)
    # Footer: data source + quota readout from EITHER payload (they're
    # the same source by construction since they share the conn).
    source = payload_a.data_source
    quota_used = payload_a.matchstat_quota_used
    quota_cap = payload_a.matchstat_quota_cap
    quota_part = ""
    if source == "matchstat" and quota_used is not None and quota_cap is not None:
        quota_part = f" · matchstat quota: {quota_used}/{quota_cap}"
    age = datetime.now(UTC) - payload_a.fetched_at.replace(tzinfo=UTC)
    age_minutes = int(age.total_seconds() // 60)
    age_label = f"{age_minutes}m" if age_minutes < 60 else f"{age_minutes // 60}h"
    st.caption(f"Data source: {source} (fetched {age_label} ago{quota_part})")
    if source == "sackmann":
        st.caption(
            "⚠ matchstat quota exhausted or external player ID unknown — "
            "falling back to Sackmann cold data (lag up to 7 days)."
        )


# ---------------------------------------------------------------------------
# Phase 6.1: news block
# ---------------------------------------------------------------------------


def news_block(response: AgentResponse) -> None:
    """Render the LLM-discovered news items.

    Four states from `response.news_lookup_status`:
    - `ok` — render the list with [source, date] tags and links.
    - `no_results` — render a one-line "nothing found" message.
    - `failed` — render an "unavailable" message.
    - `budget_exhausted` (Phase 7) — internal status set by
      `_cached_predict` when `DAILY_LLM_BUDGET` is reached; the
      LLM was deliberately skipped and the model probability above
      came from a direct `get_model_prediction` call.
    """
    st.markdown("### Recent news (last 32 days)")
    if response.news_lookup_status == "budget_exhausted":
        st.caption(
            "⚠ Daily LLM news-lookup budget reached. The news block is paused "
            "until 00:00 UTC. Model probability, market, surface-Elo, H2H and "
            "recent-form blocks above are unaffected."
        )
        return
    if response.news_lookup_status == "failed":
        st.caption("⚠ News lookup unavailable. Prediction is based on model + DB context only.")
        return
    if response.news_lookup_status == "no_results" or not response.news_items:
        st.caption("No notable news in the last 32 days for either player.")
        return
    for item in response.news_items:
        date_label = item.published_date or "date unknown"
        subject_label = {
            "player_a": item.player_subject,
            "player_b": item.player_subject,
            "both": "both",
        }[item.player_subject]
        category_label = item.category
        st.markdown(
            f"- **[{item.source_domain}, {date_label}]** ({category_label}, "
            f"{subject_label}) [{item.title}]({item.url})"
        )
        if item.snippet:
            st.caption(item.snippet)


# ---------------------------------------------------------------------------
# Phase 6.1: surface Elo + H2H + recent form orchestrator helpers
# (these wrap the matchstat-fetcher calls + error handling so views don't
# need to reproduce the same try/except dance four times each.)
# ---------------------------------------------------------------------------


def render_h2h_for_context(
    conn: duckdb.DuckDBPyConnection,
    ctx: MatchContext,
    player_a_id: str,
    player_b_id: str,
) -> None:
    """Fetch + render the H2H block, mapping fetcher exceptions to a
    visible UI message (rather than a traceback)."""
    try:
        summary = fetch_h2h_summary(
            conn,
            ctx.tour,
            player_a_id,
            player_b_id,
            ctx.player_a_name,
            ctx.player_b_name,
            ctx.match_date,
        )
    except MatchstatBudgetExceeded as exc:
        st.warning(f"matchstat quota exhausted — H2H lookup unavailable. ({exc})")
        return
    h2h_block(summary)


def render_surface_elo_for_context(
    conn: duckdb.DuckDBPyConnection,
    ctx: MatchContext,
) -> None:
    """Fetch + render the surface-Elo block."""
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
    except PlayerResolutionError as exc:
        st.warning(f"Surface Elo lookup failed: {exc}")
        return
    surface_elo_block(summary)


def render_recent_form_for_context(
    conn: duckdb.DuckDBPyConnection,
    ctx: MatchContext,
    player_a_id: str,
    player_b_id: str,
    n: int = 8,
) -> None:
    """Fetch + render the two-column recent-form block."""
    payload_a = fetch_recent_n_matches(
        conn, ctx.tour, player_a_id, ctx.player_a_name, ctx.match_date, n=n
    )
    payload_b = fetch_recent_n_matches(
        conn, ctx.tour, player_b_id, ctx.player_b_name, ctx.match_date, n=n
    )
    recent_form_table_two_column(payload_a, payload_b)


# ---------------------------------------------------------------------------
# Phase 6.1: player autocomplete (used by Custom prediction page)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerOption:
    canonical_id: str
    display: str  # "Djokovic N. (SRB, #3)"


def query_player_autocomplete_options(
    conn: duckdb.DuckDBPyConnection, tour: Tour
) -> list[PlayerOption]:
    """Return one `PlayerOption` per player in the given tour, sorted by
    most-recent ranking ascending (so #1 is at the top of the
    dropdown). Players never ranked sort to the bottom.

    `display` format: "Surname F. (CTRY, #rank)" — same convention the
    Prediction page uses for compactness."""
    rows = conn.execute(
        """
        WITH latest_rank AS (
            SELECT player_id, rank
            FROM (
                SELECT player_id, rank,
                       ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY ranking_date DESC) AS rn
                FROM rankings
            )
            WHERE rn = 1
        )
        SELECT p.player_id, p.name_first, p.name_last, p.ioc, lr.rank
        FROM players p
        LEFT JOIN latest_rank lr ON lr.player_id = p.player_id
        WHERE p.tour = ?
        ORDER BY COALESCE(lr.rank, 9999) ASC, p.name_last ASC
        """,
        [tour],
    ).fetchall()
    out: list[PlayerOption] = []
    for player_id, first, last, ioc, rank in rows:
        first_initial = (first[:1] + ".") if first else ""
        rank_part = f"#{int(rank)}" if rank is not None else "#--"
        ioc_part = f", {ioc}" if ioc else ""
        display = f"{last or '?'} {first_initial} ({ioc_part.lstrip(', ')}{', ' if ioc_part else ''}{rank_part})".strip()
        # The bracket may end up like "(SRB, #3)" or "(#--)" depending on
        # whether IOC and rank are known; normalise the leading comma if
        # IOC is missing.
        display = display.replace("(, ", "(")
        out.append(PlayerOption(canonical_id=player_id, display=display))
    return out


def player_autocomplete(
    conn: duckdb.DuckDBPyConnection,
    tour: Tour,
    *,
    key: str,
    label: str = "Player",
    placeholder: str = "Start typing surname…",
) -> tuple[str | None, str | None]:
    """Streamlit `selectbox` populated with every player in the tour.

    Returns `(canonical_id, display)` for the selected option, or
    `(None, None)` if the user hasn't picked one yet.

    Client-side filtering is whatever Streamlit's selectbox provides
    natively — typing in the box narrows the visible options."""
    options = query_player_autocomplete_options(conn, tour)
    if not options:
        st.warning(f"No players found in DB for {tour} tour.")
        return None, None
    display_to_id = {opt.display: opt.canonical_id for opt in options}
    selected = st.selectbox(
        label,
        options=list(display_to_id.keys()),
        index=None,
        key=key,
        placeholder=placeholder,
    )
    if selected is None:
        return None, None
    return display_to_id[selected], selected


# ---------------------------------------------------------------------------
# Cached prediction runner — shared by the Prediction and Custom pages.
# ---------------------------------------------------------------------------


def _budget_exhausted_response(conn: duckdb.DuckDBPyConnection, ctx: MatchContext) -> AgentResponse:
    """Build a synthetic AgentResponse when the daily LLM budget is
    reached. Calls `get_model_prediction` directly (no LLM agent loop,
    no Tavily), returns with empty news and the internal
    `news_lookup_status='budget_exhausted'` marker so `news_block`
    can render the right message."""
    from tennis_predictor.llm.tools.model_tool import get_model_prediction
    from tennis_predictor.llm.tools.schemas import GetModelPredictionInput

    pred = get_model_prediction(
        conn,
        GetModelPredictionInput(
            player_a_name=ctx.player_a_name,
            player_b_name=ctx.player_b_name,
            tour=ctx.tour,
            surface=ctx.surface,
            tournament_level=ctx.tournament_level,
            best_of=ctx.best_of,
            match_date=ctx.match_date,
        ),
    )
    return AgentResponse(
        model_probability_player_a=pred.model_probability_player_a,
        model_probability_player_b=pred.model_probability_player_b,
        news_items=[],
        news_lookup_status="budget_exhausted",
        tools_used=["get_model_prediction"],
    )


@st.cache_data(ttl=300, show_spinner=False)
def _cached_predict(
    _conn: duckdb.DuckDBPyConnection,
    *,
    tour: Tour,
    player_a_name: str,
    player_b_name: str,
    surface: Surface,
    tournament_level: TournamentLevel,
    best_of: Literal[3, 5],
    match_date: date,
    tournament_name: str | None,
    scheduled_match_id: str | None,
) -> AgentResponse:
    """Three-layer cached predict.

    Layer 1 (this `@st.cache_data`): in-memory, per-Streamlit-process,
    5 min TTL — fast path inside a single visit.

    Layer 2 (DuckDB `prediction_cache`): cross-session, 24h TTL, shared
    by all visitors, survives Machine restarts. Only used for scheduled
    fixtures (`scheduled_match_id is not None`); Custom predictions skip
    Layer 2 and rely on Layer 1 only.

    Layer 3 (`DAILY_LLM_BUDGET` cap): before invoking the LLM agent,
    check `today_trace_count()` against the daily budget. If exhausted,
    skip the LLM and return a model-only AgentResponse (no news block,
    page still renders all deterministic blocks). The budget-exhausted
    response is NOT written to Layer 2 — the cap resets at 00:00 UTC
    and the next-day visitor should get a real news lookup."""
    from tennis_predictor.data import prediction_cache
    from tennis_predictor.llm import budget

    if scheduled_match_id is not None:
        hit = prediction_cache.get_cached(_conn, scheduled_match_id)
        if hit is not None:
            return hit

    ctx = MatchContext(
        tour=tour,
        player_a_name=player_a_name,
        player_b_name=player_b_name,
        surface=surface,
        tournament_level=tournament_level,
        best_of=best_of,
        match_date=match_date,
        tournament_name=tournament_name,
        scheduled_match_id=scheduled_match_id,
    )

    if budget.is_budget_exhausted(_conn):
        return _budget_exhausted_response(_conn, ctx)

    agent = TennisAgent(_conn)
    response = asyncio.run(agent.predict(ctx))

    if scheduled_match_id is not None:
        prediction_cache.store(_conn, scheduled_match_id, response)

    return response


def run_and_render_prediction(conn: duckdb.DuckDBPyConnection, ctx: MatchContext) -> None:
    """Run the cached agent call and render the result, mapping every
    documented failure surface (CLAUDE.md "LLM agent failure modes") to a
    user-friendly message instead of a Python traceback in the browser."""
    try:
        with st.spinner("Running model + LLM analyst… (~40-80s)"):
            response = _cached_predict(
                conn,
                tour=ctx.tour,
                player_a_name=ctx.player_a_name,
                player_b_name=ctx.player_b_name,
                surface=ctx.surface,
                tournament_level=ctx.tournament_level,
                best_of=ctx.best_of,
                match_date=ctx.match_date,
                tournament_name=ctx.tournament_name,
                scheduled_match_id=ctx.scheduled_match_id,
            )
    except ModelUnavailableError as exc:
        st.error(f"Model artifact not loaded — prediction cannot run. ({exc})")
        return
    except PlayerResolutionError as exc:
        st.error(f"Player resolution failed: {exc}")
        return
    except BudgetExceededError as exc:
        st.warning(f"Today's prediction budget reached. ({exc})")
        return
    except TavilyError as exc:
        st.warning(f"News lookup unavailable: {exc}. Prediction continues on DB context only.")
        return
    except AgentError as exc:
        st.error(f"Prediction service temporarily unavailable: {exc}")
        return

    prediction_card(response, ctx)


# ---------------------------------------------------------------------------
# Phase 6.2: Dashboard scoreboard + odds-api quota indicator
# ---------------------------------------------------------------------------


def recent_predictions_scoreboard(conn: duckdb.DuckDBPyConnection, *, limit: int = 20) -> None:
    """Render the last `limit` rows from `prediction_log` joined to
    `pre_match_odds` so the user sees the track record at a glance —
    side-by-side model probability vs market consensus, signed gap,
    and the timestamp.

    Rows without a market match show '—' in the market columns; we
    surface them anyway so the user can still see what they predicted
    (some fixtures publish odds only the day of play)."""
    from tennis_predictor.data.pre_match_odds import fixture_match_key

    rows = conn.execute(
        f"""
        SELECT log_id, ts, scheduled_match_id, tour, player_a_name, player_b_name,
               surface, match_date, model_probability_player_a
        FROM prediction_log
        ORDER BY log_id DESC
        LIMIT {int(limit)}
        """
    ).fetchall()

    if not rows:
        st.info(
            "No predictions logged yet. Run a prediction from the Home page; "
            "the scoreboard populates from `prediction_log`."
        )
        return

    output_rows: list[dict[str, Any]] = []
    for (
        _log_id,
        ts,
        _sm_id,
        tour,
        player_a,
        player_b,
        surface,
        match_date,
        model_prob_a,
    ) in rows:
        commence = (
            datetime(match_date.year, match_date.month, match_date.day, tzinfo=UTC)
            if match_date is not None
            else None
        )
        market_prob_a: float | None = None
        market_source: str | None = None
        if commence is not None:
            key = fixture_match_key(tour, player_a, player_b, commence)
            market_row = conn.execute(
                "SELECT median_implied_prob_a, source FROM pre_match_odds "
                "WHERE fixture_match_key = ?",
                [key],
            ).fetchone()
            if market_row is not None:
                market_prob_a = market_row[0]
                market_source = market_row[1]
        gap_pp = (
            f"{(float(model_prob_a) - market_prob_a) * 100:+.1f}pp"
            if market_prob_a is not None
            else "—"
        )
        output_rows.append(
            {
                "ts": ts.isoformat(sep=" ", timespec="minutes") if ts else "",
                "tour": tour,
                "match": f"{player_a} vs {player_b}",
                "surface": surface or "—",
                "model %": f"{float(model_prob_a):.1%}",
                "market %": f"{market_prob_a:.1%}" if market_prob_a is not None else "—",
                "gap (model - market)": gap_pp,
                "market source": market_source or "—",
            }
        )

    st.dataframe(
        output_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ts": st.column_config.TextColumn("Time", width="small"),
            "tour": st.column_config.TextColumn("Tour", width="small"),
            "match": st.column_config.TextColumn("Match", width="large"),
            "surface": st.column_config.TextColumn("Surface", width="small"),
            "model %": st.column_config.TextColumn("Model", width="small"),
            "market %": st.column_config.TextColumn("Market", width="small"),
            "gap (model - market)": st.column_config.TextColumn("Δ pp", width="small"),
            "market source": st.column_config.TextColumn("Src", width="small"),
        },
    )


def odds_api_quota_block(conn: duckdb.DuckDBPyConnection) -> None:
    """Render a single-row indicator showing The Odds API month-to-date
    quota usage."""
    from tennis_predictor.data.pre_match_odds import quota_status

    used, cap = quota_status(conn)
    pct = (used / cap) * 100 if cap else 0
    color = "🟢" if pct < 80 else ("🟡" if pct < 95 else "🔴")
    st.markdown(f"{color} **The Odds API** — {used}/{cap} credits this month (~{pct:.0f}% used)")
    st.caption(
        "Counts only billable `/sports/{key}/odds` calls (1 credit each at "
        "`regions=eu`). The `/sports` discovery call is free per The Odds "
        "API docs and not counted here. Daily refresh consumes ~4-6 "
        "credits; lazy refresh on Prediction-page load adds ~1 per cache "
        "miss. Expected total: 150-210/month."
    )


def query_matchstat_usage_month(
    conn: duckdb.DuckDBPyConnection, now: datetime | None = None
) -> tuple[int, int]:
    """Authoritative month-to-date matchstat API usage.

    Two disjoint sources count toward the matchstat 500/month free-tier
    cap and we have historically split them across two tables:

    - `ingestion_runs(source='matchstat').requests_used` — every hot
      refresh (`scripts/refresh_hot.py`) logs its total HTTP-call count
      here. This is the dominant source (~13-15 calls per refresh).
    - `matchstat_quota.requests_used` — incremented by the Phase 6.1
      live fetcher (`MatchstatLiveFetcher`) on per-prediction H2H or
      past-match calls that miss the 24h DuckDB cache.

    Neither source overlaps with the other; sum gives the real total
    against the 500/month cap. Returns `(used_total, cap)`."""
    moment = now or datetime.now(UTC)
    month_start = datetime(moment.year, moment.month, 1, tzinfo=UTC).replace(tzinfo=None)
    ingestion_sum = conn.execute(
        """
        SELECT COALESCE(SUM(requests_used), 0) FROM ingestion_runs
        WHERE source = 'matchstat' AND started_at >= ?
        """,
        [month_start],
    ).fetchone()
    ingestion = int(ingestion_sum[0]) if ingestion_sum else 0

    month_key = f"{moment.year:04d}-{moment.month:02d}"
    quota_row = conn.execute(
        "SELECT requests_used, cap FROM matchstat_quota WHERE month = ?", [month_key]
    ).fetchone()
    quota_used = int(quota_row[0]) if quota_row else 0
    cap = int(quota_row[1]) if quota_row else 500
    return ingestion + quota_used, cap


def matchstat_quota_block(conn: duckdb.DuckDBPyConnection) -> None:
    """Render a matchstat usage indicator that unifies the two count
    sources (hot-refresh `ingestion_runs` + per-prediction `matchstat_quota`)
    so the user sees ONE authoritative `M/500` number."""
    used, cap = query_matchstat_usage_month(conn)
    pct = (used / cap) * 100 if cap else 0
    color = "🟢" if pct < 80 else ("🟡" if pct < 95 else "🔴")
    st.markdown(
        f"{color} **matchstat (RapidAPI)** — {used}/{cap} requests this month (~{pct:.0f}% used)"
    )
    st.caption(
        "Sum of: hot-refresh calls logged in `ingestion_runs` (daily ~13-15 "
        "credits — calendar + per-tournament fixtures + rankings + Slam "
        "results) AND per-prediction H2H/past-matches calls logged in "
        "`matchstat_quota` (3 credits per fresh prediction, 0 on 24h cache "
        "hit). At ≥480/500 the live fetcher raises `MatchstatBudgetExceeded` "
        "and falls back to Sackmann cold data."
    )


__all__ = [
    "COMPARISON_DIFF_THRESHOLD_PP",
    "STALE_THRESHOLD_HOURS",
    "ComparisonRow",
    "CostSummary",
    "PlayerOption",
    "back_to_home_button",
    "cost_footer",
    "cost_monitor_block",
    "format_match_time_for_display",
    "freshness_indicator",
    "h2h_block",
    "is_data_stale",
    "matchstat_quota_block",
    "news_block",
    "odds_api_quota_block",
    "player_autocomplete",
    "prediction_card",
    "query_cost_summary",
    "query_last_hot_refresh",
    "query_matchstat_usage_month",
    "query_player_autocomplete_options",
    "recent_form_table_two_column",
    "recent_predictions_scoreboard",
    "render_h2h_for_context",
    "render_recent_form_for_context",
    "render_surface_elo_for_context",
    "run_and_render_prediction",
    "signal_comparison_block",
    "stale_data_banner",
    "surface_elo_block",
    "why_model_differs_block",
]
