"""Home page: upcoming fixtures from `scheduled_matches`.

Sorted by scheduled start, grouped by tournament. Each row carries a
"Predict" button that stashes the `scheduled_match_id` into session state
and switches to the prediction page. The prediction page reads it back
on entry (with a fallback to `st.query_params['match_id']`)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import streamlit as st

from tennis_predictor.app.context import infer_tournament_level
from tennis_predictor.app.db import DuckDBLockError, get_connection
from tennis_predictor.app.widgets import (
    query_last_hot_refresh,
    stale_data_banner,
)

PENDING_MATCH_KEY = "pending_match_id"
# Path is resolved relative to the entry-point script (`app/main.py`), the
# same root `st.Page(...)` declarations use.
PREDICTION_PAGE_PATH = "views/prediction.py"


st.title("Upcoming matches")
try:
    conn = get_connection()
except DuckDBLockError as exc:
    st.error(str(exc))
    st.stop()
stale_data_banner(conn)

# Show "last refresh" age only — no manual trigger. The daily refresh
# runs in-process via APScheduler at ~05:00 UTC (plus catch-up-on-wake;
# see app/scheduler.py). Letting users click a Refresh button from the
# public URL exposes us to budget-draining click spam against the
# 500/month matchstat free tier, so the path is removed entirely.
last_refresh = query_last_hot_refresh(conn)
if last_refresh is not None:
    age = datetime.now(UTC) - last_refresh
    mins = int(age.total_seconds() // 60)
    age_label = f"{mins}m" if mins < 60 else f"{mins // 60}h {mins % 60}m"
    st.caption(f"Last fixtures refresh: {age_label} ago (auto, daily ~05:00 UTC).")
else:
    st.caption("No matchstat refresh on record yet.")

now = datetime.now(UTC)
today_utc = now.date()
horizon_date = today_utc + timedelta(days=7)
# Phase 6.2 follow-up: filter is now DATE-based, not time-based.
# Reason: matchstat returns `T12:00:00Z` for most Slam outside-court
# matches as a day-level placeholder (verified via /fixtures/tournament
# probe — see prompts.py rationale). The previous time-based 5h
# lookback dropped a match scheduled for evening as soon as 16:00 local
# rolled past, because its stored `scheduled_start_utc` was the
# misleading 09:00 UTC placeholder. Switching to DATE comparison keeps
# every match for "today" visible all day, regardless of when matchstat
# claims it starts.

raw_rows = conn.execute(
    """
    SELECT scheduled_match_id, tour, tournament_name, tournament_tier,
           round_name, surface, player1_name, player2_name,
           scheduled_start_utc
    FROM scheduled_matches
    WHERE scheduled_start_utc IS NULL
       OR DATE(scheduled_start_utc) BETWEEN ? AND ?
    ORDER BY scheduled_start_utc NULLS LAST, tournament_name, round_name
    """,
    [today_utc, horizon_date],
).fetchall()

# Drop fixtures whose tournament we can't map onto a model
# `TournamentLevel` (Challengers, ITF, M15/W100, etc.). `infer_tournament_level`
# also rescues active Slams whose tier got dropped from the matchstat
# calendar (e.g., French Open after R1 starts) by falling back to a name match.
rows = [r for r in raw_rows if infer_tournament_level(r[3], r[2], r[1]) is not None]

if not rows:
    st.info(
        "No matches scheduled in the next 7 days. Try the **Custom prediction** "
        "page from the sidebar to build a what-if fixture by hand."
    )
else:
    # Tour filter — defaults to "All" so the page still surfaces both
    # circuits at once. A typical week runs 2 ATP + 3 WTA tournaments
    # concurrently; without sectioning the same tournament name appears
    # twice (e.g. ATP and WTA French Open) and rows interleave by start
    # time across tours, which is hard to scan.
    available_tours = sorted({r[1] for r in rows if r[1]})
    tour_filter_options = ["All", *available_tours]
    selected_tour = st.radio(
        "Tour",
        options=tour_filter_options,
        horizontal=True,
        key="home_tour_filter",
    )

    visible_rows = [r for r in rows if r[1] == selected_tour] if selected_tour != "All" else rows

    # Group: tour → tournament_name → list[row]. Both keys preserve
    # insertion order (Python 3.7+ dict semantics) and the source
    # SELECT was ordered by `scheduled_start_utc`, so within each
    # bucket fixtures are still chronological.
    by_tour: dict[str, dict[str, list[tuple]]] = {}
    for row in visible_rows:
        tour_key = row[1] or "(unknown tour)"
        tournament_key = row[2] or "(unknown tournament)"
        by_tour.setdefault(tour_key, {}).setdefault(tournament_key, []).append(row)

    # Render ATP before WTA so the section order is predictable on
    # mixed days; falls back to sort order for anything else.
    tour_render_order = sorted(
        by_tour.keys(),
        key=lambda t: (0 if t == "ATP" else 1 if t == "WTA" else 2, t),
    )

    for tour_key in tour_render_order:
        tournaments = by_tour[tour_key]
        st.markdown(f"## {tour_key}")
        for tournament_name, group in tournaments.items():
            tier_label = group[0][3] or ""
            st.markdown(f"### {tournament_name}")
            if tier_label:
                st.caption(tier_label)

            for (
                scheduled_match_id,
                _tour,
                _tournament_name,
                _tier,
                round_name,
                surface,
                player1_name,
                player2_name,
                scheduled_start_utc,
            ) in group:
                # Button column kept wide enough that "Predict" never
                # wraps to two lines at mid-width viewports (~1920px with
                # the sidebar open); a 1/10 share was too narrow there.
                cols = st.columns([4, 2, 3, 2])
                cols[0].write(f"**{player1_name}** vs **{player2_name}**")
                cols[1].write(round_name or "—")
                # Phase 6.2 follow-up: render DATE only, no start time.
                # matchstat's start-time payload is unreliable for most
                # Slam outside-court matches (returns `T12:00:00Z` as a
                # day-level default for everything not on a featured
                # court). Showing 17 matches all at "11:00 CEST"
                # misleads more than it informs, and showing nothing
                # at all gives the user a clean "this match is today
                # / tomorrow / ..." signal.
                date_label = (
                    scheduled_start_utc.strftime("%a, %b %-d")
                    if scheduled_start_utc is not None
                    else "TBD"
                )
                cols[2].write(f"{surface or '—'} · {date_label}")
                if cols[3].button("Predict", key=f"predict::{scheduled_match_id}"):
                    st.session_state[PENDING_MATCH_KEY] = scheduled_match_id
                    st.query_params["match_id"] = scheduled_match_id
                    st.switch_page(PREDICTION_PAGE_PATH)
