"""Custom prediction page — minimal 3-input flow (Phase 6.1).

Reduced inputs: tour + player A (autocomplete) + player B (autocomplete) +
surface. `tournament_level` and `best_of` are inferred from the surface
(non-Slam ATP defaults to ATP250 / best_of=3; WTA → WTA250 / 3).
`match_date` is implicitly today (UTC). The output stack matches the
Prediction page exactly — same model card, H2H, surface Elo, recent
form, and news block.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast, get_args

import streamlit as st

from tennis_predictor.app.context import (
    ContextBuildError,
    load_context_from_freeform,
)
from tennis_predictor.app.db import DuckDBLockError, get_connection
from tennis_predictor.app.views.prediction import render_prediction_page
from tennis_predictor.app.widgets import (
    back_to_home_button,
    player_autocomplete,
    stale_data_banner,
)
from tennis_predictor.features.schema import Surface, TournamentLevel
from tennis_predictor.llm.tools.schemas import Tour

# Surface → tournament_level default. The intent is "what tournament
# class is most likely for a generic match on this surface?":
# clay/hard/IHard/grass → 250 series (the busiest tier). Users with a
# specific tournament in mind can still pin the level via the
# `?level=Slam` query param (see the radio at the bottom of the form).
_SURFACE_TO_DEFAULT_LEVEL: dict[tuple[Tour, Surface], TournamentLevel] = {
    ("ATP", "Clay"): "ATP250",
    ("ATP", "Hard"): "ATP250",
    ("ATP", "IHard"): "ATP250",
    ("ATP", "Grass"): "ATP250",
    ("WTA", "Clay"): "WTA250",
    ("WTA", "Hard"): "WTA250",
    ("WTA", "IHard"): "WTA250",
    ("WTA", "Grass"): "WTA250",
}


back_to_home_button(key="custom_back_to_home")
st.title("Custom match (any two players + surface)")
try:
    conn = get_connection()
except DuckDBLockError as exc:
    st.error(str(exc))
    st.stop()
stale_data_banner(conn)

st.write(
    "Enter a hypothetical match. Pick two players from the dropdowns "
    "(start typing a surname to filter). All other context defaults to "
    "today, generic 250-series tournament, best-of-3."
)

tour = cast(Tour, st.radio("Tour", options=list(get_args(Tour)), horizontal=True))

# Autocomplete lists scope to the chosen tour — switching ATP↔WTA
# resets the player selection (Streamlit keys include `tour` so the
# selectbox re-renders).
col_a, col_b = st.columns(2)
with col_a:
    player_a_id, player_a_display = player_autocomplete(
        conn, tour, key=f"player_a_{tour}", label="Player A"
    )
with col_b:
    player_b_id, player_b_display = player_autocomplete(
        conn, tour, key=f"player_b_{tour}", label="Player B"
    )

surface = cast(
    Surface,
    st.selectbox("Surface", options=list(get_args(Surface)), index=0),
)

with st.expander("Advanced (tournament level / best of)"):
    level_options = list(get_args(TournamentLevel))
    default_level = _SURFACE_TO_DEFAULT_LEVEL.get((tour, surface), level_options[0])
    tournament_level = cast(
        TournamentLevel,
        st.selectbox(
            "Tournament level",
            options=level_options,
            index=level_options.index(default_level),
        ),
    )
    best_of_choice = st.radio(
        "Best of",
        options=("Auto", "3", "5"),
        horizontal=True,
    )


def _resolve_full_name(canonical_id: str) -> str | None:
    # Prefer the seeded `full_name` alias; fall back to "first last" when it
    # is missing (some ranking-overlay player rows carry name_first/name_last
    # but a NULL full_name, which previously failed resolution outright). The
    # agent resolves the resulting name via AliasIndex downstream either way.
    row = conn.execute(
        """
        SELECT COALESCE(
                   NULLIF(TRIM(full_name), ''),
                   NULLIF(TRIM(COALESCE(name_first, '') || ' ' || COALESCE(name_last, '')), '')
               )
        FROM players WHERE player_id = ?
        """,
        [canonical_id],
    ).fetchone()
    return row[0] if row and row[0] else None


if st.button("Predict", type="primary", use_container_width=True):
    if player_a_id is None or player_b_id is None:
        st.error("Pick both players from the dropdowns.")
    elif player_a_id == player_b_id:
        st.error("Player A and Player B must be different.")
    else:
        # The agent's DB tools resolve via AliasIndex.lookup(name), which
        # matches against `players.full_name` (a seeded alias). The
        # autocomplete display "Surname F. (CTRY, #rank)" is NOT an
        # alias, so we feed full_name to the agent instead.
        full_name_a = _resolve_full_name(player_a_id)
        full_name_b = _resolve_full_name(player_b_id)
        if not full_name_a or not full_name_b:
            st.error("Could not resolve full name for one of the players.")
        else:
            best_of_value = None if best_of_choice == "Auto" else int(best_of_choice)
            match_date = datetime.now(UTC).date()
            try:
                ctx = load_context_from_freeform(
                    tour=tour,
                    player_a_name=full_name_a,
                    player_b_name=full_name_b,
                    surface=surface,
                    tournament_level=tournament_level,
                    match_date=match_date,
                    best_of=cast("int | None", best_of_value),  # type: ignore[arg-type]
                    tournament_name=None,
                )
            except ContextBuildError as exc:
                st.error(str(exc))
            else:
                render_prediction_page(conn, ctx)
