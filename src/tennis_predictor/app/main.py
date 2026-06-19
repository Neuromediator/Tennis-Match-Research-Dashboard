"""Streamlit entry point. Run with:

    uv run streamlit run src/tennis_predictor/app/main.py

Defines the sidebar navigation (Home / Prediction / Custom / Dashboard),
the global cost footer, and a freshness indicator in the sidebar. Each
page module's top-level code is executed by Streamlit when the user
navigates to it; this file does NOT call into the pages directly.

Order matters: `st.navigation(...)` is declared BEFORE any DB-touching
sidebar widget renders. If `get_connection()` raised here, Streamlit
would never see the `st.Page` registrations and would fall back to its
legacy `pages/` directory auto-discovery — which is why the sidebar
DB calls are guarded with their own try/except.
"""

from __future__ import annotations

import logging

import streamlit as st

from tennis_predictor.app.db import DuckDBLockError, get_connection
from tennis_predictor.app.scheduler import get_scheduler, maybe_catch_up_refresh
from tennis_predictor.app.widgets import cost_footer, freshness_indicator

# Configure root logging so [scheduler] / [refresh_jobs] INFO messages
# land in `fly logs`. Streamlit otherwise leaves third-party loggers at
# the Python default (WARNING) and we lose the daily-refresh trail.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

st.set_page_config(
    page_title="Tennis match research dashboard",
    layout="wide",
)

# Reserve the vertical scrollbar gutter so it never toggles. Without this,
# a width-responsive element whose height changes with width (e.g. the
# calibration-plot images on Model evaluation) creates a scrollbar↔width↔
# height feedback loop that makes the page — and any wide table on it —
# visibly oscillate ("shake"). Stable gutter breaks the loop.
st.markdown(
    "<style>html, body, [data-testid='stAppViewContainer'], "
    "[data-testid='stMain'] { scrollbar-gutter: stable; }</style>",
    unsafe_allow_html=True,
)

PAGES = [
    st.Page(
        "views/home.py",
        title="Home",
        icon=":material/home:",
        default=True,
    ),
    st.Page(
        "views/prediction.py",
        title="Match dashboard",
        icon=":material/insights:",
    ),
    st.Page(
        "views/custom.py",
        title="Custom match",
        icon=":material/edit_note:",
    ),
    st.Page(
        "views/dashboard.py",
        title="Model evaluation",
        icon=":material/dashboard:",
    ),
]

# Declare navigation FIRST so a downstream DB error never disables the
# st.Page registrations (Streamlit would otherwise treat absence of a
# st.navigation call as "use the pages/ directory" and show raw file
# names in the sidebar).
navigation = st.navigation(PAGES)


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### Tennis match dashboard")
        st.caption(
            "Model + market + surface-Elo + LLM news context for upcoming "
            "ATP / WTA matches. The model is one signal, not the answer."
        )
        try:
            freshness_indicator(get_connection())
        except DuckDBLockError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.warning(f"Freshness indicator unavailable: {exc}")


def _footer() -> None:
    try:
        cost_footer(get_connection())
    except DuckDBLockError:
        # The page body already surfaced the same error; no need to repeat.
        return
    except Exception:
        return


# Start the daily-refresh scheduler if ENABLE_SCHEDULER is on. Cached
# via @st.cache_resource — fires exactly once per process, no-op on
# subsequent script reruns. Returns None when gated off (local dev).
_scheduler = get_scheduler()

# Catch-up-on-wake: on a host that sleeps when idle (HF Spaces), the
# 21:00 cron cannot fire while asleep, so the first visit after a wake
# triggers a background refresh if the hot data is stale. Runs at most
# once per process and never blocks the render. No-op when gated off.
maybe_catch_up_refresh(_scheduler)

_sidebar()
navigation.run()
_footer()
