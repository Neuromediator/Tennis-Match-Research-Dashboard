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

import streamlit as st

from tennis_predictor.app.db import DuckDBLockError, get_connection
from tennis_predictor.app.widgets import cost_footer, freshness_indicator

st.set_page_config(
    page_title="Tennis match research dashboard",
    layout="wide",
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


_sidebar()
navigation.run()
_footer()
