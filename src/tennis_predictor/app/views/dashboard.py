"""Dashboard — calibration plots, recent traces, cost monitor.

Phase 6.2 added a top-of-page **scoreboard** showing the 20 most recent
predictions side-by-side with the market line that was visible at the
time of the prediction. The point: track record visible at a glance,
not buried behind average-Brier metrics that hide tail failures.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from tennis_predictor.app.db import DuckDBLockError, get_connection
from tennis_predictor.app.widgets import (
    back_to_home_button,
    cost_monitor_block,
    odds_api_quota_block,
    recent_predictions_scoreboard,
    stale_data_banner,
)
from tennis_predictor.config import MODELS_DIR

back_to_home_button(key="dashboard_back_to_home")
st.title("Model evaluation")
try:
    conn = get_connection()
except DuckDBLockError as exc:
    st.error(str(exc))
    st.stop()
stale_data_banner(conn)

# ---------------------------------------------------------------------------
# Phase 6.2: track-record scoreboard + odds-api quota
# ---------------------------------------------------------------------------

st.subheader("Recent predictions vs market")
recent_predictions_scoreboard(conn)

st.subheader("Pre-match odds quota")
odds_api_quota_block(conn)

# ---------------------------------------------------------------------------
# Cost monitor
# ---------------------------------------------------------------------------

st.subheader("Cost")
cost_monitor_block(conn)

# ---------------------------------------------------------------------------
# Calibration plots
# ---------------------------------------------------------------------------

st.subheader("Calibration")
st.caption(
    "LightGBM (post-calibration) overlaid with the market closing price. "
    "Diagonal = perfect calibration."
)

plot_cols = st.columns(2)
for col, tour in zip(plot_cols, ("ATP", "WTA"), strict=True):
    plot_path: Path = MODELS_DIR / tour / "lightgbm" / "latest" / "calibration_plot.png"
    with col:
        st.markdown(f"**{tour}**")
        if plot_path.exists():
            st.image(str(plot_path), use_container_width=True)
        else:
            st.info(
                f"Calibration plot unavailable for {tour}. "
                "Run `uv run python scripts/train_models.py` to generate it."
            )

# ---------------------------------------------------------------------------
# Recent traces
# ---------------------------------------------------------------------------

st.subheader("Recent LLM calls")
st.caption("Most recent 50 rows from `llm_traces`. One row per `LLMClient.acall`.")

traces_df = conn.execute(
    """
    SELECT trace_id, ts, model, tokens_in, tokens_out, cache_read_tokens,
           cache_creation_tokens, web_search_count, fetch_url_count,
           estimated_cost_usd, latency_ms, error
    FROM llm_traces
    ORDER BY trace_id DESC
    LIMIT 50
    """
).fetchdf()

if traces_df.empty:
    st.info("No LLM calls recorded yet. Run a prediction to populate this table.")
else:
    st.dataframe(traces_df, use_container_width=True, hide_index=True)
