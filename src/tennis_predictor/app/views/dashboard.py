"""Dashboard — calibration plots, recent traces, cost monitor.

Phase 6.2 added a top-of-page **scoreboard** showing the 20 most recent
predictions side-by-side with the market line that was visible at the
time of the prediction. The point: track record visible at a glance,
not buried behind average-Brier metrics that hide tail failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from tennis_predictor.app.db import DuckDBLockError, get_connection
from tennis_predictor.app.widgets import (
    back_to_home_button,
    cost_monitor_block,
    matchstat_quota_block,
    odds_api_quota_block,
    recent_predictions_scoreboard,
    stale_data_banner,
)
from tennis_predictor.config import MODELS_DIR


def _calibration_metrics(tour: str) -> tuple[float | None, float | None, int]:
    """Return `(all_folds_brier, last5_weighted_brier, n_folds)` from the
    LightGBM `latest/metadata.json` walk-forward block. None on missing /
    malformed metadata. Reads from disk on every render so a retrain
    that flips the `latest` symlink shows up immediately."""
    meta_path = MODELS_DIR / tour / "lightgbm" / "latest" / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
        wf = meta.get("walk_forward") or {}
        agg = (wf.get("metrics_post_calibration_aggregate") or {}).get("brier")
        per_fold = wf.get("per_fold") or []
        last5 = per_fold[-5:]
        if last5:
            total = sum(f["n_validate"] * f["metrics_post_calibration"]["brier"] for f in last5)
            n = sum(f["n_validate"] for f in last5)
            last5_brier = total / n if n else None
        else:
            last5_brier = None
        return agg, last5_brier, len(per_fold)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ZeroDivisionError):
        return None, None, 0


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

st.subheader("External API usage (month-to-date)")
matchstat_quota_block(conn)
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
            all_brier, last5_brier, n_folds = _calibration_metrics(tour)
            if all_brier is not None and last5_brier is not None:
                st.caption(
                    f"Post-calibration Brier — all {n_folds} folds: "
                    f"**{all_brier:.4f}** · last 5 folds (sample-weighted): "
                    f"**{last5_brier:.4f}** · market reference ≈ 0.20."
                )
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
    st.dataframe(
        traces_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "trace_id": st.column_config.NumberColumn("ID", width="small"),
            "ts": st.column_config.DatetimeColumn("Time", width="small"),
            "model": st.column_config.TextColumn("Model", width="medium"),
            "tokens_in": st.column_config.NumberColumn("Tok in", width="small"),
            "tokens_out": st.column_config.NumberColumn("Tok out", width="small"),
            "cache_read_tokens": st.column_config.NumberColumn("Cache read", width="small"),
            "cache_creation_tokens": st.column_config.NumberColumn("Cache write", width="small"),
            "web_search_count": st.column_config.NumberColumn("Web", width="small"),
            "fetch_url_count": st.column_config.NumberColumn("Fetch", width="small"),
            "estimated_cost_usd": st.column_config.NumberColumn("$", format="$%.4f", width="small"),
            "latency_ms": st.column_config.NumberColumn("ms", width="small"),
            "error": st.column_config.TextColumn("Error", width="medium"),
        },
    )
