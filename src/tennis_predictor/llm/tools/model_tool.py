"""The `get_model_prediction` tool — the only sanctioned source of a win
probability in the entire stack (CLAUDE.md hard rule #4).

Loading discipline:

- The model artifact lives at ``models/<tour>/lightgbm/latest/model.joblib``.
- We resolve `latest` lazily on first call so a freshly-retrained model is
  picked up without restarting any long-lived process; the file is loaded
  once per `(tour, artifact_path)` and memoised in `_PREDICTOR_CACHE`.
- If the artifact is missing or unloadable for any reason, we raise
  `ModelUnavailableError`. CLAUDE.md hard rule #10: the agent must not be
  invoked when this fires — the CLI catches it before entering the loop.

Probability orientation:

`compute_features` builds the FeatureVector with `p1 = lex-smaller player_id`.
The model predicts `P(p1 wins)`. The agent's user-facing labels are
`player_a` / `player_b`, which may match (p1, p2) or be reversed. We swap
back here so `model_probability_player_a` is always the probability that
the *user-facing* `player_a` wins. Re-orientation is a single floating-point
flip, no re-inference, and keeps every downstream consumer ignorant of the
internal lex-ordering convention.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from tennis_predictor.config import MODELS_DIR
from tennis_predictor.data.reconcile import AliasIndex
from tennis_predictor.features.inference import compute_features
from tennis_predictor.features.schema import FeatureVector
from tennis_predictor.llm.tools.schemas import (
    GetModelPredictionInput,
    ModelFeatureSummary,
    ModelPrediction,
    ModelUnavailableError,
    PlayerResolutionError,
    Tour,
)
from tennis_predictor.models.calibration import CalibratedPredictor
from tennis_predictor.models.feature_spec import (
    CATEGORICAL_COLUMNS,
    CATEGORY_VALUES,
    FEATURE_COLUMNS,
)

# In-process memoisation: re-loading a joblib artifact on every tool call
# would add ~1s of latency on commodity hardware for no benefit. Keyed by
# the resolved artifact path so `latest` pointing to a new directory
# invalidates the cache automatically.
_PREDICTOR_CACHE: dict[Path, CalibratedPredictor] = {}

# Confidence floor — same as `db_tools._MIN_RESOLUTION_CONFIDENCE`. Kept
# local rather than imported so the two tool families stay swappable.
_MIN_RESOLUTION_CONFIDENCE: float = 0.85


def _resolve_player(conn: duckdb.DuckDBPyConnection, tour: Tour, name: str) -> str:
    """Resolve `name` to a canonical player_id or raise PlayerResolutionError."""
    index = AliasIndex(conn, tour)
    result = index.lookup(name)
    if result.canonical_player_id is None or result.confidence < _MIN_RESOLUTION_CONFIDENCE:
        raise PlayerResolutionError(
            f"could not resolve player name {name!r} on {tour} tour "
            f"(best candidate {result.candidate_name!r}, confidence {result.confidence:.2f})"
        )
    return result.canonical_player_id


def _resolve_artifact_path(tour: Tour, models_root: Path | None = None) -> Path:
    """Return the absolute path to `model.joblib` for the latest LightGBM
    artifact of `tour`. Raises `ModelUnavailableError` if the symlink or
    file is missing."""
    root = models_root or MODELS_DIR
    artifact_dir = root / tour / "lightgbm" / "latest"
    model_path = artifact_dir / "model.joblib"
    if not model_path.exists():
        raise ModelUnavailableError(
            f"LightGBM artifact missing for {tour}: expected {model_path}. "
            "Run `uv run python scripts/train_models.py` to produce one."
        )
    return model_path.resolve()


def _load_predictor(model_path: Path) -> CalibratedPredictor:
    cached = _PREDICTOR_CACHE.get(model_path)
    if cached is not None:
        return cached
    try:
        loaded = joblib.load(model_path)
    except Exception as exc:
        raise ModelUnavailableError(
            f"failed to load LightGBM artifact {model_path}: {exc}"
        ) from exc
    if not isinstance(loaded, CalibratedPredictor):
        raise ModelUnavailableError(
            f"artifact at {model_path} is not a CalibratedPredictor "
            f"(got {type(loaded).__name__}); the production run may be corrupted."
        )
    _PREDICTOR_CACHE[model_path] = loaded
    return loaded


def _feature_vector_to_frame(fv: FeatureVector) -> pd.DataFrame:
    """Build the single-row DataFrame the predictor expects, in the
    canonical FEATURE_COLUMNS order with categoricals typed.

    Numeric columns are forced to float: when a single-row DataFrame
    contains `None` in an int-like field (e.g. `h2h_recency_days=None`
    for never-met pairs), pandas falls back to `object` dtype, which
    LightGBM rejects with "pandas dtypes must be int, float or bool".
    Training reads via DuckDB which already yields proper NaN-bearing
    floats; inference has to recreate that dtype profile explicitly.
    """
    record = fv.model_dump()
    row = {col: record[col] for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row], columns=list(FEATURE_COLUMNS))
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            df[col] = pd.Categorical(df[col], categories=CATEGORY_VALUES[col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _orient_for_user(
    fv: FeatureVector,
    p1_id: str,
    user_a_id: str,
    prob_p1_wins: float,
) -> tuple[float, float, ModelFeatureSummary]:
    """Re-label canonical (p1, p2) outputs onto the user-facing (a, b).

    `compute_features` sorts player IDs lex-ascending, so p1 may or may
    not be the user's `player_a`. Here we swap (or pass through) the
    probability and the feature-summary fields so the consumer sees
    everything from the user-facing perspective.
    """
    a_is_p1 = user_a_id == p1_id
    if a_is_p1:
        prob_a = prob_p1_wins
        elo_a, elo_b = fv.elo_p1_surface, fv.elo_p2_surface
        rank_a, rank_b = fv.rank_p1, fv.rank_p2
        h2h_a, h2h_b = fv.h2h_p1_wins, fv.h2h_p2_wins
        wp10_a, wp10_b = fv.win_pct_last10_p1, fv.win_pct_last10_p2
        wp25s_a, wp25s_b = fv.win_pct_last25_surface_p1, fv.win_pct_last25_surface_p2
        fat_a, fat_b = fv.fatigue_matches_7d_p1, fv.fatigue_matches_7d_p2
        rest_a, rest_b = fv.days_since_last_match_p1, fv.days_since_last_match_p2
    else:
        prob_a = 1.0 - prob_p1_wins
        elo_a, elo_b = fv.elo_p2_surface, fv.elo_p1_surface
        rank_a, rank_b = fv.rank_p2, fv.rank_p1
        h2h_a, h2h_b = fv.h2h_p2_wins, fv.h2h_p1_wins
        wp10_a, wp10_b = fv.win_pct_last10_p2, fv.win_pct_last10_p1
        wp25s_a, wp25s_b = fv.win_pct_last25_surface_p2, fv.win_pct_last25_surface_p1
        fat_a, fat_b = fv.fatigue_matches_7d_p2, fv.fatigue_matches_7d_p1
        rest_a, rest_b = fv.days_since_last_match_p2, fv.days_since_last_match_p1
    summary = ModelFeatureSummary(
        elo_player_a=elo_a,
        elo_player_b=elo_b,
        elo_diff_a_minus_b=elo_a - elo_b,
        rank_player_a=rank_a,
        rank_player_b=rank_b,
        h2h_player_a_wins=h2h_a,
        h2h_player_b_wins=h2h_b,
        win_pct_last10_player_a=wp10_a,
        win_pct_last10_player_b=wp10_b,
        win_pct_last25_surface_player_a=wp25s_a,
        win_pct_last25_surface_player_b=wp25s_b,
        fatigue_matches_7d_player_a=fat_a,
        fatigue_matches_7d_player_b=fat_b,
        days_since_last_match_player_a=rest_a,
        days_since_last_match_player_b=rest_b,
    )
    return prob_a, 1.0 - prob_a, summary


def get_model_prediction(
    conn: duckdb.DuckDBPyConnection,
    payload: GetModelPredictionInput,
    *,
    models_root: Path | None = None,
    compute_features_fn: Callable[..., FeatureVector] | None = None,
) -> ModelPrediction:
    """Run the production LightGBM model on a hypothetical fixture.

    `models_root` and `compute_features_fn` are seams for testing — the
    happy path uses defaults. Tests can swap in a small artifact directory
    or a stub feature builder without ever touching real DuckDB matches.
    """
    a_id = _resolve_player(conn, payload.tour, payload.player_a_name)
    b_id = _resolve_player(conn, payload.tour, payload.player_b_name)
    if a_id == b_id:
        raise PlayerResolutionError(
            f"player_a_name and player_b_name resolve to the same player "
            f"({a_id!r}) — refusing to score a self-match"
        )

    model_path = _resolve_artifact_path(payload.tour, models_root)
    predictor = _load_predictor(model_path)

    builder = compute_features_fn or compute_features
    fv = builder(
        conn,
        player_id=a_id,
        opponent_id=b_id,
        surface=payload.surface,
        tour=payload.tour,
        as_of_date=payload.match_date,
        tournament_level=payload.tournament_level,
        best_of=payload.best_of,
    )

    frame = _feature_vector_to_frame(fv)
    try:
        proba = predictor.predict_proba(frame)
    except Exception as exc:
        raise ModelUnavailableError(
            f"LightGBM predict_proba failed for {payload.tour} "
            f"({payload.player_a_name} vs {payload.player_b_name}): {exc}"
        ) from exc
    prob_p1_wins = float(proba[0, 1])

    # `compute_features` orders by lex-smaller id; recover the p1 id so we
    # can flip (or pass through) onto the user-facing labelling.
    p1_id = a_id if a_id < b_id else b_id
    prob_a, prob_b, summary = _orient_for_user(fv, p1_id, a_id, prob_p1_wins)

    return ModelPrediction(
        player_a_name=payload.player_a_name,
        player_b_name=payload.player_b_name,
        tour=payload.tour,
        surface=payload.surface,
        tournament_level=payload.tournament_level,
        best_of=payload.best_of,
        match_date=payload.match_date,
        model_probability_player_a=prob_a,
        model_probability_player_b=prob_b,
        model_artifact_version=model_path.parent.name,
        feature_summary=summary,
    )


def clear_predictor_cache() -> None:
    """Test helper: drop every cached predictor so the next call reloads.
    Production code never needs this — `latest` rotation is path-keyed."""
    _PREDICTOR_CACHE.clear()


__all__ = ["clear_predictor_cache", "get_model_prediction"]
