"""End-to-end smoke test: build a small synthetic DB and run the CLI.

This is the integration-level cousin of ``test_model_roundtrip``: it
exercises the full Phase 4 pipeline (data load → walk-forward harness →
calibration → artifact persistence → round-trip) on a 4-year synthetic
``training_features`` slice, then asserts:

- 2 artifact directories land on disk per (tour, model_type) under
  ``--models-root``.
- ``model.joblib`` round-trips against ``roundtrip_fixture.json``.
- ``metadata.json`` and ``report.md`` contain the expected sections.
- ``calibration_plot.png`` is non-empty.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest

from tennis_predictor import config
from tennis_predictor.data import schema
from tennis_predictor.models.calibration import CalibratedPredictor
from tennis_predictor.models.feature_spec import CATEGORICAL_COLUMNS, CATEGORY_VALUES

VALIDATE_YEARS = [2022, 2023]
LAST_FULL_YEAR = 2023
TRAIN_YEARS = [2020, 2021, 2022, 2023]  # range needed for 2-fold walk-forward
MATCHES_PER_YEAR = 120


def _synthetic_features(
    rng: np.random.Generator,
    match_date: date,
    elo_p1: float,
    elo_p2: float,
) -> dict[str, object]:
    """Phase 4.1 v2 synthetic features. Returns all 39 FeatureVector fields
    with realistic, in-bounds random values. The metadata block mirrors what
    `compute_features` would emit after JOINing `players` — heights and
    days_since_last_match occasionally None to exercise the NaN paths."""
    height_p1 = int(rng.integers(165, 205)) if rng.random() < 0.7 else None
    height_p2 = int(rng.integers(165, 205)) if rng.random() < 0.7 else None
    age_p1 = float(rng.uniform(18.0, 36.0))
    age_p2 = float(rng.uniform(18.0, 36.0))
    days_p1 = int(rng.integers(2, 60)) if rng.random() < 0.9 else None
    days_p2 = int(rng.integers(2, 60)) if rng.random() < 0.9 else None
    return {
        "elo_p1_surface": elo_p1,
        "elo_p2_surface": elo_p2,
        "elo_diff_surface": elo_p1 - elo_p2,
        "win_pct_last10_p1": float(rng.uniform(0.3, 0.7)),
        "win_pct_last10_p2": float(rng.uniform(0.3, 0.7)),
        "win_pct_last25_surface_p1": float(rng.uniform(0.3, 0.7)),
        "win_pct_last25_surface_p2": float(rng.uniform(0.3, 0.7)),
        "first_serve_win_pct_p1": float(rng.uniform(0.55, 0.80)),
        "first_serve_win_pct_p2": float(rng.uniform(0.55, 0.80)),
        "second_serve_win_pct_p1": float(rng.uniform(0.40, 0.60)),
        "second_serve_win_pct_p2": float(rng.uniform(0.40, 0.60)),
        "bp_saved_pct_p1": float(rng.uniform(0.40, 0.75)),
        "bp_saved_pct_p2": float(rng.uniform(0.40, 0.75)),
        "bp_converted_pct_p1": float(rng.uniform(0.30, 0.55)),
        "bp_converted_pct_p2": float(rng.uniform(0.30, 0.55)),
        "h2h_p1_wins": int(rng.integers(0, 5)),
        "h2h_p2_wins": int(rng.integers(0, 5)),
        "h2h_recency_days": int(rng.integers(30, 500)),
        "fatigue_matches_7d_p1": int(rng.integers(0, 4)),
        "fatigue_matches_7d_p2": int(rng.integers(0, 4)),
        "fatigue_sets_14d_p1": int(rng.integers(0, 12)),
        "fatigue_sets_14d_p2": int(rng.integers(0, 12)),
        "rank_p1": int(rng.integers(1, 150)),
        "rank_p2": int(rng.integers(1, 150)),
        "rank_diff": 0,  # rewritten below
        "tournament_level": rng.choice(["Slam", "M1000", "ATP500", "ATP250"]),
        "best_of": int(rng.choice([3, 5])),
        "surface": rng.choice(["Hard", "Clay", "Grass", "IHard"]),
        # --- Phase 4.1 v2 metadata + recovery ----------------------------
        "hand_p1": rng.choice(["R", "L", "U"]),
        "hand_p2": rng.choice(["R", "L", "U"]),
        "age_p1": age_p1,
        "age_p2": age_p2,
        "age_vs_peak_p1": age_p1 - 26.0,
        "age_vs_peak_p2": age_p2 - 26.0,
        "height_p1": height_p1,
        "height_p2": height_p2,
        "height_diff_cm": (height_p1 - height_p2) if (height_p1 and height_p2) else None,
        "days_since_last_match_p1": days_p1,
        "days_since_last_match_p2": days_p2,
    }


def _build_synthetic_db(db_path: Path) -> None:
    conn = duckdb.connect(str(db_path))
    schema.create_all_tables(conn)

    rng = np.random.default_rng(seed=42)

    # Phase 4.1 v2 contract: every player_id referenced in training_features
    # must exist in `players` so the JOIN inside `compute_features` doesn't
    # produce silent unknowns. The 30 IDs match _synthetic_features's
    # player-id sampling range.
    player_rows: list[dict[str, object]] = []
    for i in range(1, 30):
        pid = f"ATP_P{i:03d}"
        hand = ["R", "L", "U"][i % 3]
        dob = date(2000 - (i % 12), 6, 1)  # ~18-30 years old at 2018+ matches
        height = 165 + (i * 3) % 40 if i % 4 != 0 else None
        player_rows.append(
            {
                "player_id": pid,
                "tour": "ATP",
                "sackmann_id": i,
                "name_first": None,
                "name_last": None,
                "full_name": None,
                "hand": hand,
                "dob": dob,
                "ioc": None,
                "height": height,
                "wikidata_id": None,
            }
        )
    players_df = pd.DataFrame(player_rows)
    conn.register("players_df", players_df)
    conn.execute("INSERT INTO players SELECT * FROM players_df")
    conn.unregister("players_df")

    match_rows = []
    feat_rows = []
    market_rows = []
    match_counter = 0
    for year in TRAIN_YEARS:
        for i in range(MATCHES_PER_YEAR):
            match_counter += 1
            mid = f"smoke::M-{match_counter:05d}"
            p1 = f"ATP_P{int(rng.integers(1, 30)):03d}"
            p2 = f"ATP_P{int(rng.integers(1, 30)):03d}"
            while p2 == p1:
                p2 = f"ATP_P{int(rng.integers(1, 30)):03d}"
            # Canonical lex order (p1 < p2).
            if p1 > p2:
                p1, p2 = p2, p1
            elo1 = float(1500 + rng.normal(0, 200))
            elo2 = float(1500 + rng.normal(0, 200))
            # Label sampled from the Elo formula so the model has signal.
            p_p1 = 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400))
            label = int(rng.random() < p_p1)
            mdate = date(year, 1, 1) + timedelta(days=int(rng.integers(0, 360)))
            feats = _synthetic_features(rng, mdate, elo1, elo2)
            feats["rank_diff"] = int(feats["rank_p1"]) - int(feats["rank_p2"])  # type: ignore[arg-type]
            winner_pid = p1 if label == 1 else p2
            loser_pid = p2 if label == 1 else p1
            match_rows.append(
                {
                    "match_id": mid,
                    "source": "smoke",
                    "match_external_id": mid,
                    "tour": "ATP",
                    "match_tier": "main",
                    "tourney_id": "smoke::T-0001",
                    "tourney_date": mdate,
                    "match_num": i + 1,
                    "match_status": "completed",
                    "winner_player_id": winner_pid,
                    "loser_player_id": loser_pid,
                }
            )
            feat_rows.append(
                {
                    "match_id": mid,
                    "tour": "ATP",
                    "match_date": mdate,
                    "p1_player_id": p1,
                    "p2_player_id": p2,
                    "label_winner_is_p1": label,
                    **feats,
                    "schema_version": 2,
                }
            )
            # Market odds for ~80% of matches.
            if rng.random() < 0.8:
                p_winner = float(np.clip(p_p1 if label == 1 else 1 - p_p1, 0.05, 0.95))
                p_loser = 1.0 - p_winner
                market_rows.append(
                    {
                        "match_id": mid,
                        "odds_source": "smoke",
                        "odds_winner_close": 1.0 / p_winner,
                        "odds_loser_close": 1.0 / p_loser,
                        "p_winner_close": p_winner,
                        "p_loser_close": p_loser,
                    }
                )

    matches_df = pd.DataFrame(match_rows)
    feat_df = pd.DataFrame(feat_rows)
    market_df = pd.DataFrame(market_rows)
    conn.register("matches_df", matches_df)
    conn.execute(
        "INSERT INTO matches (match_id, source, match_external_id, tour, match_tier, "
        "tourney_id, tourney_date, match_num, match_status, winner_player_id, loser_player_id) "
        "SELECT match_id, source, match_external_id, tour, match_tier, tourney_id, "
        "tourney_date, match_num, match_status, winner_player_id, loser_player_id FROM matches_df"
    )
    conn.unregister("matches_df")
    conn.register("feat_df", feat_df)
    conn.execute("INSERT INTO training_features SELECT * FROM feat_df")
    conn.unregister("feat_df")
    conn.register("market_df", market_df)
    conn.execute("INSERT INTO market_implied_probabilities SELECT * FROM market_df")
    conn.unregister("market_df")
    conn.close()


@pytest.fixture
def smoke_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, Path]]:
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    db_path = processed / "smoke.duckdb"
    _build_synthetic_db(db_path)
    models_root = tmp_path / "models"

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "DUCKDB_PATH", db_path)
    monkeypatch.setattr(config, "MODELS_DIR", models_root)

    yield db_path, models_root


def _restore_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = pd.Categorical(out[col], categories=CATEGORY_VALUES[col])
    return out


def test_train_models_cli_end_to_end(smoke_env: tuple[Path, Path]) -> None:
    _, models_root = smoke_env
    # Import after monkeypatch so the script binds to the patched config.
    from scripts.train_models import main as train_main

    rc = train_main(
        [
            "--tour",
            "ATP",
            "--validate-years",
            *[str(y) for y in VALIDATE_YEARS],
            "--last-full-year",
            str(LAST_FULL_YEAR),
            "--models-root",
            str(models_root),
            "--comparison-last-n",
            "2",
        ]
    )
    assert rc == 0

    # Two artifact dirs land on disk: ATP/elo/<ts> and ATP/lightgbm/<ts>.
    for model_type in ("elo", "lightgbm"):
        latest = models_root / "ATP" / model_type / "latest"
        assert latest.exists(), f"{latest} missing"
        target = latest.resolve()
        assert (target / "model.joblib").exists()
        assert (target / "metadata.json").exists()
        assert (target / "report.md").exists()
        assert (target / "calibration_plot.png").exists()
        assert (target / "roundtrip_fixture.json").exists()
        assert (target / "calibration_plot.png").stat().st_size > 0

        meta = json.loads((target / "metadata.json").read_text())
        assert meta["tour"] == "ATP"
        assert meta["model_type"] == model_type
        assert meta["walk_forward"]["n_folds"] == len(VALIDATE_YEARS)
        assert len(meta["features"]) == 39
        assert meta["calibration_method"] in ("isotonic", "platt")

        # Round-trip: model.joblib + fixture.
        predictor: CalibratedPredictor = joblib.load(target / "model.joblib")
        fixture = json.loads((target / "roundtrip_fixture.json").read_text())
        feats = _restore_categoricals(pd.DataFrame(fixture["features"]))
        expected = np.array(fixture["expected_p1_proba"], dtype=float)
        actual = predictor.predict_proba(feats)[:, 1]
        np.testing.assert_allclose(actual, expected, atol=1e-9)
