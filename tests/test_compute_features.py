"""Tests for the inference-time `compute_features`.

The headline test is the **training-vs-inference equivalence** check:
    compute_features(p, q, surface, tour, as_of_date, ...)
must produce the same FeatureVector values as the corresponding row in
`training_features` for the same match — that is the Phase 3 exit
criterion.

Other tests cover the obvious invariants (player ordering, defaults,
argument validation, no future leakage).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.features import (
    FeatureVector,
    build_training_features,
    compute_features,
)


@pytest.fixture
def fresh_db(tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect(str(tmp_path / "compute_features_test.duckdb"))
    schema.create_all_tables(conn)
    yield conn
    conn.close()


_counter = {"n": 0}


def _next_match_id() -> str:
    _counter["n"] += 1
    return f"sackmann::CF-{_counter['n']:04d}"


def _reset() -> None:
    _counter["n"] = 0


def _insert_match(
    conn: duckdb.DuckDBPyConnection,
    *,
    winner: str,
    loser: str,
    match_date: date,
    surface: str | None = "Hard",
    tour: str = "ATP",
    match_tier: str = "main",
    match_status: str = "completed",
    tourney_level: str | None = "A",
    tourney_name: str = "Test Open",
    best_of: int = 3,
    score: str = "6-4 6-3",
    match_num: int = 1,
) -> str:
    match_id = _next_match_id()
    conn.execute(
        """
        INSERT INTO matches (
            match_id, source, match_external_id, tour, match_tier,
            tourney_id, tourney_name, tourney_level, tourney_date, surface,
            match_num, best_of, score, match_status,
            winner_player_id, loser_player_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            match_id,
            "sackmann",
            match_id.split("::")[1],
            tour,
            match_tier,
            f"2020-{match_num:04d}",
            tourney_name,
            tourney_level,
            match_date,
            surface,
            match_num,
            best_of,
            score,
            match_status,
            winner,
            loser,
        ],
    )
    return match_id


def _seed_history(
    conn: duckdb.DuckDBPyConnection,
    player: str,
    others: list[str],
    start: date,
    *,
    surface: str = "Hard",
) -> None:
    for i, opp in enumerate(others):
        _insert_match(
            conn,
            winner=player,
            loser=opp,
            match_date=start + timedelta(days=i),
            surface=surface,
            match_num=100 + i,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_argument_validation(fresh_db: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(ValueError, match="differ"):
        compute_features(
            fresh_db,
            player_id="ATP_X",
            opponent_id="ATP_X",
            surface="Hard",
            tour="ATP",
            as_of_date=date(2020, 1, 1),
            tournament_level="ATP250",
            best_of=3,
        )

    with pytest.raises(ValueError, match="inconsistent with tour"):
        compute_features(
            fresh_db,
            player_id="WTA_1",
            opponent_id="ATP_2",
            surface="Hard",
            tour="ATP",
            as_of_date=date(2020, 1, 1),
            tournament_level="ATP250",
            best_of=3,
        )


def test_empty_db_returns_defaults(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Brand new DB → both players unseen → defaults everywhere."""
    fv = compute_features(
        fresh_db,
        player_id="ATP_1",
        opponent_id="ATP_2",
        surface="Hard",
        tour="ATP",
        as_of_date=date(2020, 1, 1),
        tournament_level="ATP250",
        best_of=3,
    )
    assert fv.elo_p1_surface == 1500.0
    assert fv.elo_p2_surface == 1500.0
    assert fv.elo_diff_surface == 0.0
    assert fv.win_pct_last10_p1 is None
    assert fv.h2h_p1_wins == 0
    assert fv.h2h_p2_wins == 0
    assert fv.h2h_recency_days is None
    assert fv.fatigue_matches_7d_p1 == 0
    assert fv.rank_p1 == 9999
    assert fv.rank_p2 == 9999
    assert fv.rank_diff == 0
    assert fv.first_serve_win_pct_p1 is None  # below MIN_STAT_MATCHES


def test_returns_pydantic_feature_vector(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Caller contract: return a Pydantic FeatureVector, not a dict."""
    fv = compute_features(
        fresh_db,
        player_id="ATP_1",
        opponent_id="ATP_2",
        surface="Hard",
        tour="ATP",
        as_of_date=date(2020, 1, 1),
        tournament_level="ATP250",
        best_of=3,
    )
    assert isinstance(fv, FeatureVector)


def test_player_order_does_not_affect_vector(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """compute_features(A, B, ...) and compute_features(B, A, ...) must
    produce IDENTICAL FeatureVectors — p1/p2 are sorted internally."""
    _reset()
    # Asymmetric history: A wins a lot, B loses.
    _seed_history(fresh_db, "ATP_A", [f"ATP_X{i}" for i in range(8)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_Y{i}" for i in range(8)], date(2020, 2, 1))
    _insert_match(fresh_db, winner="ATP_A", loser="ATP_B", match_date=date(2020, 3, 1), match_num=1)

    fv_ab = compute_features(
        fresh_db,
        player_id="ATP_A",
        opponent_id="ATP_B",
        surface="Hard",
        tour="ATP",
        as_of_date=date(2020, 4, 1),
        tournament_level="ATP250",
        best_of=3,
    )
    fv_ba = compute_features(
        fresh_db,
        player_id="ATP_B",
        opponent_id="ATP_A",
        surface="Hard",
        tour="ATP",
        as_of_date=date(2020, 4, 1),
        tournament_level="ATP250",
        best_of=3,
    )
    assert fv_ab == fv_ba


def test_equivalence_with_training_replay(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """The Phase 3 exit-criterion test: a FeatureVector reached via the
    training replay must equal the one produced by compute_features for
    the same match.

    We engineer the fixture so the target match (`ATP_A` vs `ATP_B` on
    2020-04-01) has a STRICTLY LATER date than any other match. That way
    `as_of_date = 2020-04-01` excludes only the target — matching the
    replay's snapshot-before-update semantics exactly.
    """
    _reset()
    # Build asymmetric histories so the FeatureVector has non-default values.
    _seed_history(fresh_db, "ATP_A", [f"ATP_X{i}" for i in range(10)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_Y{i}" for i in range(10)], date(2020, 2, 1))
    # One prior A vs B meeting on Hard.
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_num=200,
    )
    # Target match — strictly later than anything else.
    target_id = _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 4, 1),
        match_num=999,
    )

    # Training replay populates training_features.
    build_training_features(fresh_db)

    cols = (
        "elo_p1_surface, elo_p2_surface, elo_diff_surface, "
        "win_pct_last10_p1, win_pct_last10_p2, "
        "win_pct_last25_surface_p1, win_pct_last25_surface_p2, "
        "h2h_p1_wins, h2h_p2_wins, h2h_recency_days, "
        "fatigue_matches_7d_p1, fatigue_matches_7d_p2, "
        "fatigue_sets_14d_p1, fatigue_sets_14d_p2, "
        "rank_p1, rank_p2, rank_diff, "
        "tournament_level, best_of, surface, "
        "p1_player_id, p2_player_id"
    )
    training_row = fresh_db.execute(
        f"SELECT {cols} FROM training_features WHERE match_id = ?",
        [target_id],
    ).fetchone()
    assert training_row is not None, "Training replay did not produce a row for the target match"

    # Now ask compute_features for the SAME match instance.
    inferred = compute_features(
        fresh_db,
        player_id="ATP_A",
        opponent_id="ATP_B",
        surface="Hard",
        tour="ATP",
        as_of_date=date(2020, 4, 1),
        tournament_level="ATP250",
        best_of=3,
    )

    # Compare every numeric / categorical field.
    expected = {
        "elo_p1_surface": training_row[0],
        "elo_p2_surface": training_row[1],
        "elo_diff_surface": training_row[2],
        "win_pct_last10_p1": training_row[3],
        "win_pct_last10_p2": training_row[4],
        "win_pct_last25_surface_p1": training_row[5],
        "win_pct_last25_surface_p2": training_row[6],
        "h2h_p1_wins": training_row[7],
        "h2h_p2_wins": training_row[8],
        "h2h_recency_days": training_row[9],
        "fatigue_matches_7d_p1": training_row[10],
        "fatigue_matches_7d_p2": training_row[11],
        "fatigue_sets_14d_p1": training_row[12],
        "fatigue_sets_14d_p2": training_row[13],
        "rank_p1": training_row[14],
        "rank_p2": training_row[15],
        "rank_diff": training_row[16],
        "tournament_level": training_row[17],
        "best_of": training_row[18],
        "surface": training_row[19],
    }
    actual = inferred.model_dump()
    for k, v in expected.items():
        if isinstance(v, float):
            assert actual[k] == pytest.approx(v), f"{k} drifted: {actual[k]} vs {v}"
        else:
            assert actual[k] == v, f"{k} mismatch: {actual[k]} vs {v}"


def test_elo_loaded_from_persisted_snapshot(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """If we run build_training_features then query compute_features for a
    NEW match with no further matches in between, Elo must come from
    elo_state (not default 1500)."""
    _reset()
    _seed_history(fresh_db, "ATP_A", [f"ATP_X{i}" for i in range(6)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_Y{i}" for i in range(6)], date(2020, 2, 1))
    # No A-vs-B yet. Just want the warm-up matches in state.
    build_training_features(fresh_db)

    fv = compute_features(
        fresh_db,
        player_id="ATP_A",
        opponent_id="ATP_B",
        surface="Hard",
        tour="ATP",
        as_of_date=date(2020, 3, 1),
        tournament_level="ATP250",
        best_of=3,
    )
    # A won all 6 warm-ups → Elo above default.
    assert fv.elo_p1_surface > 1500.0
    # B also won all 6 → also above default. Both above 1500.
    assert fv.elo_p2_surface > 1500.0


def test_elo_rolls_forward_past_snapshot(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Matches inserted AFTER the persisted Elo snapshot must update Elo
    when compute_features is called with as_of_date past those matches."""
    _reset()
    _seed_history(fresh_db, "ATP_A", [f"ATP_X{i}" for i in range(6)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_Y{i}" for i in range(6)], date(2020, 2, 1))
    build_training_features(fresh_db)

    elo_a_before = fresh_db.execute(
        "SELECT rating FROM elo_state WHERE player_id='ATP_A' AND surface='Hard'"
    ).fetchone()
    assert elo_a_before is not None

    # Now insert a fresh match where A wins — NOT yet replayed into elo_state.
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_NEW",
        match_date=date(2020, 3, 5),
        match_num=400,
    )

    fv = compute_features(
        fresh_db,
        player_id="ATP_A",
        opponent_id="ATP_B",
        surface="Hard",
        tour="ATP",
        as_of_date=date(2020, 3, 10),
        tournament_level="ATP250",
        best_of=3,
    )
    # A's Elo at inference time should be > snapshot's saved value because
    # of the roll-forward through the 2020-03-05 win.
    assert fv.elo_p1_surface > elo_a_before[0]


def test_does_not_read_future_matches(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """as_of_date = 2020-03-01 must not see a 2020-04-01 match."""
    _reset()
    _seed_history(fresh_db, "ATP_A", [f"ATP_X{i}" for i in range(6)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_Y{i}" for i in range(6)], date(2020, 2, 1))

    # Snapshot the features at 2020-03-01 with no later matches in DB.
    fv_before = compute_features(
        fresh_db,
        player_id="ATP_A",
        opponent_id="ATP_B",
        surface="Hard",
        tour="ATP",
        as_of_date=date(2020, 3, 1),
        tournament_level="ATP250",
        best_of=3,
    )

    # Add a future match that should NOT influence the 2020-03-01 snapshot.
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 4, 1),
        match_num=500,
    )

    fv_after = compute_features(
        fresh_db,
        player_id="ATP_A",
        opponent_id="ATP_B",
        surface="Hard",
        tour="ATP",
        as_of_date=date(2020, 3, 1),
        tournament_level="ATP250",
        best_of=3,
    )

    assert fv_before == fv_after, "compute_features leaked the future match"


def test_indoor_surface_normalization_applied(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """When asking for IHard, history played on Paris Bercy (raw Hard +
    indoor whitelist hit) must show up in the IHard window — surface
    normalization must be consistent between replay and inference."""
    _reset()
    # Bercy = indoor hard, raw_surface='Hard'. Six wins for A there.
    for i in range(6):
        _insert_match(
            fresh_db,
            winner="ATP_A",
            loser=f"ATP_OP_{i}",
            match_date=date(2020, 1, 1) + timedelta(days=i),
            surface="Hard",
            tourney_name="Paris Masters",
            match_num=100 + i,
        )

    fv = compute_features(
        fresh_db,
        player_id="ATP_A",
        opponent_id="ATP_B",
        surface="IHard",
        tour="ATP",
        as_of_date=date(2020, 6, 1),
        tournament_level="M1000",
        best_of=3,
    )
    # IHard Elo should reflect the 6 wins, not be the default.
    assert fv.elo_p1_surface > 1500.0
