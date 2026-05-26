"""Integration tests for `build_training_features`.

These tests build a small synthetic DB from scratch and assert the
end-to-end orchestrator behaves correctly against the two eligibility
gates documented in `src/tennis_predictor/features/build.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.features.build import build_training_features


@pytest.fixture
def fresh_db(tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect(str(tmp_path / "build_test.duckdb"))
    schema.create_all_tables(conn)
    yield conn
    conn.close()


# Counter used by `_insert_match` to keep match_id unique without forcing
# the caller to invent IDs.
_counter = {"n": 0}


def _next_match_id() -> str:
    _counter["n"] += 1
    return f"sackmann::FIX-{_counter['n']:04d}"


def _reset_counter() -> None:
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
    tourney_id: str | None = None,
) -> str:
    """Insert one match into the test DB. Returns the generated match_id."""
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
            tourney_id or f"2020-{match_num:04d}",
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


def _insert_ranking(
    conn: duckdb.DuckDBPyConnection, player_id: str, ranking_date: date, rank: int
) -> None:
    conn.execute(
        "INSERT INTO rankings (player_id, ranking_date, rank, points) VALUES (?, ?, ?, ?)",
        [player_id, ranking_date, rank, 1000 - rank],
    )


def _seed_history(
    conn: duckdb.DuckDBPyConnection,
    player: str,
    others: list[str],
    start: date,
    *,
    surface: str = "Hard",
    match_tier: str = "main",
    tourney_level: str = "A",
    tourney_name: str = "Warm-up Open",
    each_wins: bool = True,
) -> None:
    """Insert `len(others)` matches between `player` and each opponent so
    `player` accumulates history. Helper for tests that want to bring a
    player above the history floor."""
    for i, opp in enumerate(others):
        d = start + timedelta(days=i)
        w, lo = (player, opp) if each_wins else (opp, player)
        _insert_match(
            conn,
            winner=w,
            loser=lo,
            match_date=d,
            surface=surface,
            match_tier=match_tier,
            tourney_level=tourney_level,
            tourney_name=tourney_name,
            match_num=100 + i,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_matches_yields_empty_outputs(fresh_db: duckdb.DuckDBPyConnection) -> None:
    _reset_counter()
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 0
    n_features = fresh_db.execute("SELECT count(*) FROM training_features").fetchone()
    n_elo = fresh_db.execute("SELECT count(*) FROM elo_state").fetchone()
    assert n_features == (0,)
    assert n_elo == (0,)


def test_below_history_floor_produces_no_label_rows(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Two players with no warm-up history → all eligible matches between
    them are skipped (history floor not met)."""
    _reset_counter()
    for i in range(3):
        _insert_match(
            fresh_db,
            winner="ATP_A",
            loser="ATP_B",
            match_date=date(2020, 1, 1) + timedelta(days=i),
            match_num=i + 1,
        )
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 0
    assert summary.skipped_history_floor == 3
    # State was still updated (3 completed matches feed Elo).
    assert summary.state_updates_applied == 3


def test_above_history_floor_produces_label_rows(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Build enough warm-up matches that both players cross the 5-match floor,
    then a 'real' match between them produces a label row."""
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_num=1,
    )
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 1

    row = fresh_db.execute(
        "SELECT p1_player_id, p2_player_id, label_winner_is_p1, tour, surface, "
        "tournament_level, best_of "
        "FROM training_features"
    ).fetchone()
    assert row is not None
    p1, p2, label, tour, surface, t_level, best_of = row
    # Lex-ordered: ATP_A < ATP_B.
    assert p1 == "ATP_A"
    assert p2 == "ATP_B"
    assert label == 1  # ATP_A is winner and p1 → label is 1
    assert tour == "ATP"
    assert surface == "Hard"
    assert t_level == "ATP250"
    assert best_of == 3


def test_label_zero_when_lex_smaller_player_loses(fresh_db: duckdb.DuckDBPyConnection) -> None:
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    # B beats A → A is still p1 (lex), but label=0
    _insert_match(
        fresh_db,
        winner="ATP_B",
        loser="ATP_A",
        match_date=date(2020, 3, 1),
        match_num=1,
    )
    build_training_features(fresh_db)
    row = fresh_db.execute(
        "SELECT p1_player_id, p2_player_id, label_winner_is_p1 FROM training_features"
    ).fetchone()
    assert row == ("ATP_A", "ATP_B", 0)


def test_davis_cup_does_not_write_label_but_feeds_state(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    """Davis Cup match has tourney_level='D' → no label row, but Elo
    still updates."""
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        tourney_level="D",
        tourney_name="Davis Cup",
        match_num=1,
    )
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 0
    assert summary.skipped_excluded_level == 1
    # Elo of A and B should have moved.
    elo_rows = fresh_db.execute(
        "SELECT player_id, rating FROM elo_state WHERE player_id IN ('ATP_A', 'ATP_B')"
    ).fetchall()
    assert dict(elo_rows) != {"ATP_A": 1500.0, "ATP_B": 1500.0}


def test_walkover_does_not_feed_state(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """W/O matches are not real outcomes — no state update, no label."""
    _reset_counter()
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_status="W/O",
        score="W/O",
        match_num=1,
    )
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 0
    assert summary.skipped_non_completed == 1
    assert summary.state_updates_applied == 0
    assert fresh_db.execute("SELECT count(*) FROM elo_state").fetchone() == (0,)


def test_null_surface_skipped_entirely(fresh_db: duckdb.DuckDBPyConnection) -> None:
    _reset_counter()
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        surface=None,
        match_num=1,
    )
    summary = build_training_features(fresh_db)
    assert summary.skipped_null_surface == 1
    assert summary.state_updates_applied == 0


def test_atp_main_draw_qualifying_writes_label(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """ATP main-draw qualifying lives in `qual_chall` files mixed with
    Challengers. Tour-level codes (G/M/A) inside qual_chall ARE label-eligible
    per user decision in Phase 3 design discussion."""
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_tier="qual_chall",
        tourney_level="A",  # ATP 250/500 main-draw qualifying
        tourney_name="Acapulco",
        match_num=1,
    )
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 1


def test_wta_main_draw_qualifying_writes_label(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """WTA main-draw qualifying lives in `qual_itf` files mixed with ITF
    events. Tour-level codes (G/PM/P/I/T1/T2/W) inside qual_itf ARE
    label-eligible."""
    _reset_counter()
    _seed_history(fresh_db, "WTA_A", [f"WTA_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "WTA_B", [f"WTA_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="WTA_A",
        loser="WTA_B",
        match_date=date(2020, 3, 1),
        tour="WTA",
        match_tier="qual_itf",
        tourney_level="P",  # WTA 500 main-draw qualifying
        tourney_name="Doha",
        match_num=1,
    )
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 1


def test_challenger_inside_qual_chall_is_excluded(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Inside `qual_chall`, tourney_level='C' (Challenger) must NOT be
    label-eligible — Challengers stay out of the prediction surface."""
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_tier="qual_chall",
        tourney_level="C",  # Challenger
        tourney_name="Bratislava CH",
        match_num=1,
    )
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 0
    assert summary.skipped_non_main_tier == 1
    # Elo still updates: 10 seed history matches + the 1 Challenger match.
    assert summary.state_updates_applied == 11


def test_itf_prize_money_tier_in_qual_itf_is_excluded(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Inside `qual_itf`, numeric prize-money codes (10/15/25/.../100) are
    ITF women's circuit events — not tour-level qualifying."""
    _reset_counter()
    _seed_history(fresh_db, "WTA_A", [f"WTA_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "WTA_B", [f"WTA_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="WTA_A",
        loser="WTA_B",
        match_date=date(2020, 3, 1),
        tour="WTA",
        match_tier="qual_itf",
        tourney_level="25",  # ITF $25K event
        tourney_name="W25 Anywhere",
        match_num=1,
    )
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 0
    assert summary.skipped_non_main_tier == 1


def test_futures_tier_feeds_state_but_no_label(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Futures matches should not produce labels but should update Elo so
    young players climbing through Challengers / Futures arrive at the tour
    with non-default ratings."""
    _reset_counter()
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_tier="futures",
        tourney_level="C",  # not in canonical set
        match_num=1,
    )
    summary = build_training_features(fresh_db)
    assert summary.training_rows_written == 0
    assert summary.skipped_non_main_tier == 1
    assert summary.state_updates_applied == 1


def test_idempotency_full_replace(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Re-running the orchestrator must produce identical training_features
    rows — full DELETE + re-INSERT, not append."""
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_num=1,
    )

    s1 = build_training_features(fresh_db)
    rows_after_first = fresh_db.execute(
        "SELECT match_id, p1_player_id, label_winner_is_p1 FROM training_features ORDER BY match_id"
    ).fetchall()

    s2 = build_training_features(fresh_db)
    rows_after_second = fresh_db.execute(
        "SELECT match_id, p1_player_id, label_winner_is_p1 FROM training_features ORDER BY match_id"
    ).fetchall()

    assert s1.training_rows_written == s2.training_rows_written
    assert rows_after_first == rows_after_second
    assert len(rows_after_first) == len({r[0] for r in rows_after_first})  # no dups


def test_elo_state_persisted(fresh_db: duckdb.DuckDBPyConnection) -> None:
    _reset_counter()
    for i in range(3):
        _insert_match(
            fresh_db,
            winner="ATP_A",
            loser="ATP_B",
            match_date=date(2020, 1, 1) + timedelta(days=i),
            match_num=i + 1,
        )
    build_training_features(fresh_db)
    rows = fresh_db.execute(
        "SELECT player_id, surface, rating FROM elo_state ORDER BY player_id, surface"
    ).fetchall()
    # 2 players x 1 surface (Hard) = 2 rows
    assert len(rows) == 2
    by_player = {pid: r for pid, _surf, r in rows}
    assert by_player["ATP_A"] > 1500.0
    assert by_player["ATP_B"] < 1500.0


def test_v3_sentinel_columns_present(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Phase 4.2 v3 smoke: after build, the two surface-specific recovery
    columns must exist on training_features and be populated for the
    label-eligible row."""
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_num=1,
    )
    build_training_features(fresh_db)
    row = fresh_db.execute(
        "SELECT days_since_last_match_surface_p1, days_since_last_match_surface_p2 "
        "FROM training_features"
    ).fetchone()
    assert row is not None
    p1_gap, p2_gap = row
    # Both seeded their history on Hard; the target match is also Hard.
    # Each player's most recent Hard match is the LAST of their 5 warm-ups.
    # ATP_A: warm-ups 2020-01-01..05; gap to 2020-03-01 is 56 days (leap year).
    # ATP_B: warm-ups 2020-02-01..05; gap to 2020-03-01 is 25 days.
    assert p1_gap == (date(2020, 3, 1) - date(2020, 1, 5)).days
    assert p2_gap == (date(2020, 3, 1) - date(2020, 2, 5)).days


def test_v3_surface_recovery_independent_of_other_surface_history(
    fresh_db: duckdb.DuckDBPyConnection,
) -> None:
    """The Phase 4.2 invariant: a player's recent Clay history must NOT
    reset their Hard surface-gap. This is the whole point of the feature."""
    _reset_counter()
    # ATP_A warmup history split: 5 Clay matches earlier, then 1 Hard match later.
    _seed_history(
        fresh_db,
        "ATP_A",
        [f"ATP_OP_{i}" for i in range(5)],
        date(2020, 1, 1),
        surface="Clay",
        tourney_name="Warm-up Clay",
    )
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_HARD_OPP",
        match_date=date(2020, 1, 10),
        surface="Hard",
        tourney_name="Warm-up Hard",
        match_num=190,
    )
    # ATP_B warmup history on Clay.
    _seed_history(
        fresh_db,
        "ATP_B",
        [f"ATP_OQ_{i}" for i in range(5)],
        date(2020, 4, 1),
        surface="Clay",
        tourney_name="Warm-up Clay",
    )
    # Bring ATP_A above the history floor on Clay (5 warm-ups + 1 Hard = 6, OK).
    # Target match on Clay between A and B, late June.
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 6, 1),
        surface="Clay",
        tourney_name="Target Clay Open",
        match_num=999,
    )
    build_training_features(fresh_db)
    # ATP_A < ATP_B lexicographically.
    row = fresh_db.execute(
        "SELECT days_since_last_match_p1, days_since_last_match_surface_p1 "
        "FROM training_features WHERE match_id LIKE '%FIX-%' "
        "ORDER BY match_date DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    global_gap, surface_gap = row
    # Global gap: most recent ATP_A match was 2020-01-10 (Hard).
    # Gap to 2020-06-01 = 143 days.
    assert global_gap == 143
    # Surface (Clay) gap: most recent ATP_A Clay match was 2020-01-05.
    # Gap to 2020-06-01 = 148 days — strictly greater than global, since
    # A played Hard more recently.
    assert surface_gap == 148
    assert surface_gap > global_gap


def test_ranking_sentinel_for_unranked_player(fresh_db: duckdb.DuckDBPyConnection) -> None:
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_num=1,
    )
    # No ranking rows inserted → both players get sentinel.
    build_training_features(fresh_db)
    row = fresh_db.execute("SELECT rank_p1, rank_p2, rank_diff FROM training_features").fetchone()
    assert row == (9999, 9999, 0)


def test_ranking_lookup_used_correctly(fresh_db: duckdb.DuckDBPyConnection) -> None:
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_ranking(fresh_db, "ATP_A", date(2020, 2, 24), rank=10)
    _insert_ranking(fresh_db, "ATP_B", date(2020, 2, 24), rank=80)
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_num=1,
    )
    build_training_features(fresh_db)
    row = fresh_db.execute("SELECT rank_p1, rank_p2, rank_diff FROM training_features").fetchone()
    assert row == (10, 80, -70)


def test_replay_survives_batch_flushes(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Regression: with a single connection, `executemany(INSERT)` mid-replay
    would clobber the read cursor. Verify a run that triggers at least one
    INSERT-batch flush still completes."""
    from tennis_predictor.features import build as build_mod

    # Force the batch threshold low so the test does not have to insert
    # thousands of matches to reach it.
    original_batch = build_mod.BATCH_SIZE
    build_mod.BATCH_SIZE = 4
    try:
        _reset_counter()
        # 10 players, each with 5 warm-up matches → all cross the floor.
        players = [f"ATP_P{i}" for i in range(10)]
        for p in players:
            _seed_history(fresh_db, p, [f"OPP_{p}_{j}" for j in range(5)], date(2020, 1, 1))
        # 20 label-eligible matches between known players → forces ≥5
        # successful INSERT-batch flushes through the orchestrator.
        for i in range(20):
            _insert_match(
                fresh_db,
                winner=players[i % 10],
                loser=players[(i + 1) % 10],
                match_date=date(2021, 1, 1) + timedelta(days=i),
                match_num=200 + i,
            )

        summary = build_training_features(fresh_db)
        assert summary.training_rows_written == 20
        n = fresh_db.execute("SELECT count(*) FROM training_features").fetchone()
        assert n == (20,)
    finally:
        build_mod.BATCH_SIZE = original_batch


def test_required_columns_non_null(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Per the FeatureVector contract, certain columns must never be NULL.
    Trip the DDL's NOT NULL constraints by trying to write — they would
    fail the run rather than silently corrupt the dataset."""
    _reset_counter()
    _seed_history(fresh_db, "ATP_A", [f"ATP_OP_{i}" for i in range(5)], date(2020, 1, 1))
    _seed_history(fresh_db, "ATP_B", [f"ATP_OQ_{i}" for i in range(5)], date(2020, 2, 1))
    _insert_match(
        fresh_db,
        winner="ATP_A",
        loser="ATP_B",
        match_date=date(2020, 3, 1),
        match_num=1,
    )
    build_training_features(fresh_db)
    required = [
        "elo_p1_surface",
        "elo_p2_surface",
        "elo_diff_surface",
        "h2h_p1_wins",
        "h2h_p2_wins",
        "fatigue_matches_7d_p1",
        "fatigue_matches_7d_p2",
        "fatigue_sets_14d_p1",
        "fatigue_sets_14d_p2",
        "rank_p1",
        "rank_p2",
        "rank_diff",
        "tournament_level",
        "best_of",
        "surface",
        "tour",
        "match_date",
        "p1_player_id",
        "p2_player_id",
        "label_winner_is_p1",
    ]
    for col in required:
        result = fresh_db.execute(
            f"SELECT count(*) FROM training_features WHERE {col} IS NULL"
        ).fetchone()
        assert result == (0,), f"{col} unexpectedly NULL"
