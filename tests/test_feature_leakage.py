"""Anti-leakage tests for the feature layer.

The contract: `compute_features(..., as_of_date=D)` must NEVER read data
whose effective timestamp is `>= D`. These tests assert that by mutating
future rows and confirming the FeatureVector is unchanged.

If a test here fails, the feature is wrong — do not "fix" the test. Per
CLAUDE.md hard rule #1 and the feature-engineering skill.

# Coverage matrix

Each test targets one state object's read path. Together they cover every
field in the FeatureVector that could in principle be tainted by future
data:

| State          | Mutation                              | Fields protected            |
|----------------|---------------------------------------|-----------------------------|
| EloState       | swap winner/loser of future match     | elo_p1/p2_surface, diff     |
| RollingForm    | swap winner/loser of future match     | win_pct_last10 + last25_surf|
| H2HState       | swap p1-vs-p2 future match winner     | h2h_p1/p2_wins, recency_days|
| FatigueState   | change score string of future match   | fatigue_matches/sets        |
| RankingLookup  | insert a future ranking row           | rank_p1/p2, rank_diff       |
| ServeReturn    | change w_* stats of future match      | first/second/bp_*           |
| LastMatchState | swap winner / insert future match     | days_since_last_match_*     |
| ALL            | insert a brand new future match       | should not change anything  |
| ALL            | delete a future match                 | should not change anything  |

Phase 4.1 player-metadata fields (`hand_*`, `age_*`, `height_*`) come from
a static `players` JOIN and don't depend on future-match data — they
cannot leak by construction. They're covered indirectly by the
'baseline has nontrivial values' assertion: if a bug ever moved them onto
a future-reading path, they'd show drift in the existing tamper tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.features import FeatureVector, compute_features
from tennis_predictor.features.schema import FEATURE_FIELD_NAMES

# ---------------------------------------------------------------------------
# Contract-shape tests (run regardless of fixture)
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_feature_vector_has_exactly_39_fields() -> None:
    """Guards against silent additions/removals diverging from
    `.claude/skills/feature-engineering/SKILL.md` and `docs/phases.md`.
    Phase 4.1 bumped v1's 28 fields to v2's 39 (v1 + 11 metadata/recovery).
    """
    assert len(FeatureVector.model_fields) == 39, (
        f"FeatureVector must have 39 fields (Phase 4.1 v2 — see feature-engineering skill); "
        f"got {len(FeatureVector.model_fields)}"
    )


@pytest.mark.leakage
def test_feature_vector_family_breakdown() -> None:
    """The nine families must contain the documented counts.

    Catches accidental renames that would silently break downstream
    feature-importance grouping. Phase 4.1 added three families:
    handedness (2), age (4), height (3), recovery (2).
    """
    expected_by_family = {
        "elo": 3,
        "win_pct": 4,
        "serve_return": 8,
        "h2h": 3,
        "fatigue": 4,
        "rank": 3,
        "tournament_context": 3,
        "handedness": 2,
        "age": 4,
        "height": 3,
        "recovery": 2,
    }

    def classify(name: str) -> str:
        if name.startswith("elo_"):
            return "elo"
        if name.startswith("win_pct_"):
            return "win_pct"
        if (
            name.startswith("first_serve_")
            or name.startswith("second_serve_")
            or name.startswith("bp_")
        ):
            return "serve_return"
        if name.startswith("h2h_"):
            return "h2h"
        if name.startswith("fatigue_"):
            return "fatigue"
        if name.startswith("rank_"):
            return "rank"
        if name in {"tournament_level", "best_of", "surface"}:
            return "tournament_context"
        if name.startswith("hand_"):
            return "handedness"
        if name.startswith("age_"):
            return "age"
        if name.startswith("height_"):
            return "height"
        if name.startswith("days_since_last_match_"):
            return "recovery"
        raise AssertionError(f"Unclassified FeatureVector field: {name}")

    counts: dict[str, int] = dict.fromkeys(expected_by_family, 0)
    for name in FEATURE_FIELD_NAMES:
        counts[classify(name)] += 1

    assert (
        counts == expected_by_family
    ), f"Family counts drifted from contract: {counts} != {expected_by_family}"


# ---------------------------------------------------------------------------
# Tampered-future-rows fixture
# ---------------------------------------------------------------------------

TARGET_DATE = date(2020, 5, 1)
"""Date of the hypothetical 'predict me' match. Everything before this is
history; everything from here on is future relative to the snapshot."""

P1 = "ATP_AAA"
P2 = "ATP_BBB"
P3 = "ATP_CCC"
P4 = "ATP_DDD"


_counter = {"n": 0}


def _next_match_id() -> str:
    _counter["n"] += 1
    return f"sackmann::LEAK-{_counter['n']:04d}"


def _insert_match(
    conn: duckdb.DuckDBPyConnection,
    *,
    winner: str,
    loser: str,
    match_date: date,
    surface: str = "Hard",
    score: str = "6-4 6-3",
    match_num: int = 1,
    w_stats: tuple[int, int, int, int, int, int] = (60, 45, 25, 100, 3, 5),
    l_stats: tuple[int, int, int, int, int, int] = (55, 35, 18, 90, 4, 8),
) -> str:
    """Insert one match with sensible default stats. Stats tuple is
    (first_in, first_won, second_won, svpt, bp_saved, bp_faced) — same
    layout for winner and loser.

    Defaults reflect realistic per-match counts that satisfy the
    `first_won <= first_in` and `first_won + second_won <= svpt`
    invariants and yield rates in the unit interval."""
    match_id = _next_match_id()
    w_fi, w_fw, w_sw, w_svpt, w_bps, w_bpf = w_stats
    l_fi, l_fw, l_sw, l_svpt, l_bps, l_bpf = l_stats
    conn.execute(
        """
        INSERT INTO matches (
            match_id, source, match_external_id, tour, match_tier,
            tourney_id, tourney_name, tourney_level, tourney_date, surface,
            match_num, best_of, score, match_status,
            winner_player_id, loser_player_id,
            w_first_in, w_first_won, w_second_won, w_svpt, w_bp_saved, w_bp_faced,
            l_first_in, l_first_won, l_second_won, l_svpt, l_bp_saved, l_bp_faced
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            match_id,
            "sackmann",
            match_id.split("::")[1],
            "ATP",
            "main",
            f"2020-{match_num:04d}",
            "Test Open",
            "A",
            match_date,
            surface,
            match_num,
            3,
            score,
            "completed",
            winner,
            loser,
            w_fi,
            w_fw,
            w_sw,
            w_svpt,
            w_bps,
            w_bpf,
            l_fi,
            l_fw,
            l_sw,
            l_svpt,
            l_bps,
            l_bpf,
        ],
    )
    return match_id


def _insert_ranking(
    conn: duckdb.DuckDBPyConnection, player_id: str, ranking_date: date, rank: int
) -> None:
    conn.execute(
        "INSERT INTO rankings (player_id, ranking_date, rank, points) VALUES (?, ?, ?, ?)",
        [player_id, ranking_date, rank, 1000],
    )


def _insert_player(
    conn: duckdb.DuckDBPyConnection,
    player_id: str,
    *,
    hand: str | None,
    dob: date | None,
    height: int | None,
) -> None:
    """Insert a Phase 4.1 player-metadata row used by the PlayerMetadataLookup
    JOIN inside `compute_features`. `sackmann_id` is NOT NULL but not
    unique, so a static value is fine for fixture purposes."""
    conn.execute(
        "INSERT INTO players (player_id, tour, sackmann_id, hand, dob, height) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [player_id, "ATP", 0, hand, dob, height],
    )


@pytest.fixture
def leakage_db(tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """A DB with a realistic baseline: past matches (history floor crossed),
    plus a set of future matches we will tamper with in individual tests.

    The fixture is rebuilt per test so mutations are isolated.
    """
    _counter["n"] = 0
    conn = duckdb.connect(str(tmp_path / "leakage.duckdb"))
    schema.create_all_tables(conn)

    # --- Player metadata (static — JOINed for hand/age/height features) -------
    # P1 right-handed with full DOB + height; P2 left-handed, no height; P3 no
    # DOB; P4 no metadata except handedness. This mix exercises every
    # nullability branch in compute_age / compute_age_vs_peak / height_diff.
    _insert_player(conn, P1, hand="R", dob=date(1995, 6, 15), height=188)
    _insert_player(conn, P2, hand="L", dob=date(1992, 3, 22), height=None)
    _insert_player(conn, P3, hand="R", dob=None, height=183)
    _insert_player(conn, P4, hand="U", dob=None, height=None)

    # --- Past history (well below TARGET_DATE) --------------------------------
    # Each of p1, p2 gets 8 warm-up matches across both surfaces.
    for i in range(8):
        d = date(2020, 1, 1) + timedelta(days=i)
        _insert_match(conn, winner=P1, loser=P3, match_date=d, match_num=100 + i)
        _insert_match(conn, winner=P2, loser=P4, match_date=d, surface="Clay", match_num=200 + i)
    # One past h2h meeting between p1 and p2 — gives nontrivial h2h state.
    _insert_match(conn, winner=P1, loser=P2, match_date=date(2020, 4, 15), match_num=300)

    # Match within the 14-day fatigue window of TARGET_DATE.
    _insert_match(conn, winner=P1, loser=P3, match_date=date(2020, 4, 25), match_num=350)
    _insert_match(conn, winner=P2, loser=P4, match_date=date(2020, 4, 23), match_num=360)

    # --- Future rows (strictly after TARGET_DATE) -----------------------------
    # We'll mutate these to test that compute_features ignores them.
    _insert_match(
        conn,
        winner=P1,
        loser=P3,
        match_date=date(2020, 5, 10),
        match_num=500,
        score="6-4 7-5",
    )
    _insert_match(
        conn,
        winner=P2,
        loser=P4,
        match_date=date(2020, 5, 12),
        surface="Clay",
        match_num=510,
        score="6-3 6-2",
    )
    _insert_match(
        conn,
        winner=P1,
        loser=P2,
        match_date=date(2020, 5, 20),
        match_num=520,
        score="6-2 6-3",
    )
    _insert_match(
        conn,
        winner=P3,
        loser=P1,
        match_date=date(2020, 6, 1),
        match_num=530,
        score="6-7 7-6 6-4",
        w_stats=(70, 55, 25, 110, 4, 6),
        l_stats=(65, 48, 22, 100, 3, 8),
    )

    # Rankings — historical only for now. Future rankings used in leak tests.
    _insert_ranking(conn, P1, date(2020, 4, 6), 25)
    _insert_ranking(conn, P2, date(2020, 4, 6), 40)
    _insert_ranking(conn, P3, date(2020, 4, 6), 100)
    _insert_ranking(conn, P4, date(2020, 4, 6), 120)

    yield conn
    conn.close()


def _baseline_fv(conn: duckdb.DuckDBPyConnection) -> FeatureVector:
    """The FeatureVector for the (P1 vs P2) target match on TARGET_DATE."""
    return compute_features(
        conn,
        player_id=P1,
        opponent_id=P2,
        surface="Hard",
        tour="ATP",
        as_of_date=TARGET_DATE,
        tournament_level="ATP250",
        best_of=3,
    )


# ---------------------------------------------------------------------------
# Tampered-future-rows tests — one per state object + structural ones
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_baseline_has_nontrivial_values(leakage_db: duckdb.DuckDBPyConnection) -> None:
    """Sanity: the fixture is rich enough that the FV is NOT all defaults.
    A test that compares 'before' and 'after' is useless if both are
    just the empty-state defaults."""
    fv = _baseline_fv(leakage_db)
    # Elo updated by past matches (P1, P2 both won 8 warm-ups + had one H2H).
    assert fv.elo_p1_surface != 1500.0 or fv.elo_p2_surface != 1500.0
    # H2H reflects the past 2020-04-15 meeting.
    assert (fv.h2h_p1_wins, fv.h2h_p2_wins) == (1, 0)
    assert fv.h2h_recency_days is not None
    # Fatigue: P1 played 2020-04-25 (6 days before TARGET) — inside the
    # 7-day window. P2 played 2020-04-23 (8 days before) — inside the
    # 14-day sets window but not the 7-day matches window.
    assert fv.fatigue_matches_7d_p1 >= 1
    assert fv.fatigue_sets_14d_p1 >= 1
    assert fv.fatigue_sets_14d_p2 >= 1
    # Rankings loaded.
    assert fv.rank_p1 == 25
    assert fv.rank_p2 == 40

    # --- Phase 4.1 metadata + recovery features -------------------------------
    # P1 = right-handed, P2 = left-handed.
    assert fv.hand_p1 == "R"
    assert fv.hand_p2 == "L"
    # Ages computed from inserted DOBs.
    assert fv.age_p1 is not None and 24.5 < fv.age_p1 < 25.5
    assert fv.age_p2 is not None and 27.5 < fv.age_p2 < 28.5
    # Heights: P1 has 188cm; P2 has none → diff is None.
    assert fv.height_p1 == 188
    assert fv.height_p2 is None
    assert fv.height_diff_cm is None
    # Recovery: P1's last completed match before TARGET_DATE was 2020-04-25
    # (matches insertion order); P2's was 2020-04-23.
    assert fv.days_since_last_match_p1 == (TARGET_DATE - date(2020, 4, 25)).days
    assert fv.days_since_last_match_p2 == (TARGET_DATE - date(2020, 4, 23)).days


@pytest.mark.leakage
def test_swapping_future_winner_loser_does_not_change_fv(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """A future P3 win over P1 (2020-06-01) — swap so P1 wins instead.
    If Elo / RollingForm leaked, the FV would change."""
    before = _baseline_fv(leakage_db)
    leakage_db.execute(
        "UPDATE matches SET winner_player_id = ?, loser_player_id = ? WHERE match_num = ?",
        [P1, P3, 530],
    )
    after = _baseline_fv(leakage_db)
    assert before == after


@pytest.mark.leakage
def test_swapping_future_h2h_match_does_not_change_h2h_fields(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """A future P1 win over P2 (2020-05-20) — swap so P2 wins instead.
    If H2H leaked, h2h_p1_wins / h2h_p2_wins / h2h_recency_days would change."""
    before = _baseline_fv(leakage_db)
    leakage_db.execute(
        "UPDATE matches SET winner_player_id = ?, loser_player_id = ? WHERE match_num = ?",
        [P2, P1, 520],
    )
    after = _baseline_fv(leakage_db)
    assert before.h2h_p1_wins == after.h2h_p1_wins
    assert before.h2h_p2_wins == after.h2h_p2_wins
    assert before.h2h_recency_days == after.h2h_recency_days
    assert before == after


@pytest.mark.leakage
def test_changing_future_score_does_not_change_fatigue_fields(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """Mutate the score string of a future match — sets count would change.
    Fatigue must ignore it."""
    before = _baseline_fv(leakage_db)
    leakage_db.execute(
        "UPDATE matches SET score = ? WHERE match_num = ?",
        ["6-4 4-6 7-6(5) 4-6 6-4", 500],  # 5-set instead of 2
    )
    after = _baseline_fv(leakage_db)
    assert before.fatigue_matches_7d_p1 == after.fatigue_matches_7d_p1
    assert before.fatigue_sets_14d_p1 == after.fatigue_sets_14d_p1
    assert before == after


@pytest.mark.leakage
def test_changing_future_serve_stats_does_not_change_serve_return_fields(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """Mutate the w_* counts of a future match. ServeReturn must ignore it."""
    before = _baseline_fv(leakage_db)
    leakage_db.execute(
        """
        UPDATE matches
        SET w_first_in = 80, w_first_won = 75, w_second_won = 5, w_svpt = 100,
            w_bp_saved = 0, w_bp_faced = 0
        WHERE match_num = ?
        """,
        [500],
    )
    after = _baseline_fv(leakage_db)
    assert before.first_serve_win_pct_p1 == after.first_serve_win_pct_p1
    assert before.second_serve_win_pct_p1 == after.second_serve_win_pct_p1
    assert before.bp_saved_pct_p1 == after.bp_saved_pct_p1
    assert before.bp_converted_pct_p1 == after.bp_converted_pct_p1
    assert before == after


@pytest.mark.leakage
def test_inserting_future_ranking_does_not_change_rank_fields(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """A new ranking row dated AFTER TARGET_DATE — rank_p1 must keep the
    older (pre-target) snapshot."""
    before = _baseline_fv(leakage_db)
    _insert_ranking(leakage_db, P1, date(2020, 5, 11), 8)  # jumped to 8
    _insert_ranking(leakage_db, P2, date(2020, 5, 11), 15)
    after = _baseline_fv(leakage_db)
    assert before.rank_p1 == after.rank_p1
    assert before.rank_p2 == after.rank_p2
    assert before.rank_diff == after.rank_diff
    assert before == after


@pytest.mark.leakage
def test_inserting_brand_new_future_match_does_not_change_fv(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """Adding an entirely new future row — should not perturb any field."""
    before = _baseline_fv(leakage_db)
    _insert_match(
        leakage_db,
        winner=P2,
        loser=P1,
        match_date=date(2020, 7, 1),
        match_num=900,
        score="6-3 6-3",
    )
    after = _baseline_fv(leakage_db)
    assert before == after


@pytest.mark.leakage
def test_deleting_future_match_does_not_change_fv(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """Removing a future row — must also not perturb anything."""
    before = _baseline_fv(leakage_db)
    leakage_db.execute("DELETE FROM matches WHERE match_num = ?", [530])
    after = _baseline_fv(leakage_db)
    assert before == after


@pytest.mark.leakage
def test_changing_future_surface_does_not_change_fv(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """If we mutated a future match's surface (Hard → Clay), surface-Elo
    and surface-form would shift. Must not happen."""
    before = _baseline_fv(leakage_db)
    leakage_db.execute(
        "UPDATE matches SET surface = ? WHERE match_num = ?",
        ["Clay", 500],
    )
    after = _baseline_fv(leakage_db)
    assert before == after


@pytest.mark.leakage
def test_changing_future_tourney_name_to_indoor_does_not_change_fv(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """If we relabel a future Hard match as 'Paris Masters' (indoor whitelist
    hit), the surface normalizer would map it to IHard. Must not leak."""
    before = _baseline_fv(leakage_db)
    leakage_db.execute(
        "UPDATE matches SET tourney_name = ? WHERE match_num = ?",
        ["Paris Masters", 530],
    )
    after = _baseline_fv(leakage_db)
    assert before == after


# ---------------------------------------------------------------------------
# Phase 4.1 — LastMatchState leakage tests
# ---------------------------------------------------------------------------
#
# Note: `tourney_date` is the field that classifies a row as past or
# future. Mutating it is not a leakage scenario — the system trusts the
# stored date and will treat a moved-backward row as a legitimate past
# observation. The realistic attack vectors are mutating winner/loser,
# inserting brand-new future rows, and deleting future rows. Each is
# covered below or by the structural tests further up the file.


@pytest.mark.leakage
def test_changing_future_winner_does_not_change_days_since(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """A future P3-over-P1 result (2020-06-01) — if days_since_last_match
    leaked through it, the recovery snapshot would shorten for P1. Must
    stay nailed to the most recent PAST match."""
    before = _baseline_fv(leakage_db)
    leakage_db.execute(
        "UPDATE matches SET winner_player_id = ?, loser_player_id = ? WHERE match_num = ?",
        [P1, P3, 530],
    )
    after = _baseline_fv(leakage_db)
    assert before.days_since_last_match_p1 == after.days_since_last_match_p1
    assert before.days_since_last_match_p2 == after.days_since_last_match_p2


@pytest.mark.leakage
def test_inserting_future_player_match_does_not_shrink_days_since(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """Inserting a brand-new future match for P1 must NOT make
    days_since_last_match_p1 smaller — the recovery feature is locked to
    the most recent PAST match (2020-04-25), regardless of what happens
    after TARGET_DATE."""
    before = _baseline_fv(leakage_db)
    _insert_match(
        leakage_db,
        winner=P1,
        loser=P3,
        match_date=date(2020, 5, 8),  # would be 7 days before TARGET if it leaked
        match_num=999,
    )
    after = _baseline_fv(leakage_db)
    assert before.days_since_last_match_p1 == after.days_since_last_match_p1
    assert before.days_since_last_match_p2 == after.days_since_last_match_p2


# ---------------------------------------------------------------------------
# Phase 4.1 — player-metadata fields are static (sanity)
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_static_metadata_fields_pinned_to_players_row(
    leakage_db: duckdb.DuckDBPyConnection,
) -> None:
    """`hand_*`, `age_*`, `height_*` come from a `players` JOIN. They are
    static per (player, as_of_date) and don't read from `matches`. This
    test pins the contract: mutating any future MATCH row leaves them
    bit-identical. (Mutating the `players` row WOULD change them — that's
    intended; `players` is a Phase 1 cold-data table and stale entries
    are surfaced via the reconciliation pipeline, not the leakage layer.)
    """
    before = _baseline_fv(leakage_db)
    # Tamper several future rows in different ways.
    leakage_db.execute(
        "UPDATE matches SET winner_player_id = ?, loser_player_id = ? WHERE match_num = ?",
        [P3, P1, 530],
    )
    leakage_db.execute(
        "UPDATE matches SET score = ?, surface = ? WHERE match_num = ?",
        ["6-4 4-6 7-6(5) 4-6 6-4", "Clay", 500],
    )
    _insert_match(
        leakage_db,
        winner=P1,
        loser=P3,
        match_date=date(2020, 7, 1),
        match_num=901,
    )
    after = _baseline_fv(leakage_db)
    assert before.hand_p1 == after.hand_p1
    assert before.hand_p2 == after.hand_p2
    assert before.age_p1 == after.age_p1
    assert before.age_p2 == after.age_p2
    assert before.age_vs_peak_p1 == after.age_vs_peak_p1
    assert before.age_vs_peak_p2 == after.age_vs_peak_p2
    assert before.height_p1 == after.height_p1
    assert before.height_p2 == after.height_p2
    assert before.height_diff_cm == after.height_diff_cm
