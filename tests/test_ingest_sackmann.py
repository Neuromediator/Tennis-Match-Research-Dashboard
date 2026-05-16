"""Unit tests for Sackmann CSV ingestion.

Uses tiny synthetic CSV fixtures in tmp_path. Real Sackmann data lives under
data/raw/tennis_atp/, and is not loaded here — that would be slow and would
turn unit tests into integration tests. Integration coverage lives in
test_data_loading.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from tennis_predictor.data import ingest_sackmann, schema

MATCH_HEADER = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,"
    "match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,"
    "winner_ht,winner_ioc,winner_age,loser_id,loser_seed,loser_entry,"
    "loser_name,loser_hand,loser_ht,loser_ioc,loser_age,score,best_of,round,"
    "minutes,w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,"
    "w_bpFaced,l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,"
    "l_bpFaced,winner_rank,winner_rank_points,loser_rank,loser_rank_points\n"
)


def _match_row(
    tourney_id: str = "2023-9900",
    match_num: int = 300,
    winner_id: int = 126203,
    loser_id: int = 126610,
    score: str = "6-3 6-4",
    tourney_date: str = "20230102",
) -> str:
    return (
        f"{tourney_id},United Cup,Hard,18,A,{tourney_date},{match_num},"
        f"{winner_id},3,,Taylor Fritz,R,196,USA,25.1,"
        f"{loser_id},5,,Matteo Berrettini,R,196,ITA,26.7,"
        f"{score},3,F,135,15,2,85,52,45,16,12,0,0,"
        f"7,2,97,62,47,15,12,9,9,9,3355,16,2375\n"
    )


@pytest.fixture
def fake_sackmann_root(tmp_path: Path):
    """Create a fake data/raw/tennis_atp directory with tiny CSVs.

    Patches config.RAW_DIR so ingest_sackmann reads from this tmp tree.
    """
    raw = tmp_path / "raw"
    atp_dir = raw / "tennis_atp"
    atp_dir.mkdir(parents=True)

    # Players
    (atp_dir / "atp_players.csv").write_text(
        "player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id\n"
        "126203,Taylor,Fritz,R,19971028,USA,196,Q12345\n"
        "126610,Matteo,Berrettini,R,19960412,ITA,196,Q67890\n"
        "100001,Old,Player,R,,USA,,Q00001\n"  # dob missing
    )

    # One year of matches (main tier)
    (atp_dir / "atp_matches_2023.csv").write_text(
        MATCH_HEADER + _match_row() + _match_row(match_num=299, score="6-2 0-0 RET")
    )

    # Rankings
    (atp_dir / "atp_rankings_current.csv").write_text(
        "ranking_date,rank,player,points\n20230102,1,126203,5000\n20230102,2,126610,4500\n"
    )

    with patch.object(ingest_sackmann.config, "RAW_DIR", raw):
        yield raw


@pytest.fixture
def fresh_db_with_schema(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    schema.create_all_tables(conn)
    yield conn
    conn.close()


def test_load_players_inserts_canonical_ids(
    fresh_db_with_schema: duckdb.DuckDBPyConnection,
    fake_sackmann_root: Path,
) -> None:
    inserted = ingest_sackmann.load_players(fresh_db_with_schema, "ATP")
    assert inserted == 3
    rows = fresh_db_with_schema.execute(
        "SELECT player_id, tour, sackmann_id, full_name, dob FROM players ORDER BY player_id"
    ).fetchall()
    assert rows[0][0] == "ATP_100001"
    assert rows[0][1] == "ATP"
    assert rows[0][2] == 100001
    assert rows[0][3] == "Old Player"
    assert rows[0][4] is None  # missing dob
    fritz = next(r for r in rows if r[0] == "ATP_126203")
    assert fritz[3] == "Taylor Fritz"


def test_load_players_is_idempotent(
    fresh_db_with_schema: duckdb.DuckDBPyConnection,
    fake_sackmann_root: Path,
) -> None:
    ingest_sackmann.load_players(fresh_db_with_schema, "ATP")
    second = ingest_sackmann.load_players(fresh_db_with_schema, "ATP")
    assert second == 0
    total = fresh_db_with_schema.execute("SELECT COUNT(*) FROM players").fetchone()
    assert total is not None and total[0] == 3


def test_load_matches_derives_status_and_composite_ids(
    fresh_db_with_schema: duckdb.DuckDBPyConnection,
    fake_sackmann_root: Path,
) -> None:
    inserted = ingest_sackmann.load_matches(fresh_db_with_schema, "ATP", "main")
    assert inserted == 2
    rows = fresh_db_with_schema.execute(
        "SELECT match_id, source, tour, match_tier, match_status, "
        "winner_player_id, loser_player_id, tourney_date, w_first_in, w_bp_saved "
        "FROM matches ORDER BY match_num"
    ).fetchall()

    completed = next(r for r in rows if r[4] == "completed")
    assert completed[0] == "sackmann::ATP-2023-9900-300"
    assert completed[1] == "sackmann"
    assert completed[2] == "ATP"
    assert completed[3] == "main"
    assert completed[5] == "ATP_126203"
    assert completed[6] == "ATP_126610"
    assert str(completed[7]) == "2023-01-02"
    assert completed[8] == 52  # w_1stIn -> w_first_in
    assert completed[9] == 0  # w_bpSaved -> w_bp_saved (the test row has bpSaved=0)

    ret_match = next(r for r in rows if "RET" in r[4])
    assert ret_match[4] == "RET"


def test_load_matches_is_idempotent(
    fresh_db_with_schema: duckdb.DuckDBPyConnection,
    fake_sackmann_root: Path,
) -> None:
    ingest_sackmann.load_matches(fresh_db_with_schema, "ATP", "main")
    second = ingest_sackmann.load_matches(fresh_db_with_schema, "ATP", "main")
    assert second == 0


def test_load_rankings(
    fresh_db_with_schema: duckdb.DuckDBPyConnection,
    fake_sackmann_root: Path,
) -> None:
    inserted = ingest_sackmann.load_rankings(fresh_db_with_schema, "ATP")
    assert inserted == 2
    rows = fresh_db_with_schema.execute(
        "SELECT ranking_date, player_id, rank, points FROM rankings ORDER BY rank"
    ).fetchall()
    assert str(rows[0][0]) == "2023-01-02"
    assert rows[0][1] == "ATP_126203"
    assert rows[0][2] == 1
    assert rows[0][3] == 5000


def test_invalid_tour_raises() -> None:
    with pytest.raises(ValueError, match="tour must be"):
        ingest_sackmann._validate_tour("MIXED")


def test_invalid_tier_raises() -> None:
    with pytest.raises(ValueError, match="tier for ATP must be"):
        ingest_sackmann._validate_tier("ATP", "qualifying")


def test_available_tiers_differ_by_tour() -> None:
    assert ingest_sackmann.available_tiers("ATP") == ["futures", "main", "qual_chall"]
    assert ingest_sackmann.available_tiers("WTA") == ["main", "qual_itf"]
