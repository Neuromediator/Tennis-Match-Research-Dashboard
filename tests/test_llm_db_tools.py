"""Unit tests for the DB-backed LLM tools.

A small synthetic DuckDB is seeded with two players, three matches, and
a couple of ranking rows. Every tool is exercised end-to-end — alias
lookup → DuckDB query → Pydantic response.

Player resolution is the highest-risk area because `AliasIndex` runs
fuzzy matching; the tests pin both the happy path (canonical name) and
the failure path (unknown name → `PlayerResolutionError`).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.data.reconcile import seed_aliases_from_players
from tennis_predictor.llm.tools.db_tools import (
    get_head_to_head,
    get_player_ranking,
    get_player_stats,
    get_recent_form,
)
from tennis_predictor.llm.tools.schemas import (
    GetHeadToHeadInput,
    GetPlayerRankingInput,
    GetPlayerStatsInput,
    GetRecentFormInput,
    PlayerResolutionError,
)

PLAYER_A_ID = "ATP_900001"
PLAYER_B_ID = "ATP_900002"


@pytest.fixture
def seeded_db(tmp_path: Path):
    conn = duckdb.connect(str(tmp_path / "llm_db_tools.duckdb"))
    schema.create_all_tables(conn)
    _seed_players(conn)
    _seed_matches(conn)
    _seed_rankings(conn)
    seed_aliases_from_players(conn, "ATP")
    yield conn
    conn.close()


def _seed_players(conn: duckdb.DuckDBPyConnection) -> None:
    conn.executemany(
        """
        INSERT INTO players (
            player_id, tour, sackmann_id, name_first, name_last, full_name, hand, dob, ioc, height
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                PLAYER_A_ID,
                "ATP",
                900001,
                "Carlos",
                "Alcaraz",
                "Carlos Alcaraz",
                "R",
                date(2003, 5, 5),
                "ESP",
                183,
            ),
            (
                PLAYER_B_ID,
                "ATP",
                900002,
                "Jannik",
                "Sinner",
                "Jannik Sinner",
                "R",
                date(2001, 8, 16),
                "ITA",
                188,
            ),
        ],
    )


def _seed_matches(conn: duckdb.DuckDBPyConnection) -> None:
    rows = [
        # Alcaraz beats Sinner on clay, 2024
        (
            "stub::M1",
            "stub",
            "M1",
            "ATP",
            "main",
            "stub-tourney-1",
            "Madrid",
            "M",
            date(2024, 5, 8),
            "Clay",
            1,
            "F",
            3,
            "completed",
            PLAYER_A_ID,
            PLAYER_B_ID,
            "6-4 6-3",
        ),
        # Sinner beats Alcaraz on hard, 2025
        (
            "stub::M2",
            "stub",
            "M2",
            "ATP",
            "main",
            "stub-tourney-2",
            "Indian Wells",
            "M",
            date(2025, 3, 15),
            "Hard",
            2,
            "F",
            3,
            "completed",
            PLAYER_B_ID,
            PLAYER_A_ID,
            "7-6 6-1",
        ),
        # Alcaraz beats Sinner on grass, 2025
        (
            "stub::M3",
            "stub",
            "M3",
            "ATP",
            "main",
            "stub-tourney-3",
            "Wimbledon",
            "G",
            date(2025, 7, 14),
            "Grass",
            3,
            "F",
            5,
            "completed",
            PLAYER_A_ID,
            PLAYER_B_ID,
            "7-5 6-2 6-4",
        ),
    ]
    conn.executemany(
        """
        INSERT INTO matches (
            match_id, source, match_external_id, tour, match_tier,
            tourney_id, tourney_name, tourney_level, tourney_date, surface,
            match_num, round, best_of, match_status,
            winner_player_id, loser_player_id, score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _seed_rankings(conn: duckdb.DuckDBPyConnection) -> None:
    conn.executemany(
        "INSERT INTO rankings (ranking_date, player_id, rank, points) VALUES (?, ?, ?, ?)",
        [
            (date(2025, 1, 6), PLAYER_A_ID, 1, 11000),
            (date(2025, 7, 7), PLAYER_A_ID, 2, 9000),
            (date(2025, 1, 6), PLAYER_B_ID, 2, 10000),
            (date(2025, 7, 7), PLAYER_B_ID, 1, 12000),
        ],
    )


# ---------------------------------------------------------------------------


def test_get_player_stats_returns_career_and_per_surface_breakdown(seeded_db) -> None:
    out = get_player_stats(
        seeded_db,
        GetPlayerStatsInput(player_name="Carlos Alcaraz", tour="ATP", as_of_date=date(2026, 1, 1)),
    )
    assert out.canonical_player_id == PLAYER_A_ID
    assert out.career_matches == 3
    assert out.career_wins == 2
    assert out.career_losses == 1
    # Clay (1-0), Hard (0-1), Grass (1-0)
    assert out.surface_matches == {"Clay": 1, "Hard": 1, "Grass": 1}
    assert out.surface_win_pct == {"Clay": 1.0, "Hard": 0.0, "Grass": 1.0}


def test_get_player_stats_unknown_name_raises(seeded_db) -> None:
    with pytest.raises(PlayerResolutionError):
        get_player_stats(
            seeded_db,
            GetPlayerStatsInput(
                player_name="Definitely Not A Player",
                tour="ATP",
                as_of_date=date(2026, 1, 1),
            ),
        )


def test_get_head_to_head_aggregates_wins_and_lists_meetings(seeded_db) -> None:
    out = get_head_to_head(
        seeded_db,
        GetHeadToHeadInput(
            player_a_name="Carlos Alcaraz",
            player_b_name="Jannik Sinner",
            tour="ATP",
            as_of_date=date(2026, 1, 1),
        ),
    )
    assert out.player_a_wins == 2
    assert out.player_b_wins == 1
    assert len(out.matches) == 3
    # Returned oldest-first.
    assert out.matches[0].match_date == date(2024, 5, 8)
    assert out.matches[0].winner_name == "Carlos Alcaraz"


def test_get_head_to_head_respects_as_of_date(seeded_db) -> None:
    out = get_head_to_head(
        seeded_db,
        GetHeadToHeadInput(
            player_a_name="Carlos Alcaraz",
            player_b_name="Jannik Sinner",
            tour="ATP",
            as_of_date=date(2025, 1, 1),
        ),
    )
    # Only the 2024 Madrid meeting counts.
    assert len(out.matches) == 1
    assert out.player_a_wins == 1
    assert out.player_b_wins == 0


def test_get_recent_form_returns_newest_first_with_results(seeded_db) -> None:
    out = get_recent_form(
        seeded_db,
        GetRecentFormInput(
            player_name="Carlos Alcaraz",
            tour="ATP",
            as_of_date=date(2026, 1, 1),
            n_matches=5,
        ),
    )
    assert out.n_returned == 3
    assert out.wins == 2
    assert out.losses == 1
    assert [m.match_date for m in out.last_matches] == [
        date(2025, 7, 14),
        date(2025, 3, 15),
        date(2024, 5, 8),
    ]
    # The 2025-03-15 Indian Wells final was a loss to Sinner.
    middle = out.last_matches[1]
    assert middle.result == "L"
    assert middle.opponent_name == "Jannik Sinner"


def test_get_player_ranking_returns_most_recent_snapshot(seeded_db) -> None:
    out = get_player_ranking(
        seeded_db,
        GetPlayerRankingInput(
            player_name="Carlos Alcaraz", tour="ATP", as_of_date=date(2025, 12, 1)
        ),
    )
    assert out.rank == 2
    assert out.snapshot_date == date(2025, 7, 7)


def test_get_player_ranking_returns_null_when_no_history(seeded_db) -> None:
    out = get_player_ranking(
        seeded_db,
        GetPlayerRankingInput(
            player_name="Carlos Alcaraz", tour="ATP", as_of_date=date(2024, 1, 1)
        ),
    )
    assert out.rank is None
    assert out.snapshot_date is None
