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
    _elo_logistic,
    get_head_to_head,
    get_player_ranking,
    get_player_stats,
    get_recent_form,
    get_surface_elo,
)
from tennis_predictor.llm.tools.schemas import (
    GetHeadToHeadInput,
    GetPlayerRankingInput,
    GetPlayerStatsInput,
    GetRecentFormInput,
    GetSurfaceEloInput,
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

    # 2025-07-14 is the newest, asked as-of 2026-01-01 → ~171 days stale.
    assert out.latest_match_date == date(2025, 7, 14)
    assert out.data_freshness_warning is not None
    assert "2025-07-14" in out.data_freshness_warning


def test_get_recent_form_no_freshness_warning_when_recent(seeded_db) -> None:
    """Asked one day after the newest match — newest is 2025-07-14, asked
    as-of 2025-07-15. Gap is 1 day, well within the 7-day Sackmann lag."""
    out = get_recent_form(
        seeded_db,
        GetRecentFormInput(
            player_name="Carlos Alcaraz",
            tour="ATP",
            as_of_date=date(2025, 7, 15),
            n_matches=5,
        ),
    )
    assert out.latest_match_date == date(2025, 7, 14)
    assert out.data_freshness_warning is None


def test_get_recent_form_empty_player_has_no_freshness_warning(seeded_db) -> None:
    """Querying for matches strictly before the earliest stored one returns
    zero matches → no freshness warning (no anchor to compare against)."""
    out = get_recent_form(
        seeded_db,
        GetRecentFormInput(
            player_name="Carlos Alcaraz",
            tour="ATP",
            as_of_date=date(2020, 1, 1),
            n_matches=5,
        ),
    )
    assert out.n_returned == 0
    assert out.latest_match_date is None
    assert out.data_freshness_warning is None


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


# ---------------------------------------------------------------------------
# Phase 6.1: get_surface_elo
# ---------------------------------------------------------------------------


def _seed_surface_elo(conn: duckdb.DuckDBPyConnection) -> None:
    # PLAYER_A_ID = Alcaraz, PLAYER_B_ID = Sinner per the fixture above.
    conn.executemany(
        "INSERT INTO elo_state (player_id, surface, rating, matches_played, last_updated_date) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (PLAYER_B_ID, "Clay", 2000.0, 50, date(2026, 5, 10)),  # Sinner
            (PLAYER_A_ID, "Clay", 1800.0, 40, date(2026, 5, 9)),  # Alcaraz
            (PLAYER_B_ID, "Hard", 1900.0, 60, date(2026, 4, 1)),  # Sinner
            # Alcaraz has no Hard row — exercises the 1500 default below.
        ],
    )


def test_get_surface_elo_returns_both_players_and_diff(seeded_db) -> None:
    _seed_surface_elo(seeded_db)
    out = get_surface_elo(
        seeded_db,
        GetSurfaceEloInput(
            player_a_name="Jannik Sinner",
            player_b_name="Carlos Alcaraz",
            tour="ATP",
            surface="Clay",
            as_of_date=date(2026, 5, 24),
        ),
    )
    assert out.player_a_elo == 2000.0
    assert out.player_b_elo == 1800.0
    assert out.diff_a_minus_b == 200.0
    # Logistic of +200 ≈ 0.76
    assert abs(out.baseline_prob_a - _elo_logistic(200.0)) < 1e-9
    assert out.elo_state_snapshot_date == date(2026, 5, 10)


def test_get_surface_elo_uses_1500_default_when_no_row(seeded_db) -> None:
    """Player without an `elo_state` row on the queried surface gets the
    same 1500 prior `EloState` uses for first appearance."""
    _seed_surface_elo(seeded_db)
    out = get_surface_elo(
        seeded_db,
        GetSurfaceEloInput(
            player_a_name="Jannik Sinner",
            player_b_name="Carlos Alcaraz",
            tour="ATP",
            surface="Hard",
            as_of_date=date(2026, 5, 24),
        ),
    )
    assert out.player_a_elo == 1900.0  # Sinner has Hard row
    assert out.player_b_elo == 1500.0  # Alcaraz missing — default
    assert out.diff_a_minus_b == 400.0


def test_get_surface_elo_refuses_self_match(seeded_db) -> None:
    with pytest.raises(PlayerResolutionError, match="self-match"):
        get_surface_elo(
            seeded_db,
            GetSurfaceEloInput(
                player_a_name="Jannik Sinner",
                player_b_name="Jannik Sinner",
                tour="ATP",
                surface="Clay",
                as_of_date=date(2026, 5, 24),
            ),
        )
