"""Tests for the ranking lookup."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import schema
from tennis_predictor.features.ranking import RankingLookup


@pytest.fixture
def db_with_rankings(tmp_path: Path):
    db = tmp_path / "ranks.duckdb"
    conn = duckdb.connect(str(db))
    schema.create_all_tables(conn)
    rows = [
        # Player A: 3 weekly snapshots
        ("ATP_A", date(2020, 1, 6), 10, 5000),
        ("ATP_A", date(2020, 1, 13), 8, 5200),
        ("ATP_A", date(2020, 1, 20), 7, 5400),
        # Player B: single snapshot
        ("ATP_B", date(2020, 1, 13), 50, 1000),
        # Player C: late entrant
        ("ATP_C", date(2020, 2, 3), 150, 400),
    ]
    conn.executemany(
        "INSERT INTO rankings (player_id, ranking_date, rank, points) VALUES (?, ?, ?, ?)",
        rows,
    )
    yield conn
    conn.close()


def test_empty_lookup_returns_sentinel() -> None:
    lookup = RankingLookup()
    assert lookup.get("ATP_X", date(2020, 1, 1)) == RankingLookup.SENTINEL_UNRANKED


def test_sentinel_value_matches_pydantic_bound() -> None:
    """The FeatureVector schema bounds rank_p1/p2 to `le=9999`. The sentinel
    must match — otherwise validation fails on unranked players."""
    assert RankingLookup.SENTINEL_UNRANKED == 9999


def test_get_returns_most_recent_rank(db_with_rankings: duckdb.DuckDBPyConnection) -> None:
    lookup = RankingLookup.from_db(db_with_rankings)
    # On 2020-01-15, A's most recent snapshot is 2020-01-13 with rank 8.
    assert lookup.get("ATP_A", date(2020, 1, 15)) == 8
    # On 2020-01-25, A's most recent snapshot is 2020-01-20 with rank 7.
    assert lookup.get("ATP_A", date(2020, 1, 25)) == 7


def test_get_returns_rank_when_date_matches_snapshot(
    db_with_rankings: duckdb.DuckDBPyConnection,
) -> None:
    """as_of_date == ranking_date must return that snapshot's rank."""
    lookup = RankingLookup.from_db(db_with_rankings)
    assert lookup.get("ATP_A", date(2020, 1, 13)) == 8


def test_get_returns_sentinel_when_as_of_date_predates_first_ranking(
    db_with_rankings: duckdb.DuckDBPyConnection,
) -> None:
    """Player A's first snapshot is 2020-01-06; asking on 2020-01-05
    must return the sentinel (player not yet ranked)."""
    lookup = RankingLookup.from_db(db_with_rankings)
    assert lookup.get("ATP_A", date(2020, 1, 5)) == RankingLookup.SENTINEL_UNRANKED


def test_get_returns_sentinel_for_unseen_player(
    db_with_rankings: duckdb.DuckDBPyConnection,
) -> None:
    lookup = RankingLookup.from_db(db_with_rankings)
    assert lookup.get("ATP_Z", date(2025, 6, 1)) == RankingLookup.SENTINEL_UNRANKED


def test_get_returns_latest_after_all_snapshots(
    db_with_rankings: duckdb.DuckDBPyConnection,
) -> None:
    """A long time after the last snapshot, we still return the last known
    rank — there is no "expiry"; that is for upstream policy if needed."""
    lookup = RankingLookup.from_db(db_with_rankings)
    assert lookup.get("ATP_A", date(2025, 12, 31)) == 7


def test_different_players_independent(db_with_rankings: duckdb.DuckDBPyConnection) -> None:
    lookup = RankingLookup.from_db(db_with_rankings)
    assert lookup.get("ATP_A", date(2020, 1, 20)) == 7
    assert lookup.get("ATP_B", date(2020, 1, 20)) == 50
    assert lookup.get("ATP_C", date(2020, 1, 20)) == RankingLookup.SENTINEL_UNRANKED
    assert lookup.get("ATP_C", date(2020, 2, 3)) == 150


def test_has_ranking_history(db_with_rankings: duckdb.DuckDBPyConnection) -> None:
    lookup = RankingLookup.from_db(db_with_rankings)
    assert lookup.has_ranking_history("ATP_A") is True
    assert lookup.has_ranking_history("ATP_B") is True
    assert lookup.has_ranking_history("ATP_NOT_FOUND") is False


def test_total_rows_loaded(db_with_rankings: duckdb.DuckDBPyConnection) -> None:
    lookup = RankingLookup.from_db(db_with_rankings)
    assert len(lookup) == 5  # 3 + 1 + 1 from the fixture
