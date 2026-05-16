"""Unit tests for the tennis-data.co.uk market-data loader.

Uses a tiny synthetic xlsx fixture so no network is hit. End-to-end
coverage against the live site happens in integration runs (see
scripts/refresh_data.py).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from tennis_predictor.data import load_market, reconcile, schema


def _write_market_xlsx(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, engine="openpyxl")


@pytest.fixture
def db_with_match(tmp_path: Path):
    """A DB pre-loaded with players, aliases, and one matches row."""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    schema.create_all_tables(conn)
    conn.executemany(
        """
        INSERT INTO players (
            player_id, tour, sackmann_id, name_first, name_last, full_name, hand
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("ATP_104925", "ATP", 104925, "Roger", "Federer", "Roger Federer", "R"),
            ("ATP_104745", "ATP", 104745, "Rafael", "Nadal", "Rafael Nadal", "L"),
        ],
    )
    reconcile.seed_aliases_from_players(conn, "ATP")
    conn.execute(
        """
        INSERT INTO matches (
            match_id, source, match_external_id, tour, match_tier,
            tourney_id, tourney_date, match_num, match_status,
            winner_player_id, loser_player_id, surface
        ) VALUES (
            'sackmann::ATP-2023-9001-1', 'sackmann', 'ATP-2023-9001-1',
            'ATP', 'main',
            '2023-9001', DATE '2023-06-04', 1, 'completed',
            'ATP_104925', 'ATP_104745', 'Grass'
        )
        """
    )
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# unit-level helpers


def test_normalize_overround_sums_to_one() -> None:
    p_w, p_l = load_market._normalize_overround(1.93, 1.95)
    assert p_w + p_l == pytest.approx(1.0)
    assert 0.45 < p_w < 0.55
    assert 0.45 < p_l < 0.55


def test_pick_best_odds_prefers_pinnacle() -> None:
    row: dict[str, object] = {
        "PSW": 1.50,
        "PSL": 2.50,
        "B365W": 1.45,
        "B365L": 2.60,
        "AvgW": 1.48,
        "AvgL": 2.55,
    }
    result = load_market._pick_best_odds(row)
    assert result is not None
    odds_w, odds_l, source = result
    assert source == "pinnacle"
    assert odds_w == 1.50
    assert odds_l == 2.50


def test_pick_best_odds_falls_through_when_pinnacle_missing() -> None:
    row: dict[str, object] = {
        "PSW": None,
        "PSL": None,
        "AvgW": 1.48,
        "AvgL": 2.55,
        "B365W": 1.45,
        "B365L": 2.60,
    }
    result = load_market._pick_best_odds(row)
    assert result is not None
    assert result[2] == "market_avg"


def test_pick_best_odds_returns_none_when_all_missing() -> None:
    row: dict[str, object] = {"PSW": None, "PSL": None}
    assert load_market._pick_best_odds(row) is None


def test_pick_best_odds_ignores_invalid_odds_under_one() -> None:
    # Odds below 1.0 are physically impossible decimal odds. Skip.
    row: dict[str, object] = {"PSW": 0.5, "PSL": 0.6, "AvgW": 1.5, "AvgL": 2.5}
    result = load_market._pick_best_odds(row)
    assert result is not None
    assert result[2] == "market_avg"


# ---------------------------------------------------------------------------
# end-to-end with synthetic xlsx


def test_load_market_file_inserts_matched_row(
    db_with_match: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    xlsx = tmp_path / "2023.xlsx"
    _write_market_xlsx(
        xlsx,
        [
            {
                "ATP": 1,
                "Location": "London",
                "Tournament": "Wimbledon",
                "Date": pd.Timestamp("2023-06-05"),
                "Series": "Grand Slam",
                "Court": "Outdoor",
                "Surface": "Grass",
                "Round": "1st Round",
                "Best of": 5,
                "Winner": "Federer R.",
                "Loser": "Nadal R.",
                "WRank": 1,
                "LRank": 2,
                "Wsets": 3,
                "Lsets": 0,
                "Comment": "Completed",
                "B365W": 1.45,
                "B365L": 2.60,
                "PSW": 1.50,
                "PSL": 2.50,
                "MaxW": 1.55,
                "MaxL": 2.65,
                "AvgW": 1.48,
                "AvgL": 2.55,
            }
        ],
    )
    idx = reconcile.AliasIndex(db_with_match, "ATP")
    stats = load_market.load_market_file(
        db_with_match,
        xlsx,
        "ATP",
        idx,
        unmatched_csv=tmp_path / "unmatched.csv",
        review_csv=tmp_path / "review.csv",
    )
    assert stats.loaded == 1
    assert stats.unmatched == 0
    assert stats.by_odds_source == {"pinnacle": 1}

    rows = db_with_match.execute(
        "SELECT match_id, odds_source, odds_winner_close, p_winner_close "
        "FROM market_implied_probabilities"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "sackmann::ATP-2023-9001-1"
    assert rows[0][1] == "pinnacle"
    assert rows[0][2] == 1.50
    # Implied prob for winner with odds 1.50 vs loser 2.50: ~0.625 before overround
    assert 0.55 < rows[0][3] < 0.65


def test_load_market_file_writes_unmatched_csv(
    db_with_match: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    xlsx = tmp_path / "2023.xlsx"
    _write_market_xlsx(
        xlsx,
        [
            {
                "Date": pd.Timestamp("2023-06-05"),
                "Winner": "Completely Unknown Person",
                "Loser": "Another Unknown",
                "PSW": 1.50,
                "PSL": 2.50,
            }
        ],
    )
    idx = reconcile.AliasIndex(db_with_match, "ATP")
    unmatched_csv = tmp_path / "unmatched.csv"
    stats = load_market.load_market_file(
        db_with_match,
        xlsx,
        "ATP",
        idx,
        unmatched_csv=unmatched_csv,
        review_csv=tmp_path / "review.csv",
    )
    assert stats.loaded == 0
    assert stats.unmatched == 1
    assert unmatched_csv.exists()
    content = unmatched_csv.read_text()
    assert "name_unknown" in content


def test_load_market_file_is_idempotent(
    db_with_match: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    xlsx = tmp_path / "2023.xlsx"
    _write_market_xlsx(
        xlsx,
        [
            {
                "Date": pd.Timestamp("2023-06-05"),
                "Winner": "Federer R.",
                "Loser": "Nadal R.",
                "PSW": 1.50,
                "PSL": 2.50,
            }
        ],
    )
    idx = reconcile.AliasIndex(db_with_match, "ATP")
    load_market.load_market_file(
        db_with_match,
        xlsx,
        "ATP",
        idx,
        unmatched_csv=tmp_path / "u.csv",
        review_csv=tmp_path / "r.csv",
    )
    load_market.load_market_file(
        db_with_match,
        xlsx,
        "ATP",
        idx,
        unmatched_csv=tmp_path / "u.csv",
        review_csv=tmp_path / "r.csv",
    )
    count = db_with_match.execute("SELECT COUNT(*) FROM market_implied_probabilities").fetchone()
    assert count is not None and count[0] == 1


def test_load_market_file_unknown_match_row(
    db_with_match: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Names resolve but the date is so far off that no matches row joins."""
    xlsx = tmp_path / "2023.xlsx"
    _write_market_xlsx(
        xlsx,
        [
            {
                "Date": pd.Timestamp("2022-01-01"),  # nowhere near the match in our DB
                "Winner": "Federer R.",
                "Loser": "Nadal R.",
                "PSW": 1.50,
                "PSL": 2.50,
            }
        ],
    )
    idx = reconcile.AliasIndex(db_with_match, "ATP")
    unmatched_csv = tmp_path / "unmatched.csv"
    stats = load_market.load_market_file(
        db_with_match,
        xlsx,
        "ATP",
        idx,
        unmatched_csv=unmatched_csv,
        review_csv=tmp_path / "review.csv",
    )
    assert stats.loaded == 0
    assert stats.unmatched == 1
    content = unmatched_csv.read_text()
    assert "no_match_row" in content


# ---------------------------------------------------------------------------
# URL composition


def test_download_archive_returns_existing_file_without_network(
    tmp_path: Path,
) -> None:
    # Pre-create the destination so download is a no-op
    dest = tmp_path / "2023.xlsx"
    dest.write_bytes(b"local")
    out = load_market.download_archive(2023, "ATP", dest_dir=tmp_path)
    assert out == dest
    assert dest.read_bytes() == b"local"  # not overwritten
