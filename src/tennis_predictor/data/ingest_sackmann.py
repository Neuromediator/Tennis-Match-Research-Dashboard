"""Sackmann CSV -> DuckDB ingestion.

Three entry points: load_players, load_matches, load_rankings.
All idempotent: ON CONFLICT DO NOTHING on the documented primary keys, so
re-running the script never produces duplicates.

Transformations applied at ingest:
- Sackmann's integer YYYYMMDD dates -> DuckDB DATE.
- Composite player_id: tour prefix + sackmann integer ID (e.g. ATP_104925).
- match_id: 'sackmann::' + match_external_id where
  match_external_id = '{tour}-{tourney_id}-{match_num}'.
- match_status derived from `score` string ('completed' / 'RET' / 'W/O' / 'DEF').
- Sackmann's CamelCase stat columns (w_1stIn) renamed to snake_case (w_first_in).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import duckdb

from tennis_predictor import config

Tour = Literal["ATP", "WTA"]
# Tier names are per-tour (see _TIER_GLOBS_BY_TOUR below). They are stored
# verbatim in matches.match_tier so the original source is recoverable.
Tier = Literal["main", "qual_chall", "futures", "qual_itf"]

VALID_TOURS: frozenset[str] = frozenset({"ATP", "WTA"})

# Per-tour glob patterns. Tier names differ across tours because Sackmann
# organizes the WTA lower-tier files differently (a single qual+ITF stream)
# than the ATP files (qual+Challenger + separate Futures). The "main" tier
# is the tour-level singles file in both cases.
#
# Patterns are anchored on a digit after the prefix so "main" does not
# accidentally pick up `*_qual_chall_*`, `*_futures_*`, `*_qual_itf_*`,
# `*_doubles_*`, or `*_amateur.csv` — those have different column layouts.
_TIER_GLOBS_BY_TOUR: dict[Tour, dict[str, str]] = {
    "ATP": {
        "main": "atp_matches_[0-9]*.csv",
        "qual_chall": "atp_matches_qual_chall_[0-9]*.csv",
        "futures": "atp_matches_futures_[0-9]*.csv",
    },
    "WTA": {
        "main": "wta_matches_[0-9]*.csv",
        "qual_itf": "wta_matches_qual_itf_[0-9]*.csv",
    },
}


def _sackmann_dir(tour: Tour) -> Path:
    return config.RAW_DIR / ("tennis_atp" if tour == "ATP" else "tennis_wta")


def _file_prefix(tour: Tour) -> str:
    return "atp" if tour == "ATP" else "wta"


def _validate_tour(tour: str) -> None:
    if tour not in VALID_TOURS:
        raise ValueError(f"tour must be one of {sorted(VALID_TOURS)}; got {tour!r}")


def _validate_tier(tour: Tour, tier: str) -> None:
    valid = sorted(_TIER_GLOBS_BY_TOUR[tour].keys())
    if tier not in valid:
        raise ValueError(f"tier for {tour} must be one of {valid}; got {tier!r}")


def available_tiers(tour: Tour) -> list[str]:
    """Return the tier names available for a given tour."""
    _validate_tour(tour)
    return sorted(_TIER_GLOBS_BY_TOUR[tour].keys())


# ---------------------------------------------------------------------------
# Players


def load_players(conn: duckdb.DuckDBPyConnection, tour: Tour) -> int:
    """Load all players for a tour from atp_players.csv or wta_players.csv.

    Returns the number of rows newly inserted (excludes pre-existing rows).
    """
    _validate_tour(tour)
    csv_path = _sackmann_dir(tour) / f"{_file_prefix(tour)}_players.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    before = _row_count(conn, "players")
    conn.execute(
        f"""
        INSERT INTO players BY NAME
        SELECT
            '{tour}_' || CAST(player_id AS VARCHAR) AS player_id,
            '{tour}' AS tour,
            CAST(player_id AS INTEGER) AS sackmann_id,
            name_first,
            name_last,
            TRIM(COALESCE(name_first, '') || ' ' || COALESCE(name_last, '')) AS full_name,
            hand,
            TRY_CAST(try_strptime(CAST(dob AS VARCHAR), '%Y%m%d') AS DATE) AS dob,
            ioc,
            TRY_CAST(height AS INTEGER) AS height,
            wikidata_id
        FROM read_csv_auto(?, header=True)
        ON CONFLICT (player_id) DO NOTHING
        """,
        [str(csv_path)],
    )
    return _row_count(conn, "players") - before


# ---------------------------------------------------------------------------
# Matches


def load_matches(conn: duckdb.DuckDBPyConnection, tour: Tour, tier: Tier) -> int:
    """Load every match CSV for the given (tour, tier).

    Iterates over all per-year files; idempotent via ON CONFLICT.
    Returns the total number of rows newly inserted.
    """
    _validate_tour(tour)
    _validate_tier(tour, tier)

    glob_pat = _TIER_GLOBS_BY_TOUR[tour][tier]
    files = sorted(_sackmann_dir(tour).glob(glob_pat))

    total_inserted = 0
    for path in files:
        total_inserted += _ingest_match_file(conn, path, tour, tier)
    return total_inserted


def _ingest_match_file(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
    tour: Tour,
    tier: Tier,
) -> int:
    before = _row_count(conn, "matches")
    # Use a subquery alias so SELECT aliases don't collide with source
    # column names. (DuckDB resolves SELECT-clause aliases eagerly, which
    # otherwise causes "column referenced before defined" errors when an
    # alias shares a name with the underlying column.)
    sql = f"""
    INSERT INTO matches BY NAME
    SELECT
        'sackmann::{tour}-' || raw.tourney_id || '-' || raw.match_num AS match_id,
        'sackmann' AS source,
        '{tour}-' || raw.tourney_id || '-' || raw.match_num AS match_external_id,
        '{tour}' AS tour,
        '{tier}' AS match_tier,
        raw.tourney_id AS tourney_id,
        raw.tourney_name AS tourney_name,
        raw.tourney_level AS tourney_level,
        CAST(try_strptime(raw.tourney_date, '%Y%m%d') AS DATE) AS tourney_date,
        raw.surface AS surface,
        TRY_CAST(raw.draw_size AS INTEGER) AS draw_size,
        CAST(raw.match_num AS INTEGER) AS match_num,
        raw.round AS round,
        TRY_CAST(raw.best_of AS INTEGER) AS best_of,
        TRY_CAST(raw.minutes AS INTEGER) AS minutes,
        raw.score AS score,
        CASE
            WHEN raw.score IS NULL OR raw.score = '' THEN 'unknown'
            WHEN raw.score LIKE '%RET%' THEN 'RET'
            WHEN raw.score LIKE '%W/O%' THEN 'W/O'
            WHEN raw.score LIKE '%DEF%' THEN 'DEF'
            ELSE 'completed'
        END AS match_status,
        '{tour}_' || raw.winner_id AS winner_player_id,
        '{tour}_' || raw.loser_id AS loser_player_id,
        raw.winner_seed AS winner_seed,
        raw.winner_entry AS winner_entry,
        TRY_CAST(raw.winner_rank AS INTEGER) AS winner_rank,
        TRY_CAST(raw.winner_rank_points AS INTEGER) AS winner_rank_points,
        TRY_CAST(raw.winner_age AS DOUBLE) AS winner_age,
        raw.loser_seed AS loser_seed,
        raw.loser_entry AS loser_entry,
        TRY_CAST(raw.loser_rank AS INTEGER) AS loser_rank,
        TRY_CAST(raw.loser_rank_points AS INTEGER) AS loser_rank_points,
        TRY_CAST(raw.loser_age AS DOUBLE) AS loser_age,
        TRY_CAST(raw.w_ace AS INTEGER) AS w_ace,
        TRY_CAST(raw.w_df AS INTEGER) AS w_df,
        TRY_CAST(raw.w_svpt AS INTEGER) AS w_svpt,
        TRY_CAST(raw.w_1stIn AS INTEGER) AS w_first_in,
        TRY_CAST(raw.w_1stWon AS INTEGER) AS w_first_won,
        TRY_CAST(raw.w_2ndWon AS INTEGER) AS w_second_won,
        TRY_CAST(raw.w_SvGms AS INTEGER) AS w_sv_gms,
        TRY_CAST(raw.w_bpSaved AS INTEGER) AS w_bp_saved,
        TRY_CAST(raw.w_bpFaced AS INTEGER) AS w_bp_faced,
        TRY_CAST(raw.l_ace AS INTEGER) AS l_ace,
        TRY_CAST(raw.l_df AS INTEGER) AS l_df,
        TRY_CAST(raw.l_svpt AS INTEGER) AS l_svpt,
        TRY_CAST(raw.l_1stIn AS INTEGER) AS l_first_in,
        TRY_CAST(raw.l_1stWon AS INTEGER) AS l_first_won,
        TRY_CAST(raw.l_2ndWon AS INTEGER) AS l_second_won,
        TRY_CAST(raw.l_SvGms AS INTEGER) AS l_sv_gms,
        TRY_CAST(raw.l_bpSaved AS INTEGER) AS l_bp_saved,
        TRY_CAST(raw.l_bpFaced AS INTEGER) AS l_bp_faced
    FROM read_csv_auto(?, header=True, all_varchar=True) AS raw
    ON CONFLICT (match_id) DO NOTHING
    """
    conn.execute(sql, [str(csv_path)])
    return _row_count(conn, "matches") - before


# ---------------------------------------------------------------------------
# Rankings


def load_rankings(conn: duckdb.DuckDBPyConnection, tour: Tour) -> int:
    """Load every per-decade rankings file plus the current file for a tour."""
    _validate_tour(tour)
    prefix = _file_prefix(tour)
    files = sorted(_sackmann_dir(tour).glob(f"{prefix}_rankings_*.csv"))

    total_inserted = 0
    for path in files:
        total_inserted += _ingest_rankings_file(conn, path, tour)
    return total_inserted


def _ingest_rankings_file(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
    tour: Tour,
) -> int:
    before = _row_count(conn, "rankings")
    conn.execute(
        f"""
        INSERT INTO rankings BY NAME
        SELECT
            CAST(try_strptime(CAST(ranking_date AS VARCHAR), '%Y%m%d') AS DATE) AS ranking_date,
            '{tour}_' || CAST(player AS VARCHAR) AS player_id,
            CAST("rank" AS INTEGER) AS "rank",
            TRY_CAST(points AS INTEGER) AS points
        FROM read_csv_auto(?, header=True, all_varchar=True)
        ON CONFLICT (ranking_date, player_id) DO NOTHING
        """,
        [str(csv_path)],
    )
    return _row_count(conn, "rankings") - before


# ---------------------------------------------------------------------------
# Helpers


def _row_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row is not None else 0
