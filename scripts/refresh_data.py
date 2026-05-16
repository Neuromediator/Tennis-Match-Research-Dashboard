"""Cold-layer refresh orchestration.

One command, refresh everything that phase 1 owns:

    uv run python scripts/refresh_data.py            # incremental
    uv run python scripts/refresh_data.py --clean    # full rebuild

Default behavior is **incremental**: existing rows are kept and only new
matches/rankings/aliases are inserted (via ON CONFLICT DO NOTHING).
Sackmann updates its current-year CSV weekly, tennis-data.co.uk updates
its xlsx files daily — incremental picks the new data in minutes.

Pass --clean to delete the DuckDB file and rebuild from scratch. Use that
when the schema changed, or after fixing an ingestion bug whose effect
needs to be unwound.

Steps:
1. git submodule update (unless --skip-submodules)
2. Open DuckDB at config.DUCKDB_PATH, ensure schema
3. For each tour: load_players, load_matches (all tiers), load_rankings,
   seed_aliases_from_players
4. For each tour: download + load tennis-data.co.uk archives for the year
   range (default 2001-current). Failures per-year are logged, not fatal.
5. Print coverage summary

Every step is idempotent in either mode.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
import urllib.error
from typing import cast

import duckdb

from tennis_predictor import config
from tennis_predictor.data import db, ingest_sackmann, load_market, reconcile, schema

DEFAULT_MARKET_YEAR_START = 2001
DEFAULT_MARKET_YEAR_END = dt.date.today().year


def update_submodules() -> None:
    """Pull latest commit on each Sackmann submodule."""
    print("[1/5] Updating Sackmann submodules...")
    subprocess.run(
        ["git", "submodule", "update", "--init", "--remote"],
        check=True,
    )


def load_sackmann_for_tour(conn: duckdb.DuckDBPyConnection, tour: ingest_sackmann.Tour) -> None:
    print(f"\n=== {tour}: Sackmann ===")
    t0 = time.time()
    n_players = ingest_sackmann.load_players(conn, tour)
    print(f"  players:        {n_players:>8d} new   ({time.time() - t0:.1f}s)")

    for tier in ingest_sackmann.available_tiers(tour):
        t1 = time.time()
        n_matches = ingest_sackmann.load_matches(
            conn,
            tour,
            cast(ingest_sackmann.Tier, tier),
        )
        print(f"  matches/{tier:<10s}: {n_matches:>8d} new   ({time.time() - t1:.1f}s)")

    t1 = time.time()
    n_rankings = ingest_sackmann.load_rankings(conn, tour)
    print(f"  rankings:       {n_rankings:>8d} new   ({time.time() - t1:.1f}s)")

    t1 = time.time()
    n_aliases = reconcile.seed_aliases_from_players(conn, tour)
    print(f"  aliases:        {n_aliases:>8d} new   ({time.time() - t1:.1f}s)")


def load_market_for_tour(
    conn: duckdb.DuckDBPyConnection,
    tour: ingest_sackmann.Tour,
    year_start: int,
    year_end: int,
) -> None:
    print(f"\n=== {tour}: tennis-data.co.uk ({year_start}-{year_end}) ===")
    idx = reconcile.AliasIndex(conn, tour)

    totals = load_market.LoadStats()
    for year in range(year_start, year_end + 1):
        t0 = time.time()
        try:
            path = load_market.download_archive(year, tour)
        except urllib.error.HTTPError as e:
            print(f"  {year}: HTTP {e.code} (missing for this tour) - skipping")
            continue
        except urllib.error.URLError as e:
            print(f"  {year}: network error {e.reason} - skipping")
            continue
        except ValueError as e:
            print(f"  {year}: {e} - skipping")
            continue

        try:
            stats = load_market.load_market_file(conn, path, tour, idx)
        except Exception as e:
            print(f"  {year}: parse failed ({type(e).__name__}: {e}) - skipping")
            continue
        matched = stats.loaded + stats.review
        total = stats.total()
        rate = matched * 100 / max(total, 1)
        print(
            f"  {year}: loaded={stats.loaded:>4d}  review={stats.review:>3d}  "
            f"unmatched={stats.unmatched:>4d}  total={total:>4d}  "
            f"rate={rate:>5.1f}%  ({time.time() - t0:.1f}s)"
        )
        totals.loaded += stats.loaded
        totals.review += stats.review
        totals.unmatched += stats.unmatched
        totals.no_odds += stats.no_odds
        totals.skipped += stats.skipped
        for src, n in stats.by_odds_source.items():
            totals.by_odds_source[src] = totals.by_odds_source.get(src, 0) + n

    print(
        f"  TOTAL {tour}: loaded={totals.loaded}  review={totals.review}  "
        f"unmatched={totals.unmatched}  no_odds={totals.no_odds}  "
        f"by_source={dict(totals.by_odds_source)}"
    )


def print_coverage(conn: duckdb.DuckDBPyConnection) -> None:
    print("\n=== Coverage report ===")

    print("\nRows per table:")
    for table in (
        "players",
        "matches",
        "rankings",
        "player_aliases",
        "market_implied_probabilities",
    ):
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        n = int(row[0]) if row else 0
        print(f"  {table:>32s}: {n:>10,d}")

    print("\nMatches by tour x tier:")
    rows = conn.execute(
        "SELECT tour, match_tier, COUNT(*) "
        "FROM matches GROUP BY tour, match_tier ORDER BY tour, match_tier"
    ).fetchall()
    for r in rows:
        print(f"  {r[0]} {r[1]:>12s}: {r[2]:>10,d}")

    print("\nMatch_status distribution (main tier):")
    rows = conn.execute(
        "SELECT match_status, COUNT(*) FROM matches WHERE match_tier='main' "
        "GROUP BY match_status ORDER BY 2 DESC"
    ).fetchall()
    for r in rows:
        print(f"  {r[0]:>12s}: {r[1]:>10,d}")

    print("\nMarket coverage of ATP/WTA main matches (2001+):")
    rows = conn.execute(
        """
        SELECT
            m.tour,
            EXTRACT(YEAR FROM m.tourney_date) AS year,
            COUNT(DISTINCT m.match_id) AS total,
            COUNT(DISTINCT mp.match_id) AS with_market,
            ROUND(100.0 * COUNT(DISTINCT mp.match_id) /
                  NULLIF(COUNT(DISTINCT m.match_id), 0), 1) AS pct
        FROM matches m
        LEFT JOIN market_implied_probabilities mp ON m.match_id = mp.match_id
        WHERE m.match_tier = 'main'
          AND EXTRACT(YEAR FROM m.tourney_date) >= 2001
        GROUP BY m.tour, EXTRACT(YEAR FROM m.tourney_date)
        ORDER BY m.tour, year
        """
    ).fetchall()
    print("  tour  year   total  market   pct")
    for r in rows:
        print(f"  {r[0]:>4s}  {int(r[1]):>4d}  {int(r[2]):>6d}  {int(r[3]):>6d}  {r[4] or 0:>5}%")

    namesakes_atp = len(reconcile.find_namesakes(conn, "ATP"))
    namesakes_wta = len(reconcile.find_namesakes(conn, "WTA"))
    print(f"\nSame-full-name players: ATP={namesakes_atp}  WTA={namesakes_wta}")

    aliases_review = config.PROCESSED_DIR / "aliases_review.csv"
    if aliases_review.exists():
        lines = sum(1 for _ in aliases_review.open()) - 1  # minus header
        print(f"\naliases_review.csv has {lines:,d} rows queued for manual review")

    unmatched = config.PROCESSED_DIR / "unmatched_market_rows.csv"
    if unmatched.exists():
        lines = sum(1 for _ in unmatched.open()) - 1
        print(f"unmatched_market_rows.csv has {lines:,d} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-submodules",
        action="store_true",
        help="Don't `git submodule update` (useful when offline or for re-runs)",
    )
    parser.add_argument(
        "--skip-market",
        action="store_true",
        help="Don't download or load tennis-data.co.uk archives",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete the DuckDB file before rebuilding (full reset). Use this "
            "when the schema changes or an ingestion bug needs the data re-derived. "
            "Default behavior is incremental: existing rows are kept, only new "
            "ones added via ON CONFLICT DO NOTHING."
        ),
    )
    parser.add_argument(
        "--tours",
        nargs="+",
        choices=["ATP", "WTA"],
        default=["ATP", "WTA"],
        help="Which tours to refresh",
    )
    parser.add_argument(
        "--market-years",
        type=int,
        nargs=2,
        default=[DEFAULT_MARKET_YEAR_START, DEFAULT_MARKET_YEAR_END],
        metavar=("START", "END"),
        help=f"Inclusive year range for tennis-data.co.uk archives "
        f"(default {DEFAULT_MARKET_YEAR_START}-{DEFAULT_MARKET_YEAR_END})",
    )
    args = parser.parse_args()

    print(f"DuckDB target: {config.DUCKDB_PATH}")

    if args.clean and config.DUCKDB_PATH.exists():
        print(f"--clean: removing {config.DUCKDB_PATH}")
        config.DUCKDB_PATH.unlink()

    if not args.skip_submodules:
        update_submodules()
    else:
        print("[1/5] Skipping submodule update")

    print("\n[2/5] Opening DuckDB and ensuring schema")
    conn = db.open_connection()
    schema.create_all_tables(conn)

    print("\n[3/5] Loading Sackmann data")
    for tour in args.tours:
        load_sackmann_for_tour(conn, cast(ingest_sackmann.Tour, tour))

    if not args.skip_market:
        print("\n[4/5] Loading tennis-data.co.uk archives")
        year_start, year_end = args.market_years
        for tour in args.tours:
            load_market_for_tour(conn, cast(ingest_sackmann.Tour, tour), year_start, year_end)
    else:
        print("\n[4/5] Skipping market data")

    print("\n[5/5] Coverage")
    print_coverage(conn)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
