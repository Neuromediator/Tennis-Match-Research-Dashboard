"""tennis-data.co.uk historical archive loader.

Per-year Excel files from http://www.tennis-data.co.uk contain pre-match
closing odds from several books (Pinnacle, Bet365, market max/avg). For each
row we:

1. Resolve `Winner`/`Loser` names to canonical Sackmann player_ids via
   AliasIndex (rapidfuzz on `Last F.`/`Last First`/`First Last` seeds).
2. Look up the corresponding match row in our `matches` table (date ±3 days,
   both player_ids match, same tour).
3. Pick the best available odds source (Pinnacle preferred, then market
   average, then Bet365, then market max).
4. Normalize overround: `p_w = (1/odds_w) / ((1/odds_w) + (1/odds_l))`.
5. INSERT into `market_implied_probabilities`.

Rows that fail at any step are written to two reports under
`data/processed/`:
- `unmatched_market_rows.csv` — name resolution or match lookup failed.
- `aliases_review.csv` — fuzzy lookup returned 'review' status (low
  confidence or ambiguous), so a human should confirm before the row is
  treated as authoritative in a future run.

Coverage of ~85-95% is realistic and acceptable for the calibration
benchmark. We do not chase 100%.
"""

from __future__ import annotations

import csv
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from tennis_predictor import config
from tennis_predictor.data.ingest_sackmann import Tour, _validate_tour
from tennis_predictor.data.reconcile import AliasIndex, ReconciliationResult

BASE_URL = "http://www.tennis-data.co.uk"

# URL template per tour (year is interpolated). ATP and WTA archives live at
# different paths on the site.
_URL_TEMPLATE: dict[Tour, str] = {
    "ATP": "{base}/{year}/{year}.xlsx",
    "WTA": "{base}/{year}w/{year}.xlsx",
}

# Odds source priority. Pinnacle has the sharpest line and the widest year
# coverage in the archive (~2007+). Bet365 covers earlier years. Market
# avg/max are last resorts.
_ODDS_SOURCES: list[tuple[str, str, str]] = [
    ("PSW", "PSL", "pinnacle"),
    ("AvgW", "AvgL", "market_avg"),
    ("B365W", "B365L", "bet365"),
    ("MaxW", "MaxL", "market_max"),
]

# Match date tolerance: tennis-data uses the start-of-tournament date or the
# match date depending on the year; our matches table uses tourney_date
# (Monday of tournament week). Allow ±10 days slack so 2-week Grand Slams
# don't fall outside the window.
_DATE_TOLERANCE_DAYS = 10

OutcomeStatus = Literal["loaded", "unmatched", "review", "no_odds", "skipped"]


@dataclass
class LoadStats:
    loaded: int = 0
    unmatched: int = 0
    review: int = 0  # auto-loaded but with sub-threshold name confidence
    no_odds: int = 0
    skipped: int = 0  # rows we couldn't even read
    by_odds_source: dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        return self.loaded + self.unmatched + self.review + self.no_odds + self.skipped


# ---------------------------------------------------------------------------
# Download


def download_archive(year: int, tour: Tour, dest_dir: Path | None = None) -> Path:
    """Download one year's xlsx archive to `dest_dir/{year}.xlsx`.

    If the file already exists locally it is not re-downloaded. Idempotent.
    Returns the local path either way.
    """
    _validate_tour(tour)
    if dest_dir is None:
        dest_dir = config.RAW_DIR / "tennis_data_co_uk" / tour
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{year}.xlsx"
    if dest.exists():
        return dest
    url = _URL_TEMPLATE[tour].format(base=BASE_URL, year=year)
    urllib.request.urlretrieve(url, dest)
    return dest


# ---------------------------------------------------------------------------
# Parse


def _pick_best_odds(row: dict[str, object]) -> tuple[float, float, str] | None:
    """Return (odds_winner, odds_loser, odds_source) for this row, or None.

    Accepts a plain dict (one record from the xlsx) so the typing is clean.
    """
    for col_w, col_l, src in _ODDS_SOURCES:
        w = row.get(col_w)
        lo = row.get(col_l)
        if w is None or lo is None:
            continue
        # pd.isna is overloaded; for scalars it returns bool but pyright
        # widens to ndarray. Cast through bool() to keep typing happy.
        if bool(pd.isna(w)) or bool(pd.isna(lo)):  # pyright: ignore[reportArgumentType]
            continue
        try:
            w_f = float(w)  # type: ignore[arg-type]
            l_f = float(lo)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if w_f > 1.0 and l_f > 1.0:
            return w_f, l_f, src
    return None


def _normalize_overround(odds_w: float, odds_l: float) -> tuple[float, float]:
    """Convert decimal odds to implied probabilities, normalized to sum to 1."""
    p_w_raw = 1.0 / odds_w
    p_l_raw = 1.0 / odds_l
    total = p_w_raw + p_l_raw
    return p_w_raw / total, p_l_raw / total


# ---------------------------------------------------------------------------
# Loader


def load_market_file(
    conn: duckdb.DuckDBPyConnection,
    xlsx_path: Path,
    tour: Tour,
    alias_index: AliasIndex,
    *,
    unmatched_csv: Path | None = None,
    review_csv: Path | None = None,
) -> LoadStats:
    """Load one tennis-data.co.uk Excel file into market_implied_probabilities.

    Returns counts by outcome. Writes unmatched/review rows to CSVs under
    data/processed/ unless overridden.
    """
    _validate_tour(tour)
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    if unmatched_csv is None:
        unmatched_csv = config.PROCESSED_DIR / "unmatched_market_rows.csv"
    if review_csv is None:
        review_csv = config.PROCESSED_DIR / "aliases_review.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    stats = LoadStats()
    unmatched_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []

    # Stage 1: resolve names + pick odds in Python. Build a DataFrame keyed
    # for a single SQL JOIN, instead of issuing one match-lookup query per
    # row (the naive loop ran ~290ms/row x 2700 rows = 13 minutes per year).
    staging: list[dict[str, object]] = []

    records: list[dict[str, object]] = df.to_dict(orient="records")  # pyright: ignore[reportAssignmentType]
    for row_idx, raw_row in enumerate(records):
        if "Date" not in raw_row or "Winner" not in raw_row or "Loser" not in raw_row:
            stats.skipped += 1
            continue
        date_value = raw_row["Date"]
        if date_value is None or bool(pd.isna(date_value)):  # pyright: ignore[reportArgumentType]
            stats.skipped += 1
            continue
        match_date = pd.Timestamp(date_value)  # type: ignore[arg-type]
        winner_raw = str(raw_row["Winner"])
        loser_raw = str(raw_row["Loser"])

        winner_lookup: ReconciliationResult = alias_index.lookup(winner_raw)
        loser_lookup: ReconciliationResult = alias_index.lookup(loser_raw)

        if winner_lookup.status == "unknown" or loser_lookup.status == "unknown":
            stats.unmatched += 1
            unmatched_rows.append(
                {
                    "reason": "name_unknown",
                    "tour": tour,
                    "date": match_date.date().isoformat(),
                    "winner_raw": winner_raw,
                    "loser_raw": loser_raw,
                    "winner_status": winner_lookup.status,
                    "loser_status": loser_lookup.status,
                }
            )
            continue

        odds_pick = _pick_best_odds(raw_row)
        if odds_pick is None:
            stats.no_odds += 1
            continue
        odds_w, odds_l, source = odds_pick
        p_w, p_l = _normalize_overround(odds_w, odds_l)

        assert winner_lookup.canonical_player_id is not None
        assert loser_lookup.canonical_player_id is not None
        staging.append(
            {
                "row_idx": row_idx,
                "date": match_date.date(),
                "winner_player_id": winner_lookup.canonical_player_id,
                "loser_player_id": loser_lookup.canonical_player_id,
                "winner_raw": winner_raw,
                "loser_raw": loser_raw,
                "winner_status": winner_lookup.status,
                "loser_status": loser_lookup.status,
                "winner_matched": winner_lookup.candidate_name,
                "loser_matched": loser_lookup.candidate_name,
                "winner_confidence": winner_lookup.confidence,
                "loser_confidence": loser_lookup.confidence,
                "odds_source": source,
                "odds_winner_close": odds_w,
                "odds_loser_close": odds_l,
                "p_winner_close": p_w,
                "p_loser_close": p_l,
            }
        )

    if not staging:
        _append_rows_csv(unmatched_csv, unmatched_rows)
        return stats

    # Stage 2: JOIN against matches to find a single canonical match_id per
    # staging row. When both 'main' and 'qual_chall' have a candidate for
    # the same (winner_id, loser_id, date), prefer 'main' — that's the
    # tour-level row tennis-data is reporting on.
    staging_df = pd.DataFrame(staging)
    conn.register("market_staging", staging_df)
    try:
        joined = conn.execute(
            f"""
            SELECT
                s.*,
                m.match_id,
                m.match_tier
            FROM market_staging s
            LEFT JOIN matches m
                ON  m.tour = ?
                AND m.winner_player_id = s.winner_player_id
                AND m.loser_player_id  = s.loser_player_id
                AND m.tourney_date BETWEEN s.date - INTERVAL {_DATE_TOLERANCE_DAYS} DAY
                                       AND s.date + INTERVAL {_DATE_TOLERANCE_DAYS} DAY
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY s.row_idx
                ORDER BY CASE m.match_tier
                    WHEN 'main' THEN 1
                    WHEN 'qual_chall' THEN 2
                    WHEN 'qual_itf' THEN 2
                    WHEN 'futures' THEN 3
                    ELSE 4 END NULLS LAST
            ) = 1
            """,
            [tour],
        ).fetchdf()
    finally:
        conn.unregister("market_staging")

    # Stage 3: handle each joined row via dict access (typed cleanly).
    matched_rows: list[tuple[str, str, float, float, float, float]] = []
    joined_records: list[dict[str, object]] = joined.to_dict(orient="records")  # pyright: ignore[reportAssignmentType]
    for row in joined_records:
        match_id = row.get("match_id")
        date_str = _to_iso_date(row.get("date"))
        if match_id is None or (isinstance(match_id, float) and pd.isna(match_id)):
            stats.unmatched += 1
            unmatched_rows.append(
                {
                    "reason": "no_match_row",
                    "tour": tour,
                    "date": date_str,
                    "winner_raw": row["winner_raw"],
                    "loser_raw": row["loser_raw"],
                    "winner_player_id": row["winner_player_id"],
                    "loser_player_id": row["loser_player_id"],
                }
            )
            continue
        matched_rows.append(
            (
                str(match_id),
                str(row["odds_source"]),
                float(row["odds_winner_close"]),  # type: ignore[arg-type]
                float(row["odds_loser_close"]),  # type: ignore[arg-type]
                float(row["p_winner_close"]),  # type: ignore[arg-type]
                float(row["p_loser_close"]),  # type: ignore[arg-type]
            )
        )
        src = str(row["odds_source"])
        stats.by_odds_source[src] = stats.by_odds_source.get(src, 0) + 1
        if row["winner_status"] == "review" or row["loser_status"] == "review":
            stats.review += 1
            review_rows.append(
                {
                    "tour": tour,
                    "date": date_str,
                    "winner_raw": row["winner_raw"],
                    "winner_matched": row["winner_matched"],
                    "winner_confidence": row["winner_confidence"],
                    "winner_player_id": row["winner_player_id"],
                    "loser_raw": row["loser_raw"],
                    "loser_matched": row["loser_matched"],
                    "loser_confidence": row["loser_confidence"],
                    "loser_player_id": row["loser_player_id"],
                }
            )
        else:
            stats.loaded += 1

    if matched_rows:
        conn.executemany(
            """
            INSERT INTO market_implied_probabilities (
                match_id, odds_source, odds_winner_close, odds_loser_close,
                p_winner_close, p_loser_close
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (match_id, odds_source) DO NOTHING
            """,
            matched_rows,
        )

    _append_rows_csv(unmatched_csv, unmatched_rows)
    _append_rows_csv(review_csv, review_rows)

    return stats


def load_market_data(
    conn: duckdb.DuckDBPyConnection,
    tour: Tour,
    years: range,
    alias_index: AliasIndex,
    *,
    archive_dir: Path | None = None,
) -> LoadStats:
    """Download and load every year in `years` for one tour.

    Aggregates stats across years.
    """
    _validate_tour(tour)
    totals = LoadStats()
    for year in years:
        path = download_archive(year, tour, dest_dir=archive_dir)
        stats = load_market_file(conn, path, tour, alias_index)
        totals.loaded += stats.loaded
        totals.unmatched += stats.unmatched
        totals.review += stats.review
        totals.no_odds += stats.no_odds
        totals.skipped += stats.skipped
        for src, n in stats.by_odds_source.items():
            totals.by_odds_source[src] = totals.by_odds_source.get(src, 0) + n
    return totals


# ---------------------------------------------------------------------------
# Helpers


def _append_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _to_iso_date(value: object) -> str:
    """Best-effort ISO 8601 date string from whatever pandas/DuckDB returned."""
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[union-attr]
    return str(value)
