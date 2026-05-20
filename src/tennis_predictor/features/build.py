"""Chronological training-features replay.

`build_training_features(conn)` is one of the two sanctioned entry points
into the feature layer. It walks every row in `matches` in chronological
order, maintains five in-memory state objects (Elo, recent form, H2H,
fatigue, serve/return) plus an in-memory ranking lookup, and writes one
row per eligible match into the `training_features` table.

# Eligibility tiers

Each match goes through two gates:

**State-update gate** (does the match feed the state objects?)
    - `match_status == 'completed'`         — RET / W/O / DEF are not real outcomes.
    - Normalized `surface` is not None     — can't update surface-Elo without surface.

If the match passes this gate, *every* state object is updated regardless
of whether a label row is written. This is how Challengers / Futures /
Davis Cup matches sharpen ratings for top players without polluting the
training labels.

**Label-write gate** (does the match produce a training_features row?)
    Adds, on top of state-update gate:
    - `match_tier == 'main'` OR the row is a tour-level main-draw qualifying
      match — Sackmann stores Q1/Q2/Q3 matches at tour-level events in the
      qualifying-tier files (`qual_chall` for ATP, `qual_itf` for WTA),
      mixed with Challengers / Futures / ITF / juniors. See
      `_is_main_draw_qualifying` for the per-tour level whitelist.
    - Normalized `tournament_level` is not None — excludes D, O, WTA-OOS, WTA-125.
    - `best_of in (3, 5)`                  — FeatureVector contract.
    - Both players have `>= 5` completed matches in history — hard floor.

# Determinism

Matches are ordered `(tourney_date, tourney_id, match_num, match_id)` —
the same input DB always produces the same `training_features`. p1 / p2
are assigned lexicographically on `player_id`, so the label distribution
does not depend on the original winner/loser column order.

# Persistence

- `training_features` is fully overwritten on each run (one transaction).
- `elo_state` is persisted via `EloState.save_to_db` at the end —
  inference path will load this snapshot and roll forward.
- Other state objects are NOT persisted; they rebuild from scratch each
  run (cheap on our data volume, per the feature-engineering skill).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import duckdb

from tennis_predictor.features.elo import EloState
from tennis_predictor.features.fatigue import FatigueState, count_sets
from tennis_predictor.features.h2h import H2HState
from tennis_predictor.features.ranking import RankingLookup
from tennis_predictor.features.rolling_form import RollingFormState
from tennis_predictor.features.schema import FeatureVector
from tennis_predictor.features.serve_return import MatchStats, ServeReturnState
from tennis_predictor.features.surface import normalize_surface
from tennis_predictor.features.tournament_level import normalize_tournament_level

logger = logging.getLogger(__name__)

MIN_HISTORY_FOR_LABEL = 5
"""Both players must have at least this many completed matches in history
for the match to write a label row. Below the floor, recent-form windows
and H2H stats are mostly noise. See user decision in the Phase 3
design-discussion turn."""

# Sackmann file layout puts tour-level main-draw QUALIFYING matches into
# the qualifying-tier files (`atp_matches_qual_chall_*.csv`,
# `wta_matches_qual_itf_*.csv`), mixed in with Challengers / ITF / juniors.
# These sets identify the `tourney_level` codes inside those files that
# correspond to tour-level main-draw qualifying (Q1/Q2/Q3 before the main
# draw of a Slam / Masters / 250-500 / Premier event). Per user decision in
# the Phase 3 design discussion, those qualifying matches are treated as
# label-eligible — same as main-draw matches.
ATP_QUAL_TOUR_LEVELS: frozenset[str] = frozenset({"G", "M", "A"})
WTA_QUAL_TOUR_LEVELS: frozenset[str] = frozenset({"G", "PM", "P", "I", "T1", "T2", "W"})

BATCH_SIZE = 5_000


def _is_main_draw_qualifying(tour: str, match_tier: str, tourney_level: str | None) -> bool:
    """True if a row from a Sackmann qualifying file is actually a tour-level
    main-draw qualifying match (rather than a Challenger / ITF / junior row
    that lives in the same file)."""
    if tour == "ATP" and match_tier == "qual_chall":
        return tourney_level in ATP_QUAL_TOUR_LEVELS
    if tour == "WTA" and match_tier == "qual_itf":
        return tourney_level in WTA_QUAL_TOUR_LEVELS
    return False


_TRAINING_FEATURES_COLUMNS: tuple[str, ...] = (
    "match_id",
    "tour",
    "match_date",
    "p1_player_id",
    "p2_player_id",
    "label_winner_is_p1",
    "elo_p1_surface",
    "elo_p2_surface",
    "elo_diff_surface",
    "win_pct_last10_p1",
    "win_pct_last10_p2",
    "win_pct_last25_surface_p1",
    "win_pct_last25_surface_p2",
    "first_serve_win_pct_p1",
    "first_serve_win_pct_p2",
    "second_serve_win_pct_p1",
    "second_serve_win_pct_p2",
    "bp_saved_pct_p1",
    "bp_saved_pct_p2",
    "bp_converted_pct_p1",
    "bp_converted_pct_p2",
    "h2h_p1_wins",
    "h2h_p2_wins",
    "h2h_recency_days",
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
)

_INSERT_SQL = (
    "INSERT INTO training_features ("
    + ", ".join(_TRAINING_FEATURES_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" * len(_TRAINING_FEATURES_COLUMNS))
    + ")"
)

# Column order in the SELECT must match the unpacking in _iter_match_rows.
_MATCH_SELECT_SQL = """
    SELECT
        match_id, tour, match_tier, match_status,
        tourney_id, tourney_name, tourney_level, tourney_date, surface,
        match_num, best_of, score,
        winner_player_id, loser_player_id,
        w_first_in, w_first_won, w_second_won, w_svpt, w_bp_saved, w_bp_faced,
        l_first_in, l_first_won, l_second_won, l_svpt, l_bp_saved, l_bp_faced
    FROM matches
    ORDER BY tourney_date ASC, tourney_id ASC, match_num ASC, match_id ASC
"""


@dataclass(frozen=True, slots=True)
class _MatchRow:
    """Typed view of one `matches` row. Built in `_iter_match_rows`."""

    match_id: str
    tour: str
    match_tier: str
    match_status: str
    tourney_id: str
    tourney_name: str | None
    tourney_level: str | None
    match_date: date
    raw_surface: str | None
    match_num: int
    best_of: int | None
    score: str | None
    winner_id: str
    loser_id: str
    stats: MatchStats | None


@dataclass
class BuildSummary:
    """Returned by `build_training_features` for logging / smoke tests."""

    matches_scanned: int = 0
    state_updates_applied: int = 0
    training_rows_written: int = 0
    skipped_non_completed: int = 0
    skipped_null_surface: int = 0
    skipped_non_main_tier: int = 0
    skipped_excluded_level: int = 0
    skipped_history_floor: int = 0
    skipped_bad_best_of: int = 0


def build_training_features(conn: duckdb.DuckDBPyConnection) -> BuildSummary:
    """Run the full chronological replay → populate `training_features` and
    persist `elo_state`. Returns counters for visibility into how many rows
    landed where."""
    ranking_lookup = RankingLookup.from_db(conn)
    elo = EloState()
    form = RollingFormState()
    h2h = H2HState()
    fatigue = FatigueState()
    serve_return = ServeReturnState()
    summary = BuildSummary()

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("DELETE FROM training_features")
        batch: list[tuple[object, ...]] = []

        for m in _iter_match_rows(conn):
            summary.matches_scanned += 1

            # --- State-update gate -----------------------------------------
            if m.match_status != "completed":
                summary.skipped_non_completed += 1
                continue

            surface = normalize_surface(m.raw_surface, m.tourney_name)
            if surface is None:
                summary.skipped_null_surface += 1
                continue

            # --- Label-write gate (eligibility) ----------------------------
            # Tour-level main-draw matches AND tour-level main-draw qualifying
            # are both label-eligible. Qualifying matches live in Sackmann's
            # qual_chall / qual_itf files but only the rows whose
            # tourney_level matches a tour-level code count as qualifying for
            # us — Challengers / Futures / ITF prize-money tiers / juniors
            # are excluded here.
            is_main_tier = m.match_tier == "main"
            is_qualifying = _is_main_draw_qualifying(m.tour, m.match_tier, m.tourney_level)
            label_eligible = is_main_tier or is_qualifying
            if not label_eligible:
                summary.skipped_non_main_tier += 1

            t_level = None
            if label_eligible:
                t_level = normalize_tournament_level(m.tour, m.tourney_level, m.tourney_name)
                if t_level is None:
                    summary.skipped_excluded_level += 1
                    label_eligible = False

            if label_eligible and m.best_of not in (3, 5):
                summary.skipped_bad_best_of += 1
                label_eligible = False

            if label_eligible and (
                form.matches_played(m.winner_id) < MIN_HISTORY_FOR_LABEL
                or form.matches_played(m.loser_id) < MIN_HISTORY_FOR_LABEL
            ):
                summary.skipped_history_floor += 1
                label_eligible = False

            # --- Snapshot phase --------------------------------------------
            if label_eligible:
                assert t_level is not None  # narrowed by the eligibility flow
                assert m.best_of is not None
                p1, p2 = sorted([m.winner_id, m.loser_id])
                label = 1 if m.winner_id == p1 else 0
                fv = _build_feature_vector(
                    p1=p1,
                    p2=p2,
                    surface=surface,
                    match_date=m.match_date,
                    tournament_level=t_level,
                    best_of=m.best_of,
                    elo=elo,
                    form=form,
                    h2h=h2h,
                    fatigue=fatigue,
                    serve_return=serve_return,
                    ranking_lookup=ranking_lookup,
                )
                batch.append(_to_insert_row(m.match_id, m.tour, m.match_date, p1, p2, label, fv))
                summary.training_rows_written += 1

                if len(batch) >= BATCH_SIZE:
                    conn.executemany(_INSERT_SQL, batch)
                    batch.clear()

            # --- Update phase ----------------------------------------------
            sets_played = count_sets(m.score)
            elo.update(m.winner_id, m.loser_id, surface, m.match_date)
            form.update(m.winner_id, m.loser_id, surface, m.match_date)
            h2h.update(m.winner_id, m.loser_id, m.match_date)
            fatigue.update(m.winner_id, m.loser_id, sets_played, m.match_date)
            serve_return.update(m.winner_id, m.loser_id, surface, m.match_date, m.stats)
            summary.state_updates_applied += 1

        if batch:
            conn.executemany(_INSERT_SQL, batch)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    elo.save_to_db(conn)

    logger.info(
        "build_training_features done: scanned=%d, updates=%d, training_rows=%d, "
        "skip(non_completed=%d, null_surface=%d, non_main=%d, excl_level=%d, "
        "bad_bo=%d, history_floor=%d)",
        summary.matches_scanned,
        summary.state_updates_applied,
        summary.training_rows_written,
        summary.skipped_non_completed,
        summary.skipped_null_surface,
        summary.skipped_non_main_tier,
        summary.skipped_excluded_level,
        summary.skipped_bad_best_of,
        summary.skipped_history_floor,
    )
    return summary


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _iter_match_rows(conn: duckdb.DuckDBPyConnection) -> Iterator[_MatchRow]:
    """Stream `matches` rows in chronological order, packaged as `_MatchRow`.

    Uses `conn.cursor()` so the read cursor stays alive even when the
    caller issues writes (`executemany(INSERT)`) on the parent connection
    between fetches — DuckDB's main connection has a single execution
    slot that the next `.execute(...)` would clobber.
    """
    read_cursor = conn.cursor()
    read_cursor.execute(_MATCH_SELECT_SQL)
    while True:
        chunk = read_cursor.fetchmany(BATCH_SIZE)
        if not chunk:
            return
        for row in chunk:
            yield _row_to_match(row)


def _row_to_match(row: tuple) -> _MatchRow:
    (
        match_id,
        tour,
        match_tier,
        match_status,
        tourney_id,
        tourney_name,
        tourney_level,
        tourney_date,
        raw_surface,
        match_num,
        best_of,
        score,
        winner_id,
        loser_id,
        w_first_in,
        w_first_won,
        w_second_won,
        w_svpt,
        w_bp_saved,
        w_bp_faced,
        l_first_in,
        l_first_won,
        l_second_won,
        l_svpt,
        l_bp_saved,
        l_bp_faced,
    ) = row

    stat_cols = (
        w_first_in,
        w_first_won,
        w_second_won,
        w_svpt,
        w_bp_saved,
        w_bp_faced,
        l_first_in,
        l_first_won,
        l_second_won,
        l_svpt,
        l_bp_saved,
        l_bp_faced,
    )
    stats: MatchStats | None = None
    if all(c is not None for c in stat_cols):
        stats = MatchStats(
            w_first_in=int(w_first_in),
            w_first_won=int(w_first_won),
            w_second_won=int(w_second_won),
            w_svpt=int(w_svpt),
            w_bp_saved=int(w_bp_saved),
            w_bp_faced=int(w_bp_faced),
            l_first_in=int(l_first_in),
            l_first_won=int(l_first_won),
            l_second_won=int(l_second_won),
            l_svpt=int(l_svpt),
            l_bp_saved=int(l_bp_saved),
            l_bp_faced=int(l_bp_faced),
        )

    return _MatchRow(
        match_id=match_id,
        tour=tour,
        match_tier=match_tier,
        match_status=match_status,
        tourney_id=tourney_id,
        tourney_name=tourney_name,
        tourney_level=tourney_level,
        match_date=tourney_date,
        raw_surface=raw_surface,
        match_num=int(match_num),
        best_of=int(best_of) if best_of is not None else None,
        score=score,
        winner_id=winner_id,
        loser_id=loser_id,
        stats=stats,
    )


def _build_feature_vector(
    *,
    p1: str,
    p2: str,
    surface: str,
    match_date: date,
    tournament_level: str,
    best_of: int,
    elo: EloState,
    form: RollingFormState,
    h2h: H2HState,
    fatigue: FatigueState,
    serve_return: ServeReturnState,
    ranking_lookup: RankingLookup,
) -> FeatureVector:
    """Snapshot every state object and assemble one FeatureVector.

    Pydantic construction goes through `model_validate` so all bounds /
    literal-set checks run — the orchestrator pays the validation cost
    on every label row, which is the contract.
    """
    elo_p1 = elo.get(p1, surface)
    elo_p2 = elo.get(p2, surface)

    wp10_p1, wp25_p1 = form.snapshot(p1, surface)
    wp10_p2, wp25_p2 = form.snapshot(p2, surface)

    h2h_p1_wins, h2h_p2_wins, h2h_recency = h2h.snapshot(p1, p2, match_date)

    fat_m1, fat_s1 = fatigue.snapshot(p1, match_date)
    fat_m2, fat_s2 = fatigue.snapshot(p2, match_date)

    rank_p1 = ranking_lookup.get(p1, match_date)
    rank_p2 = ranking_lookup.get(p2, match_date)

    sr1 = serve_return.snapshot(p1, surface)
    sr2 = serve_return.snapshot(p2, surface)

    return FeatureVector.model_validate(
        {
            "elo_p1_surface": elo_p1,
            "elo_p2_surface": elo_p2,
            "elo_diff_surface": elo_p1 - elo_p2,
            "win_pct_last10_p1": wp10_p1,
            "win_pct_last10_p2": wp10_p2,
            "win_pct_last25_surface_p1": wp25_p1,
            "win_pct_last25_surface_p2": wp25_p2,
            "first_serve_win_pct_p1": sr1[0],
            "first_serve_win_pct_p2": sr2[0],
            "second_serve_win_pct_p1": sr1[1],
            "second_serve_win_pct_p2": sr2[1],
            "bp_saved_pct_p1": sr1[2],
            "bp_saved_pct_p2": sr2[2],
            "bp_converted_pct_p1": sr1[3],
            "bp_converted_pct_p2": sr2[3],
            "h2h_p1_wins": h2h_p1_wins,
            "h2h_p2_wins": h2h_p2_wins,
            "h2h_recency_days": h2h_recency,
            "fatigue_matches_7d_p1": fat_m1,
            "fatigue_matches_7d_p2": fat_m2,
            "fatigue_sets_14d_p1": fat_s1,
            "fatigue_sets_14d_p2": fat_s2,
            "rank_p1": rank_p1,
            "rank_p2": rank_p2,
            "rank_diff": rank_p1 - rank_p2,
            "tournament_level": tournament_level,
            "best_of": best_of,
            "surface": surface,
        }
    )


def _to_insert_row(
    match_id: str,
    tour: str,
    match_date: date,
    p1: str,
    p2: str,
    label: int,
    fv: FeatureVector,
) -> tuple[object, ...]:
    return (
        match_id,
        tour,
        match_date,
        p1,
        p2,
        label,
        fv.elo_p1_surface,
        fv.elo_p2_surface,
        fv.elo_diff_surface,
        fv.win_pct_last10_p1,
        fv.win_pct_last10_p2,
        fv.win_pct_last25_surface_p1,
        fv.win_pct_last25_surface_p2,
        fv.first_serve_win_pct_p1,
        fv.first_serve_win_pct_p2,
        fv.second_serve_win_pct_p1,
        fv.second_serve_win_pct_p2,
        fv.bp_saved_pct_p1,
        fv.bp_saved_pct_p2,
        fv.bp_converted_pct_p1,
        fv.bp_converted_pct_p2,
        fv.h2h_p1_wins,
        fv.h2h_p2_wins,
        fv.h2h_recency_days,
        fv.fatigue_matches_7d_p1,
        fv.fatigue_matches_7d_p2,
        fv.fatigue_sets_14d_p1,
        fv.fatigue_sets_14d_p2,
        fv.rank_p1,
        fv.rank_p2,
        fv.rank_diff,
        fv.tournament_level,
        fv.best_of,
        fv.surface,
    )
