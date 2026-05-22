"""Inference-time feature computation.

`compute_features(...)` is the second sanctioned entry point into the
feature layer (CLAUDE.md hard rule #2). It returns a Pydantic
`FeatureVector` for a single hypothetical `(player_id, opponent_id,
surface, as_of_date)` instance — exactly the vector the LightGBM model
will score.

# How it stays consistent with training replay

The hard contract from the feature-engineering skill:
    `compute_features` returns identical values for the same
    `(player, opponent, surface, as_of_date)` whether reached via training
    replay or inference path.

To achieve this:

1. Load `EloState` from the persisted `elo_state` table (snapshot taken at
   the end of the most recent `build_training_features` run).
2. Build the other four state objects (`RollingFormState`, `H2HState`,
   `FatigueState`, `ServeReturnState`) in-memory from scratch — they were
   not persisted, per the skill.
3. Query every match involving `player_id` OR `opponent_id` with
   `match_date < as_of_date` and `match_status='completed'` and
   `surface IS NOT NULL`. Replay them in chronological order into the
   in-memory states. For Elo, skip matches whose `match_date` is on or
   before the persisted snapshot date — those updates are already in the
   loaded ratings.
4. Take snapshots from each state, look up ranking, and assemble a
   `FeatureVector`. p1 / p2 are sorted lexicographically so the canonical
   ordering matches training-time replay.

# Performance

Naive loads scan the full `elo_state` (~100k rows) and `rankings`
(~5.6M rows) on every call. For the Phase 6 Streamlit integration, the
caller can pass pre-loaded `EloState` / `RankingLookup` via keyword
arguments to amortize cost across many predictions.

# Limitations

- Assumes `as_of_date >= elo_snapshot_date`. Calling with a much earlier
  `as_of_date` would require rolling Elo backward, which is impossible
  from a forward-only snapshot. For historical analysis, regenerate
  `training_features` (which already produces FeatureVectors at every
  historical match date).
- The "rolled forward" Elo includes matches with `match_date < as_of_date`.
  Same-day prior matches (if any) are NOT included — chronological
  match-num ordering within a day is not reconstructable from a date-only
  filter. In practice we predict tomorrow's match, so this is a non-issue.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import duckdb

from tennis_predictor.features.elo import EloState
from tennis_predictor.features.fatigue import FatigueState, count_sets
from tennis_predictor.features.h2h import H2HState
from tennis_predictor.features.last_match import LastMatchState
from tennis_predictor.features.player_metadata import (
    PlayerMetadataLookup,
    compute_age,
    compute_age_vs_peak,
    compute_height_diff,
)
from tennis_predictor.features.ranking import RankingLookup
from tennis_predictor.features.rolling_form import RollingFormState
from tennis_predictor.features.schema import FeatureVector, Surface, TournamentLevel
from tennis_predictor.features.serve_return import MatchStats, ServeReturnState
from tennis_predictor.features.surface import normalize_surface

_HISTORY_SELECT_SQL = """
    SELECT
        tourney_id, tourney_date, match_num, match_id,
        tourney_name, surface, score,
        winner_player_id, loser_player_id,
        w_first_in, w_first_won, w_second_won, w_svpt, w_bp_saved, w_bp_faced,
        l_first_in, l_first_won, l_second_won, l_svpt, l_bp_saved, l_bp_faced
    FROM matches
    WHERE (winner_player_id IN (?, ?) OR loser_player_id IN (?, ?))
      AND match_status = 'completed'
      AND surface IS NOT NULL
      AND tourney_date < ?
    ORDER BY tourney_date ASC, tourney_id ASC, match_num ASC, match_id ASC
"""


@dataclass(frozen=True, slots=True)
class _HistoryRow:
    """One row pulled from `matches` for the inference-time replay."""

    match_date: date
    surface: str
    winner_id: str
    loser_id: str
    score: str | None
    stats: MatchStats | None


def compute_features(
    conn: duckdb.DuckDBPyConnection,
    player_id: str,
    opponent_id: str,
    surface: Surface,
    tour: str,
    as_of_date: date,
    tournament_level: TournamentLevel,
    best_of: int,
    *,
    elo: EloState | None = None,
    ranking_lookup: RankingLookup | None = None,
    player_metadata: PlayerMetadataLookup | None = None,
) -> FeatureVector:
    """Return the FeatureVector for `(player_id, opponent_id)` as of
    `as_of_date`.

    `elo`, `ranking_lookup`, and `player_metadata` may be passed pre-loaded
    (Phase 6 cache); omitting them triggers a fresh load from DB.
    """
    if player_id == opponent_id:
        raise ValueError("player_id and opponent_id must differ")
    if not player_id.startswith(f"{tour}_") or not opponent_id.startswith(f"{tour}_"):
        raise ValueError(f"player IDs ({player_id}, {opponent_id}) inconsistent with tour={tour}")
    if best_of not in (3, 5):
        raise ValueError(f"best_of must be 3 or 5, got {best_of}")

    # Canonical ordering — matches training_features p1/p2 assignment.
    p1, p2 = sorted([player_id, opponent_id])

    if ranking_lookup is None:
        ranking_lookup = RankingLookup.from_db(conn)
    if player_metadata is None:
        player_metadata = PlayerMetadataLookup.from_db(conn)

    # Elo + LastMatch: load the persisted snapshot when safe; otherwise
    # rebuild from scratch by replaying every relevant match in the history
    # loop below. The persisted snapshots reflect state AFTER every match
    # in the DB, so they leak information when `as_of_date <= snapshot_date`
    # — for that case (e.g. the training-vs-inference equivalence test, or
    # any historical inference) we deliberately discard the snapshot.
    persisted_snapshot_date = _elo_snapshot_date(conn)
    if elo is None:
        if persisted_snapshot_date is not None and persisted_snapshot_date < as_of_date:
            elo = EloState.from_db(conn)
            elo_baseline_date: date | None = persisted_snapshot_date
        else:
            elo = EloState()
            elo_baseline_date = None
    else:
        # Caller supplied a pre-loaded snapshot — trust them and pair it
        # with the persisted date (they presumably loaded it from the same DB).
        elo_baseline_date = persisted_snapshot_date

    # LastMatchState uses the same baseline-date discipline as Elo: the
    # persisted snapshot is post-every-match, so it's only safe to seed
    # from when `as_of_date` is strictly later than the snapshot.
    if elo_baseline_date is not None:
        last_match = LastMatchState.from_db(conn)
        last_match_baseline_date: date | None = elo_baseline_date
    else:
        last_match = LastMatchState()
        last_match_baseline_date = None

    # Build the rest of the states fresh by replaying every completed
    # match either player has played before `as_of_date`.
    form = RollingFormState()
    h2h = H2HState()
    fatigue = FatigueState()
    serve_return = ServeReturnState()

    for h in _load_history(conn, p1, p2, as_of_date):
        form.update(h.winner_id, h.loser_id, h.surface, h.match_date)
        h2h.update(h.winner_id, h.loser_id, h.match_date)
        fatigue.update(h.winner_id, h.loser_id, count_sets(h.score), h.match_date)
        serve_return.update(h.winner_id, h.loser_id, h.surface, h.match_date, h.stats)
        # Elo + LastMatch: skip matches already in the persisted snapshot.
        if elo_baseline_date is None or h.match_date > elo_baseline_date:
            elo.update(h.winner_id, h.loser_id, h.surface, h.match_date)
        if last_match_baseline_date is None or h.match_date > last_match_baseline_date:
            last_match.update(h.winner_id, h.match_date)
            last_match.update(h.loser_id, h.match_date)

    # Snapshots.
    elo_p1 = elo.get(p1, surface)
    elo_p2 = elo.get(p2, surface)
    wp10_p1, wp25_p1 = form.snapshot(p1, surface)
    wp10_p2, wp25_p2 = form.snapshot(p2, surface)
    h2h_p1_wins, h2h_p2_wins, h2h_recency = h2h.snapshot(p1, p2, as_of_date)
    fat_m1, fat_s1 = fatigue.snapshot(p1, as_of_date)
    fat_m2, fat_s2 = fatigue.snapshot(p2, as_of_date)
    rank_p1 = ranking_lookup.get(p1, as_of_date)
    rank_p2 = ranking_lookup.get(p2, as_of_date)
    sr1 = serve_return.snapshot(p1, surface)
    sr2 = serve_return.snapshot(p2, surface)

    meta_p1 = player_metadata.get(p1)
    meta_p2 = player_metadata.get(p2)
    age_p1 = compute_age(meta_p1.dob, as_of_date)
    age_p2 = compute_age(meta_p2.dob, as_of_date)
    days_since_p1 = last_match.days_since(p1, as_of_date)
    days_since_p2 = last_match.days_since(p2, as_of_date)

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
            "hand_p1": meta_p1.hand,
            "hand_p2": meta_p2.hand,
            "age_p1": age_p1,
            "age_p2": age_p2,
            "age_vs_peak_p1": compute_age_vs_peak(age_p1, tour),
            "age_vs_peak_p2": compute_age_vs_peak(age_p2, tour),
            "height_p1": meta_p1.height,
            "height_p2": meta_p2.height,
            "height_diff_cm": compute_height_diff(meta_p1.height, meta_p2.height),
            "days_since_last_match_p1": days_since_p1,
            "days_since_last_match_p2": days_since_p2,
        }
    )


def _elo_snapshot_date(conn: duckdb.DuckDBPyConnection) -> date | None:
    """Return max(last_updated_date) across `elo_state`, or None if empty."""
    row = conn.execute("SELECT max(last_updated_date) FROM elo_state").fetchone()
    if row is None or row[0] is None:
        return None
    return row[0]


def _load_history(
    conn: duckdb.DuckDBPyConnection,
    p1: str,
    p2: str,
    as_of_date: date,
) -> Iterable[_HistoryRow]:
    """Yield every completed, surface-resolved match involving `p1` OR `p2`
    with `match_date < as_of_date`, in chronological order."""
    rows = conn.execute(_HISTORY_SELECT_SQL, [p1, p2, p1, p2, as_of_date]).fetchall()
    for r in rows:
        (
            _tourney_id,
            tourney_date,
            _match_num,
            _match_id,
            tourney_name,
            raw_surface,
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
        ) = r

        # Normalize surface the same way build_training_features did — the
        # raw "Hard" from a Paris Bercy row becomes "IHard" here, otherwise
        # the surface-keyed snapshots (Elo, surface form, serve/return) miss.
        canonical_surface = normalize_surface(raw_surface, tourney_name)
        if canonical_surface is None:
            # Should not happen — SQL filter excludes NULL surfaces. Defensive.
            continue

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

        yield _HistoryRow(
            match_date=tourney_date,
            surface=canonical_surface,
            winner_id=winner_id,
            loser_id=loser_id,
            score=score,
            stats=stats,
        )
