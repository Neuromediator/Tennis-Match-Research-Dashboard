"""Market-implied probability benchmark.

For each set of validation match_ids we look up the closing-price implied
probabilities in ``market_implied_probabilities`` and compute the market's
Brier score and calibration curve over the matches where odds are
available. Used as the overlay on each model's calibration plot —
methodology framing: "our model approaches market calibration without
reaching it", never "we beat the market".

The market table stores ``p_winner_close`` / ``p_loser_close`` keyed by
``match_id``. We map that to ``P(p1 wins)`` using the original Sackmann
``winner_player_id`` from the ``matches`` table, so the orientation matches
how ``label_winner_is_p1`` was generated in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import numpy.typing as npt

from tennis_predictor.models.metrics import ClassificationMetrics, compute_metrics


@dataclass(frozen=True)
class MarketProbabilities:
    """Market probabilities aligned to a set of validation match_ids.

    ``y_true`` and ``y_prob`` are subset to matches where odds exist; the
    aligned-row count may be smaller than the input ``match_ids`` length.
    """

    match_ids: npt.NDArray[np.str_]
    y_true: npt.NDArray[np.int_]
    y_prob: npt.NDArray[np.float64]

    @property
    def n(self) -> int:
        return len(self.match_ids)


# Threshold below which the market overlay is omitted with a note in the
# report (e.g., pre-2013 folds have no odds in our DB at all).
MIN_MARKET_OVERLAY_ROWS: int = 1000


def fetch_market_probabilities(
    conn: duckdb.DuckDBPyConnection,
    match_ids: npt.NDArray[np.str_],
    labels: npt.NDArray[np.int_],
    p1_player_ids: npt.NDArray[np.str_],
) -> MarketProbabilities:
    """Join validation match_ids against market_implied_probabilities.

    Orientation: market stores ``p_winner_close`` / ``p_loser_close``. We
    convert to ``P(p1 wins)`` using the Sackmann ``winner_player_id`` field
    in ``matches`` (which player won), independent of the lex-ordered
    ``p1_player_id`` used by feature engineering.
    """
    if len(match_ids) == 0:
        return MarketProbabilities(
            match_ids=np.array([], dtype=str),
            y_true=np.array([], dtype=int),
            y_prob=np.array([], dtype=float),
        )

    input_df = _build_input_df(match_ids, labels, p1_player_ids)
    conn.register("market_probe_input", input_df)
    try:
        rows = conn.execute(
            """
            SELECT
                i.match_id,
                i.label,
                CASE
                    WHEN m.winner_player_id = i.p1_player_id THEN p.p_winner_close
                    ELSE p.p_loser_close
                END AS p_p1_close
            FROM market_probe_input AS i
            JOIN market_implied_probabilities AS p USING (match_id)
            JOIN matches AS m ON m.match_id = i.match_id
            WHERE p.p_winner_close IS NOT NULL
              AND p.p_loser_close IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.unregister("market_probe_input")

    if not rows:
        return MarketProbabilities(
            match_ids=np.array([], dtype=str),
            y_true=np.array([], dtype=int),
            y_prob=np.array([], dtype=float),
        )
    mids = np.array([r[0] for r in rows], dtype=str)
    y_true = np.array([r[1] for r in rows], dtype=int)
    y_prob = np.array([r[2] for r in rows], dtype=float)
    return MarketProbabilities(match_ids=mids, y_true=y_true, y_prob=y_prob)


def _build_input_df(
    match_ids: npt.NDArray[np.str_],
    labels: npt.NDArray[np.int_],
    p1_player_ids: npt.NDArray[np.str_],
):
    import pandas as pd

    return pd.DataFrame(
        {
            "match_id": match_ids,
            "label": labels,
            "p1_player_id": p1_player_ids,
        }
    )


def market_metrics(probs: MarketProbabilities) -> ClassificationMetrics | None:
    if probs.n == 0:
        return None
    return compute_metrics(probs.y_true, probs.y_prob)
