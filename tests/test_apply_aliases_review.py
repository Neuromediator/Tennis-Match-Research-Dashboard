"""Unit tests for the manual-review apply logic."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from tennis_predictor.data import manual_review, schema


@pytest.fixture
def fresh_db(tmp_path: Path):
    conn = duckdb.connect(str(tmp_path / "test.duckdb"))
    schema.create_all_tables(conn)
    yield conn
    conn.close()


def _write_review_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_apply_inserts_unique_winner_and_loser_pairs(
    fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    csv = tmp_path / "review.csv"
    _write_review_csv(
        csv,
        [
            {
                "tour": "ATP",
                "date": "2023-06-05",
                "winner_raw": "Federer R.",
                "winner_matched": "Federer R",
                "winner_confidence": 1.0,
                "winner_player_id": "ATP_104925",
                "loser_raw": "Nadal R.",
                "loser_matched": "Nadal R",
                "loser_confidence": 0.95,
                "loser_player_id": "ATP_104745",
            },
            # Same winner appears in a second match — should dedupe.
            {
                "tour": "ATP",
                "date": "2023-07-03",
                "winner_raw": "Federer R.",
                "winner_matched": "Federer R",
                "winner_confidence": 1.0,
                "winner_player_id": "ATP_104925",
                "loser_raw": "Djokovic N.",
                "loser_matched": "Djokovic N",
                "loser_confidence": 1.0,
                "loser_player_id": "ATP_104918",
            },
        ],
    )

    stats = manual_review.apply_review(fresh_db, csv)
    assert stats["csv_rows"] == 2
    assert stats["unique_pairs"] == 3  # Federer, Nadal, Djokovic
    assert stats["newly_inserted"] == 3

    rows = fresh_db.execute(
        "SELECT alias_text, tour, canonical_player_id, source, confidence "
        "FROM player_aliases WHERE source = 'manual_review' ORDER BY alias_text"
    ).fetchall()
    aliases = {(r[0], r[2]) for r in rows}
    assert ("Federer R.", "ATP_104925") in aliases
    assert ("Nadal R.", "ATP_104745") in aliases
    assert ("Djokovic N.", "ATP_104918") in aliases
    for r in rows:
        assert r[3] == "manual_review"
        assert r[4] == 1.0


def test_apply_is_idempotent(fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    csv = tmp_path / "review.csv"
    _write_review_csv(
        csv,
        [
            {
                "tour": "ATP",
                "winner_raw": "Federer R.",
                "winner_player_id": "ATP_104925",
                "loser_raw": "Nadal R.",
                "loser_player_id": "ATP_104745",
            }
        ],
    )
    manual_review.apply_review(fresh_db, csv)
    stats2 = manual_review.apply_review(fresh_db, csv)
    assert stats2["newly_inserted"] == 0
    assert stats2["already_present"] == 2


def test_apply_skips_rows_with_missing_canonical_id(
    fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    csv = tmp_path / "review.csv"
    _write_review_csv(
        csv,
        [
            {
                "tour": "ATP",
                "winner_raw": "Federer R.",
                "winner_player_id": "ATP_104925",
                "loser_raw": "Mystery X.",
                "loser_player_id": None,  # not resolved
            }
        ],
    )
    stats = manual_review.apply_review(fresh_db, csv)
    assert stats["unique_pairs"] == 1
    assert stats["newly_inserted"] == 1


def test_apply_raises_on_unknown_schema(
    fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    csv = tmp_path / "review.csv"
    pd.DataFrame([{"foo": "bar"}]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="match neither"):
        manual_review.apply_review(fresh_db, csv)


def test_apply_raises_on_missing_file(fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        manual_review.apply_review(fresh_db, tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# matchstat per-player format (Phase 2)


def test_matchstat_apply_promotes_only_y_verdicts(
    fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    csv = tmp_path / "review.csv"
    _write_review_csv(
        csv,
        [
            {
                "raw_name": "Pedro Martinez Portero",
                "tour": "ATP",
                "candidate_name": "Martinez P",
                "candidate_canonical_id": "ATP_111111",
                "confidence": 0.90,
                "runner_up_confidence": 0.90,
                "verdict": "y",  # accept
            },
            {
                "raw_name": "Mystery Person",
                "tour": "ATP",
                "candidate_name": "Other Mystery",
                "candidate_canonical_id": "ATP_222222",
                "confidence": 0.80,
                "runner_up_confidence": 0.0,
                "verdict": "",  # reject (blank)
            },
            {
                "raw_name": "Niksa Smiljanic",
                "tour": "ATP",
                "candidate_name": "Suka Niksa",
                "candidate_canonical_id": "ATP_333333",
                "confidence": 0.855,
                "runner_up_confidence": 0.0,
                "verdict": "n",  # explicit reject
            },
        ],
    )

    stats = manual_review.apply_review(fresh_db, csv)
    assert stats["csv_rows"] == 3
    assert stats["unique_pairs"] == 1  # only the verdict='y' row
    assert stats["newly_inserted"] == 1

    rows = fresh_db.execute(
        "SELECT alias_text, canonical_player_id FROM player_aliases WHERE source = 'manual_review'"
    ).fetchall()
    assert rows == [("Pedro Martinez Portero", "ATP_111111")]


def test_matchstat_apply_handles_case_and_whitespace_in_verdict(
    fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Reviewer might type 'Y ', '  y', 'yes' — only literal 'y'/'Y' (trimmed)
    counts as accept. 'yes' is rejected (forces unambiguous verdict)."""
    csv = tmp_path / "review.csv"
    _write_review_csv(
        csv,
        [
            {
                "raw_name": "A",
                "tour": "ATP",
                "candidate_name": "Aa",
                "candidate_canonical_id": "ATP_AAA",
                "confidence": 0.8,
                "runner_up_confidence": 0.0,
                "verdict": "Y ",
            },
            {
                "raw_name": "B",
                "tour": "ATP",
                "candidate_name": "Bb",
                "candidate_canonical_id": "ATP_BBB",
                "confidence": 0.8,
                "runner_up_confidence": 0.0,
                "verdict": "  y",
            },
            {
                "raw_name": "C",
                "tour": "ATP",
                "candidate_name": "Cc",
                "candidate_canonical_id": "ATP_CCC",
                "confidence": 0.8,
                "runner_up_confidence": 0.0,
                "verdict": "yes",  # not 'y' -> rejected (avoid ambiguity)
            },
        ],
    )
    stats = manual_review.apply_review(fresh_db, csv)
    assert stats["newly_inserted"] == 2


def test_matchstat_apply_is_idempotent(fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    csv = tmp_path / "review.csv"
    _write_review_csv(
        csv,
        [
            {
                "raw_name": "Pedro Martinez Portero",
                "tour": "ATP",
                "candidate_name": "Martinez P",
                "candidate_canonical_id": "ATP_111111",
                "confidence": 0.90,
                "runner_up_confidence": 0.0,
                "verdict": "y",
            }
        ],
    )
    manual_review.apply_review(fresh_db, csv)
    stats2 = manual_review.apply_review(fresh_db, csv)
    assert stats2["newly_inserted"] == 0
    assert stats2["already_present"] == 1


def test_matchstat_apply_skips_missing_candidate_canonical_id(
    fresh_db: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """If the resolver couldn't even surface a candidate id, an empty verdict
    must NOT promote a blank link."""
    csv = tmp_path / "review.csv"
    _write_review_csv(
        csv,
        [
            {
                "raw_name": "Lonely Stranger",
                "tour": "ATP",
                "candidate_name": "",
                "candidate_canonical_id": "",
                "confidence": 0.0,
                "runner_up_confidence": 0.0,
                "verdict": "y",  # even with y, no canonical to link to
            }
        ],
    )
    stats = manual_review.apply_review(fresh_db, csv)
    assert stats["newly_inserted"] == 0
