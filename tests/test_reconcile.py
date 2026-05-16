"""Player reconciliation tests.

Covers normalization, identity-seed, and the four classes of lookup:
- exact / canonical-name match (auto)
- diacritics and abbreviated formats (auto)
- ambiguous same-surname pairs (review)
- unknown name (unknown)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from tennis_predictor.data import reconcile, schema


@pytest.fixture
def db_with_players(tmp_path: Path):
    """A fresh DB with a few hand-crafted players for fuzzy tests."""
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
            ("ATP_104918", "ATP", 104918, "Novak", "Djokovic", "Novak Djokovic", "R"),
            # Two same-surname players for ambiguity test
            ("ATP_103970", "ATP", 103970, "Guillermo", "Coria", "Guillermo Coria", "R"),
            ("ATP_104267", "ATP", 104267, "Federico", "Coria", "Federico Coria", "R"),
        ],
    )
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# normalize_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Roger Federer", "roger federer"),
        ("Đoković, Novak", "dokovic novak"),
        ("Federer R.", "federer r"),
        ("Hsieh Su-wei", "hsieh su wei"),
        ("  Spaces   Out ", "spaces out"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_name(raw: str | None, expected: str) -> None:
    assert reconcile.normalize_name(raw) == expected


# ---------------------------------------------------------------------------
# seeding


def test_seed_aliases_from_players_inserts_multiple_forms_per_player(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    inserted = reconcile.seed_aliases_from_players(db_with_players, "ATP")
    # 5 players x 3 forms (canonical / reversed / abbreviated)
    assert inserted == 15
    rows = db_with_players.execute(
        "SELECT alias_text, canonical_player_id FROM player_aliases WHERE tour = 'ATP' "
        "ORDER BY alias_text"
    ).fetchall()
    aliases = {a for a, _ in rows}
    assert "Roger Federer" in aliases
    assert "Federer Roger" in aliases
    assert "Federer R" in aliases
    assert ("Federico Coria", "ATP_104267") in rows


def test_seed_skips_unknown_placeholder_players(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    db_with_players.execute(
        "INSERT INTO players (player_id, tour, sackmann_id, name_first, name_last, "
        "full_name) VALUES ('ATP_99', 'ATP', 99, 'Unknown', 'Smith', 'Unknown Smith')"
    )
    reconcile.seed_aliases_from_players(db_with_players, "ATP")
    rows = db_with_players.execute(
        "SELECT COUNT(*) FROM player_aliases WHERE alias_text LIKE 'Unknown%'"
    ).fetchone()
    assert rows is not None and rows[0] == 0


def test_seed_aliases_is_idempotent(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    reconcile.seed_aliases_from_players(db_with_players, "ATP")
    second = reconcile.seed_aliases_from_players(db_with_players, "ATP")
    assert second == 0


# ---------------------------------------------------------------------------
# AliasIndex lookups


def test_lookup_empty_index_returns_unknown(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    index = reconcile.AliasIndex(db_with_players, "ATP")
    assert len(index) == 0
    result = index.lookup("Roger Federer")
    assert result.status == "unknown"
    assert result.canonical_player_id is None


def test_lookup_exact_canonical_name_auto(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    reconcile.seed_aliases_from_players(db_with_players, "ATP")
    index = reconcile.AliasIndex(db_with_players, "ATP")
    result = index.lookup("Roger Federer")
    assert result.status == "auto"
    assert result.canonical_player_id == "ATP_104925"
    assert result.confidence == 1.0


def test_lookup_diacritics_match_auto(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    reconcile.seed_aliases_from_players(db_with_players, "ATP")
    index = reconcile.AliasIndex(db_with_players, "ATP")
    result = index.lookup("Novak Đoković")
    assert result.status == "auto"
    assert result.canonical_player_id == "ATP_104918"
    assert result.confidence >= reconcile.AUTO_THRESHOLD


def test_lookup_reverse_order_match_auto(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    reconcile.seed_aliases_from_players(db_with_players, "ATP")
    index = reconcile.AliasIndex(db_with_players, "ATP")
    result = index.lookup("Federer, Roger")
    assert result.status == "auto"
    assert result.canonical_player_id == "ATP_104925"


def test_lookup_abbreviated_initial_match_auto(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    """tennis-data.co.uk uses 'Last F.' format — this is the format that
    motivated the multi-form seeding strategy. Should hit exact match."""
    reconcile.seed_aliases_from_players(db_with_players, "ATP")
    index = reconcile.AliasIndex(db_with_players, "ATP")
    result = index.lookup("Federer R.")
    assert result.status == "auto"
    assert result.canonical_player_id == "ATP_104925"
    assert result.confidence == 1.0


def test_lookup_ambiguous_same_surname_flags_review(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    reconcile.seed_aliases_from_players(db_with_players, "ATP")
    index = reconcile.AliasIndex(db_with_players, "ATP")
    # "Coria" alone is ambiguous between Guillermo and Federico
    result = index.lookup("Coria")
    assert result.status == "review"
    # runner-up should be high (close to best) since both Corias match
    assert result.runner_up_confidence >= reconcile.REVIEW_THRESHOLD


def test_lookup_unknown_below_threshold(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    reconcile.seed_aliases_from_players(db_with_players, "ATP")
    index = reconcile.AliasIndex(db_with_players, "ATP")
    result = index.lookup("Completely Unrelated Person")
    assert result.status == "unknown"
    assert result.canonical_player_id is None
    assert result.confidence < reconcile.REVIEW_THRESHOLD


# ---------------------------------------------------------------------------
# find_namesakes


def test_find_namesakes_returns_empty_when_no_dupes(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    assert reconcile.find_namesakes(db_with_players, "ATP") == []


def test_find_namesakes_detects_duplicate_full_name(
    db_with_players: duckdb.DuckDBPyConnection,
) -> None:
    # Inject a duplicate full_name
    db_with_players.execute(
        "INSERT INTO players (player_id, tour, sackmann_id, full_name) "
        "VALUES ('ATP_999999', 'ATP', 999999, 'Roger Federer')"
    )
    namesakes = reconcile.find_namesakes(db_with_players, "ATP")
    assert len(namesakes) == 1
    name, ids = namesakes[0]
    assert name == "Roger Federer"
    assert set(ids) == {"ATP_104925", "ATP_999999"}
