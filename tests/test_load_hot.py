"""Tests for the matchstat load layer.

In-memory DuckDB + a stub resolver — no HTTP, no real alias index.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from tennis_predictor.data import load_hot, schema
from tennis_predictor.data.load_hot import (
    LoadCounts,
    PlayerResolver,
    insert_completed_matches,
    insert_market_odds_from_matches,
    insert_scheduled_matches,
    promote_completed_fixtures,
    upsert_ranking_overlay,
)
from tennis_predictor.data.matchstat import Fixture, Match, RankingEntry


@pytest.fixture
def db(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "test.duckdb"))
    schema.create_all_tables(conn)
    return conn


def make_resolver(mapping: dict[str, str | None]) -> PlayerResolver:
    """Stub resolver: name -> canonical_player_id (or None if missing)."""

    def resolve(name: str, tour: str) -> str | None:
        return mapping.get(name)

    return resolve


def make_fixture(**overrides: Any) -> Fixture:
    defaults: dict[str, Any] = {
        "id": 1215,
        "date": "2026-05-19T13:00:00.000Z",
        "roundId": 4,
        "player1Id": 37741,
        "player2Id": 87277,
        "tournamentId": 21327,
        "seed1": None,
        "seed2": "q",
        "player1": {"id": 37741, "name": "Zizou Bergs", "countryAcr": "BEL"},
        "player2": {"id": 87277, "name": "Arthur Gea", "countryAcr": "FRA"},
        "tournament": {
            "id": 21327,
            "name": "Geneva Open",
            "court": {"id": 2, "name": "Clay"},
            "rank": {"id": 2, "name": "Main tour"},
            "countryAcr": "SUI",
        },
        "round": {"id": 4, "name": "R32"},
    }
    defaults.update(overrides)
    return Fixture.model_validate(defaults)


def make_match(**overrides: Any) -> Match:
    defaults: dict[str, Any] = {
        "id": "84752520",
        "date": "2026-05-17T17:15:00.000Z",
        "roundId": 4,
        "player1Id": 29935,
        "player2Id": 82269,
        "tournamentId": 21327,
        "match_winner": 29935,
        "result": "6-1 6-3",
        "best_of": None,
        "odd1": "1.38",
        "odd2": "3.04",
        "player1": {"id": 29935, "name": "Tommy Paul", "countryAcr": "USA"},
        "player2": {"id": 82269, "name": "Ethan Quinn", "countryAcr": "USA"},
    }
    defaults.update(overrides)
    return Match.model_validate(defaults)


def make_ranking_entry(**overrides: Any) -> RankingEntry:
    defaults: dict[str, Any] = {
        "id": 1,
        "date": "2026-05-18T00:00:00.000Z",
        "point": 14700,
        "position": 1,
        "player": {
            "id": 47275,
            "name": "Jannik Sinner",
            "countryAcr": "ITA",
            "currentRank": 1,
            "points": 14700,
        },
    }
    defaults.update(overrides)
    return RankingEntry.model_validate(defaults)


# ---------------------------------------------------------------------------
# LoadCounts


def test_load_counts_addition_is_pointwise() -> None:
    a = LoadCounts(added=2, skipped=1, failed=0)
    b = LoadCounts(added=3, skipped=0, failed=4)
    assert a + b == LoadCounts(added=5, skipped=1, failed=4)


# ---------------------------------------------------------------------------
# insert_scheduled_matches


def test_insert_scheduled_matches_inserts_row(db: duckdb.DuckDBPyConnection) -> None:
    resolver = make_resolver({"Zizou Bergs": "ATP_37741", "Arthur Gea": "ATP_87277"})
    counts = insert_scheduled_matches(
        db,
        [make_fixture()],
        tour="ATP",
        resolve_player=resolver,
    )
    assert counts == LoadCounts(added=1)
    row = db.execute(
        "SELECT tour, surface, round_name, player1_canonical_id, player2_canonical_id "
        "FROM scheduled_matches"
    ).fetchone()
    assert row == ("ATP", "Clay", "R32", "ATP_37741", "ATP_87277")


def test_insert_scheduled_matches_dedupes_on_rerun(db: duckdb.DuckDBPyConnection) -> None:
    resolver = make_resolver({"Zizou Bergs": "x", "Arthur Gea": "y"})
    first = insert_scheduled_matches(db, [make_fixture()], tour="ATP", resolve_player=resolver)
    second = insert_scheduled_matches(db, [make_fixture()], tour="ATP", resolve_player=resolver)
    assert first == LoadCounts(added=1)
    assert second == LoadCounts(skipped=1)
    total = db.execute("SELECT COUNT(*) FROM scheduled_matches").fetchone()
    assert total is not None and total[0] == 1


def test_insert_scheduled_matches_leaves_canonical_null_when_unresolved(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Unresolved players don't block a scheduled_matches insert — canonical
    IDs are nullable for this table by design."""
    resolver = make_resolver({"Zizou Bergs": "ATP_X"})  # only one resolves
    counts = insert_scheduled_matches(db, [make_fixture()], tour="ATP", resolve_player=resolver)
    assert counts == LoadCounts(added=1)
    row = db.execute(
        "SELECT player1_canonical_id, player2_canonical_id FROM scheduled_matches"
    ).fetchone()
    assert row == ("ATP_X", None)


def test_insert_scheduled_matches_uses_tier_lookup(db: duckdb.DuckDBPyConnection) -> None:
    """tournament_tier comes from the orchestrator's calendar cache, since the
    fixture payload doesn't carry the human-readable tier directly."""
    resolver = make_resolver({"Zizou Bergs": "a", "Arthur Gea": "b"})
    insert_scheduled_matches(
        db,
        [make_fixture()],
        tour="ATP",
        resolve_player=resolver,
        tournament_tier_by_id={21327: "ATP 250"},
    )
    tier = db.execute("SELECT tournament_tier FROM scheduled_matches").fetchone()
    assert tier == ("ATP 250",)


# ---------------------------------------------------------------------------
# insert_completed_matches


def test_insert_completed_matches_inserts_with_winner_canonical(
    db: duckdb.DuckDBPyConnection,
) -> None:
    resolver = make_resolver({"Tommy Paul": "ATP_29935", "Ethan Quinn": "ATP_82269"})
    counts = insert_completed_matches(
        db,
        [make_match()],
        tour="ATP",
        tournament_name="Geneva Open",
        tournament_tier="ATP 250",
        surface="Clay",
        tourney_date=date(2026, 5, 11),
        resolve_player=resolver,
    )
    assert counts == LoadCounts(added=1)
    row = db.execute(
        "SELECT winner_player_id, loser_player_id, score, surface, tour, match_status FROM matches"
    ).fetchone()
    assert row == ("ATP_29935", "ATP_82269", "6-1 6-3", "Clay", "ATP", "completed")


def test_insert_completed_matches_handles_winner_being_player2(
    db: duckdb.DuckDBPyConnection,
) -> None:
    resolver = make_resolver({"Tommy Paul": "ATP_29935", "Ethan Quinn": "ATP_82269"})
    insert_completed_matches(
        db,
        [make_match(match_winner=82269)],  # player2 wins
        tour="ATP",
        tournament_name="X",
        tournament_tier="ATP 250",
        surface="Clay",
        tourney_date=date(2026, 5, 11),
        resolve_player=resolver,
    )
    row = db.execute("SELECT winner_player_id, loser_player_id FROM matches").fetchone()
    assert row == ("ATP_82269", "ATP_29935")


def test_insert_completed_matches_fails_on_unresolved_player(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """matches.winner_player_id is NOT NULL — refuse to insert if either
    player can't be resolved. The orchestrator can write these to
    aliases_review.csv on its own."""
    resolver = make_resolver({"Tommy Paul": "ATP_X"})  # only one resolves
    counts = insert_completed_matches(
        db,
        [make_match()],
        tour="ATP",
        tournament_name="X",
        tournament_tier="ATP 250",
        surface="Clay",
        tourney_date=date(2026, 5, 11),
        resolve_player=resolver,
    )
    assert counts == LoadCounts(failed=1)
    count = db.execute("SELECT COUNT(*) FROM matches").fetchone()
    assert count is not None and count[0] == 0


def test_insert_completed_matches_dedupes_on_rerun(db: duckdb.DuckDBPyConnection) -> None:
    resolver = make_resolver({"Tommy Paul": "a", "Ethan Quinn": "b"})
    kwargs: dict[str, Any] = {
        "tour": "ATP",
        "tournament_name": "X",
        "tournament_tier": "ATP 250",
        "surface": "Clay",
        "tourney_date": date(2026, 5, 11),
        "resolve_player": resolver,
    }
    first = insert_completed_matches(db, [make_match()], **kwargs)
    second = insert_completed_matches(db, [make_match()], **kwargs)
    assert first == LoadCounts(added=1)
    assert second == LoadCounts(skipped=1)


def test_insert_completed_matches_skips_when_winner_id_missing(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Matches without a clear winner (rare but possible — e.g., walkover
    rows the API hasn't populated yet) are skipped, not failed."""
    resolver = make_resolver({"Tommy Paul": "a", "Ethan Quinn": "b"})
    counts = insert_completed_matches(
        db,
        [make_match(match_winner=None)],
        tour="ATP",
        tournament_name="X",
        tournament_tier="ATP 250",
        surface="Clay",
        tourney_date=date(2026, 5, 11),
        resolve_player=resolver,
    )
    assert counts == LoadCounts(skipped=1)


# ---------------------------------------------------------------------------
# insert_market_odds_from_matches


def test_market_odds_normalized_probabilities(db: duckdb.DuckDBPyConnection) -> None:
    """odd1=1.38, odd2=3.04, winner=player1 → p_winner ≈ 0.687 after overround."""
    counts = insert_market_odds_from_matches(db, [make_match()])
    assert counts == LoadCounts(added=1)
    row = db.execute(
        "SELECT odds_winner_close, odds_loser_close, p_winner_close, p_loser_close, odds_source "
        "FROM market_implied_probabilities"
    ).fetchone()
    assert row is not None
    odds_w, odds_l, p_w, p_l, source = row
    assert odds_w == pytest.approx(1.38)
    assert odds_l == pytest.approx(3.04)
    assert p_w + p_l == pytest.approx(1.0, abs=1e-9)
    assert p_w == pytest.approx((1 / 1.38) / ((1 / 1.38) + (1 / 3.04)))
    assert source == "matchstat"


def test_market_odds_handles_winner_being_player2(db: duckdb.DuckDBPyConnection) -> None:
    """When player2 wins, odds_winner_close must be odd2, not odd1."""
    insert_market_odds_from_matches(db, [make_match(match_winner=82269)])
    row = db.execute(
        "SELECT odds_winner_close, odds_loser_close FROM market_implied_probabilities"
    ).fetchone()
    assert row == pytest.approx((3.04, 1.38))


def test_market_odds_skipped_when_odds_missing(db: duckdb.DuckDBPyConnection) -> None:
    counts = insert_market_odds_from_matches(db, [make_match(odd1=None)])
    assert counts == LoadCounts(skipped=1)
    count = db.execute("SELECT COUNT(*) FROM market_implied_probabilities").fetchone()
    assert count is not None and count[0] == 0


def test_market_odds_skipped_when_odds_invalid(db: duckdb.DuckDBPyConnection) -> None:
    """Decimal odds ≤ 1.0 are nonsensical (implied probability > 1) — skip."""
    counts = insert_market_odds_from_matches(db, [make_match(odd1="0.5")])
    assert counts == LoadCounts(skipped=1)


def test_market_odds_dedupes_on_rerun(db: duckdb.DuckDBPyConnection) -> None:
    first = insert_market_odds_from_matches(db, [make_match()])
    second = insert_market_odds_from_matches(db, [make_match()])
    assert first == LoadCounts(added=1)
    assert second == LoadCounts(skipped=1)


# ---------------------------------------------------------------------------
# upsert_ranking_overlay


def test_ranking_overlay_inserts_row(db: duckdb.DuckDBPyConnection) -> None:
    resolver = make_resolver({"Jannik Sinner": "ATP_47275"})
    counts = upsert_ranking_overlay(db, [make_ranking_entry()], tour="ATP", resolve_player=resolver)
    assert counts == LoadCounts(added=1)
    row = db.execute("SELECT ranking_date, player_id, rank, points FROM rankings").fetchone()
    assert row == (date(2026, 5, 18), "ATP_47275", 1, 14700)


def test_ranking_overlay_fails_on_unresolved_player(db: duckdb.DuckDBPyConnection) -> None:
    """rankings.player_id is NOT NULL — refuse and count as failed."""
    resolver = make_resolver({})  # nothing resolves
    counts = upsert_ranking_overlay(db, [make_ranking_entry()], tour="ATP", resolve_player=resolver)
    assert counts == LoadCounts(failed=1)


def test_ranking_overlay_first_write_wins_on_same_day(db: duckdb.DuckDBPyConnection) -> None:
    """If the same player is fetched twice the same day (re-run), first wins."""
    resolver = make_resolver({"Jannik Sinner": "ATP_47275"})
    upsert_ranking_overlay(
        db, [make_ranking_entry(position=1)], tour="ATP", resolve_player=resolver
    )
    counts = upsert_ranking_overlay(
        db, [make_ranking_entry(position=2)], tour="ATP", resolve_player=resolver
    )
    assert counts == LoadCounts(skipped=1)
    row = db.execute("SELECT rank FROM rankings").fetchone()
    assert row == (1,)


# ---------------------------------------------------------------------------
# promote_completed_fixtures


def test_promote_completed_fixtures_removes_matched_rows(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A fixture in scheduled_matches whose composite (tournament + players
    + round) appears in `matches` is removed — the prediction surface only
    needs the not-yet-played list."""
    resolver = make_resolver({"Tommy Paul": "ATP_29935", "Ethan Quinn": "ATP_82269"})

    # Stage a scheduled fixture with both canonicals resolved.
    insert_scheduled_matches(
        db,
        [
            make_fixture(
                id=9999,
                tournamentId=21327,
                roundId=4,
                player1Id=29935,
                player2Id=82269,
                player1={"id": 29935, "name": "Tommy Paul", "countryAcr": "USA"},
                player2={"id": 82269, "name": "Ethan Quinn", "countryAcr": "USA"},
            )
        ],
        tour="ATP",
        resolve_player=resolver,
    )
    # Insert the corresponding completed match.
    insert_completed_matches(
        db,
        [make_match()],
        tour="ATP",
        tournament_name="Geneva Open",
        tournament_tier="ATP 250",
        surface="Clay",
        tourney_date=date(2026, 5, 11),
        resolve_player=resolver,
    )

    removed = promote_completed_fixtures(db)
    assert removed == 1
    remaining = db.execute("SELECT COUNT(*) FROM scheduled_matches").fetchone()
    assert remaining is not None and remaining[0] == 0


def test_promote_completed_fixtures_handles_winner_player_order_swap(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Even when the result row has player2 as winner (winner becomes
    matches.winner_player_id), the unordered-pair match in the DELETE
    predicate must still fire."""
    resolver = make_resolver({"Tommy Paul": "ATP_29935", "Ethan Quinn": "ATP_82269"})
    insert_scheduled_matches(
        db,
        [
            make_fixture(
                id=8888,
                tournamentId=21327,
                roundId=4,
                player1Id=29935,
                player2Id=82269,
                player1={"id": 29935, "name": "Tommy Paul", "countryAcr": "USA"},
                player2={"id": 82269, "name": "Ethan Quinn", "countryAcr": "USA"},
            )
        ],
        tour="ATP",
        resolve_player=resolver,
    )
    # Player2 wins in the matches row.
    insert_completed_matches(
        db,
        [make_match(match_winner=82269)],
        tour="ATP",
        tournament_name="X",
        tournament_tier="ATP 250",
        surface="Clay",
        tourney_date=date(2026, 5, 11),
        resolve_player=resolver,
    )
    removed = promote_completed_fixtures(db)
    assert removed == 1


def test_promote_completed_fixtures_leaves_unmatched_rows_alone(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A scheduled fixture without a corresponding matches row stays."""
    resolver = make_resolver({"Zizou Bergs": "ATP_X", "Arthur Gea": "ATP_Y"})
    insert_scheduled_matches(db, [make_fixture()], tour="ATP", resolve_player=resolver)
    removed = promote_completed_fixtures(db)
    assert removed == 0
    count = db.execute("SELECT COUNT(*) FROM scheduled_matches").fetchone()
    assert count is not None and count[0] == 1


# ---------------------------------------------------------------------------
# Smoke: SURFACE_MAP coverage of observed values


def test_surface_map_covers_observed_matchstat_courts() -> None:
    """Probe rounds saw `Clay` and `I.hard`. Both should map to known surfaces.
    Unknowns pass through unchanged — better than dropping signal."""
    assert load_hot._map_surface("Clay") == "Clay"
    assert load_hot._map_surface("I.hard") == "Hard"
    assert load_hot._map_surface("Hard") == "Hard"
    assert load_hot._map_surface("Grass") == "Grass"
    assert load_hot._map_surface("Carpet") == "Carpet"
    assert load_hot._map_surface(None) is None
    # Unknown surfaces are passed through, not dropped.
    assert load_hot._map_surface("SomeNewSurface2030") == "SomeNewSurface2030"


def test_parse_odd_handles_edge_cases() -> None:
    assert load_hot._parse_odd("1.38") == pytest.approx(1.38)
    assert load_hot._parse_odd(None) is None
    assert load_hot._parse_odd("") is None
    assert load_hot._parse_odd("not-a-number") is None
    assert load_hot._parse_odd("0.5") is None  # implied probability > 1
    assert load_hot._parse_odd("1.0") is None  # boundary: no juice possible


def test_ranking_overlay_uses_explicit_as_of_date(db: duckdb.DuckDBPyConnection) -> None:
    """Caller can pin the date instead of trusting entry.date — useful when
    the orchestrator wants 'today' regardless of what the payload claims."""
    resolver = make_resolver({"Jannik Sinner": "ATP_47275"})
    upsert_ranking_overlay(
        db,
        [make_ranking_entry()],
        tour="ATP",
        resolve_player=resolver,
        as_of_date=date(2026, 5, 19),
    )
    row = db.execute("SELECT ranking_date FROM rankings").fetchone()
    assert row == (date(2026, 5, 19),)


def test_ranking_overlay_uses_payload_date_when_no_override(
    db: duckdb.DuckDBPyConnection,
) -> None:
    resolver = make_resolver({"Jannik Sinner": "ATP_47275"})
    upsert_ranking_overlay(db, [make_ranking_entry()], tour="ATP", resolve_player=resolver)
    row = db.execute("SELECT ranking_date FROM rankings").fetchone()
    assert row == (date(2026, 5, 18),)


def test_ranking_overlay_falls_back_to_now_when_payload_date_missing(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """If matchstat ever omits `date` AND caller doesn't pin one, fall back
    to today UTC so the row is still write-able."""
    resolver = make_resolver({"Jannik Sinner": "ATP_47275"})
    upsert_ranking_overlay(
        db,
        [make_ranking_entry(date=None)],
        tour="ATP",
        resolve_player=resolver,
    )
    row = db.execute("SELECT ranking_date FROM rankings").fetchone()
    assert row is not None
    assert row[0] == datetime.now(UTC).date()
