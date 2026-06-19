"""Unit tests for `app.context`.

`load_context_from_match_id` is exercised against a synthetic DuckDB
seeded with two `scheduled_matches` rows (one mapped tier, one out-of-scope).
`load_context_from_freeform` is exercised against synthetic inputs only —
it doesn't touch the database.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import duckdb
import pytest

from tennis_predictor.app.context import (
    ContextBuildError,
    infer_tournament_level,
    load_context_from_freeform,
    load_context_from_match_id,
)
from tennis_predictor.data import schema


@pytest.fixture
def conn(tmp_path: Path):
    db = duckdb.connect(str(tmp_path / "ctx.duckdb"))
    schema.create_all_tables(db)
    yield db
    db.close()


def _insert_scheduled_match(
    conn: duckdb.DuckDBPyConnection,
    *,
    scheduled_match_id: str,
    tour: str,
    surface: str | None,
    tier: str | None,
    tournament_name: str | None = "Roland Garros",
    scheduled_start_utc: datetime | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO scheduled_matches (
            scheduled_match_id, source, fixture_external_id,
            tour, tournament_external_id, tournament_name, tournament_tier,
            surface, round_name,
            player1_external_id, player2_external_id,
            player1_name, player2_name,
            scheduled_start_utc, ingested_at
        ) VALUES (?, 'matchstat', ?, ?, 'T123', ?, ?, ?, 'R32', 'P1', 'P2',
                  'Carlos Alcaraz', 'Jannik Sinner', ?, CURRENT_TIMESTAMP)
        """,
        [
            scheduled_match_id,
            scheduled_match_id.split("::")[-1],
            tour,
            tournament_name,
            tier,
            surface,
            scheduled_start_utc,
        ],
    )


def test_load_context_from_match_id_happy_path(conn: duckdb.DuckDBPyConnection) -> None:
    _insert_scheduled_match(
        conn,
        scheduled_match_id="matchstat::42",
        tour="ATP",
        surface="Clay",
        tier="Grand Slam",
        scheduled_start_utc=datetime(2026, 6, 8, 14, 0),
    )

    ctx = load_context_from_match_id(conn, "matchstat::42")

    assert ctx.tour == "ATP"
    assert ctx.player_a_name == "Carlos Alcaraz"
    assert ctx.player_b_name == "Jannik Sinner"
    assert ctx.surface == "Clay"
    assert ctx.tournament_level == "Slam"
    assert ctx.tournament_name == "Roland Garros"
    assert ctx.best_of == 5
    assert ctx.match_date == date(2026, 6, 8)
    assert ctx.scheduled_match_id == "matchstat::42"


def test_load_context_from_match_id_unknown_id_raises(conn: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(ContextBuildError, match="no scheduled match found"):
        load_context_from_match_id(conn, "matchstat::nope")


def test_load_context_from_match_id_unsupported_tier_raises(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _insert_scheduled_match(
        conn,
        scheduled_match_id="matchstat::99",
        tour="ATP",
        surface="Hard",
        tier="Challenger",
        # Use a tournament name that does NOT match a Slam pattern so the
        # name-fallback resolver doesn't rescue this row.
        tournament_name="Phoenix Challenger 75",
    )
    with pytest.raises(ContextBuildError, match="does not map to a model tournament_level"):
        load_context_from_match_id(conn, "matchstat::99")


def test_load_context_from_match_id_slam_name_fallback(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """matchstat calendar drops active tournaments (Phase 2 known limit),
    so `tournament_tier` ends up NULL for in-progress Grand Slams. The
    name-fallback resolver must rescue these so users can predict the
    actual Slam draws."""
    _insert_scheduled_match(
        conn,
        scheduled_match_id="matchstat::rg-2026",
        tour="ATP",
        surface="Clay",
        tier=None,
        tournament_name="French Open - Paris",
        scheduled_start_utc=datetime(2026, 5, 24, 13, 0),
    )
    ctx = load_context_from_match_id(conn, "matchstat::rg-2026")
    assert ctx.tournament_level == "Slam"
    assert ctx.best_of == 5
    assert ctx.surface == "Clay"


def test_load_context_from_match_id_itf_with_null_tier_still_rejected(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """A tier-less M15 fixture must NOT be promoted to a tour-level event
    by the name fallback — only Slam names trigger the rescue."""
    _insert_scheduled_match(
        conn,
        scheduled_match_id="matchstat::m15-fake",
        tour="ATP",
        surface="Hard",
        tier=None,
        tournament_name="M15 Gimcheon",
    )
    with pytest.raises(ContextBuildError, match="does not map to a model tournament_level"):
        load_context_from_match_id(conn, "matchstat::m15-fake")


def test_load_context_from_match_id_unsupported_surface_raises(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _insert_scheduled_match(
        conn,
        scheduled_match_id="matchstat::100",
        tour="ATP",
        surface="Carpet",
        tier="ATP 500",
    )
    with pytest.raises(ContextBuildError, match="not in supported set"):
        load_context_from_match_id(conn, "matchstat::100")


def test_load_context_from_match_id_wta_masters_aliases(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _insert_scheduled_match(
        conn,
        scheduled_match_id="matchstat::wta-m1000",
        tour="WTA",
        surface="Hard",
        tier="WTA 1000",
        tournament_name="Madrid Open",
    )
    ctx = load_context_from_match_id(conn, "matchstat::wta-m1000")
    assert ctx.tour == "WTA"
    assert ctx.tournament_level == "M1000"
    # WTA is always best-of-3.
    assert ctx.best_of == 3


def test_load_context_from_match_id_null_start_defaults_to_today(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _insert_scheduled_match(
        conn,
        scheduled_match_id="matchstat::null-start",
        tour="ATP",
        surface="Hard",
        tier="ATP 250",
        scheduled_start_utc=None,
    )
    ctx = load_context_from_match_id(conn, "matchstat::null-start")
    assert ctx.match_date == date.today()


def test_load_context_from_freeform_infers_best_of_for_atp_slam() -> None:
    ctx = load_context_from_freeform(
        tour="ATP",
        player_a_name="A",
        player_b_name="B",
        surface="Grass",
        tournament_level="Slam",
        match_date=date(2026, 7, 1),
    )
    assert ctx.best_of == 5
    assert ctx.scheduled_match_id is None


def test_load_context_from_freeform_respects_explicit_best_of() -> None:
    ctx = load_context_from_freeform(
        tour="ATP",
        player_a_name="A",
        player_b_name="B",
        surface="Hard",
        tournament_level="ATP250",
        match_date=date(2026, 1, 1),
        best_of=5,
    )
    assert ctx.best_of == 5


# ---------------------------------------------------------------------------
# infer_tournament_level direct unit tests
# ---------------------------------------------------------------------------


def test_infer_tournament_level_exact_tier_match() -> None:
    assert infer_tournament_level("Grand Slam", None) == "Slam"
    assert infer_tournament_level("ATP Masters 1000", None) == "M1000"
    assert infer_tournament_level("WTA 500", None) == "WTA500"


def test_infer_tournament_level_slam_name_fallback() -> None:
    assert infer_tournament_level(None, "French Open - Paris") == "Slam"
    assert infer_tournament_level(None, "Wimbledon") == "Slam"
    assert infer_tournament_level(None, "Roland Garros 2026") == "Slam"
    assert infer_tournament_level("", "US Open") == "Slam"


def test_infer_tournament_level_unknown_returns_none() -> None:
    assert infer_tournament_level(None, None) is None
    assert infer_tournament_level("Challenger", "Phoenix Challenger") is None
    assert infer_tournament_level(None, "M15 Gimcheon") is None
    assert infer_tournament_level(None, "ATP Cup") is None


def test_infer_tournament_level_main_tour_defaults_by_tour() -> None:
    # matchstat collapses every active non-Slam event's tier to "Main tour";
    # with the tour known we default to the 500 tier so the event isn't dropped.
    assert infer_tournament_level("Main tour", "Terra Wortmann Open - Halle", "ATP") == "ATP500"
    assert infer_tournament_level("Main tour", "Berlin Tennis Open - Berlin", "WTA") == "WTA500"


def test_infer_tournament_level_main_tour_without_tour_is_none() -> None:
    # No tour → can't pick ATP500 vs WTA500, so it stays out-of-scope.
    assert infer_tournament_level("Main tour", "Terra Wortmann Open - Halle") is None
    assert infer_tournament_level("Main tour", "Some Event", "ITF") is None


def test_infer_tournament_level_slam_name_beats_main_tour_tier() -> None:
    # An active Slam can also arrive with tier="Main tour"; the name fallback
    # must still win so it's a Slam (best-of-5 on the men's side), not ATP500.
    assert infer_tournament_level("Main tour", "Wimbledon", "ATP") == "Slam"


def test_load_context_from_freeform_tournament_name_passthrough() -> None:
    ctx = load_context_from_freeform(
        tour="WTA",
        player_a_name="A",
        player_b_name="B",
        surface="Clay",
        tournament_level="WTA500",
        match_date=date(2026, 4, 4),
        tournament_name="Madrid",
    )
    assert ctx.tournament_name == "Madrid"
    assert ctx.best_of == 3
