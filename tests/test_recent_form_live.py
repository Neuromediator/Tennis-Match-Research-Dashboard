"""Unit tests for the view-layer recent-form + H2H helpers.

Covers the two decision branches that matter most:
- matchstat available + quota OK → matchstat-sourced payload.
- no matchstat external ID known for the player → Sackmann fallback.

H2H summary aggregation (by_surface breakdown) is pinned separately
because the rendering on the Prediction page reads `by_surface` directly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import duckdb
import httpx

from tennis_predictor.data.matchstat import BASE_URL, MatchstatClient
from tennis_predictor.data.recent_form_live import (
    fetch_h2h_summary,
    fetch_recent_n_matches,
)
from tennis_predictor.data.schema import create_all_tables

ATP_A = "ATP_AAA"
ATP_B = "ATP_BBB"
ATP_A_EXTERNAL = 100
ATP_B_EXTERNAL = 200


def _make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_all_tables(conn)
    _seed_players(conn)
    return conn


def _seed_players(conn: duckdb.DuckDBPyConnection) -> None:
    conn.executemany(
        "INSERT INTO players (player_id, tour, sackmann_id, name_first, name_last, full_name) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (ATP_A, "ATP", 1, "Alice", "Alpha", "Alice Alpha"),
            (ATP_B, "ATP", 2, "Bob", "Beta", "Bob Beta"),
        ],
    )


def _seed_scheduled_with_external_ids(conn: duckdb.DuckDBPyConnection) -> None:
    """Insert a scheduled fixture so the external-id lookup helper has
    a row to find."""
    conn.execute(
        """
        INSERT INTO scheduled_matches (
            scheduled_match_id, source, fixture_external_id, tour,
            tournament_external_id, tournament_name, surface,
            player1_external_id, player2_external_id,
            player1_canonical_id, player2_canonical_id,
            player1_name, player2_name, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "matchstat::F1",
            "matchstat",
            "F1",
            "ATP",
            "TOURN1",
            "Test Open",
            "Clay",
            str(ATP_A_EXTERNAL),
            str(ATP_B_EXTERNAL),
            ATP_A,
            ATP_B,
            "Alice Alpha",
            "Bob Beta",
            datetime(2026, 5, 25, 9, 0),
        ],
    )


def _rich_match(
    match_id: str,
    *,
    p1: int,
    p2: int,
    winner: int,
    result: str = "6-3 6-4",
    surface: str = "Clay",
) -> dict[str, Any]:
    return {
        "id": match_id,
        "date": "2026-05-20T12:00:00.000Z",
        "roundId": 1,
        "round": {"id": 1, "name": "R32"},
        "tournamentId": 999,
        "tournament": {
            "id": 999,
            "name": "Test Tournament",
            "court": {"id": 1, "name": surface},
            "rank": {"id": 1, "name": "ATP 250"},
        },
        "player1Id": p1,
        "player2Id": p2,
        "player1": {"id": p1, "name": "Alice Alpha"},
        "player2": {"id": p2, "name": "Bob Beta"},
        "matchWinner": winner,
        "result": result,
        "bestOf": 3,
        "odd1": "1.65",
        "odd2": "2.20",
    }


def _make_client(handler: Any) -> MatchstatClient:
    transport = httpx.MockTransport(handler)
    inner = httpx.Client(base_url=BASE_URL, transport=transport, headers={})
    return MatchstatClient(api_key="test-key", client=inner)


# ---------------------------------------------------------------------------
# Recent form — matchstat path.
# ---------------------------------------------------------------------------


def test_recent_form_uses_matchstat_when_external_id_known() -> None:
    conn = _make_conn()
    _seed_scheduled_with_external_ids(conn)
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        assert "player/past-matches/100" in request.url.path
        return httpx.Response(
            200,
            json={
                "data": [
                    _rich_match("M1", p1=100, p2=300, winner=1),
                    _rich_match("M2", p1=400, p2=100, winner=1, result="3-6 6-2 ret."),
                ],
                "hasNextPage": False,
            },
        )

    # We can't intercept the real client from inside fetch_recent_n_matches
    # — but we can pre-seed the cache with the matchstat payload, which
    # exercises the matchstat-sourced branch end-to-end (the cache hit
    # path uses the same parser code).
    import json as _json

    payload = {
        "data": [
            _rich_match("M1", p1=100, p2=300, winner=1),
            _rich_match("M2", p1=400, p2=100, winner=1, result="3-6 6-2 ret."),
        ],
        "hasNextPage": False,
    }
    conn.execute(
        "INSERT INTO matchstat_player_recent_cache (tour, player_id, fetched_at, payload) "
        "VALUES (?, ?, ?, ?)",
        ["atp", ATP_A_EXTERNAL, now.replace(tzinfo=None), _json.dumps(payload)],
    )

    payload_out = fetch_recent_n_matches(
        conn, "ATP", ATP_A, "Alice Alpha", date(2026, 5, 25), now=now
    )
    assert payload_out.data_source == "matchstat"
    assert len(payload_out.matches) == 2
    # First match — A is player1 and won → W.
    assert payload_out.matches[0].result == "W"
    assert payload_out.matches[0].opponent_name == "Bob Beta"
    assert payload_out.matches[0].completion_status == "W"
    # Second match — A is player2 and player1 (id 400) won → L; completion=RET.
    assert payload_out.matches[1].result == "L"
    assert payload_out.matches[1].completion_status == "RET"

    # Should also unused this once (cache hit, not via handler).
    _ = handler  # mark used to keep ruff happy


def test_recent_form_falls_back_to_sackmann_when_no_external_id() -> None:
    """Player has no row in scheduled_matches → no matchstat external ID
    → falls back to Sackmann `matches`."""
    conn = _make_conn()
    # Note: NOT calling _seed_scheduled_with_external_ids — A is unknown.

    # Seed one Sackmann match: A beat B on 2026-05-20.
    conn.execute(
        "INSERT INTO matches (match_id, source, match_external_id, tour, match_tier, "
        "tourney_id, tourney_date, match_num, match_status, "
        "winner_player_id, loser_player_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "sackmann::1",
            "sackmann",
            "1",
            "ATP",
            "main",
            "T1",
            date(2026, 5, 20),
            1,
            "completed",
            ATP_A,
            ATP_B,
        ],
    )
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
    payload = fetch_recent_n_matches(conn, "ATP", ATP_A, "Alice Alpha", date(2026, 5, 25), now=now)
    assert payload.data_source == "sackmann"
    assert len(payload.matches) == 1
    assert payload.matches[0].result == "W"
    assert payload.matches[0].opponent_name == "Bob Beta"


# ---------------------------------------------------------------------------
# H2H summary.
# ---------------------------------------------------------------------------


def test_h2h_summary_aggregates_by_surface() -> None:
    """Two matches: A wins on Clay, B wins on Hard → by_surface should
    show Clay (1,0) and Hard (0,1)."""
    conn = _make_conn()
    _seed_scheduled_with_external_ids(conn)
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)

    import json as _json

    payload = {
        "data": [
            _rich_match("H1", p1=ATP_A_EXTERNAL, p2=ATP_B_EXTERNAL, winner=1, surface="Clay"),
            _rich_match(
                "H2",
                p1=ATP_B_EXTERNAL,
                p2=ATP_A_EXTERNAL,
                winner=1,
                surface="Hard",
            ),
        ],
        "hasNextPage": False,
    }
    # Cache by canonical (smaller, larger).
    conn.execute(
        "INSERT INTO matchstat_h2h_cache (tour, p1_id, p2_id, fetched_at, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        ["atp", ATP_A_EXTERNAL, ATP_B_EXTERNAL, now.replace(tzinfo=None), _json.dumps(payload)],
    )

    summary = fetch_h2h_summary(
        conn, "ATP", ATP_A, ATP_B, "Alice Alpha", "Bob Beta", date(2026, 5, 25), now=now
    )
    assert summary.data_source == "matchstat"
    assert summary.player_a_wins == 1
    assert summary.player_b_wins == 1
    assert summary.by_surface == {"Clay": (1, 0), "Hard": (0, 1)}
    assert len(summary.matches) == 2


def test_h2h_summary_falls_back_to_sackmann_when_matchstat_returns_empty() -> None:
    """Regression: matchstat's H2H endpoint occasionally returns
    `data: []` even for famous matchups (Sinner-Djokovic observed live).
    Previously we trusted that as authoritative "never met" — wrong.
    Now we fall through to Sackmann when matchstat returns nothing."""
    import json as _json

    conn = _make_conn()
    _seed_scheduled_with_external_ids(conn)
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
    # Pre-seed matchstat H2H cache with an empty payload — simulates the
    # 200 OK + data:[] response we saw against /h2h/11517/68627.
    conn.execute(
        "INSERT INTO matchstat_h2h_cache (tour, p1_id, p2_id, fetched_at, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            "atp",
            ATP_A_EXTERNAL,
            ATP_B_EXTERNAL,
            now.replace(tzinfo=None),
            _json.dumps({"data": [], "hasNextPage": False}),
        ],
    )
    # And seed a real Sackmann match they played.
    conn.execute(
        "INSERT INTO matches (match_id, source, match_external_id, tour, match_tier, "
        "tourney_id, tourney_date, match_num, match_status, surface, "
        "winner_player_id, loser_player_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "sackmann::EMPTY_FALLBACK_1",
            "sackmann",
            "EFB1",
            "ATP",
            "main",
            "TX",
            date(2025, 5, 1),
            1,
            "completed",
            "Clay",
            ATP_A,
            ATP_B,
        ],
    )

    summary = fetch_h2h_summary(
        conn,
        "ATP",
        ATP_A,
        ATP_B,
        "Alice Alpha",
        "Bob Beta",
        date(2026, 5, 25),
        now=now,
    )
    # The empty matchstat result must NOT be advertised as authoritative.
    # Sackmann fallback should surface the real meeting.
    assert summary.data_source == "sackmann"
    assert summary.player_a_wins == 1
    assert summary.player_b_wins == 0
    assert len(summary.matches) == 1


def test_h2h_summary_drops_non_completed_result_type_rows() -> None:
    """Phase 6.2 defensive filter: any row matchstat tags with a
    non-"completed" `resultType` must be dropped before aggregation.
    The legacy Phase 6.1 URL `/fixtures/h2h/...` returned upcoming
    fixtures (no `resultType`) as the entire payload — even after the
    URL fix this filter is a belt-and-braces guard against future
    schema drift."""
    import json as _json

    conn = _make_conn()
    _seed_scheduled_with_external_ids(conn)
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)

    completed = _rich_match("H1", p1=ATP_A_EXTERNAL, p2=ATP_B_EXTERNAL, winner=1, surface="Clay")
    completed["resultType"] = "completed"
    upcoming = _rich_match("H2", p1=ATP_A_EXTERNAL, p2=ATP_B_EXTERNAL, winner=1, surface="Hard")
    upcoming["resultType"] = "upcoming"

    payload = {"data": [completed, upcoming], "hasNextPage": False}
    conn.execute(
        "INSERT INTO matchstat_h2h_cache (tour, p1_id, p2_id, fetched_at, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        ["atp", ATP_A_EXTERNAL, ATP_B_EXTERNAL, now.replace(tzinfo=None), _json.dumps(payload)],
    )

    summary = fetch_h2h_summary(
        conn, "ATP", ATP_A, ATP_B, "Alice Alpha", "Bob Beta", date(2026, 5, 25), now=now
    )
    assert summary.data_source == "matchstat"
    # Only the completed row should survive the filter.
    assert len(summary.matches) == 1
    assert summary.player_a_wins == 1
    assert summary.player_b_wins == 0
    assert summary.by_surface == {"Clay": (1, 0)}


def test_h2h_summary_infers_winner_from_score_when_match_winner_null() -> None:
    """matchstat regularly returns `matchWinner: null` on older H2H rows
    (observed live on the 2016 Barcelona Open Q3 Khachanov-Trungelliti
    row that triggered this fix). The Phase 6.2 fallback parses the
    score string to derive the winner side."""
    import json as _json

    conn = _make_conn()
    _seed_scheduled_with_external_ids(conn)
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)

    # ATP_A is player1, ATP_B is player2 in the raw row. Khachanov-style
    # score 6-3 4-6 7-6(4): player1 wins 2 sets, player2 wins 1.
    row = _rich_match("HM1", p1=ATP_A_EXTERNAL, p2=ATP_B_EXTERNAL, winner=1, surface="Clay")
    row["matchWinner"] = None  # matchstat omits the winner
    row["result"] = "6-3 4-6 7-6(4)"

    payload = {"data": [row], "hasNextPage": False}
    conn.execute(
        "INSERT INTO matchstat_h2h_cache (tour, p1_id, p2_id, fetched_at, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        ["atp", ATP_A_EXTERNAL, ATP_B_EXTERNAL, now.replace(tzinfo=None), _json.dumps(payload)],
    )

    summary = fetch_h2h_summary(
        conn, "ATP", ATP_A, ATP_B, "Alice Alpha", "Bob Beta", date(2026, 5, 25), now=now
    )
    assert summary.data_source == "matchstat"
    # Player A (= matchstat player1) wins 2 sets → infer wins.
    assert summary.player_a_wins == 1
    assert summary.player_b_wins == 0
    assert summary.matches[0].winner_name == "Alice Alpha"


def test_h2h_summary_falls_back_to_sackmann_when_no_external_ids() -> None:
    conn = _make_conn()
    # No scheduled_matches row — no external IDs → Sackmann fallback.
    conn.execute(
        "INSERT INTO matches (match_id, source, match_external_id, tour, match_tier, "
        "tourney_id, tourney_date, match_num, match_status, surface, "
        "winner_player_id, loser_player_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "sackmann::H1",
            "sackmann",
            "H1",
            "ATP",
            "main",
            "T1",
            date(2025, 6, 1),
            1,
            "completed",
            "Clay",
            ATP_A,
            ATP_B,
        ],
    )
    now = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
    summary = fetch_h2h_summary(
        conn, "ATP", ATP_A, ATP_B, "Alice Alpha", "Bob Beta", date(2026, 5, 25), now=now
    )
    assert summary.data_source == "sackmann"
    assert summary.player_a_wins == 1
    assert summary.player_b_wins == 0
    assert summary.by_surface == {"Clay": (1, 0)}
