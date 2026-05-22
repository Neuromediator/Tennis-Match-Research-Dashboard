"""DuckDB schema definitions.

All tables created via CREATE TABLE IF NOT EXISTS, so `create_all_tables` is
idempotent and safe to call on every connection open.

Conventions:
- `match_id` is `f"{source}::{match_external_id}"` — stable across reruns,
  used as the join key for every per-match table.
- `player_id` is `f"{tour}_{sackmann_id}"` (e.g. `"ATP_104925"`).
  ATP and WTA have separate integer ID spaces in Sackmann; this composite
  key keeps them disambiguated.
- Statistic columns are renamed from Sackmann's CamelCase (`w_1stIn`) to
  snake_case (`w_first_in`) for SQL-friendliness. The mapping happens in
  ingestion code, not here.
"""

from __future__ import annotations

import duckdb

MATCHES_DDL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id            VARCHAR PRIMARY KEY,
    source              VARCHAR NOT NULL,
    match_external_id   VARCHAR NOT NULL,

    tour                VARCHAR NOT NULL,
    match_tier          VARCHAR NOT NULL,

    tourney_id          VARCHAR NOT NULL,
    tourney_name        VARCHAR,
    tourney_level       VARCHAR,
    tourney_date        DATE NOT NULL,
    surface             VARCHAR,
    draw_size           INTEGER,

    match_num           INTEGER NOT NULL,
    round               VARCHAR,
    best_of             INTEGER,
    minutes             INTEGER,
    score               VARCHAR,
    match_status        VARCHAR NOT NULL,

    winner_player_id    VARCHAR NOT NULL,
    loser_player_id     VARCHAR NOT NULL,
    winner_seed         VARCHAR,
    winner_entry        VARCHAR,
    winner_rank         INTEGER,
    winner_rank_points  INTEGER,
    winner_age          DOUBLE,
    loser_seed          VARCHAR,
    loser_entry         VARCHAR,
    loser_rank          INTEGER,
    loser_rank_points   INTEGER,
    loser_age           DOUBLE,

    w_ace               INTEGER,
    w_df                INTEGER,
    w_svpt              INTEGER,
    w_first_in          INTEGER,
    w_first_won         INTEGER,
    w_second_won        INTEGER,
    w_sv_gms            INTEGER,
    w_bp_saved          INTEGER,
    w_bp_faced          INTEGER,

    l_ace               INTEGER,
    l_df                INTEGER,
    l_svpt              INTEGER,
    l_first_in          INTEGER,
    l_first_won         INTEGER,
    l_second_won        INTEGER,
    l_sv_gms            INTEGER,
    l_bp_saved          INTEGER,
    l_bp_faced          INTEGER,

    UNIQUE (source, match_external_id)
);
"""

PLAYERS_DDL = """
CREATE TABLE IF NOT EXISTS players (
    player_id    VARCHAR PRIMARY KEY,
    tour         VARCHAR NOT NULL,
    sackmann_id  INTEGER NOT NULL,
    name_first   VARCHAR,
    name_last    VARCHAR,
    full_name    VARCHAR,
    hand         VARCHAR,
    dob          DATE,
    ioc          VARCHAR,
    height       INTEGER,
    wikidata_id  VARCHAR
);
"""

RANKINGS_DDL = """
CREATE TABLE IF NOT EXISTS rankings (
    ranking_date  DATE NOT NULL,
    player_id     VARCHAR NOT NULL,
    rank          INTEGER NOT NULL,
    points        INTEGER,
    PRIMARY KEY (ranking_date, player_id)
);
"""

PLAYER_ALIASES_DDL = """
CREATE TABLE IF NOT EXISTS player_aliases (
    alias_text           VARCHAR NOT NULL,
    tour                 VARCHAR NOT NULL,
    source               VARCHAR NOT NULL,
    canonical_player_id  VARCHAR NOT NULL,
    confidence           DOUBLE NOT NULL,
    PRIMARY KEY (alias_text, tour, source)
);
"""

MARKET_IMPLIED_PROBABILITIES_DDL = """
CREATE TABLE IF NOT EXISTS market_implied_probabilities (
    match_id           VARCHAR NOT NULL,
    odds_source        VARCHAR NOT NULL,
    odds_winner_close  DOUBLE,
    odds_loser_close   DOUBLE,
    p_winner_close     DOUBLE,
    p_loser_close      DOUBLE,
    PRIMARY KEY (match_id, odds_source)
);
"""

LLM_TRACES_SEQUENCE_DDL = "CREATE SEQUENCE IF NOT EXISTS seq_llm_traces START 1;"

# Phase 5: `web_search_count` and `estimated_cost_usd` are added so the
# Streamlit dashboard (Phase 6) can surface "spent today / this month" and
# cache-hit hygiene without re-deriving from token counts. The migration
# below ALTERs an existing table to add the columns when they're missing
# (DuckDB ALTER TABLE ADD COLUMN preserves existing rows).
LLM_TRACES_DDL = """
CREATE TABLE IF NOT EXISTS llm_traces (
    trace_id               BIGINT PRIMARY KEY DEFAULT nextval('seq_llm_traces'),
    ts                     TIMESTAMP NOT NULL,
    model                  VARCHAR NOT NULL,
    system_prompt_hash     VARCHAR,
    input_messages         JSON,
    tool_calls             JSON,
    output                 JSON,
    tokens_in              INTEGER,
    tokens_out             INTEGER,
    cache_read_tokens      INTEGER,
    cache_creation_tokens  INTEGER,
    latency_ms             INTEGER,
    error                  VARCHAR,
    web_search_count       INTEGER,
    estimated_cost_usd     DOUBLE
);
"""

ELO_STATE_DDL = """
CREATE TABLE IF NOT EXISTS elo_state (
    player_id          VARCHAR NOT NULL,
    surface            VARCHAR NOT NULL,
    rating             DOUBLE NOT NULL,
    matches_played     INTEGER NOT NULL,
    last_updated_date  DATE NOT NULL,
    PRIMARY KEY (player_id, surface)
);
"""

# Phase 4.1: one row per player carrying the date of their most recent
# completed match (across surfaces). Persisted by `LastMatchState` at the
# end of `build_training_features`; inference loads the snapshot and rolls
# forward — same pattern as `elo_state`. See
# `src/tennis_predictor/features/last_match.py`.
LAST_MATCH_STATE_DDL = """
CREATE TABLE IF NOT EXISTS last_match_state (
    player_id        VARCHAR PRIMARY KEY,
    last_match_date  DATE NOT NULL
);
"""

# Upcoming fixtures pulled from the hot API. One row per fixture the API
# currently exposes (round-by-round visibility — see docs/phases.md Phase 2).
# Rows are removed (or matched out) once the corresponding `matches` row
# appears with a result. Linkage to `matches` is the composite
# (tournament_external_id, player1_external_id, player2_external_id,
# round_external_id) — NOT a shared external id.
SCHEDULED_MATCHES_DDL = """
CREATE TABLE IF NOT EXISTS scheduled_matches (
    scheduled_match_id        VARCHAR PRIMARY KEY,
    source                    VARCHAR NOT NULL,
    fixture_external_id       VARCHAR NOT NULL,

    tour                      VARCHAR NOT NULL,
    tournament_external_id    VARCHAR NOT NULL,
    tournament_name           VARCHAR,
    tournament_tier           VARCHAR,
    tournament_country_acr    VARCHAR,
    surface                   VARCHAR,
    round_external_id         VARCHAR,
    round_name                VARCHAR,

    player1_external_id       VARCHAR NOT NULL,
    player2_external_id       VARCHAR NOT NULL,
    player1_canonical_id      VARCHAR,
    player2_canonical_id      VARCHAR,
    player1_name              VARCHAR NOT NULL,
    player2_name              VARCHAR NOT NULL,
    player1_country_acr       VARCHAR,
    player2_country_acr       VARCHAR,
    player1_seed              VARCHAR,
    player2_seed              VARCHAR,

    scheduled_start_utc       TIMESTAMP,
    ingested_at               TIMESTAMP NOT NULL,

    UNIQUE (source, fixture_external_id)
);
"""

# One row per refresh execution against any source. Drives the
# "data is N hours stale" warning in the UI and tracks per-source
# request usage against quota caps.
INGESTION_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id          VARCHAR PRIMARY KEY,
    source          VARCHAR NOT NULL,
    tour            VARCHAR,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    status          VARCHAR NOT NULL,
    rows_added      INTEGER,
    rows_skipped    INTEGER,
    rows_failed     INTEGER,
    requests_used   INTEGER,
    error_message   VARCHAR,
    notes           VARCHAR
);
"""

# Phase 4.1: full FeatureVector layout — v2, 39 features. See
# `src/tennis_predictor/features/schema.py` for the Pydantic contract.
# Required columns (NOT NULL): Elo (3), H2H wins (2), fatigue (4), ranking
# (3), tournament context (3), handedness (2 — default 'U' when unknown),
# plus identifying/label columns. Nullable: recent form (4), serve/return
# rolling (8), h2h_recency_days, age (4), height (3), recovery (2). The
# set of nullable columns matches the FeatureVector fields that allow None.
TRAINING_FEATURES_DDL = """
CREATE TABLE IF NOT EXISTS training_features (
    match_id                      VARCHAR PRIMARY KEY,
    tour                          VARCHAR NOT NULL,
    match_date                    DATE NOT NULL,
    p1_player_id                  VARCHAR NOT NULL,
    p2_player_id                  VARCHAR NOT NULL,
    label_winner_is_p1            INTEGER NOT NULL,

    -- Surface-Elo (3)
    elo_p1_surface                DOUBLE NOT NULL,
    elo_p2_surface                DOUBLE NOT NULL,
    elo_diff_surface              DOUBLE NOT NULL,

    -- Recent form (4) — nullable when window < 3 matches
    win_pct_last10_p1             DOUBLE,
    win_pct_last10_p2             DOUBLE,
    win_pct_last25_surface_p1     DOUBLE,
    win_pct_last25_surface_p2     DOUBLE,

    -- Serve/return rolling (8) — nullable when < 5 stat-rich matches in window
    first_serve_win_pct_p1        DOUBLE,
    first_serve_win_pct_p2        DOUBLE,
    second_serve_win_pct_p1       DOUBLE,
    second_serve_win_pct_p2       DOUBLE,
    bp_saved_pct_p1               DOUBLE,
    bp_saved_pct_p2               DOUBLE,
    bp_converted_pct_p1           DOUBLE,
    bp_converted_pct_p2           DOUBLE,

    -- H2H (3) — recency_days nullable when never met
    h2h_p1_wins                   INTEGER NOT NULL,
    h2h_p2_wins                   INTEGER NOT NULL,
    h2h_recency_days              INTEGER,

    -- Fatigue (4)
    fatigue_matches_7d_p1         INTEGER NOT NULL,
    fatigue_matches_7d_p2         INTEGER NOT NULL,
    fatigue_sets_14d_p1           INTEGER NOT NULL,
    fatigue_sets_14d_p2           INTEGER NOT NULL,

    -- Ranking (3) — 9999 sentinel for unranked
    rank_p1                       INTEGER NOT NULL,
    rank_p2                       INTEGER NOT NULL,
    rank_diff                     INTEGER NOT NULL,

    -- Tournament context (3)
    tournament_level              VARCHAR NOT NULL,
    best_of                       INTEGER NOT NULL,
    surface                       VARCHAR NOT NULL,

    -- Phase 4.1: handedness (2) — default 'U' when missing from players JOIN
    hand_p1                       VARCHAR NOT NULL DEFAULT 'U',
    hand_p2                       VARCHAR NOT NULL DEFAULT 'U',

    -- Phase 4.1: age (4) — nullable when players.dob is missing
    age_p1                        DOUBLE,
    age_p2                        DOUBLE,
    age_vs_peak_p1                DOUBLE,
    age_vs_peak_p2                DOUBLE,

    -- Phase 4.1: height (3) — nullable when players.height is missing
    height_p1                     INTEGER,
    height_p2                     INTEGER,
    height_diff_cm                INTEGER,

    -- Phase 4.1: recovery (2) — nullable when no prior completed match; capped 365
    days_since_last_match_p1      INTEGER,
    days_since_last_match_p2      INTEGER,

    schema_version                INTEGER NOT NULL DEFAULT 2
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_matches_winner_date "
    "ON matches(winner_player_id, tourney_date);",
    "CREATE INDEX IF NOT EXISTS idx_matches_loser_date ON matches(loser_player_id, tourney_date);",
    "CREATE INDEX IF NOT EXISTS idx_matches_date_tour ON matches(tourney_date, tour);",
    "CREATE INDEX IF NOT EXISTS idx_matches_tier ON matches(match_tier);",
    "CREATE INDEX IF NOT EXISTS idx_players_tour ON players(tour);",
    "CREATE INDEX IF NOT EXISTS idx_rankings_player_date ON rankings(player_id, ranking_date);",
    "CREATE INDEX IF NOT EXISTS idx_llm_traces_ts ON llm_traces(ts);",
    "CREATE INDEX IF NOT EXISTS idx_scheduled_matches_start "
    "ON scheduled_matches(scheduled_start_utc);",
    "CREATE INDEX IF NOT EXISTS idx_scheduled_matches_composite "
    "ON scheduled_matches(tournament_external_id, player1_external_id, "
    "player2_external_id, round_external_id);",
    "CREATE INDEX IF NOT EXISTS idx_ingestion_runs_source_started "
    "ON ingestion_runs(source, started_at);",
    "CREATE INDEX IF NOT EXISTS idx_training_features_tour_date "
    "ON training_features(tour, match_date);",
]

TABLE_DDL: list[str] = [
    MATCHES_DDL,
    PLAYERS_DDL,
    RANKINGS_DDL,
    PLAYER_ALIASES_DDL,
    MARKET_IMPLIED_PROBABILITIES_DDL,
    LLM_TRACES_SEQUENCE_DDL,
    LLM_TRACES_DDL,
    ELO_STATE_DDL,
    LAST_MATCH_STATE_DDL,
    TRAINING_FEATURES_DDL,
    SCHEDULED_MATCHES_DDL,
    INGESTION_RUNS_DDL,
]

EXPECTED_TABLES: frozenset[str] = frozenset(
    {
        "matches",
        "players",
        "rankings",
        "player_aliases",
        "market_implied_probabilities",
        "llm_traces",
        "elo_state",
        "last_match_state",
        "training_features",
        "scheduled_matches",
        "ingestion_runs",
    }
)

# Phase 4.1 v2 column added that is the cleanest "did v2 already land" marker.
# Picked from the player-metadata block — `days_since_last_match_p1` is one
# of the new columns and is not a name that could pre-exist in any earlier
# placeholder shape.
_V2_SENTINEL_COLUMN: str = "days_since_last_match_p1"


def _migrate_training_features(conn: duckdb.DuckDBPyConnection) -> None:
    """Idempotent migration of `training_features` to the v2 (Phase 4.1) shape.

    Two situations to handle:

    1. **Phase 1 placeholder** — original `(match_id, label_winner_is_p1,
       schema_version)` skeleton, no `tournament_level` column. Always
       empty. DROP it.

    2. **Phase 3 shape (v1, 28 features)** — populated, but missing the
       Phase 4.1 v2 columns (handedness, age, height, recovery). Per the
       Phase 4.1 design doc we always re-run `scripts/build_features.py`
       after a feature change, so the rows are about to be rewritten —
       DROP and re-create with the v2 layout. Detection: table has
       `tournament_level` but no `days_since_last_match_p1`.

    Once the v2 layout is in place, this function is a no-op.
    """
    table_exists = (
        conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'training_features'"
        ).fetchone()
        is not None
    )
    if not table_exists:
        return

    has_tournament_level = (
        conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'training_features' AND column_name = 'tournament_level'"
        ).fetchone()
        is not None
    )
    if not has_tournament_level:
        # Phase 1 placeholder — never populated, safe to drop.
        conn.execute("DROP TABLE training_features")
        return

    has_v2_sentinel = (
        conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'training_features' AND column_name = ?",
            [_V2_SENTINEL_COLUMN],
        ).fetchone()
        is not None
    )
    if not has_v2_sentinel:
        # Phase 3 v1 shape — feature set changed in Phase 4.1, rebuild.
        # `scripts/build_features.py` is always re-run after a feature
        # change, so the rows are about to be regenerated.
        conn.execute("DROP TABLE training_features")


# Phase 5 columns added to `llm_traces`. Listed as (column_name, DDL_type)
# tuples so the migration helper can ALTER an existing table without doing
# a full drop-and-recreate (the table accumulates user-visible history).
_LLM_TRACES_PHASE5_COLUMNS: tuple[tuple[str, str], ...] = (
    ("web_search_count", "INTEGER"),
    ("estimated_cost_usd", "DOUBLE"),
)


def _migrate_llm_traces(conn: duckdb.DuckDBPyConnection) -> None:
    """Idempotent migration of `llm_traces` to the Phase 5 shape.

    Adds `web_search_count` and `estimated_cost_usd` to a pre-existing
    `llm_traces` table when they're missing. Existing rows survive with
    NULLs in the new columns (back-fill is unnecessary — Phase 4 and
    earlier never logged web-search counts or estimated cost). No-op on a
    fresh DB; `CREATE TABLE IF NOT EXISTS` then materialises the up-to-date
    layout.
    """
    table_exists = (
        conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'llm_traces'"
        ).fetchone()
        is not None
    )
    if not table_exists:
        return
    existing_cols = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'llm_traces'"
        ).fetchall()
    }
    for col_name, col_type in _LLM_TRACES_PHASE5_COLUMNS:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE llm_traces ADD COLUMN {col_name} {col_type}")


def create_all_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create every table and index. Idempotent."""
    _migrate_training_features(conn)
    _migrate_llm_traces(conn)
    for ddl in TABLE_DDL:
        conn.execute(ddl)
    for idx in INDEXES:
        conn.execute(idx)
