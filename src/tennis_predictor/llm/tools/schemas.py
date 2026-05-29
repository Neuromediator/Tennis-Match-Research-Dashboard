"""Pydantic I/O contracts for every LLM tool.

These models are the source of truth for every tool's input and output. The
Anthropic tool-definition layer (built in `llm/tools/*.py`) consumes the
Pydantic JSON schema; the agent loop validates tool inputs through Pydantic
on the way in and serialises Pydantic outputs to JSON on the way back.

# Why frozen + extra="forbid" everywhere

- `frozen=True` ensures a validated input can't be mutated mid-iteration —
  the LLM's tool-call dict is treated as an immutable record.
- `extra="forbid"` catches a typo / hallucinated field at the Pydantic
  layer before it silently becomes a no-op (CLAUDE.md hard rule #4 lives
  on the same principle for AgentResponse).

# Canonical player resolution

All non-`web_search` tools accept human-friendly player names — the agent
sees and reasons about names, not internal IDs. Resolution to
`canonical_player_id` happens inside each tool via `player_aliases`. If a
name is ambiguous (two active players share `full_name`) or unknown, the
tool raises `PlayerResolutionError`; the agent surfaces the failure in
`caveats` rather than fabricating a match.

# Tour scoping

Every input carries `tour` (`"ATP"` or `"WTA"`). The `player_aliases` table
is keyed by tour, so resolution would be ambiguous without it. The agent
infers `tour` from the `MatchContext` it receives at the start of the
prediction.

# AgentResponse lives elsewhere

The final structured output (`AgentResponse`) is defined alongside the
`submit_analysis` tool in `llm/tools/submit.py` so its Pydantic shape and
JSON-schema shape stay in lockstep. This file deliberately has no
`AgentResponse` import — keeping that direction of dependency one-way
prevents accidental coupling of tool-input validation to the output
contract.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tennis_predictor.features.schema import Surface, TournamentLevel

Tour = Literal["ATP", "WTA"]

# Phase 6.1 — completion-status sentinel for per-match rows. Mirrors
# `matchstat.parse_completion_status` so the view layer can render a
# small badge (RET / W/O / DEF) next to the score.
MatchCompletionStatus = Literal["W", "RET", "WO", "DEF"]

# Phase 6.1 — news category whitelist for the bounded LLM job. `other`
# is a fallback the agent may emit when it can't fit an item; the agent
# loop drops `other`-tagged items before returning to the caller.
NewsCategory = Literal[
    "injury",
    "withdrawal",
    "illness",
    "result",
    "coach_change",
    "personal",
    "other",
]

# Phase 6.1 — which player(s) a news item refers to.
NewsPlayerSubject = Literal["player_a", "player_b", "both"]

# Phase 6.1 — how the LLM agent classifies its news lookup outcome.
NewsLookupStatus = Literal["ok", "no_results", "failed", "budget_exhausted"]
# Phase 7: `budget_exhausted` is an INTERNAL value never emitted by the
# LLM. It's set by `_cached_predict` when `DAILY_LLM_BUDGET` is reached
# and the agent loop is skipped in favour of a direct model-only call.
# The `submit_analysis` JSON-schema enum in submit.py deliberately
# excludes it so the LLM can't fabricate this status.


# ---------------------------------------------------------------------------
# Agent input — the match the user wants a prediction for.
# ---------------------------------------------------------------------------


class MatchContext(BaseModel):
    """Single fixture handed to `TennisAgent.predict`. Built from either
    a `scheduled_matches` row or the CLI's free-form `--player-a / -b ...`
    arguments. The agent never edits this object — it threads it through
    tool calls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tour: Tour
    player_a_name: str = Field(min_length=1)
    player_b_name: str = Field(min_length=1)
    surface: Surface
    tournament_level: TournamentLevel
    tournament_name: str | None = None
    best_of: Literal[3, 5]
    match_date: date
    # `scheduled_match_id` is informational only — included so the trace
    # row can be JOINed back to `scheduled_matches` later. Free-form CLI
    # invocations leave it None.
    scheduled_match_id: str | None = None


# ---------------------------------------------------------------------------
# Tool inputs (one model per tool — these become the tool's input_schema).
# ---------------------------------------------------------------------------


class GetPlayerStatsInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    player_name: str = Field(min_length=1)
    tour: Tour
    as_of_date: date


class GetHeadToHeadInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    player_a_name: str = Field(min_length=1)
    player_b_name: str = Field(min_length=1)
    tour: Tour
    as_of_date: date


class GetRecentFormInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    player_name: str = Field(min_length=1)
    tour: Tour
    as_of_date: date
    n_matches: int = Field(default=10, ge=1, le=25)


class GetPlayerRankingInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    player_name: str = Field(min_length=1)
    tour: Tour
    as_of_date: date


class GetModelPredictionInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    player_a_name: str = Field(min_length=1)
    player_b_name: str = Field(min_length=1)
    tour: Tour
    surface: Surface
    tournament_level: TournamentLevel
    best_of: Literal[3, 5]
    match_date: date


class GetSurfaceEloInput(BaseModel):
    """Phase 6.1: LLM-callable tool returning both players' surface Elo
    in a single round-trip (saves one tool iteration vs. issuing two
    separate `get_player_stats`-style calls)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_a_name: str = Field(min_length=1)
    player_b_name: str = Field(min_length=1)
    tour: Tour
    surface: Surface
    as_of_date: date


# ---------------------------------------------------------------------------
# Tool outputs — the structured payload the tool returns to the LLM.
# ---------------------------------------------------------------------------


class PlayerStats(BaseModel):
    """Career-level summary for one player, as of `as_of_date`. Surfaces
    only completed tour-level matches (the same population used to train
    the model)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_player_id: str
    player_name: str
    tour: Tour
    as_of_date: date

    career_matches: int = Field(ge=0)
    career_wins: int = Field(ge=0)
    career_losses: int = Field(ge=0)
    career_win_pct: float | None = Field(default=None, ge=0.0, le=1.0)

    # Per-surface tallies. Keys are the canonical `Surface` literal values.
    # A missing surface key means the player has zero completed matches
    # there; surfaces with non-zero count carry both fields.
    surface_matches: dict[str, int] = Field(default_factory=dict)
    surface_win_pct: dict[str, float] = Field(default_factory=dict)


class HeadToHeadMatch(BaseModel):
    """One row in the H2H history list, oldest-to-newest within the parent
    `HeadToHeadResult`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    match_date: date
    surface: str | None = None
    tournament_name: str | None = None
    tournament_level: str | None = None
    round_name: str | None = None
    winner_name: str
    score: str | None = None


class HeadToHeadResult(BaseModel):
    """H2H wrapper carrying the aggregate win counts and the per-meeting
    detail rows. An empty `matches` list paired with both counts at 0 is a
    legitimate "never met" signal — the agent should phrase it that way,
    not as a missing-data error (failure-mode 2 in CLAUDE.md)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_a_name: str
    player_b_name: str
    tour: Tour
    player_a_wins: int = Field(ge=0)
    player_b_wins: int = Field(ge=0)
    matches: list[HeadToHeadMatch] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 6.1: detailed H2H + Surface Elo for the new bounded agent.
# ---------------------------------------------------------------------------


class H2HMatchDetail(BaseModel):
    """One row in `H2HSummary.matches`. Extends the legacy `HeadToHeadMatch`
    with pre-match odds and a parsed completion-status sentinel so the
    Prediction page can render `RET` / `W/O` / `DEF` badges and surface
    upset-by-odds context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    match_date: date
    tournament_name: str | None = None
    round_name: str | None = None
    surface: str | None = None
    winner_player_id: str | None = None
    winner_name: str | None = None
    score: str | None = None
    odds_winner: float | None = Field(default=None, ge=1.0)
    odds_loser: float | None = Field(default=None, ge=1.0)
    completion_status: MatchCompletionStatus = "W"


class H2HSummary(BaseModel):
    """Phase 6.1 replacement for `HeadToHeadResult`. Carries the same
    aggregate counts plus a per-surface breakdown and an explicit
    `data_source` tag so the UI can show a "matchstat quota exhausted —
    using Sackmann" banner when the cold-layer fallback fires."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_a_name: str
    player_b_name: str
    player_a_id: str
    player_b_id: str
    tour: Tour
    player_a_wins: int = Field(ge=0)
    player_b_wins: int = Field(ge=0)
    # Per-surface (a_wins, b_wins) tuples. Keys are canonical Surface
    # literals; only surfaces with at least one meeting appear.
    by_surface: dict[str, tuple[int, int]] = Field(default_factory=dict)
    matches: list[H2HMatchDetail] = Field(default_factory=list)
    data_source: Literal["matchstat", "sackmann"]
    fetched_at: datetime


class SurfaceEloSummary(BaseModel):
    """Output of `get_surface_elo`. `baseline_prob_a` is the logistic
    transform of the rating diff (1 / (1 + 10**(-diff/400))) — the same
    formula `EloState.expected_score` uses internally so the agent and
    the trained model are looking at the same baseline."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    player_a_name: str
    player_b_name: str
    player_a_id: str
    player_b_id: str
    tour: Tour
    surface: Surface
    as_of_date: date
    player_a_elo: float
    player_b_elo: float
    diff_a_minus_b: float
    baseline_prob_a: float = Field(ge=0.0, le=1.0)
    elo_state_snapshot_date: date | None = None


class RecentMatchDetail(BaseModel):
    """Phase 6.1: enriched per-match row used by the view-layer
    `recent_form_table_two_column` widget. Same shape as `RecentMatch`
    plus odds + completion-status badge. Not an LLM-callable schema —
    the view layer builds it directly from matchstat or Sackmann data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    match_date: date
    opponent_name: str
    result: Literal["W", "L"]
    surface: str | None = None
    tournament_name: str | None = None
    round_name: str | None = None
    score: str | None = None
    odds_self: float | None = Field(default=None, ge=1.0)
    odds_opponent: float | None = Field(default=None, ge=1.0)
    completion_status: MatchCompletionStatus = "W"


class RecentFormPayload(BaseModel):
    """Wrapper returned by `fetch_recent_n_matches` so the view layer can
    render a "matchstat (18m ago)" / "Sackmann (lag up to 7d)" footnote
    without re-deriving freshness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: str
    player_name: str
    tour: Tour
    as_of_date: date
    matches: list[RecentMatchDetail] = Field(default_factory=list)
    data_source: Literal["matchstat", "sackmann"]
    fetched_at: datetime
    matchstat_quota_used: int | None = None
    matchstat_quota_cap: int | None = None


# ---------------------------------------------------------------------------
# Phase 6.1: NewsItem — the LLM's only output channel for prose.
# ---------------------------------------------------------------------------


class NewsItem(BaseModel):
    """One piece of news returned by the bounded LLM analyst. Title + URL
    + snippet come from Tavily; `source_domain`, `published_date`,
    `player_subject`, `category` are the LLM's structured tags.

    `published_date` is a free-form string (not parsed `date`) because
    Tavily occasionally returns partial dates like "2026" or "May 2026"
    that don't fit `date.fromisoformat`. The post-validate filter in the
    agent loop parses these heuristically when deciding whether to drop
    the item for being older than 32 days."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=1, max_length=2000)
    snippet: str = Field(min_length=1, max_length=2000)
    published_date: str | None = Field(default=None, max_length=64)
    source_domain: str = Field(min_length=1, max_length=200)
    player_subject: NewsPlayerSubject
    category: NewsCategory


class RecentMatch(BaseModel):
    """One row in `RecentFormSummary.last_matches`. Result is from the
    perspective of the player named in the parent summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    match_date: date
    opponent_name: str
    result: Literal["W", "L"]
    surface: str | None = None
    tournament_name: str | None = None
    tournament_level: str | None = None
    round_name: str | None = None
    score: str | None = None


class RecentFormSummary(BaseModel):
    """Recent-N form, newest-first. `n_returned` may be less than the
    requested `n_matches` for debutants or returning players — the agent
    handles that as context (failure-mode 2).

    `latest_match_date` and `data_freshness_warning` exist so the agent
    can detect the Phase 2 cold-data lag (Sackmann lags 1-7 days for active
    tour players). Without them, the agent treats whatever is in the DB
    as "current form" even when the player has played matches in the gap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_player_id: str
    player_name: str
    tour: Tour
    as_of_date: date

    n_requested: int = Field(ge=1, le=25)
    n_returned: int = Field(ge=0, le=25)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    win_pct: float | None = Field(default=None, ge=0.0, le=1.0)

    last_matches: list[RecentMatch] = Field(default_factory=list)

    # Newest `match_date` across `last_matches`. None for debutants.
    latest_match_date: date | None = Field(default=None)
    # Set when `as_of_date - latest_match_date > 7 days`. The agent MUST
    # surface this in `caveats` and prefer web search over the DB record
    # for any "current form" claim. Phase 2 documented a 1-7 day Sackmann
    # ingestion lag; this field exposes it to the prompt.
    data_freshness_warning: str | None = Field(default=None)


class RankingSnapshot(BaseModel):
    """ATP / WTA ranking on or just before `as_of_date`. `rank` is None
    when the player is unranked at that date (Sackmann rankings only cover
    players with at least one official ATP / WTA point)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_player_id: str
    player_name: str
    tour: Tour
    as_of_date: date
    rank: int | None = Field(default=None, ge=1)
    points: int | None = Field(default=None, ge=0)
    # The `rankings` table stores weekly snapshots; we record the actual
    # snapshot date so the agent can mention "ranked X as of D" with the
    # right date instead of pretending `as_of_date` is exact.
    snapshot_date: date | None = None


class ModelFeatureSummary(BaseModel):
    """Compact view of the FeatureVector for narrative use. Field set is
    intentionally small and stable — names and meanings rarely change so
    prompt-cache invalidation stays a non-issue.

    `player_a_*` / `player_b_*` here refer to the user-facing players in
    `MatchContext`, NOT the internal lex-canonical p1/p2. The
    `get_model_prediction` tool rewrites the canonical p1/p2 vector into
    the user-facing labelling before returning."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    elo_player_a: float
    elo_player_b: float
    elo_diff_a_minus_b: float

    rank_player_a: int = Field(ge=1, le=9999)
    rank_player_b: int = Field(ge=1, le=9999)

    h2h_player_a_wins: int = Field(ge=0)
    h2h_player_b_wins: int = Field(ge=0)

    # Optional / nullable mirrors of the underlying FeatureVector fields.
    win_pct_last10_player_a: float | None = Field(default=None, ge=0.0, le=1.0)
    win_pct_last10_player_b: float | None = Field(default=None, ge=0.0, le=1.0)
    win_pct_last25_surface_player_a: float | None = Field(default=None, ge=0.0, le=1.0)
    win_pct_last25_surface_player_b: float | None = Field(default=None, ge=0.0, le=1.0)

    fatigue_matches_7d_player_a: int = Field(ge=0)
    fatigue_matches_7d_player_b: int = Field(ge=0)

    days_since_last_match_player_a: int | None = Field(default=None, ge=0, le=365)
    days_since_last_match_player_b: int | None = Field(default=None, ge=0, le=365)


class ModelPrediction(BaseModel):
    """Output of `get_model_prediction` — the source of truth for the
    probability the user sees (CLAUDE.md hard rule #4).

    `feature_summary` exposes a small, named subset of the FeatureVector
    so the LLM can build a narrative ("Elo edge on clay for player A is
    +120, never met on clay before") without us shipping all 39 fields and
    re-deriving them on the LLM side. Keeping it a Pydantic submodel
    (rather than `dict[str, Any]`) keeps the JSON-schema declaration tight.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    player_a_name: str
    player_b_name: str
    tour: Tour
    surface: Surface
    tournament_level: TournamentLevel
    best_of: Literal[3, 5]
    match_date: date

    model_probability_player_a: float = Field(ge=0.0, le=1.0)
    model_probability_player_b: float = Field(ge=0.0, le=1.0)

    model_artifact_version: str
    feature_summary: ModelFeatureSummary


# ---------------------------------------------------------------------------
# Errors a tool can raise. The agent loop catches each below and surfaces
# them as either tool error blocks (DB empty, name not resolvable) or
# fatal exceptions (model missing — CLAUDE.md hard rule #10).
# ---------------------------------------------------------------------------


class PlayerResolutionError(Exception):
    """Raised when `player_aliases` cannot uniquely resolve a name to a
    canonical_player_id. Caught by the agent loop and turned into a tool
    error block so the LLM can mention the unresolved player in caveats."""


class ModelUnavailableError(Exception):
    """Raised when the model artifact is missing, unloadable, or the
    `predict_proba` call fails for any reason. CLAUDE.md hard rule #10:
    the LLM agent MUST NOT be invoked when this fires. The CLI catches
    it before the agent loop starts."""


class TavilyError(Exception):
    """Raised by `web_search` / `fetch_url` on any Tavily HTTP failure
    (5xx, 4xx, JSON parse error). Caught by the agent loop and surfaced
    as a tool_result error block so the LLM can mention the lookup
    failure in `caveats` — CLAUDE.md failure-mode #1 (web search error)
    and #7 (fetch_url error)."""


# ---------------------------------------------------------------------------
# Phase 5.1: web_search and fetch_url — Tavily-backed client-side tools.
# Replaces Anthropic native `web_search_20250305` (Phase 5).
# ---------------------------------------------------------------------------


class WebSearchInput(BaseModel):
    """LLM-supplied arguments for one Tavily search.

    `max_results` is bounded at 10 — Tavily's basic plan returns up to 10
    per call and going higher just costs us more without surfacing
    additional value (the agent rarely uses beyond the top 5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchHit(BaseModel):
    """One result row in `WebSearchOutput.results`. Snippet is the cleaned
    text fragment Tavily returns (~150-300 chars). Anthropic's native
    search hid this in `encrypted_content`; Tavily exposes it so the
    `llm_traces` dashboard can show the actual content the agent saw."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    url: str
    snippet: str
    # Tavily returns ISO-8601 strings when known; left as `str | None`
    # rather than `date | None` because the field is heuristic (Tavily
    # parses publication timestamps from page meta, occasionally wrong).
    published_date: str | None = None


class WebSearchOutput(BaseModel):
    """Output of `web_search`. `cost_usd` is the per-call Tavily charge,
    accumulated by the agent loop into `llm_traces.estimated_cost_usd`
    alongside the Anthropic line items."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    results: list[WebSearchHit] = Field(default_factory=list)
    cost_usd: float = Field(ge=0.0)


class FetchUrlInput(BaseModel):
    """LLM-supplied URL for the `fetch_url` tool. Used when a `web_search`
    snippet truncates an important detail and the agent wants the cleaned
    full article body — see CLAUDE.md "Phase 5.1" guidance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1, max_length=2000)


class FetchUrlOutput(BaseModel):
    """Output of `fetch_url`. `content` is the cleaned article body Tavily
    Extract returns (typically 2-5k chars). `extraction_success` is False
    on paywalls / JS-rendered SPAs Tavily can't parse — the agent should
    mention that in `caveats` rather than fabricating around the empty body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    content: str
    extraction_success: bool
    cost_usd: float = Field(ge=0.0)
