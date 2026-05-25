# CLAUDE.md — Project conventions

This file is the contract for any Claude session working on this repo. Keep it terse, keep it accurate, update it when rules change.

## Project at a glance

A tennis match research dashboard for ATP and WTA tour-level singles. For each upcoming match it surfaces a comparison row — **market consensus odds (auto-fetched), our trained LightGBM model's probability, a surface-Elo baseline** — alongside a detailed H2H card, last-8-matches per player, and LLM-discovered recent news. When the model disagrees with the market by > 10pp, the dashboard renders a structural explanation (stale surface-Elo, activity asymmetry, returning veteran). **The model is one signal, not the answer.** Built to demonstrate end-to-end LLM agent engineering on a real domain, not to claim a market edge.

Project history: Phase 6.1 close-out exposed that the calibrated LightGBM probability is unreliable on top matches (inverted favourites at the Cina-Opelka / Sinner-Djokovic level). Phase 6.2 re-scoped from "predictor" to "context dashboard" — model output stays visible but stops being the headline framing. See `docs/tutorials/phase_6_2_notes.md` for the full rescope rationale.

Phase tracking lives in `docs/phases.md`. Architecture in `docs/architecture.md`. Evaluation philosophy in `docs/methodology.md`. Read all three before non-trivial work.

## Environment

- **Python:** 3.12+ (3.13 acceptable). Pinned in `.python-version`.
- **Package manager:** `uv`. Add deps with `uv add <pkg>`; lockfile is `uv.lock`.
- **Run anything:** `uv run <cmd>`.

## Quality gates (run before claiming work is done)

```bash
uv run ruff check .              # lint
uv run ruff format --check .     # format check
uv run pyright                   # type check
uv run pytest                    # tests
```

CI runs the same four commands. Pre-commit hook runs `ruff` and `nbstripout`.

## Hard rules (do not violate without explicit user approval)

1. **Point-in-time correctness.** Feature computation may only read data with `match_date < as_of_date`. There are tests asserting this (`tests/test_feature_leakage.py`). If a test fails, the bug is in the code, not the test.
2. **Only sanctioned feature entry points:**
   - `build_training_features()` — chronological replay producing the `training_features` table.
   - `compute_features(player_id, opponent_id, surface, tour, as_of_date) -> FeatureVector` — inference-time API. Returns a Pydantic `FeatureVector`, **not** a `dict`.
   Any feature math elsewhere is a bug.
3. **Betting odds are not training features.** Historical market-implied probabilities in `market_implied_probabilities`, AND pre-match odds in `pre_match_odds` (Phase 6.2 — auto-fetched from The Odds API for upcoming fixtures), are used **only** for UI display (the comparison row) and calibration evaluation (Dashboard reports). Never input to model training. The model and the market remain independent signals so the user can see the gap between them. If the model ever needs to be retrained, this rule still holds.
4. **The LLM does not emit a probability, and no longer emits prose.** The model's number is the only probability shown. After Phase 6.1, the LLM's only output is `news_items: list[NewsItem]` + `news_lookup_status` — every fact carries source + date + URL + category as adjacent structured metadata. No `narrative`, `confidence_band`, `caveats`, or `key_factors`. Schema rejects any LLM-supplied probability field AND any free-text synthesis field. Determinism (H2H, surface-Elo, recent form) is rendered by the view layer from typed tool outputs; only news discovery — which requires the open web — remains an LLM task.
5. **Every LLM call is logged.** The `LLMClient` writes a row to `llm_traces` (inputs, tool calls, outputs, tokens, cache stats, latency). No bypass.
6. **Prompt caching is on by default.** System prompt and tool definitions are sent as cacheable blocks in every call.
7. **Player reconciliation has a manual-review checkpoint.** Auto-match only at confidence ≥ 0.90. Confidence 0.75–0.90 → `data/processed/aliases_review.csv` for human review. Approved entries are promoted to `player_aliases` (with `source='manual_review'`) by `scripts/apply_aliases_review.py`. Never silently merge ambiguous matches.
8. **Calibration method depends on sample size.** Isotonic when held-out calibration set has ≥ 1000 samples; Platt scaling otherwise.
9. **No notebook outputs in commits.** `nbstripout` is enforced via pre-commit.
10. **`get_model_prediction` is mandatory IF the agent is invoked.** When the model artifact is unloadable or missing, the agent must NOT be invoked. However: Phase 6.2 makes the LLM agent optional from the user's perspective — the deterministic blocks (market odds, surface-Elo, H2H, recent form) render even if the agent fails. We do not block the whole page on agent unavailability. We only block invoking the agent if the model number it needs to anchor against is missing.

11. **Honest framing in user-facing copy.** The product is not a predictor. UI text, page titles, README, and sidebar tagline say "research dashboard" / "match dashboard" / "context for the matchup," never "calibrated prediction" or "we predict." This is a content rule, not a code rule, but it's load-bearing for the product's honesty. Phase 6.2 added this rule because the prior framing (Phase 1-6.1) led to user-visible inverted-favourite outputs that embarrassed the product.

12. **Acceptance test must include 10-match reality check.** No phase that touches the prediction page or the model output is closed without manually walking through 10 current upcoming matches in the dashboard and recording: (model prob, market prob, surface-Elo prob, "why model differs" panel triggered correctly, subjective 1-5 rating of analyst usefulness). Average rating < 3.5 OR any inverted favourite that the dashboard doesn't explain = phase reopened. Phase 6.2 added this rule because Phase 4's average-Brier acceptance test masked tail-failure modes that only show up on a per-match walkthrough.

## Model artifact naming

Saved under `models/<tour>/<model_type>/<YYYYMMDD-HHMM>/`:

- `model.joblib`
- `metadata.json` — training date, data date range, feature list, metrics (pre- and post-calibration), calibration method, git commit hash
- `report.md` — Markdown summary with calibration plot
- `calibration_plot.png`

Active model symlinked at `models/<tour>/<model_type>/latest`.

## Anthropic SDK

- **Default model:** `claude-sonnet-4-6`. Picked over Opus 4.7 (~5× cost, marginal quality gain on our news-discovery + categorisation task) and Haiku 4.5 (noticeably weaker on multi-source reconciliation). Configurable via `ANTHROPIC_MODEL`. After Phase 6.1 the agent's task is bounded (call 3 tools, return a typed `NewsItem` list); Sonnet remains the right tradeoff because category-tagging + dated relevance judgment still benefit from a stronger model.
- **Provider stack:** Direct `anthropic` SDK only. No abstraction-layer wrappers (LangChain, LiteLLM, OpenRouter, Anthropic Managed Agents). Vendor flexibility comes from our own `LLMClient` ABC, not from third-party multi-provider layers. Reasoning: explicit `cache_control` markers and best-in-class tool-use reliability are features abstraction layers either lose entirely or surface poorly. If a future provider beats Anthropic on this combo, swap is a new `LLMClient` implementation (~100 lines), not a framework migration.
- **`web_search` is NOT an Anthropic native server tool** (Phase 5.1 swap). It's our client-side tool against Tavily — see the "Web search" section below. The `LLMClient` doesn't know about search providers; the agent loop dispatches `web_search` / `fetch_url` like any other client tool.
- The `LLMClient` abstract base lives in `src/tennis_predictor/llm/client.py`. Use it; do not call `anthropic.Anthropic()` directly outside that module.
- **Prompt caching is on by default.** System prompt + tool definitions are sent with `cache_control` markers. Our typical agent call has ~2000 stable input tokens that cache → ~70% input-cost reduction on the second-and-later call within the 5-minute TTL window. One marker, on the last tool definition (caches everything prefix to it).
- **Cache hit hygiene.** The cacheable prefix (system prompt + tool definitions) must be byte-stable across calls. Never inject timestamps, random IDs, or session-specific data into the system prompt — current date and match context belong in the user message. A unit test on `LLMClient._build_cacheable_blocks()` enforces byte-equality between two consecutive calls; if it fails, the cache-hit rate just dropped to ~0%.
- Web search uses Claude's native `web_search` tool (no Tavily / Brave / Serper as separate dependencies).

## Web search

- **Tool (Phase 5.1):** **Tavily Search API** (`basic` depth), wrapped by our own client-side `web_search` tool in `src/tennis_predictor/llm/tools/web_search.py`. Replaced Anthropic native `web_search_20250305` after the Phase 5 A/B (`scripts/compare_search_providers.py`) showed Tavily is 9.5x cheaper per search, 3.5x faster, and matches Anthropic on niche discovery while delivering more diverse sources (Yahoo / BBC / Reddit / local press rather than Anthropic's tour-official bias).
- **`fetch_url` retired (Phase 6.1).** The Tavily Extract companion tool was removed from the agent's tool surface: snippet-only news is sufficient for the bounded 32-day window, and deep-reading whole articles encouraged the LLM to over-synthesise (the failure mode Phase 6.1 was created to fix). The `fetch_url` code remains in the repo for potential future use but is no longer registered with the agent.
- **`exclude_domains`:** small blocklist of betting / pick-of-the-day clickbait: `draftkings.com, fanduel.com, betmgm.com, pickwise.com, actionnetwork.com`. Same list Phase 5 used.
- **No `include_domains`** — we want maximum recall on legitimate news. Tavily ranking is good enough that whitelisting would hurt more than help.
- **Preferred sources** (mentioned in system prompt, not whitelist-enforced): **ESPN, BBC, tennis.com, tennis365.com**, X/Twitter when surfaced. Tour-official `atptour.com` / `wtatennis.com` are intentionally not highlighted — they surface only news everyone already has.
- **Budget per agent call (Phase 6.1):** `max_web_searches = 2` (one per player), `max_tool_iterations = 4` (get_head_to_head + get_surface_elo + 2× web_search + submit_analysis = 5 tool calls in the absolute happy path; cap is 4 for non-terminal calls — see `agent.py`). Atomic reservation via `BudgetTracker.reserve_web_search` prevents parallel tool_uses in a single turn from overshooting the cap.
- **Recency window:** the last **32 days** (Phase 6.1, was 14 in Phase 5). Enforced two ways: (a) system-prompt instruction; (b) post-validate filter in `agent.py` drops `NewsItem`s with parsed `published_date > 32 days before match_date`. Items with `published_date = None` are kept and labelled "(date unknown)".
- **Category whitelist (Phase 6.1).** Each `NewsItem` is tagged with one of: `injury / withdrawal / illness / result / coach_change / personal`. `other` is a sentinel — any item the agent tags `other` is dropped before the response is returned. `interview / sponsorship / charity / social-media-drama-without-on-court-consequence` are forbidden in the prompt and absent from the enum.
- **Anti-fabrication clause** in the system prompt: when search returns nothing material, the agent MUST emit `news_lookup_status = "no_results"` and an empty `news_items` list. It MUST NOT fabricate plausible-sounding news.
- **Cost:** Tavily basic free tier covers 1000 searches/month (~120/month at 2 searches × 60 predictions/month). Paid plan: $0.005 per search. Total search-line cost per prediction: ~$0.01.
- **HTTP transport:** `httpx.AsyncClient` with `retries=2` (built-in transport-level retry). Tavily 5xx / 4xx / timeout → typed `TavilyError`, caught by agent loop and surfaced as `news_lookup_status = "failed"` with an empty `news_items` list (failure-mode #1).

## Pre-match odds (Phase 6.2)

- **Source:** **The Odds API** (`the-odds-api.com`). Public docs at `docs.the-odds-api.com/liveapi/guides/v4/`. Free tier: **500 credits/month**, email-only signup, no credit card. Auth via new env var `THE_ODDS_API_KEY` (separate from `X_RAPIDAPI_KEY` — different vendor, different account).
- **Pinnacle Odds API on RapidAPI was considered and rejected** — the `DataMenu/pinnacle-odds-api` wrapper has no readable public docs (JS-rendered RapidAPI UI), and live probing only surfaced 2 of N endpoints. Without a documented contract we can't build reliably. The Odds API exposes Pinnacle as one of its `bookmakers[]` for EU region, so the sharp-line preference is preserved indirectly.
- **Terminology:** "pre-match" means odds for upcoming fixtures that haven't started yet. NOT "in-play" / "live" (during-match). We never fetch in-play odds.
- **What we fetch:** ATP and WTA pre-match h2h odds for currently-active tour-level tournaments. Sport keys are per-tournament (`tennis_atp_french_open`, `tennis_wta_madrid`, etc.) — daily refresh starts by listing active tennis keys via `GET /v4/sports/?all=false`, then iterates each. Decimal odds (`oddsFormat=decimal`), `regions=eu` only (1 credit per call).
- **Storage:** new `pre_match_odds` table. Stores aggregated views (median across books, best-price across books) plus Pinnacle's specific price when present (separate `pinnacle_odds_*` columns). UI headline number = median; Pinnacle shown as subtitle.
- **Linking to scheduled fixtures:** name + UTC-date + tour matching. Both The Odds API (`home_team`/`away_team`) and matchstat (`player1_name`/`player2_name`) use canonical full names. Set-membership match on `{lowercased_a, lowercased_b}` avoids ordering issues. `tour` derived from `sport_key` prefix.
- **Refresh cadence:**
  - Daily batch via `scripts/refresh_pre_match_odds.py`: list active tennis keys + 1 odds call per key (~4-6 credits/day = ~120-180/month).
  - Lazy refresh on Prediction-page load when no row exists OR the cached row is > 24h old (~30 credits/month).
  - **Explicitly NOT doing** per-match re-fetch when match starts in < 1h — that's line-shopping, not analyst-dashboard.
  - Total expected: 150-210 credits/month against 500 cap → ~60% headroom.
- **Fallback chain when The Odds API has no row:** (1) Tavily snippet-search for the matchup with regex extraction of decimal-odds patterns (flagged `source='tavily'` in UI), (2) UI shows "Market: odds unavailable" with no number. Never user-input — the contract is the user does not enter odds.
- **Not a training feature.** See hard rule #3. Odds are display + calibration evaluation only.

## LLM agent failure modes

The agent has seven failure surfaces, each handled distinctly:

1. **`web_search` errors or returns empty** — degraded mode. The agent returns `news_lookup_status = "failed"` (Tavily error) or `"no_results"` (empty result set) with an empty `news_items` list. Never aborts the prediction; the view layer still renders model probability + H2H + Elo + recent form.
2. **DB tool returns empty data** (no H2H history, debutant with no recent form, unranked player) — normal signal, not a failure. `get_head_to_head` returns `player_a_wins=0, player_b_wins=0, matches=[]` which the view layer renders as "never met". The view layer also renders "no completed matches in DB" if recent form is empty for a debutant.
3. **DB tool raises an exception** — bubbled up out of `LLMClient`. Never silently masked. Streamlit surfaces an error page; we never ship a prediction built on partially-corrupted data.
4. **Anthropic API errors (5xx / 429)** — handled by the SDK's built-in retry with exponential backoff (`max_retries=2`). After exhaustion, the exception bubbles up to the caller.
5. **`get_model_prediction` unavailable** — **fatal**, see hard rule #10. The model number is mandatory; the agent is not invoked at all.
6. **Agent loop timeout (120 s hard cap)** — the entire agent loop is wrapped in `asyncio.timeout(120)`. On timeout we log partial state to `llm_traces` and surface "prediction timed out" to the user.
7. **matchstat free-tier quota exhausted (Phase 6.1)** — when `MatchstatLiveFetcher` sees `matchstat_quota.requests_used >= 480` or receives a 429 from the API, it raises `MatchstatBudgetExceeded`. The H2H tool catches this and falls back to Sackmann (cold layer), with `H2HSummary.data_source = "sackmann"`. The recent-form view-layer helper does the same. UI surfaces a small banner: "matchstat quota exhausted for this month — using cold data (lag up to 7 days)". Prediction proceeds.

**Twitter is not a directly usable source** — X.com is behind an auth wall and `web_search` returns either headlines or login redirects. We do not advertise Twitter in the system prompt; tweet content reaches us only when journalist sources quote or embed.

## Structured output discipline

- LLM structured output uses the **tool-use pattern**: a single `submit_analysis` tool whose `input_schema` mirrors `AgentResponse`, with `additionalProperties: false`. Forbidden fields (any `probability`-like field per hard rule #4, AND any free-text synthesis field per Phase 6.1 — `narrative`, `confidence_band`, `caveats`, `key_factors`, `summary`, `analysis`, etc.) are blocked at the JSON-schema layer before Pydantic ever sees them.
- Validation pipeline per agent call: `tool_choice` forces `submit_analysis` on the final iteration → `AgentResponse.model_validate(tool_call.input)` → Pydantic enforces Literal / min-max constraints as the second wall.
- Post-validation filter (Phase 6.1): items in `news_items` tagged `category = "other"` are dropped (they signal the agent couldn't fit the item into the real whitelist categories); items with `published_date` parsing to > 32 days before `match_date` are dropped.
- Never use `response_format={"type": "json_object"}` (unstructured) or free-text JSON parsing — both have failure modes that the tool-use pattern eliminates by construction.
- Agentic loop uses hybrid `tool_choice`: `"auto"` while the agent gathers data (`get_head_to_head`, `get_surface_elo`, `web_search` × 2), hard-forced to `submit_analysis` on the final iteration to guarantee termination within a bounded tool-call budget (Phase 6.1: 4 iterations max).

## Testing the LLM agent

Three-tier test strategy:

1. **Unit tests** mock `anthropic.Anthropic` (using `respx` or `pytest-mock`). Cover: request body construction, cache-marker placement, tool-call parsing, `llm_traces` row format. Run in CI on every push. Free.
2. **Recorded-fixture e2e tests** in `tests/fixtures/llm/*.json` cover one happy path, one degraded path (web_search error), and one edge case (empty H2H). Replayed deterministically in CI. Re-record when Anthropic API contract or our tool schemas change: `uv run pytest -m llm_record` (manual, paid).
3. **Live API tests** under `@pytest.mark.llm_live`, excluded by default (`addopts = -m "not llm_live"` in `pyproject.toml`). Run locally before pushing a phase or when validating a system-prompt change: `uv run pytest -m llm_live`. Never in CI — protects budget and avoids API-key exposure surface.

CI must NOT have `ANTHROPIC_API_KEY` as a secret. The unit + fixture tiers never need it; the live tier should fail fast with "API key not set" if accidentally invoked outside a local dev session.

## Budget discipline

Three-layer protection against runaway spend, **all must be in place**:

1. **Org-level hard cap (Anthropic console, manual setup).** A dedicated workspace for this project with a monthly budget limit — currently set to **$20/month**. This is a hard 429 ceiling, not an alert. Without this, any code-side bug or key leak can spend unbounded. Setup is the user's responsibility (one-time UI action, no code involved).
2. **Per-agent-call cap (Phase 6.1 tightened).** Each `TennisAgent.predict()` enforces an `AgentBudget` with four hard limits:
   - `max_tool_iterations = 4` (down from 6 — happy path is `get_head_to_head` + `get_surface_elo` + `web_search` × 2, then forced `submit_analysis`)
   - `max_total_tokens = 30_000` (input + output sum per agent call)
   - `max_wall_clock_seconds = 120`
   - `max_web_searches = 2` (down from 3 — one per player)

   Exceeding any limit forces `submit_analysis` on the next turn (if the iteration budget still allows it) or raises `BudgetExceededError`.
3. **matchstat free-tier accounting (Phase 6.1).** `matchstat_quota` table tracks month-to-date `requests_used`. `MatchstatLiveFetcher` increments on every fresh fetch; at `requests_used >= 480` (10-call buffer below the 500/month free-tier cap) it raises `MatchstatBudgetExceeded` before issuing the request. Callers catch → fall back to Sackmann cold layer. The hot-refresh script logs its own `requests_used` into `ingestion_runs`; the prediction path logs the running total to `llm_traces.matchstat_requests_used` so the Dashboard can show "M/500 matchstat requests used this month".
4. **Per-session / per-IP throttling (Streamlit, Phase 7).** Public-URL abuse protection. Not in Phase 6.1 scope — deferred to deployment.

The `llm_traces` row records `web_search_count` and `estimated_cost_usd` per trace so Streamlit can surface "spent today / this month" to the user.

## Time zones in `scheduled_matches`

matchstat returns fixture timestamps with a trailing `Z` but the wall-clock is empirically **Moscow time, not UTC**. We verified this against several events (including non-Moscow tournaments like Roland Garros): the offset is consistently +3h regardless of where the tournament is played, so the labelled TZ is matchstat-organiser-local-time and not per-event-venue-local-time.

- **Ingestion default (Phase 6.1):** `MATCHSTAT_SOURCE_TZ = Europe/Moscow`. Override via env var if matchstat fixes their labelling. `_to_naive_utc` in `load_hot.py` reads this and converts wall-clock → real UTC at write time.
- **Display:** never label naive timestamps as "UTC" in the UI. Use `format_match_time_for_display(utc_dt)` which returns `"Sun, May 25 — 11:00 CEST (09:00 UTC)"`. CEST is shown as the user-facing TZ; UTC kept in parentheses as the unambiguous reference.
- **One-shot migration after default flip:** existing rows are off by 3h. Run `uv run python scripts/clear_scheduled_matches.py` (truncates `scheduled_matches`) then `uv run python scripts/refresh_hot.py` once.

## How we work

- **Phased development.** Each phase has entry criteria, deliverables, and exit criteria in `docs/phases.md`. Don't start the next phase until the current phase's exit criteria are green.
- **Skills.** Domain conventions live in `.claude/skills/<name>/SKILL.md`: `data-ingestion`, `feature-engineering`, `model-training`, `llm-tools`. Read the relevant skill before touching that area.
- **Chat language.** Responses to the user are in Russian; all code, docs, comments, commit messages are in English.
- **Cold-layer refresh.** `uv run python scripts/refresh_data.py` is incremental (idempotent). Pass `--clean` to delete the DuckDB file and rebuild from scratch. Audit artefacts live in `data/processed/`: `aliases_review.csv` (low-confidence resolutions awaiting human verdict) and `unmatched_market_rows.csv` (rows tennis-data.co.uk reported but we couldn't join — analyzed in `notebooks/explore_unmatched.ipynb`).

## What to avoid
-  Never read `.env`. It contains live API keys. Reading it exposes the values in this conversation's transcript. If you need to know which api keys are inside this file, ask user about it.
- Adding features, tables, or abstractions not listed in `docs/architecture.md`. If you want one, ask first.
- Bypassing `compute_features` to "just quickly compute X" inline. There is no quick path.
- Bypassing `AgentBudget` caps "just for one experiment". All paths into the LLM go through `TennisAgent.predict()` with the bounded budget. If you need a different limit for a specific test, create a scoped subclass with explicit reasoning — don't disable the limits inline.
- Using `dict[str, Any]` where a Pydantic model fits.
- Generating Markdown reports without a calibration plot.
- Writing TODO comments without a phase tag, e.g., `# TODO(phase-3): ...`.
