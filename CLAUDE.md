# CLAUDE.md — Project orientation

A reference card for any Claude session working on this repo. Skim it, then start work; for deeper context follow the pointers at the bottom.

**Live:** https://neuromediator-tennis-research-dashboard.hf.space/ — free Hugging Face Space (Docker SDK, CPU basic, 2 vCPU / 16 GB RAM), **no persistent disk**: DuckDB + models are bootstrapped on container boot from the HF Dataset `Neuromediator/tennis-dashboard-data` (`scripts/hf_bootstrap.py`) onto the ephemeral local FS; a twice-daily GitHub Actions ping (`.github/workflows/keepalive.yml`) keeps it warm; daily refresh via in-process APScheduler at 05:00 UTC + catch-up-on-wake (`app/scheduler.py`). $0/month. Migration story in `docs/phases.md` Phase 8; prior Fly.io deployment in `docs/phases.md` Phase 7.

## What this project is

A **tennis match research dashboard** for ATP and WTA tour-level singles. For every upcoming fixture the app renders four independent signals side by side — market consensus odds (auto-fetched from The Odds API), a trained LightGBM probability, a surface-Elo baseline, and an LLM-discovered news block — plus a deterministic *"why model differs"* panel when the model-vs-market gap exceeds 10pp.

**It is not a betting tool.**

## Environment

- **Python 3.12+** (pinned in `.python-version`).
- **Dependency manager:** `uv`. Add deps with `uv add <pkg>`; lockfile is `uv.lock`. Run anything with `uv run <cmd>`.
- **Database:** single DuckDB file at `data/processed/tennis.duckdb`.

## Quality gates (run before claiming a change is done)

```bash
uv run ruff check .              # lint
uv run ruff format --check .     # format check
uv run pyright                   # type check
uv run pytest                    # tests (~500 passing)
```

CI runs the same four. Pre-commit hook runs `ruff` + `nbstripout`.

## Project conventions

- **Chat language:** responses to the user are in Russian; code, docs, comments, commit messages are in English.
- **Provenance:** every match carries `source` + `match_external_id`; every player has a single canonical `player_id`. Non-canonical names map via `player_aliases`.
- **Player reconciliation:** auto-merge only at confidence ≥ 0.90; lower goes to `data/processed/aliases_review.csv` for human review. `scripts/find_duplicate_players.py` + `scripts/apply_player_dedupe.py` collapse Sackmann's same-name-same-DOB duplicates.
- **Feature entry points:** `build_training_features()` (chronological replay → `training_features` table) and `compute_features(...) → FeatureVector` (inference). Anything else doing feature math inline is a bug.
- **LLM contracts** (Phase 6 settled state):
  - Direct `anthropic` SDK only — no framework wrappers.
  - Prompt caching by default; cacheable prefix must be byte-stable (the test in `tests/test_agent_*` enforces this).
  - Bounded budget per call (4 iter / 30k tok / 120s / 2 web searches).
  - Structured output via `tool_use` — schema rejects any LLM-emitted probability and any free-text synthesis field. Output is a typed `NewsItem` list + status enum.
  - Every call logged to `llm_traces`.
- **Betting odds are never training features.** Market data (`market_implied_probabilities`, `pre_match_odds`) is rendered alongside the model as an independent signal and used for calibration evaluation only.
- **Point-in-time correctness** is enforced by `tests/test_feature_leakage.py`, not by convention. If a leakage test fails, the bug is in the code.
- **Time zones:** matchstat returns timestamps labelled `Z` but the wall-clock is empirically Moscow time. `_to_naive_utc` in `data/load_hot.py` corrects this on ingestion. The Home page displays DATE only (matchstat's `T12:00:00Z` is unreliable as a real start time for most Slam outside-court matches).

## Anthropic SDK

- Default model `claude-sonnet-4-6` (config via `ANTHROPIC_MODEL`).
- `LLMClient` abstract base in `src/tennis_predictor/llm/client.py`. Don't instantiate `anthropic.Anthropic()` outside that module.
- Web search uses Tavily (`topic="news"`, `days=32`) — not Anthropic's native server tool.

## External quotas

| Source | Free tier | What our counter shows |
|---|---|---|
| matchstat | 500 req / month | SUM of `ingestion_runs.requests_used` (hot refresh, ~13-15/run) + `matchstat_quota.requests_used` (per-prediction H2H/past-matches) |
| The Odds API | 500 credits / calendar month | `odds_api_quota.requests_used` — only billable `/sports/{key}/odds` calls (discovery is free per docs) |
| Anthropic | $20/month workspace cap (set in console) | `llm_traces.estimated_cost_usd` aggregated daily/monthly |
| Tavily | 1000 searches / month | Counted via `llm_traces.web_search_count` |

**Quota reset cadence is NOT uniform.** matchstat (RapidAPI) resets on the
**subscription billing cycle**, not the calendar 1st — for this account the
cycle started 2026-05-18, so the monthly window rolls over ~the 18th. The
daily hot refresh alone is ~13-15 req × 30 ≈ 450/month, i.e. it nearly
exhausts the 500 free tier on its own; per-prediction H2H/past-matches
calls eat the rest. When the limit is hit, every matchstat call returns
HTTP 429 and the hot refresh fails on its first request — surfaced in the
UI as "matchstat monthly quota exhausted (429)" (see `app/widgets.py`
`is_quota_error`). The Odds API / Anthropic / Tavily counters reset on the
calendar 1st (UTC).

## Where to find things

- **`docs/architecture.md`** — five-layer module map, data flow, what's intentionally absent.
- **`docs/methodology.md`** — walk-forward, calibration choices, why we evaluate against the market.
- **`docs/phases.md`** — phase-by-phase history of what was built and why.
- **`docs/tutorials/tutorial.md`** — workflow walkthrough in plain language.
- **`.claude/skills/<name>/SKILL.md`** — domain conventions for `data-ingestion`, `feature-engineering`, `model-training`, `llm-tools`. Read the relevant skill before touching that area.

## What to avoid

- Don't read `.env`. It contains live API keys; reading it exposes values into transcripts. Ask the user which keys are present if needed.
- Don't bypass `compute_features` for "just one quick value".
- Don't bypass the agent's `AgentBudget` for "just one experiment".
- Don't generate model artifacts without a calibration plot.
- Don't use `dict[str, Any]` where a Pydantic model fits.
