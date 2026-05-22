# CLAUDE.md — Project conventions

This file is the contract for any Claude session working on this repo. Keep it terse, keep it accurate, update it when rules change.

## Project at a glance

A working tool that gives calibrated win probabilities for upcoming ATP and WTA tour-level singles matches, paired with an LLM analyst that surfaces relevant context (recent news, injuries, form) for each prediction. Built to be useful in real use, not just to illustrate techniques. **Not a betting tool — we do not claim to beat the market.**

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
3. **Betting odds are not features.** Historical market-implied probabilities are loaded into `market_implied_probabilities` but used **only** to compare our model's calibration to the market's. Never input to a model.
4. **The LLM does not emit a probability.** The model's number is the only probability shown. The LLM produces `narrative`, `key_factors`, `confidence_band` (`"low" | "medium" | "high"`), `caveats`. Schema rejects any LLM-supplied probability field.
5. **Every LLM call is logged.** The `LLMClient` writes a row to `llm_traces` (inputs, tool calls, outputs, tokens, cache stats, latency). No bypass.
6. **Prompt caching is on by default.** System prompt and tool definitions are sent as cacheable blocks in every call.
7. **Player reconciliation has a manual-review checkpoint.** Auto-match only at confidence ≥ 0.90. Confidence 0.75–0.90 → `data/processed/aliases_review.csv` for human review. Approved entries are promoted to `player_aliases` (with `source='manual_review'`) by `scripts/apply_aliases_review.py`. Never silently merge ambiguous matches.
8. **Calibration method depends on sample size.** Isotonic when held-out calibration set has ≥ 1000 samples; Platt scaling otherwise.
9. **No notebook outputs in commits.** `nbstripout` is enforced via pre-commit.
10. **`get_model_prediction` is mandatory.** If the model artifact is unloadable, missing, or the prediction call fails for any reason, the LLM agent must NOT be invoked. We do not ship predictions based on web search context alone — the calibrated probability from the trained model is non-negotiable.

## Model artifact naming

Saved under `models/<tour>/<model_type>/<YYYYMMDD-HHMM>/`:

- `model.joblib`
- `metadata.json` — training date, data date range, feature list, metrics (pre- and post-calibration), calibration method, git commit hash
- `report.md` — Markdown summary with calibration plot
- `calibration_plot.png`

Active model symlinked at `models/<tour>/<model_type>/latest`.

## Anthropic SDK

- **Default model:** `claude-sonnet-4-6`. Picked over Opus 4.7 (~5× cost, marginal quality gain on our synthesis task) and Haiku 4.5 (noticeably weaker on multi-source reconciliation in `narrative` / `caveats`). Configurable via `ANTHROPIC_MODEL`.
- **Provider stack:** Direct `anthropic` SDK only. No abstraction-layer wrappers (LangChain, LiteLLM, OpenRouter, Anthropic Managed Agents). Vendor flexibility comes from our own `LLMClient` ABC, not from third-party multi-provider layers. Reasoning: native `web_search`, explicit `cache_control` markers, and best-in-class tool-use reliability are all features abstraction layers either lose entirely or surface poorly. If a future provider beats Anthropic on this combo, swap is a new `LLMClient` implementation (~100 lines), not a framework migration.
- The `LLMClient` abstract base lives in `src/tennis_predictor/llm/client.py`. Use it; do not call `anthropic.Anthropic()` directly outside that module.
- **Prompt caching is on by default.** System prompt + tool definitions are sent with `cache_control` markers. Our typical agent call has ~2000 stable input tokens that cache → ~70% input-cost reduction on the second-and-later call within the 5-minute TTL window. One marker, on the last tool definition (caches everything prefix to it).
- **Cache hit hygiene.** The cacheable prefix (system prompt + tool definitions) must be byte-stable across calls. Never inject timestamps, random IDs, or session-specific data into the system prompt — current date and match context belong in the user message. A unit test on `LLMClient._build_cacheable_blocks()` enforces byte-equality between two consecutive calls; if it fails, the cache-hit rate just dropped to ~0%.
- Web search uses Claude's native `web_search` tool (no Tavily / Brave / Serper as separate dependencies).

## Web search

- **Tool:** Anthropic native `web_search`, configured with a small `blocked_domains` list (betting / pick-of-the-day clickbait) and **no** `allowed_domains` — we want maximum recall on legitimate news (local press, journalist tweets, dedicated tennis outlets) and rely on Anthropic's own ranking to filter out the worst.
- **Preferred sources** (mentioned in system prompt, not whitelist-enforced — Anthropic API has no `preferred_domains` parameter): **ESPN, BBC, tennis.com, tennis365.com**, X/Twitter when accessible. The two tour-official sites (`atptour.com`, `wtatennis.com`) are intentionally **not** highlighted — they surface only the news everyone already knows, almost always about top-10 players, and don't add value over the journalist sources above.
- **Budget per agent call:** `max_uses = 3` (typically one query per player, optionally one tournament query).
- **Recency window:** the last ~14 days, enforced via system-prompt instruction (Anthropic does not expose a time-range parameter).
- **Anti-fabrication clause** in the system prompt: when search returns nothing material, the agent MUST explicitly note "no recent news surfaced" in `caveats`. It MUST NOT fabricate plausible-sounding news to fill the slot.
- **Cost:** ~$10 per 1000 searches × ~2 searches per prediction ≈ $0.02 per prediction.

## LLM agent failure modes

The agent has six failure surfaces, each handled distinctly:

1. **`web_search` errors or returns empty** — degraded mode. The agent continues with DB-only context, surfaces `"news lookup unavailable"` or `"no recent news surfaced"` in `caveats`. Never aborts the prediction.
2. **DB tool returns empty data** (no H2H history, debutant with no recent form, unranked player) — normal signal, not a failure. The agent interprets emptiness as context (returning player, fresh pairing, etc.) and mentions it in `narrative` or `caveats`.
3. **DB tool raises an exception** — bubbled up out of `LLMClient`. Never silently masked. Streamlit surfaces an error page; we never ship a prediction built on partially-corrupted data.
4. **Anthropic API errors (5xx / 429)** — handled by the SDK's built-in retry with exponential backoff (`max_retries=2`). After exhaustion, the exception bubbles up to the caller.
5. **`get_model_prediction` unavailable** — **fatal**, see hard rule #10. The model number is mandatory; the agent is not invoked at all.
6. **Agent loop timeout (120 s hard cap)** — the entire agent loop is wrapped in `asyncio.timeout(120)`. On timeout we log partial state to `llm_traces` and surface "prediction timed out" to the user.

**Twitter is not a directly usable source** — X.com is behind an auth wall and `web_search` returns either headlines or login redirects. We do not advertise Twitter in the system prompt; tweet content reaches us only when journalist sources quote or embed.

## Structured output discipline

- LLM structured output uses the **tool-use pattern**: a single `submit_analysis` tool whose `input_schema` mirrors `AgentResponse`, with `additionalProperties: false`. Forbidden fields (any `probability`-like field, see hard rule #4) are blocked at the JSON-schema layer before Pydantic ever sees them.
- Validation pipeline per agent call: `tool_choice` forces `submit_analysis` on the final iteration → `AgentResponse.model_validate(tool_call.input)` → Pydantic enforces Literal / min-max constraints as the second wall.
- Never use `response_format={"type": "json_object"}` (unstructured) or free-text JSON parsing — both have failure modes that the tool-use pattern eliminates by construction.
- Agentic loop uses hybrid `tool_choice`: `"auto"` while the agent gathers data, hard-forced to `submit_analysis` on the final iteration to guarantee termination within a bounded tool-call budget.

## Testing the LLM agent

Three-tier test strategy:

1. **Unit tests** mock `anthropic.Anthropic` (using `respx` or `pytest-mock`). Cover: request body construction, cache-marker placement, tool-call parsing, `llm_traces` row format. Run in CI on every push. Free.
2. **Recorded-fixture e2e tests** in `tests/fixtures/llm/*.json` cover one happy path, one degraded path (web_search error), and one edge case (empty H2H). Replayed deterministically in CI. Re-record when Anthropic API contract or our tool schemas change: `uv run pytest -m llm_record` (manual, paid).
3. **Live API tests** under `@pytest.mark.llm_live`, excluded by default (`addopts = -m "not llm_live"` in `pyproject.toml`). Run locally before pushing a phase or when validating a system-prompt change: `uv run pytest -m llm_live`. Never in CI — protects budget and avoids API-key exposure surface.

CI must NOT have `ANTHROPIC_API_KEY` as a secret. The unit + fixture tiers never need it; the live tier should fail fast with "API key not set" if accidentally invoked outside a local dev session.

## Budget discipline

Three-layer protection against runaway spend, **all must be in place**:

1. **Org-level hard cap (Anthropic console, manual setup).** A dedicated workspace for this project with a monthly budget limit — currently set to **$20/month**. This is a hard 429 ceiling, not an alert. Without this, any code-side bug or key leak can spend unbounded. Setup is the user's responsibility (one-time UI action, no code involved).
2. **Per-agent-call cap (in code, Phase 5 deliverable).** Each `TennisAgent.predict()` enforces an `AgentBudget` with four hard limits:
   - `max_tool_iterations = 6` (tool calls before forcing `submit_analysis`)
   - `max_total_tokens = 30_000` (input + output sum per agent call)
   - `max_wall_clock_seconds = 120`
   - `max_web_searches = 3`

   Exceeding any limit forces `submit_analysis` on the next turn (if the iteration budget still allows it) or raises `BudgetExceededError`.
3. **Per-session / per-IP throttling (Streamlit, Phase 7).** Public-URL abuse protection. Not in Phase 5 scope — deferred to deployment.

The `llm_traces` row records `web_search_count` and `estimated_cost_usd` per trace so Streamlit can later surface "spent today / this month" to the user.

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
