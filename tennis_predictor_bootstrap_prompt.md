# Tennis Match Probability Predictor with LLM Analyst — Project Bootstrap

## Your role for this session

You are helping me bootstrap a new portfolio project. **Do not start writing code yet.** This first session is for setting up the project skeleton: repo structure, `CLAUDE.md`, skills, environment, dependencies, and a written project plan. We will build the actual features in subsequent sessions, one phase at a time.

## What to do this session

In order, do the following. Stop after each step and confirm with me before proceeding.

1. **Initialize the repo skeleton** above with all empty placeholder files and the directory structure. Do not write implementation code yet; just create the structure with TODO comments where appropriate.

2. **Write `CLAUDE.md`** at the repo root. It should establish project conventions: Python version, formatter (ruff), type checking (pyright), test command, the critical anti-leakage testing discipline, the rule that `compute_features` (and `build_training_features`) are the only sanctioned ways to compute features, the rule that the LLM does not emit its own probability, the naming convention for model artifacts, the requirement that every LLM call be logged to `llm_traces`, and a note that we work in phases.

3. **Write the four skill files** under `.claude/skills/`. Each skill should be focused and short. For example, `data-ingestion/SKILL.md` describes the contract for adding new data sources, the player-reconciliation pattern (including the manual-review checkpoint), and the DuckDB schema conventions. `feature-engineering/SKILL.md` describes the strict point-in-time rule, the replay-and-snapshot pattern, and the contract for the `FeatureVector` Pydantic model. `model-training/SKILL.md` describes walk-forward validation discipline, the calibration-method decision rule, and required reporting (including the market-benchmark calibration plot). `llm-tools/SKILL.md` describes the tool definition pattern, the structured output schema, the rule that the LLM does not produce its own probability, the prompt-caching requirement, and the `llm_traces` logging contract.

4. **Write `docs/phases.md`** describing a phased rollout:
   - Phase 1: Data layer (Sackmann ingestion, DuckDB schema, player reconciliation, tests)
   - Phase 2: Hot data layer (API integration, daily refresh script)
   - Phase 3: Feature engineering (compute_features function, leakage tests, surface-Elo)
   - Phase 4: Modeling (three models per tour, walk-forward validation, calibration, model reports)
   - Phase 5: LLM agent (tool definitions, client abstraction, structured outputs, single-turn agent)
   - Phase 6: Streamlit app (prediction page, dashboard page)
   - Phase 7: Deployment (Dockerfile, Fly.io/Railway setup, README polish)
   Each phase should have explicit entry criteria, deliverables, and exit criteria.

5. **Write `docs/architecture.md`** with a one-page architecture overview matching what's described in this prompt.

6. **Write `docs/methodology.md`** with the honest-evaluation framing: why calibration matters more than accuracy, why we report Brier score and log loss, why walk-forward validation, why the LLM does *not* emit its own probability (and what role it plays instead), why we use market-implied probabilities as a calibration benchmark but never as model features, the isotonic-vs-Platt decision rule, and why this project does *not* claim to beat the market.

7. **Set up `pyproject.toml`** with `uv` conventions. Include the dependencies we'll need (anthropic, duckdb, pandas, scikit-learn, lightgbm, rapidfuzz, pydantic, streamlit, pytest, ruff, pyright, nbstripout, pre-commit). Do not install yet — just declare.

8. **Write `.gitignore`** appropriately (data/raw, data/processed, .env, models/, .venv/, __pycache__, etc.).

9. **Write `.env.example`** listing the env vars we'll need (ANTHROPIC_API_KEY, optional tennis API keys).

10. **Write `.github/workflows/ci.yml`** running ruff, pyright, and pytest on push and PR. **Write `.pre-commit-config.yaml`** with ruff and nbstripout hooks. Both as skeletons — they don't need to pass on an empty repo, but the configuration should be correct.

11. **Stop and produce a summary** of what was created, what's deliberately not yet implemented, and what the next session (phase 1: data layer) will involve.

## How to behave during this session

- Ask clarifying questions before making non-trivial decisions. Default to minimal scope.
- When something is genuinely unclear, surface the decision to me with options rather than guessing.
- Do not write implementation logic this session. Skeleton, docs, and conventions only.
- After each numbered step above, briefly summarize what you did and wait for me to acknowledge before continuing.
- Keep CLAUDE.md, skill files, and docs concise — they're meant to guide future sessions, not be exhaustive specifications. Aim for terse and useful.
- If you find yourself wanting to add a feature not in this prompt, stop and ask.

## My background

I am a developer setting up a professional ML/AI engineering portfolio. I have working knowledge of Python, ML basics, and have used Claude Code before but want to practice using it well — including skills, CLAUDE.md, and structured phased development. I prefer  step-by-step hand-holding. Push back if I'm wrong about something.

At the end of this session I should have a clean, empty-but-organized project I can start phase 1 on.

## Project summary

A learning-focused portfolio project: a tennis match probability predictor (pre-match only) for ATP and WTA tour-level singles matches, with an LLM-based analyst layer that explains predictions using tool calling.

**User-facing flow:** user asks "What do you think about Alcaraz vs Djokovic tomorrow?" → LLM agent calls tools to fetch player stats, recent form, H2H, model prediction, current news → LLM returns a structured response containing: win probability for each player, key factors, narrative explanation, and confidence/caveats.

## Goals and non-goals

**Goals:**
- Demonstrate end-to-end ML+LLM engineering: data ingestion, feature engineering, model training, evaluation, LLM tool-calling integration, deployable interface.
- Honest evaluation: calibration metrics (Brier score, log loss, calibration plots) lead the analysis. Accuracy is reported but not the headline.
- Strict point-in-time correctness in feature engineering. No data leakage from the future.
- A clean, well-documented repo that a senior engineer reviewer would consider professional-grade.
- The LLM agent should be a credible demonstration of tool calling, structured outputs, and thoughtful agent design.

**Explicit non-goals for v1:**
- No betting integration, no live odds, no recommendations to wager. Betting odds are **not** used as model features.
- **Permitted exception**: historical closing-price implied probabilities (e.g., from tennis-data.co.uk archives) may be loaded as a *calibration benchmark* — we compare our model's calibration to the market's, but we never feed market prices into the model. This is the honest framing: "approaching market calibration without using market information," not "we beat Vegas."
- No in-play / live point-by-point prediction.
- No doubles.
- No user accounts, authentication, or multi-user state.
- No claim that the model "beats the market" — this is an educational artifact.
- No real-time WebSocket data; batch ingestion is sufficient.
- No microservices or distributed architecture. Single Python codebase, single deploy target.

## Technical decisions. I am open to discuss and relitigate and make changes if reasonable

- **Language**: Python 3.12+ (3.13 acceptable).
- **UI**: Streamlit (chosen for speed of development and clean ML-project aesthetic; deploys easily to Fly.io or Railway).
- **LLM provider**: Anthropic Claude Sonnet 4.6 as primary. Wrap the LLM client in an interface (`LLMClient` abstract base) with a Claude implementation, so OpenAI/OpenRouter can be added later. Use the Anthropic Python SDK directly. Use Claude's native `web_search` tool for news lookup. **Prompt caching is required from day one** — the `LLMClient` interface must support cacheable blocks (system prompt + tool definitions) so demo cost stays low and so it isn't retrofitted later.
- **Data storage**: DuckDB as the primary analytical store (file-based, no server, excellent for the read-heavy time-series workloads we'll have). Raw CSVs from Sackmann stored on disk; processed/joined data stored in DuckDB. Use Parquet for any intermediate columnar artifacts. A dedicated `llm_traces` table logs every LLM call: input messages (truncated), tool calls, tool results, output, latency, token counts. This is the agent's audit log.
- **ML libraries**: scikit-learn for baselines and logistic regression, LightGBM for gradient boosting. Use scikit-learn's `CalibratedClassifierCV` for isotonic calibration as a post-processing step, with a hard rule: if the held-out calibration set has fewer than ~1000 samples, fall back to Platt (sigmoid) calibration to avoid overfit calibration curves.
- **Tour scope**: ATP tour-level singles + WTA tour-level singles for prediction targets. Challenger/Futures match data is ingested into the database for feature computation (rating updates, recent form) but not used as prediction targets. Train separate models for ATP and WTA.
- **Player name reconciliation**: rapidfuzz library, with a one-time-built lookup table persisted to the database. **Ambiguous fuzzy matches below a confidence threshold are written to an `aliases_review.csv` checkpoint that I review manually** — automated matching alone is not trusted (same-name players exist; e.g., Coria brothers).
- **Deployment target**: Fly.io or Railway for the final deploy. Local development with `uv` for dependency management.
- **Testing**: pytest. Specific test categories required: data-loading correctness, fuzzy-match player reconciliation, **point-in-time correctness of features (anti-leakage tests)**, model serialization round-trip, LLM tool-calling schema validation, structured-output schema validation.
- **CI**: GitHub Actions running `ruff`, type-checking (pyright), and `pytest` on every push. Bootstrapped in phase 1.
- **Notebooks**: `nbstripout` configured as a pre-commit hook so notebook outputs do not pollute diffs.

## Architecture overview

Five layers, each with clean interfaces between them:

1. **Data layer**
   - Cold source: Jeff Sackmann's `tennis_atp` and `tennis_wta` GitHub repos (git submodules so we can pin a version and pull updates).
   - Hot source: a free tennis API for the last ~30 days of completed matches (recommendation: api-tennis.com free tier; final choice deferred to phase 2). Ingested daily/weekly via a script.
   - Storage: DuckDB tables `matches`, `players`, `rankings`, `player_aliases`, `market_implied_probabilities` (benchmark only, not a feature source), `llm_traces` (audit log of every LLM call). Raw CSVs preserved on disk. Schema normalized so cold and hot rows are interchangeable.
   - Player ID reconciliation across sources via the `player_aliases` table, with a manual-review checkpoint for ambiguous fuzzy matches.

2. **Feature engineering layer**
   - **Two entry points, one truth:**
     - `build_training_features()` — replays all matches in chronological order, maintains a state object (Elo per surface, rolling windows, H2H counters, fatigue counters), and for each match writes a row `(match_id, pre_match_features...)` to a `training_features` table. This is how training data is produced.
     - `compute_features(player_id, opponent_id, surface, tour, as_of_date) -> FeatureVector` — used at inference time. Reads the most recent state snapshot ≤ `as_of_date` and rolls forward through any matches in between. Returns a Pydantic `FeatureVector` model (not a dict), so schema drift between training and inference is caught at runtime.
   - Hard rule enforced by tests: at any point during feature computation, only data with `date < as_of_date` is accessible. No exceptions. Tests assert this by feeding future-tampered rows and confirming feature values do not change.
   - Feature families: surface-Elo (separate ratings pipeline), recent form (rolling windows), serve/return rolling stats, head-to-head, fatigue (matches/sets in last 7/14 days), ranking and ranking delta, tournament-level features.

3. **Modeling layer**
   - Three models trained per tour (ATP, WTA): (a) baseline surface-Elo, (b) logistic regression, (c) gradient boosting (LightGBM). Six total.
   - Walk-forward validation: train on years up to Y, validate on year Y+1, advance, repeat.
   - Evaluation: Brier score, log loss, calibration plot, accuracy, accuracy by ranking gap. **Also: calibration plot of our model overlaid against the market's closing-price implied probabilities for the same matches**, where market data is available. Output a model report (Markdown + plots) for each training run.
   - Calibration: isotonic regression as a post-processing step when held-out calibration set has ≥ ~1000 samples; otherwise Platt scaling. Show pre- and post-calibration metrics.
   - Models persisted to `models/` directory with metadata (training date, data range, metrics, feature list, calibration method, git commit hash).

4. **LLM agent layer**
   - Tools exposed to the LLM:
     - `get_player_stats(player_name, as_of_date)` — career and recent stats
     - `get_head_to_head(player_a, player_b)` — H2H record and recent meetings
     - `get_recent_form(player_name, n_matches=10)` — last N matches with results
     - `get_model_prediction(player_a, player_b, surface, tournament_level, match_date)` — returns calibrated probability from the ML model
     - `search_tennis_news(query)` — Claude's native web_search restricted to tennis-relevant queries
     - `get_player_ranking(player_name, as_of_date)` — current/historical ranking
   - **The model probability is the only probability shown to the user.** The LLM does *not* emit its own probability; it cannot override or adjust the model's number. Its job is to contextualize, narrate, and surface caveats.
   - Structured output schema (JSON, Pydantic-validated): `{model_probability_player_a, model_probability_player_b, key_factors[], narrative, confidence_band: "low" | "medium" | "high", caveats[], tools_used[]}`. The `confidence_band` is the LLM's qualitative read on how well-supported the prediction is given what tools returned (e.g., "low" if recent form is missing or news surfaces an injury). It is not a probability adjustment.
   - System prompt + tool definitions are sent as cacheable blocks (Anthropic prompt caching) so repeated calls stay cheap.
   - Every LLM call is logged to the `llm_traces` table (inputs, tool calls, outputs, tokens, latency) and is browsable in the dashboard.
   - Conversation is single-turn for v1: user asks, agent reasons, agent answers. Follow-up clarification (user can ask "why?" or "what about surface?") is a stretch goal — design the message-history threading from the start but only wire UI for it if phases allow.

5. **Interface layer**
   - Streamlit app with two pages: (a) interactive match prediction (enter two players + tournament + surface + date, get LLM-mediated answer), (b) model evaluation dashboard (calibration plots, metrics over time, feature importance).
   - Deployed to Fly.io or Railway. Dockerfile included.

## Project structure to create now

```
tennis_predictor/
├── CLAUDE.md
├── README.md                           # placeholder, polished in final phase
├── pyproject.toml                      # uv-managed
├── .python-version
├── .gitignore
├── .env.example
├── Dockerfile
├── .claude/
│   ├── skills/
│   │   ├── data-ingestion/SKILL.md
│   │   ├── feature-engineering/SKILL.md
│   │   ├── model-training/SKILL.md
│   │   └── llm-tools/SKILL.md
│   └── commands/                       # slash commands TBD per phase
├── .github/
│   └── workflows/
│       └── ci.yml                      # ruff + pyright + pytest on push
├── .pre-commit-config.yaml             # nbstripout, ruff
├── data/
│   ├── raw/                            # gitignored, holds Sackmann submodules and API dumps
│   ├── processed/                      # gitignored, DuckDB file lives here
│   └── README.md                       # explains the data layer
├── src/
│   └── tennis_predictor/
│       ├── __init__.py
│       ├── config.py                   # paths, env vars
│       ├── data/                       # ingestion, reconciliation
│       ├── features/                   # the compute_features function and helpers
│       ├── models/                     # training, evaluation, persistence
│       ├── llm/                        # client abstraction, agent, tools
│       └── app/                        # Streamlit app