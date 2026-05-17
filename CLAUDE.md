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

## Model artifact naming

Saved under `models/<tour>/<model_type>/<YYYYMMDD-HHMM>/`:

- `model.joblib`
- `metadata.json` — training date, data date range, feature list, metrics (pre- and post-calibration), calibration method, git commit hash
- `report.md` — Markdown summary with calibration plot
- `calibration_plot.png`

Active model symlinked at `models/<tour>/<model_type>/latest`.

## Anthropic SDK

- Default model: `claude-sonnet-4-6` (Sonnet 4.6). Configurable via `ANTHROPIC_MODEL`.
- The `LLMClient` abstract base lives in `src/tennis_predictor/llm/client.py`. Use it; do not call `anthropic.Anthropic()` directly outside that module.
- Prompt caching: tool definitions + system prompt are sent with `cache_control` markers.
- Web search uses Claude's native `web_search` tool.

## How we work

- **Phased development.** Each phase has entry criteria, deliverables, and exit criteria in `docs/phases.md`. Don't start the next phase until the current phase's exit criteria are green.
- **Skills.** Domain conventions live in `.claude/skills/<name>/SKILL.md`: `data-ingestion`, `feature-engineering`, `model-training`, `llm-tools`. Read the relevant skill before touching that area.
- **Chat language.** Responses to the user are in Russian; all code, docs, comments, commit messages are in English.
- **Cold-layer refresh.** `uv run python scripts/refresh_data.py` is incremental (idempotent). Pass `--clean` to delete the DuckDB file and rebuild from scratch. Audit artefacts live in `data/processed/`: `aliases_review.csv` (low-confidence resolutions awaiting human verdict) and `unmatched_market_rows.csv` (rows tennis-data.co.uk reported but we couldn't join — analyzed in `notebooks/explore_unmatched.ipynb`).

## What to avoid
-  Never read `.env`. It contains live API keys. Reading it exposes the values in this conversation's transcript. If you need to know which api keys are inside this file, ask user about it.
- Adding features, tables, or abstractions not listed in `docs/architecture.md`. If you want one, ask first.
- Bypassing `compute_features` to "just quickly compute X" inline. There is no quick path.
- Using `dict[str, Any]` where a Pydantic model fits.
- Generating Markdown reports without a calibration plot.
- Writing TODO comments without a phase tag, e.g., `# TODO(phase-3): ...`.
