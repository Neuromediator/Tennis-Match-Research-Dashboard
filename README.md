# Tennis Match Research Dashboard

A research dashboard comparing **market consensus odds** (auto-fetched from The Odds API), a trained **LightGBM model**, and a **surface-Elo baseline** for upcoming ATP and WTA tour-level singles. For each fixture it surfaces an LLM-discovered news block (injuries, withdrawals, coach changes, recent results) and detailed H2H + last-8-matches per player. When the model disagrees with the market by > 10pp, a deterministic panel explains the gap structurally (stale surface-Elo, activity asymmetry, returning veteran). **The model is one signal, not the answer.**

**Not a betting tool — we do not claim to beat the market.** Phase 6.2 explicitly reframed the product from "predictor" to "context dashboard" after Phase 6.1 close-out exposed that the calibrated LightGBM probability is unreliable on top matchups (inverted favourites at the Cina-Opelka / Sinner-Djokovic level). See [`docs/tutorials/phase_6_2_notes.md`](docs/tutorials/phase_6_2_notes.md) for the full rescope rationale.

**Status:** Phases 1-6.1 complete; Phase 6.2 (research-dashboard reframing + The Odds API integration + UI bug bundle) is the current phase — see [docs/phases.md](docs/phases.md). Phase 7 (Fly.io deployment) is next.

**Aggregate model metrics** (last-5-fold sample-weighted Brier, post-calibration): ATP LightGBM **0.2101**, WTA LightGBM **0.1954**, both ahead of the Surface-Elo baseline (ATP 0.2220, WTA 0.2180). These averages hide tail-failure on top matches — see the reality-test rationale in the Phase 6.2 notes.

See [docs/architecture.md](docs/architecture.md), [docs/methodology.md](docs/methodology.md), and [docs/phases.md](docs/phases.md) for details.

## Quick start

```bash
# Install dependencies (Python 3.12+ required, pinned via .python-version)
uv sync

# Run the full test suite
uv run pytest

# Rebuild the cold data layer (Sackmann + tennis-data.co.uk)
uv run python scripts/refresh_data.py            # incremental (default)
uv run python scripts/refresh_data.py --clean    # full rebuild from scratch

# Daily hot refresh (matchstat fixtures + rankings)
uv run python scripts/refresh_hot.py

# Promote reviewed aliases (after editing data/processed/aliases_review*.csv)
uv run python scripts/apply_aliases_review.py

# Build the training_features table + persist elo_state + last_match_state (phases 3 / 4.1)
uv run python scripts/build_features.py

# Train the four production models (Elo + LightGBM per tour) — phases 4 / 4.1
uv run python scripts/train_models.py

# (later phases)
uv run streamlit run src/tennis_predictor/app/main.py  # phase 6+
```

After `refresh_data.py`, the DuckDB file lives at `data/processed/tennis.duckdb`. Audit artefacts (low-confidence resolutions, unmatched rows) live alongside it.

## Headline numbers after phase 1

- 137,318 players (ATP + WTA, composite IDs `ATP_<id>` / `WTA_<id>`)
- 1,701,617 matches across all tiers (~360k tour-level singles)
- 5,559,400 weekly rankings
- ~52,000 market-implied probabilities for 2013-current
- 75% median match rate vs. tennis-data.co.uk archive

## What you get

- A calibrated win probability for any upcoming ATP or WTA tour-level singles match, computed from public match data only.
- An LLM-written narrative that explains the prediction's `key_factors`, surfaces recent news (withdrawals, injuries, form), and tags the result with a qualitative `confidence_band`. The LLM never overrides the model's number — it contextualizes it.
- A dashboard showing the model's calibration over time vs. the market's calibration on the same matches — so you can judge how much to trust any individual prediction.

The methodological bar — point-in-time-correct features, walk-forward validation, isotonic/Platt calibration, market-as-benchmark — is set high deliberately: a tool that misleads its users is worse than no tool at all.
