# Tennis Match Probability Predictor

A working tool that gives calibrated win probabilities for upcoming ATP and WTA tour-level singles matches, paired with an LLM analyst that surfaces relevant context (recent news, injuries, form) for each prediction. Built to be useful in real use, not just to illustrate techniques.

**Not a betting tool — we do not claim to beat the market.**

**Status:** Phases 1 (cold data), 2 (hot data), and 3 (feature engineering) complete — see [docs/phases.md](docs/phases.md). Phase 4 (modeling: walk-forward validation, LightGBM + Elo baseline, calibration) is next.

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

# Build the training_features table + persist elo_state (phase 3)
uv run python scripts/build_features.py

# (later phases)
uv run python scripts/train_models.py            # phase 4
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
