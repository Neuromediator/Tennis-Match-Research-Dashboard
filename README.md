# Tennis Match Probability Predictor

A portfolio project: pre-match win-probability predictor for ATP and WTA tour-level singles, with an LLM analyst layer that narrates predictions using tool calling.

**Status:** Phase 1 (cold data layer) complete — see [docs/phases.md](docs/phases.md). Phase 2 (hot data layer) is next.

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

# Promote reviewed aliases (after editing data/processed/aliases_review.csv)
uv run python scripts/apply_aliases_review.py

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

## Why this project

To demonstrate end-to-end ML+LLM engineering: data ingestion, point-in-time-correct feature engineering, walk-forward model evaluation with proper calibration, LLM tool calling with structured outputs, and a deployable interface.

This project does **not** claim to beat the betting market. It is an educational artifact.
