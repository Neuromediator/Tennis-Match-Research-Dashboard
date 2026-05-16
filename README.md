# Tennis Match Probability Predictor

A portfolio project: pre-match win-probability predictor for ATP and WTA tour-level singles, with an LLM analyst layer that narrates predictions using tool calling.

**Status: phase-0 (bootstrap).** Implementation begins in phase 1.

See `docs/architecture.md`, `docs/methodology.md`, and `docs/phases.md` for details.

## Quick start

```bash
uv sync                    # install dependencies
uv run pytest              # run the test suite
uv run streamlit run src/tennis_predictor/app/main.py  # (phase 6+)
```

## Why this project

To demonstrate end-to-end ML+LLM engineering: data ingestion, point-in-time-correct feature engineering, walk-forward model evaluation with proper calibration, LLM tool calling with structured outputs, and a deployable interface.

This project does **not** claim to beat the betting market. It is an educational artifact.
