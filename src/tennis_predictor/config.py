"""Centralized configuration: paths and env-var resolution.

All filesystem paths and environment variables flow through this module.
Modules elsewhere import constants from here rather than reading os.environ
directly, so deployment and local dev resolve the same way.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR: Path = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))

DUCKDB_PATH: Path = PROCESSED_DIR / "tennis.duckdb"

ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# matchstat Tennis API on RapidAPI ("Tennis API - ATP WTA ITF"), free tier 500 req/month.
X_RAPIDAPI_KEY: str | None = os.environ.get("X_RAPIDAPI_KEY")
