"""DuckDB connection helper.

All code that touches the database goes through `open_connection`. This is
the only place that knows the on-disk path comes from `config.DUCKDB_PATH`.
"""

from __future__ import annotations

import duckdb

from tennis_predictor import config


def open_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (or create) the project DuckDB file.

    The parent directory is created if missing.
    """
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(config.DUCKDB_PATH), read_only=read_only)
