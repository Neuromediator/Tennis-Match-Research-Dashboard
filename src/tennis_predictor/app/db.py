"""Process-wide DuckDB connection for the Streamlit app.

`st.cache_resource` is the canonical Streamlit pattern for database
connections: one connection shared across every session inside the
Streamlit process, lazily created on first call, never re-opened on a
script rerun. We use it instead of `st.session_state` because DuckDB
holds a write lock on the file, and a session-scoped connection would
multiply locks across browser tabs of the same app.

`DuckDBLockError` translates the raw DuckDB `IOException` into a
user-facing message that names the offending PID (Jupyter kernels and
leftover CLI runs are the usual culprits) so the operator knows what to
kill instead of being shown a Python traceback in the browser.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import streamlit as st

from tennis_predictor.config import DUCKDB_PATH
from tennis_predictor.data.schema import create_all_tables


class DuckDBLockError(RuntimeError):
    """The DuckDB file is held by another process. Carries the conflicting
    PID when DuckDB's error message provides it (which it does as of
    DuckDB 1.1+)."""


_LOCK_PID_RE = re.compile(r"PID\s+(\d+)", re.IGNORECASE)


def _humanise_lock_error(exc: duckdb.IOException, db_path: Path) -> DuckDBLockError:
    text = str(exc)
    match = _LOCK_PID_RE.search(text)
    if match:
        pid = match.group(1)
        msg = (
            f"DuckDB file `{db_path}` is locked by another Python process "
            f"(PID {pid}). Most often this is a leftover Jupyter kernel or "
            f"an earlier `predict_match.py` run. Stop it and reload this page. "
            f"To identify it: `ps -p {pid} -o pid,user,etime,command`."
        )
    else:
        msg = (
            f"DuckDB file `{db_path}` is locked by another process. "
            f"Close other Streamlit / Jupyter / CLI sessions touching this "
            f"file and reload. Raw error: {text}"
        )
    return DuckDBLockError(msg)


@st.cache_resource(show_spinner=False)
def get_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """Return the one DuckDB connection for this Streamlit process.

    Streamlit invalidates this cache on app shutdown only, so the schema
    migration runs at most once per process. `db_path` is exposed as a
    cache key so tests with a tmp path don't collide with the production
    file.
    """
    path = Path(db_path) if db_path is not None else DUCKDB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = duckdb.connect(str(path))
    except duckdb.IOException as exc:
        raise _humanise_lock_error(exc, path) from exc
    create_all_tables(conn)
    return conn


__all__ = ["DuckDBLockError", "get_connection"]
