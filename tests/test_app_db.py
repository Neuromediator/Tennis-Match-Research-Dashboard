"""Unit tests for `app.db` — specifically the lock-error humaniser.

`get_connection` itself is decorated with `st.cache_resource` which is
not usable outside a Streamlit script context (no ScriptRunContext), so
we test the error mapper directly. Real integration is the Phase 6 manual
smoke test."""

from __future__ import annotations

from pathlib import Path

import duckdb

from tennis_predictor.app.db import DuckDBLockError, _humanise_lock_error


def test_humanise_lock_error_extracts_pid() -> None:
    raw = duckdb.IOException(
        'IO Error: Could not set lock on file "/tmp/x.duckdb": '
        "Conflicting lock is held in /usr/bin/python3.12 (PID 205798)."
    )
    out = _humanise_lock_error(raw, Path("/tmp/x.duckdb"))
    assert isinstance(out, DuckDBLockError)
    msg = str(out)
    assert "PID 205798" in msg
    assert "/tmp/x.duckdb" in msg
    # The actionable command is part of the message — operators rely on it.
    assert "ps -p 205798" in msg


def test_humanise_lock_error_without_pid_falls_back() -> None:
    raw = duckdb.IOException("IO Error: Could not set lock on file (no PID surfaced)")
    out = _humanise_lock_error(raw, Path("/tmp/x.duckdb"))
    msg = str(out)
    assert "locked by another process" in msg
    assert "/tmp/x.duckdb" in msg
