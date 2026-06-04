#!/usr/bin/env bash
# Container entrypoint: populate the persistent volume on first boot (HF
# Spaces), then exec the real command (Streamlit by default).
#
# `hf_bootstrap.py` is a no-op when the volume is already populated or when
# HF_DATA_REPO is unset, so this is safe on every start and on Fly (whose
# volume persists). A bootstrap failure must not take the app down — we log
# and continue; the app will then surface a "never refreshed" banner.
set -euo pipefail

python /app/scripts/hf_bootstrap.py || echo "[entrypoint] bootstrap skipped/failed — continuing"

exec "$@"
