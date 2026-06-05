# syntax=docker/dockerfile:1.7
#
# Two-stage build for the tennis research dashboard.
#
# Builder: installs Python deps into a project venv via uv.
# Runtime: slim image with only the venv + code. No data, no models in the
# image — on the live Hugging Face Space they are downloaded to /data on
# container boot by scripts/hf_bootstrap.py (from a companion HF Dataset; see
# docs/phases.md Phase 8). The entrypoint runs that bootstrap, then the CMD.
#
# The same image is deployment-agnostic: it also ran on Fly.io (Phase 7),
# where /data was a persistent volume instead of ephemeral container disk.

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# uv from the official image — pinned by SHA-able tag at deploy time.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Resolve and install deps first, without the project itself, so the
# layer is cached unless pyproject.toml / uv.lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now the project source.
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp1: required by lightgbm at runtime.
# git: cron Machine pulls Sackmann submodule updates weekly.
# ca-certificates: HTTPS for git clone + outbound API calls.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        git \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

RUN chmod +x /app/scripts/docker-entrypoint.sh

# Data root. On HF this is the container's local (ephemeral) filesystem,
# populated on boot by scripts/hf_bootstrap.py. DuckDB + model artifacts
# live here. (On Fly it was a persistent volume mounted at the same path.)
ENV DATA_DIR=/data \
    MODELS_DIR=/data/models

# Streamlit runtime config — headless, bind to all interfaces, port 8080.
# HF routes to it via `app_port: 8080` in the README Space metadata.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

# Entrypoint bootstraps /data (no-op once populated / when HF_DATA_REPO is
# unset) then execs the CMD. CMD is the Streamlit app; it can be overridden
# (e.g. `python scripts/refresh_hot.py`) and still runs through bootstrap.
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["streamlit", "run", "src/tennis_predictor/app/main.py"]
