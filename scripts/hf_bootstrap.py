"""One-shot data bootstrap for the persistent volume.

HF Spaces (and Fly) keep the 1.3 GB DuckDB + model artifacts on a
persistent volume mounted at ``DATA_DIR`` (``/data`` in production). The
container image ships code only — the volume starts empty on a brand-new
Space. This script populates it on first boot by downloading a prebuilt
snapshot from a HF Dataset repo.

It is a **no-op** when either:
  - the DuckDB file already exists (volume already populated — every boot
    after the first, and every boot on Fly where the volume persists), or
  - ``HF_DATA_REPO`` is unset (local dev / Fly, which have no dataset).

so it is safe to run unconditionally at container start.

The dataset is expected to mirror the on-volume layout::

    processed/tennis.duckdb
    models/<tour>/<type>/...

Set ``HF_DATA_REPO=<user>/<dataset>`` and, for a private dataset,
``HF_TOKEN=<read token>`` (public datasets need no token).
"""

from __future__ import annotations

import logging
import os

from tennis_predictor import config

log = logging.getLogger("hf_bootstrap")


def main() -> None:
    if config.DUCKDB_PATH.exists():
        log.info("DuckDB already present at %s — skipping bootstrap", config.DUCKDB_PATH)
        return

    repo = os.environ.get("HF_DATA_REPO")
    if not repo:
        log.info("HF_DATA_REPO unset — skipping bootstrap")
        return

    # Imported lazily so the dependency is only needed on the bootstrap path.
    from huggingface_hub import snapshot_download

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading data snapshot from dataset %r into %s", repo, config.DATA_DIR)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(config.DATA_DIR),
        token=os.environ.get("HF_TOKEN"),
    )
    log.info("Bootstrap complete — DuckDB at %s", config.DUCKDB_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
