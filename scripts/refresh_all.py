"""Unified daily refresh wrapper for the Fly scheduled Machine.

Runs every required refresh script in sequence. Designed to be the
entrypoint of a single Scheduled Machine (`fly machine run ... --schedule daily`):

    Daily steps (every run):
      1. refresh_hot.py            — matchstat fixtures + rankings
      2. refresh_pre_match_odds.py — The Odds API pre-match odds

    Weekly step (Sundays only, UTC):
      3. git pull Sackmann submodules (tennis_atp, tennis_wta)
      4. refresh_data.py --skip-submodules — incremental cold ingest

Each step is isolated: a non-zero exit from one step is logged but does
not abort the chain. The wrapper's own exit code is non-zero only if a
daily step failed (weekly failures are noisy but don't gate the daily
ones).

Sackmann submodules live on the Fly volume at $DATA_DIR/raw/tennis_atp
and $DATA_DIR/raw/tennis_wta. They're cloned during volume bootstrap
(see docs/phase7_plan.md); this wrapper only pulls updates.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
SACKMANN_ATP_DIR: Path = DATA_DIR / "raw" / "tennis_atp"
SACKMANN_WTA_DIR: Path = DATA_DIR / "raw" / "tennis_wta"

SUNDAY: int = 6  # datetime.weekday() — Mon=0 .. Sun=6


@dataclass
class StepResult:
    name: str
    exit_code: int
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def _run_step(name: str, argv: list[str], *, cwd: Path | None = None) -> StepResult:
    """Run one subprocess step, stream its output, return a result row."""
    header = f"\n{'=' * 70}\n[refresh_all] {name}\n{'=' * 70}"
    print(header, flush=True)
    started = datetime.now(UTC)
    completed = subprocess.run(
        argv,
        cwd=cwd or REPO_ROOT,
        check=False,
    )
    duration = (datetime.now(UTC) - started).total_seconds()
    print(
        f"[refresh_all] {name} finished: exit={completed.returncode} duration={duration:.1f}s",
        flush=True,
    )
    return StepResult(name=name, exit_code=completed.returncode, duration_s=duration)


def _git_pull(repo_dir: Path) -> StepResult:
    """`git pull --ff-only` inside a Sackmann clone. Skips with exit 0 if
    the directory doesn't exist yet (first weekly run before bootstrap)."""
    name = f"git pull {repo_dir.name}"
    if not (repo_dir / ".git").exists():
        print(
            f"\n[refresh_all] {name} SKIPPED — {repo_dir} is not a git repo "
            f"(bootstrap not yet run?)",
            flush=True,
        )
        return StepResult(name=name, exit_code=0, duration_s=0.0)
    return _run_step(name, ["git", "pull", "--ff-only"], cwd=repo_dir)


def main() -> int:
    now = datetime.now(UTC)
    is_sunday = now.weekday() == SUNDAY
    py = sys.executable

    print(
        f"[refresh_all] start ts={now.isoformat()} weekday={now.strftime('%A')} "
        f"sunday_weekly={is_sunday}",
        flush=True,
    )

    results: list[StepResult] = []

    # ---- Daily ----
    results.append(_run_step("refresh_hot", [py, "scripts/refresh_hot.py"]))
    results.append(_run_step("refresh_pre_match_odds", [py, "scripts/refresh_pre_match_odds.py"]))

    # ---- Weekly (Sundays) ----
    weekly_results: list[StepResult] = []
    if is_sunday:
        weekly_results.append(_git_pull(SACKMANN_ATP_DIR))
        weekly_results.append(_git_pull(SACKMANN_WTA_DIR))
        weekly_results.append(
            _run_step("refresh_data", [py, "scripts/refresh_data.py", "--skip-submodules"])
        )

    # ---- Summary ----
    print(f"\n{'=' * 70}\n[refresh_all] summary\n{'=' * 70}", flush=True)
    all_results = [*results, *weekly_results]
    for r in all_results:
        flag = "OK" if r.ok else "FAIL"
        print(f"  [{flag}] {r.name:32s} exit={r.exit_code} duration={r.duration_s:.1f}s")

    # Exit code: non-zero ONLY if a daily (required) step failed. Weekly
    # failures are visible in the summary but don't fail the run — the
    # cold layer can lag by a week without breaking the dashboard.
    daily_failed = any(not r.ok for r in results)
    return 1 if daily_failed else 0


if __name__ == "__main__":
    sys.exit(main())
