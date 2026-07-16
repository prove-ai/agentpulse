"""Batch runner — execute a YAML-defined set of scenarios to build observability data.

Each run goes through digital-company/main.py as a subprocess, so the existing
instrumentation captures everything automatically. Failed runs don't abort the
batch; you'll get a summary at the end.

Usage
-----
    python scripts/batch_run.py                              # default scenarios
    python scripts/batch_run.py scenarios/drift_test.yaml    # custom config
    python scripts/batch_run.py --dry-run                    # just list, don't execute
    python scripts/batch_run.py --quick "your one-off question"  # single ad-hoc run

YAML format
-----------
    scenarios:
      - task_type: csv-analysis
        prompt_version: 1
        question: "Analyze data/sales.csv and summarize the findings."
        runs: 3            # how many times to repeat this scenario
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# Paths assumed: observability/ is sibling to digital-company/
_OBS_ROOT     = Path(__file__).parent.parent
_COMPANY_ROOT = _OBS_ROOT.parent / "digital-company"
_VENV_PYTHON  = _COMPANY_ROOT / ".venv" / "bin" / "python"

DEFAULT_SCENARIOS = _OBS_ROOT / "scripts" / "scenarios" / "default.yaml"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class RunSpec:
    task_type:      str
    prompt_version: int
    question:       str
    rep:            int   # 1-indexed within this scenario
    of:             int   # total reps in this scenario

    @property
    def short_question(self) -> str:
        return self.question if len(self.question) <= 55 else self.question[:52] + "…"


# ---------------------------------------------------------------------------
# Loading scenarios
# ---------------------------------------------------------------------------
def load_scenarios(path: Path) -> list[RunSpec]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "scenarios" not in cfg:
        raise ValueError(f"{path} must have a top-level 'scenarios:' key")

    runs: list[RunSpec] = []
    for s in cfg["scenarios"]:
        reps = int(s.get("runs", 1))
        for i in range(reps):
            runs.append(RunSpec(
                task_type=s["task_type"],
                prompt_version=int(s.get("prompt_version", 1)),
                question=s["question"],
                rep=i + 1, of=reps,
            ))
    return runs


# ---------------------------------------------------------------------------
# Running one scenario
# ---------------------------------------------------------------------------
def run_one(spec: RunSpec, idx: int, total: int) -> tuple[bool, float]:
    """Run a single scenario via main.py subprocess. Returns (success, elapsed_s)."""
    cmd = [
        str(_VENV_PYTHON),
        "main.py",
        "--task-type", spec.task_type,
        "--prompt-version", str(spec.prompt_version),
        spec.question,
    ]

    label = f"[{idx:>2}/{total}] {spec.task_type:18s} v{spec.prompt_version} (rep {spec.rep}/{spec.of})"
    print(f"  {label}  {spec.short_question:<55s}", end="", flush=True)

    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=_COMPANY_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        print(f"  ✗ timeout after {elapsed:>5.1f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - started
        print(f"  ✗ {type(e).__name__}: {e}")
        return False, elapsed

    elapsed = time.time() - started

    if result.returncode == 0:
        print(f"  ✓ {elapsed:>5.1f}s")
        return True, elapsed

    # Non-zero exit
    print(f"  ✗ exit {result.returncode}  ({elapsed:.1f}s)")
    err = (result.stderr or "").strip().splitlines()
    if err:
        print(f"     └ {err[-1][:90]}")
    return False, elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run scenarios through the observability-instrumented multi-agent system."
    )
    parser.add_argument(
        "config", nargs="?", default=str(DEFAULT_SCENARIOS),
        help="Path to a YAML scenarios file (default: scenarios/default.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the planned runs without executing.",
    )
    parser.add_argument(
        "--quick", metavar="QUESTION",
        help="One-off ad-hoc run with the given question. Skips the YAML config.",
    )
    parser.add_argument(
        "--quick-type", default="adhoc",
        help="task_type for --quick (default: adhoc)",
    )
    parser.add_argument(
        "--quick-version", type=int, default=1,
        help="prompt_version for --quick (default: 1)",
    )
    parser.add_argument(
        "--quick-reps", type=int, default=1,
        help="How many times to repeat --quick (default: 1)",
    )
    args = parser.parse_args()

    # Build the list of runs
    if args.quick:
        runs = [
            RunSpec(args.quick_type, args.quick_version, args.quick,
                    rep=i + 1, of=args.quick_reps)
            for i in range(args.quick_reps)
        ]
        source = f"--quick: {args.quick_reps} run(s) of {args.quick_type} v{args.quick_version}"
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (_OBS_ROOT / config_path).resolve()
        if not config_path.exists():
            print(f"❌ Scenarios file not found: {config_path}")
            sys.exit(1)
        runs = load_scenarios(config_path)
        try:
            source = str(config_path.relative_to(_OBS_ROOT))
        except ValueError:
            source = str(config_path)

    if not runs:
        print("No runs to execute.")
        return

    # Header
    print()
    print(f"  Batch source : {source}")
    print(f"  Total runs   : {len(runs)}")
    print()

    if args.dry_run:
        print("  (dry run — would execute the following)\n")
        for i, r in enumerate(runs, 1):
            print(f"    [{i:>2}/{len(runs)}] {r.task_type:18s} v{r.prompt_version} (rep {r.rep}/{r.of})  {r.short_question}")
        return

    # Sanity check: venv python exists
    if not _VENV_PYTHON.exists():
        print(f"❌ Python venv not found: {_VENV_PYTHON}")
        print(f"   Activate it manually or check the path in batch_run.py.")
        sys.exit(1)

    # Run them
    started_all = time.time()
    successes = 0
    failures  = 0

    try:
        for i, r in enumerate(runs, 1):
            ok, _ = run_one(r, i, len(runs))
            if ok:
                successes += 1
            else:
                failures += 1
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted by user. Partial results saved.")

    total_elapsed = time.time() - started_all

    print()
    print(f"  ── done ────────────────────────────────")
    print(f"  ✓ successful  : {successes}")
    print(f"  ✗ failed      : {failures}")
    print(f"  ⏱ total time  : {total_elapsed:.0f}s "
          f"({total_elapsed/max(1,successes+failures):.1f}s per run avg)")
    print()
    print(f"  Open the dashboard:  http://localhost:5001")
    print(f"  Trend view:          http://localhost:5001/trends")
    print()


if __name__ == "__main__":
    main()
