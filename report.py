"""Report CLI — view metrics and drift for captured runs.

Usage:
    python report.py              # same as --last
    python report.py --last       # last run: full metrics + drift vs baseline
    python report.py --all        # table of all stored runs
    python report.py --drift      # only runs where drift was detected
    python report.py --run <id>   # specific run by run_id prefix
    python report.py --clear      # delete all stored runs (with confirmation)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_OBS_ROOT = Path(__file__).parent
sys.path.insert(0, str(_OBS_ROOT))

from rich.console import Console
from rich.table import Table
from rich import box

from analysis.layer1_raw import (
    list_runs, get_run, get_agent_spans, get_tool_calls, get_baseline_runs
)
from analysis.run_metrics import compute_all
from analysis.run_anomaly import build_anomaly_report
from reporter.terminal import print_report
from storage.sqlite_store import get_connection

console = Console()


# ---------------------------------------------------------------------------
# --all: summary table of every run
# ---------------------------------------------------------------------------
def cmd_all() -> None:
    runs = list_runs(limit=100)
    if not runs:
        console.print("[yellow]No runs stored yet.[/]")
        return

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    table.add_column("Run ID",    width=10)
    table.add_column("Task Type", width=18)
    table.add_column("v",         width=3,  justify="right")
    table.add_column("Turns",     width=6,  justify="right")
    table.add_column("Route",     width=40)
    table.add_column("Tokens",    width=8,  justify="right")
    table.add_column("Cost $",    width=9,  justify="right")
    table.add_column("Status",    width=10)
    table.add_column("Time",      width=22)

    for r in runs:
        import json
        seq = json.loads(r.get("agent_sequence") or "[]")
        route = " → ".join(seq) if seq else "—"
        ok = "APPROVED" in (r.get("termination_reason") or "")
        status_str = "[green]✅[/]" if ok else "[red]⚠[/]"
        table.add_row(
            r["run_id"][:8],
            r.get("task_type") or "—",
            str(r.get("prompt_version") or "—"),
            str(r.get("total_turns") or "—"),
            route[:38],
            str(r.get("total_input_tokens", 0) + r.get("total_output_tokens", 0)),
            f"{r.get('total_cost_usd', 0):.4f}",
            status_str,
            (r.get("timestamp") or "")[:19].replace("T", " "),
        )

    console.print(table)
    console.print(f"[dim]{len(runs)} run(s) total[/]")


# ---------------------------------------------------------------------------
# --last / --run: full metrics + drift for one run
# ---------------------------------------------------------------------------
def cmd_run(run_id_prefix: str | None = None) -> None:
    if run_id_prefix:
        # Find by prefix
        all_runs = list_runs(limit=200)
        matched = [r for r in all_runs if r["run_id"].startswith(run_id_prefix)]
        if not matched:
            console.print(f"[red]No run found with prefix {run_id_prefix!r}[/]")
            return
        run = matched[0]
    else:
        runs = list_runs(limit=1)
        if not runs:
            console.print("[yellow]No runs stored yet.[/]")
            return
        run = runs[0]

    agent_spans = get_agent_spans(run["run_id"])
    tool_calls  = get_tool_calls(run["run_id"])
    metrics     = compute_all(run, agent_spans, tool_calls)

    # Build baseline from same task_type + same prompt version 1
    baseline_runs = [
        r for r in get_baseline_runs()
        if r["run_id"] != run["run_id"]
        and r.get("task_type") == run.get("task_type")
    ]
    baseline_metrics = []
    for br in baseline_runs:
        b_spans = get_agent_spans(br["run_id"])
        b_tools = get_tool_calls(br["run_id"])
        baseline_metrics.append(compute_all(br, b_spans, b_tools))

    baseline_version = baseline_runs[0]["prompt_version"] if baseline_runs else run.get("prompt_version", 1)
    report = build_anomaly_report(metrics, baseline_metrics, baseline_version)
    print_report(report, metrics)


# ---------------------------------------------------------------------------
# --drift: only runs where drift was detected
# ---------------------------------------------------------------------------
def cmd_drift() -> None:
    runs = list_runs(limit=100)
    if not runs:
        console.print("[yellow]No runs stored yet.[/]")
        return

    baseline_runs = get_baseline_runs()
    drifted = []

    for run in runs:
        agent_spans = get_agent_spans(run["run_id"])
        tool_calls  = get_tool_calls(run["run_id"])
        metrics     = compute_all(run, agent_spans, tool_calls)

        bl = [
            r for r in baseline_runs
            if r["run_id"] != run["run_id"]
            and r.get("task_type") == run.get("task_type")
        ]
        bl_metrics = [
            compute_all(r, get_agent_spans(r["run_id"]), get_tool_calls(r["run_id"]))
            for r in bl
        ]
        bl_version = bl[0]["prompt_version"] if bl else run.get("prompt_version", 1)
        report = build_anomaly_report(metrics, bl_metrics, bl_version)

        if report.has_anomaly:
            drifted.append((run, report))

    if not drifted:
        console.print("[green]No drift detected in any stored run.[/]")
        return

    console.print(f"[bold red]{len(drifted)} run(s) with drift detected:[/]\n")
    for run, report in drifted:
        import json
        seq = json.loads(run.get("agent_sequence") or "[]")
        console.print(
            f"  [bold]{run['run_id'][:8]}[/]  "
            f"{run.get('task_type')}  v{run.get('prompt_version')}  "
            f"{'→'.join(seq)}  "
            f"[red]{', '.join(report.anomaly_labels)}[/]"
        )


# ---------------------------------------------------------------------------
# --clear: wipe all runs
# ---------------------------------------------------------------------------
def cmd_clear() -> None:
    answer = input("Delete ALL stored runs? This cannot be undone. (yes/no): ")
    if answer.strip().lower() != "yes":
        console.print("Aborted.")
        return
    conn = get_connection()
    conn.execute("DELETE FROM tool_calls")
    conn.execute("DELETE FROM spans")
    conn.execute("DELETE FROM runs")
    conn.commit()
    console.print("[green]All runs deleted.[/]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="View observability metrics for captured agent runs."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--last",  action="store_true", help="Show last run (default)")
    group.add_argument("--all",   action="store_true", help="Table of all runs")
    group.add_argument("--drift", action="store_true", help="Only drifted runs")
    group.add_argument("--run",   metavar="ID",        help="Specific run by id prefix")
    group.add_argument("--clear", action="store_true", help="Delete all stored runs")
    parser.add_argument("--db", metavar="NAME",
                        help="Project db name under db/ (default: runs)")
    args = parser.parse_args()

    if args.db:
        from storage.sqlite_store import resolve_db_path, set_active_db_path
        set_active_db_path(resolve_db_path(args.db))

    if args.all:
        cmd_all()
    elif args.drift:
        cmd_drift()
    elif args.run:
        cmd_run(args.run)
    elif args.clear:
        cmd_clear()
    else:
        cmd_run()  # default: last run


if __name__ == "__main__":
    main()
