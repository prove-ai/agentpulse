"""Terminal reporter — renders a AnomalyReport as a rich structured output."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

from analysis.run_anomaly import AnomalyReport, MetricVerdict

console = Console()


def _verdict_row(v: MetricVerdict) -> tuple[str, str, str, str, str]:
    """Return (metric, current, baseline, delta, status) strings."""
    metric  = v.name
    current = str(v.current)
    base    = str(v.baseline)

    if isinstance(v.delta, float):
        delta = f"{v.delta:+.1f}%"
    elif isinstance(v.delta, int):
        delta = f"{v.delta:+d}"
    else:
        delta = "—"

    if v.drifted:
        status = f"[bold red]⚠ {v.label}[/]"
    else:
        status = "[green]✅ ok[/]"

    return metric, current, base, delta, status


def print_report(report: AnomalyReport, metrics: dict) -> None:
    console.rule(f"[bold]Observability Report[/]  run {report.run_id[:8]}")

    # Header
    console.print(
        f"  task type      [cyan]{report.task_type}[/]\n"
        f"  prompt version [cyan]v{report.prompt_version}[/]  "
        f"(baseline v{report.baseline_version}, "
        f"{report.baseline_run_count} baseline run(s))\n"
        f"  turns          [cyan]{metrics.get('total_turns')}[/]\n"
        f"  route          [cyan]{' → '.join(metrics.get('agent_sequence') or [])}[/]\n"
        f"  termination    [cyan]{metrics.get('termination_reason', '')}[/]\n"
        f"  total cost     [cyan]${metrics.get('total_cost_usd', 0):.6f}[/]\n"
    )

    if not report.verdicts:
        console.print(
            "[yellow]No baseline runs found — this run will serve as the baseline.[/]\n"
        )
        _print_metrics_table(metrics)
        return

    # Drift table
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    table.add_column("Metric",   style="dim", width=22)
    table.add_column("Current",  justify="right", width=16)
    table.add_column("Baseline", justify="right", width=16)
    table.add_column("Delta",    justify="right", width=10)
    table.add_column("Status",   width=24)

    for v in report.verdicts:
        metric, cur, base, delta, status = _verdict_row(v)
        table.add_row(metric, cur, base, delta, Text.from_markup(status))

    console.print(table)

    if report.has_anomaly:
        labels = ", ".join(report.anomaly_labels)
        console.print(f"[bold red]DRIFT DETECTED:[/] {labels}\n")
    else:
        console.print("[bold green]✅ All metrics within threshold — no drift detected.[/]\n")

    _print_metrics_table(metrics)


def _print_metrics_table(metrics: dict) -> None:
    """Print a secondary table with the full derived metrics."""
    console.rule("[dim]Full Derived Metrics[/]")
    table = Table(box=box.MINIMAL, show_header=False)
    table.add_column("Key",   style="dim", width=32)
    table.add_column("Value", width=50)

    skip = {"run_id", "task_type", "prompt_version", "agent_sequence",
            "unique_agents", "input_tokens_per_turn"}

    def _fmt(v):
        if isinstance(v, dict):
            return "  ".join(f"{k}: {round(x, 4) if isinstance(x, float) else x}" for k, x in v.items())
        if isinstance(v, list):
            return " → ".join(str(x) for x in v)
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for k, v in metrics.items():
        if k in skip:
            continue
        table.add_row(k, _fmt(v))

    console.print(table)
    console.print()
