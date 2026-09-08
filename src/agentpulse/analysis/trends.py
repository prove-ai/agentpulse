"""Trend analysis — gradual drift detection across many runs.

Per-run anomaly detection catches sudden outliers; per-version drift catches
prompt-change shifts. This module fills the gap in the middle: **slow drift**
across a window of runs — the kind that creeps up over weeks of deployment.

For each agent, for each signal (output tokens, latency, tool calls, etc.),
we split the window in half and compare the early-half mean to the recent-half
mean. A % change beyond threshold = potential drift.

The verdict per agent:
  stable      — all signals within threshold
  watch       — one signal drifted; could be noise
  drifting    — two or more signals drifted in the same window

We also try to attribute: if the prompt_version changed during the window AND
signals drifted, the suggestion is "changes correlate with prompt vN."
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Thresholds — when does a change count as drift?
# ---------------------------------------------------------------------------
DRIFT_PCT = 20.0          # any signal moving > 20% counts as a drift candidate
WATCH_COUNT = 1           # 1 signal moved → "watch"
DRIFT_COUNT = 2           # 2+ signals moved → "drifting"
MIN_RUNS_FOR_TREND = 4    # need at least this many runs in the window


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SignalChange:
    signal:   str
    early:    float          # mean over early half
    recent:   float          # mean over recent half
    pct:      float          # % change
    drifted:  bool           # |pct| > DRIFT_PCT
    n_early:  int
    n_recent: int


@dataclass
class AgentTrend:
    agent:        str
    signals:      dict[str, SignalChange] = field(default_factory=dict)
    verdict:      str   = "stable"   # stable | watch | drifting | insufficient
    likely_issue: Optional[str] = None
    drifted_signal_names: list[str] = field(default_factory=list)
    # Health summary (used by Agent Health cards on the trends page):
    efficiency:        float = 0.0   # recent expansion ratio (output / input tokens)
    efficiency_delta:  float = 0.0   # % change vs the early-window value
    runs_seen:         int   = 0     # how many runs the agent showed up in


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _pct_change(early: float, recent: float) -> float:
    if early == 0:
        return 0.0 if recent == 0 else 100.0   # going from zero to non-zero is a 100% jump
    return (recent - early) / early * 100.0


def _signal_change(
    signal_name: str,
    early_values: list[float],
    recent_values: list[float],
) -> SignalChange:
    early_mean  = _mean(early_values)
    recent_mean = _mean(recent_values)
    pct = _pct_change(early_mean, recent_mean)
    return SignalChange(
        signal=signal_name,
        early=round(early_mean, 2),
        recent=round(recent_mean, 2),
        pct=round(pct, 1),
        drifted=abs(pct) > DRIFT_PCT,
        n_early=len(early_values),
        n_recent=len(recent_values),
    )


def _extract_per_agent(
    metrics_list: list[dict],
    field_name: str,
    agent: str,
) -> list[float]:
    """Pull a per-agent value out of each run's metrics dict."""
    values: list[float] = []
    for m in metrics_list:
        d = m.get(field_name) or {}
        v = d.get(agent)
        if isinstance(v, (int, float)):
            values.append(float(v))
    return values


# ---------------------------------------------------------------------------
# Per-agent trend computation
# ---------------------------------------------------------------------------
_AGENT_SIGNALS = [
    ("output_tokens",        "per_agent_output_tokens"),
    ("input_tokens",         "per_agent_input_tokens"),
    ("tool_calls",           "tool_calls_per_agent"),
    ("latency_ms",           "per_agent_p50_ms"),
    ("output/input_ratio",   "per_agent_output_input_ratio"),
]


def compute_agent_trend(
    agent: str,
    early_metrics: list[dict],
    recent_metrics: list[dict],
    early_runs: list[dict],
    recent_runs: list[dict],
) -> AgentTrend:
    trend = AgentTrend(agent=agent)

    for ui_name, field_name in _AGENT_SIGNALS:
        early_v  = _extract_per_agent(early_metrics,  field_name, agent)
        recent_v = _extract_per_agent(recent_metrics, field_name, agent)
        if not early_v or not recent_v:
            continue
        change = _signal_change(ui_name, early_v, recent_v)
        trend.signals[ui_name] = change
        if change.drifted:
            trend.drifted_signal_names.append(ui_name)

    n = len(trend.drifted_signal_names)
    if n >= DRIFT_COUNT:
        trend.verdict = "drifting"
    elif n >= WATCH_COUNT:
        trend.verdict = "watch"
    else:
        trend.verdict = "stable"

    # Health snapshot — uses the output/input ratio if available, else a synthetic
    # one from input/output token means.
    ratio_sig = trend.signals.get("output/input_ratio")
    if ratio_sig:
        trend.efficiency       = round(ratio_sig.recent, 2)
        trend.efficiency_delta = round(ratio_sig.pct, 1)
    else:
        in_sig  = trend.signals.get("input_tokens")
        out_sig = trend.signals.get("output_tokens")
        if in_sig and out_sig and in_sig.recent > 0:
            trend.efficiency       = round(out_sig.recent / in_sig.recent, 2)
            if in_sig.early > 0:
                early_eff = out_sig.early / in_sig.early
                trend.efficiency_delta = round(_pct_change(early_eff, trend.efficiency), 1)
    trend.runs_seen = max(
        max((s.n_early + s.n_recent) for s in trend.signals.values()) if trend.signals else 0,
        0,
    )

    # Attribution — did prompt_version change between halves?
    early_versions  = {r.get("prompt_version") for r in early_runs}
    recent_versions = {r.get("prompt_version") for r in recent_runs}
    new_versions = recent_versions - early_versions
    if trend.verdict != "stable" and new_versions:
        new_v = sorted(new_versions)[-1]
        trend.likely_issue = (
            f"Changes correlate with prompt v{new_v} "
            f"(used in {sum(1 for r in recent_runs if r.get('prompt_version')==new_v)} "
            f"of {len(recent_runs)} recent runs)."
        )

    return trend


# ---------------------------------------------------------------------------
# Run-level trend (cost + volume over time)
# ---------------------------------------------------------------------------
@dataclass
class RunLevelTrend:
    cost:        SignalChange
    tokens:      SignalChange
    duration:    SignalChange
    anomaly_rate: SignalChange  # % of runs with at least one anomaly


def compute_run_level_trend(
    early_runs: list[dict],
    recent_runs: list[dict],
    early_metrics: list[dict],
    recent_metrics: list[dict],
) -> RunLevelTrend:
    return RunLevelTrend(
        cost=_signal_change(
            "total cost",
            [r.get("total_cost_usd", 0) or 0 for r in early_runs],
            [r.get("total_cost_usd", 0) or 0 for r in recent_runs],
        ),
        tokens=_signal_change(
            "total tokens",
            [(r.get("total_input_tokens", 0) or 0) + (r.get("total_output_tokens", 0) or 0)
             for r in early_runs],
            [(r.get("total_input_tokens", 0) or 0) + (r.get("total_output_tokens", 0) or 0)
             for r in recent_runs],
        ),
        duration=_signal_change(
            "wall clock",
            [(r.get("total_duration_ms", 0) or 0) / 1000 for r in early_runs],
            [(r.get("total_duration_ms", 0) or 0) / 1000 for r in recent_runs],
        ),
        anomaly_rate=_signal_change(
            "anomaly rate (%)",
            [100 if not m.get("terminated_cleanly", True) else 0 for m in early_metrics],
            [100 if not m.get("terminated_cleanly", True) else 0 for m in recent_metrics],
        ),
    )


# ---------------------------------------------------------------------------
# Top-level entry — used by the dashboard route
# ---------------------------------------------------------------------------
def build_trends(
    runs:    list[dict],     # sorted ascending by timestamp
    metrics: list[dict],     # parallel to runs
) -> dict:
    """Return everything the /trends template needs."""
    n = len(runs)
    if n < MIN_RUNS_FOR_TREND:
        return {
            "insufficient":  True,
            "runs_count":    n,
            "min_required":  MIN_RUNS_FOR_TREND,
        }

    # Split in half
    mid = n // 2
    early_runs,    recent_runs    = runs[:mid],    runs[mid:]
    early_metrics, recent_metrics = metrics[:mid], metrics[mid:]

    # All agents seen in this window
    agents = []
    seen = set()
    for m in metrics:
        for a in (m.get("unique_agents") or []):
            if a not in seen:
                seen.add(a)
                agents.append(a)

    # Per-agent trends
    agent_trends = {
        a: compute_agent_trend(a, early_metrics, recent_metrics, early_runs, recent_runs)
        for a in agents
    }

    # Run-level (cost, volume, anomaly rate)
    run_level = compute_run_level_trend(
        early_runs, recent_runs, early_metrics, recent_metrics,
    )

    # Versions present in this window
    versions = sorted({r.get("prompt_version") for r in runs if r.get("prompt_version") is not None})

    # Top handoff pairs across the window
    top_handoffs = compute_top_handoffs(metrics, top_n=5)

    return {
        "insufficient": False,
        "runs_count":   n,
        "n_early":      len(early_runs),
        "n_recent":     len(recent_runs),
        "agent_trends": agent_trends,
        "run_level":    run_level,
        "versions":     versions,
        "top_handoffs": top_handoffs,
        "early_end_ts": early_runs[-1]["timestamp"] if early_runs else "",
        "recent_start_ts": recent_runs[0]["timestamp"] if recent_runs else "",
    }


# ---------------------------------------------------------------------------
# Top handoffs — most common (sender → receiver) pairs across the window
# ---------------------------------------------------------------------------
def compute_top_handoffs(metrics_list: list[dict], top_n: int = 5) -> dict:
    """Count handoff pairs across all runs in the window.

    Returns: { 'pairs': [{from, to, count, pct}], 'total': int }
    Each metrics dict is expected to have a `handoffs` list of
    {agent_a, agent_b, ...} entries.
    """
    counts: dict[tuple[str, str], int] = {}
    for m in metrics_list:
        for h in (m.get("handoffs") or []):
            a = h.get("agent_a") or h.get("from")
            b = h.get("agent_b") or h.get("to")
            if not a or not b:
                continue
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    pairs = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    return {
        "total": total,
        "pairs": [
            {
                "from":  a, "to": b,
                "count": c,
                "pct":   round((c / total * 100), 1) if total else 0.0,
            }
            for (a, b), c in pairs
        ],
    }
