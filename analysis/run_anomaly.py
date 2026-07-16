"""Layer 3 — single-run ANOMALY detection (not drift).

Compares ONE run against the average of baseline runs. Labels are *_ANOMALY
because a single run differing from average is variance/anomaly, NOT drift.
Drift requires sustained, directional, multi-run change — see version_drift.py.

Per-agent anomaly checks:
  input_tokens    — unusually high/low for this agent in this run?
  output_tokens   — unusually high/low?
  tool_usage_rate — failed to call tools when it usually does (or vice versa)?
  latency         — unusually slow?

Run-level checks:
  total_turns     — unusual turn count
  agent_sequence  — different route than usual
  tool_failure    — tools failing
  termination     — didn't end cleanly (failure markers like stuck_loop, error)
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------
DEFAULTS = {
    "turn_count_delta":       1,     # absolute ± turns (still used as a floor)
    "tool_failure_ratio_max": 0.10,
    "tool_usage_rate_delta":  0.20,  # 20 percentage points (categorical-ish signal)
    # Percent-change thresholds for paired version comparison (version_drift.py).
    # A metric is flagged when its baseline→target change exceeds this fraction.
    # Only used in "paired" mode, where the two versions ran the same prompts.
    "cost_pct":               0.25,  # ±25% avg cost
    "ratio_pct":              0.30,  # ±30% verbosity (output/input) ratio
    "latency_pct":            0.30,  # ±30% latency
    "token_pct":              0.30,  # ±30% token volume (in or out)
    # The pct fields below are no longer the primary detector — the IQR/σ-based
    # normal range replaces them. They remain only as a fallback minimum spread
    # when the baseline has zero variance (e.g. all runs returned identical
    # values), so we don't flag tiny floating-point differences.
    "fallback_pct":           0.05,  # 5% — minimum tolerance when σ/IQR is 0
}


# ---------------------------------------------------------------------------
# Range-based anomaly detection
# ---------------------------------------------------------------------------
def _normal_range(values: list[float]) -> tuple[float, float] | None:
    """Return (low, high) bounds for the 'normal' range of a baseline series.

    - ≥4 values → IQR rule:  [Q1 − 1.5·IQR, Q3 + 1.5·IQR]   (boxplot whiskers)
    - 2–3 values → ±2σ rule: [mean − 2·σ, mean + 2·σ]
    - <2 values  → None (not enough baseline to judge)

    When the spread is zero (all identical), we expand the bound by a small
    fallback tolerance so floating-point noise isn't flagged.
    """
    n = len(values)
    if n < 2:
        return None

    mean = statistics.mean(values)
    fallback_tol = max(abs(mean) * DEFAULTS["fallback_pct"], 1e-6)

    if n >= 4:
        sv = sorted(values)
        q1, _med, q3 = statistics.quantiles(sv, n=4)
        iqr = q3 - q1
        if iqr == 0:
            return (mean - fallback_tol, mean + fallback_tol)
        return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    try:
        std = statistics.stdev(values)
    except statistics.StatisticsError:
        std = 0.0
    if std == 0:
        return (mean - fallback_tol, mean + fallback_tol)
    return (mean - 2 * std, mean + 2 * std)


def _outside_range(x: float, rng: tuple[float, float] | None) -> bool:
    """True if x falls outside the normal range. False if range is unknown."""
    if rng is None:
        return False
    lo, hi = rng
    return x < lo or x > hi


def _range_method(n: int) -> str:
    """Label of the rule used, for tooltips/explanations."""
    if n >= 4:
        return "IQR"
    if n >= 2:
        return "±2σ"
    return "n/a"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class MetricVerdict:
    name:     str
    current:  Any
    baseline: Any
    delta:    Any
    drifted:  bool
    label:    str
    note:     str = ""


@dataclass
class AgentDriftVerdict:
    agent_name: str
    signal:     str
    current:    float
    baseline:   float
    delta_pct:  float
    drifted:    bool
    label:      str


@dataclass
class AnomalyReport:
    """Single-run anomaly report — not drift. Compares one run to the baseline avg."""
    run_id:               str
    task_type:            str
    prompt_version:       int
    baseline_version:     int
    baseline_run_count:   int
    verdicts:             list[MetricVerdict]       = field(default_factory=list)
    per_agent_verdicts:   dict[str, list[AgentDriftVerdict]] = field(default_factory=dict)

    @property
    def has_anomaly(self) -> bool:
        if any(v.drifted for v in self.verdicts):
            return True
        return any(
            v.drifted
            for verdicts in self.per_agent_verdicts.values()
            for v in verdicts
        )

    @property
    def anomaly_labels(self) -> list[str]:
        labels = [v.label for v in self.verdicts if v.drifted]
        for agent, verdicts in self.per_agent_verdicts.items():
            for v in verdicts:
                if v.drifted:
                    labels.append(f"{agent}:{v.label}")
        return labels





# ---------------------------------------------------------------------------
# Baseline aggregation
# ---------------------------------------------------------------------------
def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _baseline_run_stats(baseline_list: list[dict]) -> dict[str, Any]:
    """Aggregate baseline run-level stats."""
    if not baseline_list:
        return {}

    sequences = [tuple(m.get("agent_sequence") or []) for m in baseline_list]
    seq_mode  = max(set(sequences), key=sequences.count) if sequences else ()

    turns_series   = [m.get("total_turns", 0) for m in baseline_list]
    latency_series = [m.get("latency_p50_ms", 0.0) for m in baseline_list]
    tokens_series  = [m.get("total_tokens", 0) for m in baseline_list]
    cost_series    = [m.get("total_cost_usd", 0.0) for m in baseline_list]
    tfr_series     = [m.get("tool_failure_ratio", 0.0) for m in baseline_list]

    return {
        "total_turns":           _mean(turns_series),
        "agent_sequence_mode":   list(seq_mode),
        "latency_p50_ms":        _mean(latency_series),
        "total_tokens":          _mean(tokens_series),
        "total_cost_usd":        _mean(cost_series),
        "tool_failure_ratio":    _mean(tfr_series),
        "termination_reasons":   [m.get("termination_reason", "") for m in baseline_list],
        # Raw series for range-based anomaly detection
        "_n":                     len(baseline_list),
        "_turns_series":          turns_series,
        "_latency_series":        latency_series,
        "_tokens_series":         tokens_series,
        "_cost_series":           cost_series,
        "_tfr_series":            tfr_series,
    }


def _baseline_agent_stats(baseline_list: list[dict]) -> dict[str, dict[str, float]]:
    """For each agent, aggregate per-agent stats across baseline runs."""
    agent_data: dict[str, list[dict]] = defaultdict(list)
    for m in baseline_list:
        agents = m.get("unique_agents") or []
        for agent in agents:
            agent_data[agent].append(m)

    result: dict[str, dict[str, float]] = {}
    for agent, runs in agent_data.items():
        inp = [(m.get("per_agent_input_tokens")  or {}).get(agent, 0)   for m in runs]
        out = [(m.get("per_agent_output_tokens") or {}).get(agent, 0)   for m in runs]
        rat = [(m.get("per_agent_output_input_ratio") or {}).get(agent, 0.0) for m in runs]
        tlr = [(m.get("tool_usage_rate_per_agent")    or {}).get(agent, 0.0) for m in runs]
        lat = [(m.get("per_agent_p50_ms")             or {}).get(agent, 0.0) for m in runs]

        result[agent] = {
            "avg_input_tokens":    _mean(inp),
            "avg_output_tokens":   _mean(out),
            "avg_ratio":           _mean(rat),
            "avg_tool_usage_rate": _mean(tlr),
            "avg_latency_ms":      _mean(lat),
            "_n":                  len(runs),
            "_input_series":       inp,
            "_output_series":      out,
            "_ratio_series":       rat,
            "_tool_series":        tlr,
            "_latency_series":     lat,
        }
    return result


# ---------------------------------------------------------------------------
# Per-metric run-level checks
# ---------------------------------------------------------------------------
def _pct(cur: float, base: float) -> float:
    return round((cur - base) / base * 100, 1) if base else 0.0


def _range_note(rng: tuple[float, float] | None, n: int, fmt: str = "{:.1f}") -> str:
    if rng is None:
        return "no baseline range"
    return f"normal: {fmt.format(rng[0])}–{fmt.format(rng[1])} ({_range_method(n)}, n={n})"


def _check_range(name: str, label: str, current: float, base_mean: float,
                 series: list[float], fmt: str = "{:.1f}") -> MetricVerdict:
    """Generic range-based anomaly check used by run-level numeric signals."""
    rng = _normal_range(series)
    flagged = _outside_range(current, rng)
    delta = round(_pct(current, base_mean), 1) if base_mean else 0.0
    return MetricVerdict(
        name=name, current=round(current, 2), baseline=round(base_mean, 2),
        delta=delta, drifted=flagged, label=label,
        note=_range_note(rng, len(series), fmt),
    )


def _check_turns(cur: dict, base: dict, t: dict) -> MetricVerdict:
    """Turns: range-based, but with a 1-turn absolute floor (never flag a ±0.5 jitter)."""
    c = cur.get("total_turns", 0)
    series = base.get("_turns_series") or []
    b_mean = base.get("total_turns", 0)
    rng = _normal_range(series)
    delta = round(c - b_mean, 1)
    # Anomaly only if outside the range AND differs by more than 1 turn
    flagged = _outside_range(c, rng) and abs(delta) > t["turn_count_delta"]
    return MetricVerdict(
        name="total_turns", current=c, baseline=round(b_mean, 1),
        delta=delta, drifted=flagged, label="TURN_ANOMALY",
        note=_range_note(rng, len(series), "{:.1f}"),
    )


def _check_route(cur: dict, base: dict) -> MetricVerdict:
    c_seq = cur.get("agent_sequence") or []
    b_seq = base.get("agent_sequence_mode") or []
    drifted = c_seq != b_seq
    return MetricVerdict(
        name="agent_sequence", current=c_seq, baseline=b_seq,
        delta=None, drifted=drifted, label="ROUTE_ANOMALY",
        note="sequence changed" if drifted else "matches most-common route",
    )


def _check_tool_failure(cur: dict, base: dict, t: dict) -> MetricVerdict:
    c, b = cur.get("tool_failure_ratio", 0.0), base.get("tool_failure_ratio", 0.0)
    return MetricVerdict(
        name="tool_failure_ratio", current=round(c, 3), baseline=round(b, 3),
        delta=round(c - b, 3), drifted=c > t["tool_failure_ratio_max"],
        label="TOOL_RELIABILITY", note=f"failure rate {c*100:.1f}%",
    )


def _check_termination(cur: dict, base: dict) -> MetricVerdict:
    # Use the shared helper so this check matches the rest of the dashboard's
    # framework-agnostic clean-finish detection (handles APPROVED, completed,
    # ok, chain_complete, etc. — and only flags real failures).
    from analysis.run_metrics import is_clean_termination
    c = cur.get("termination_reason", "") or ""
    clean = is_clean_termination(c)
    return MetricVerdict(
        name="termination_reason",
        current=c or "completed",
        baseline="clean finish (expected)",
        delta=None, drifted=not clean, label="TERMINATION_ANOMALY",
        note="did not end cleanly" if not clean else "ended cleanly",
    )


# ---------------------------------------------------------------------------
# Per-agent drift checks
# ---------------------------------------------------------------------------
def _agent_range_verdict(
    agent: str, signal: str, current: float, base_mean: float,
    series: list[float], label: str,
) -> AgentDriftVerdict:
    """Range-based per-agent anomaly check."""
    rng = _normal_range(series)
    flagged = _outside_range(current, rng)
    delta_pct = _pct(current, base_mean)
    return AgentDriftVerdict(
        agent_name=agent, signal=signal,
        current=round(current, 2), baseline=round(base_mean, 2),
        delta_pct=delta_pct, drifted=flagged, label=label,
    )


def _per_agent_drift(
    cur: dict,
    baseline_agent_stats: dict[str, dict[str, float]],
    t: dict,
) -> dict[str, list[AgentDriftVerdict]]:
    result: dict[str, list[AgentDriftVerdict]] = {}
    agents = cur.get("unique_agents") or []

    for agent in agents:
        base = baseline_agent_stats.get(agent)
        if not base:
            continue

        verdicts: list[AgentDriftVerdict] = []

        verdicts.append(_agent_range_verdict(
            agent, "input_tokens",
            (cur.get("per_agent_input_tokens") or {}).get(agent, 0),
            base["avg_input_tokens"],   base.get("_input_series", []),
            "INPUT_TOKEN_ANOMALY",
        ))
        verdicts.append(_agent_range_verdict(
            agent, "output_tokens",
            (cur.get("per_agent_output_tokens") or {}).get(agent, 0),
            base["avg_output_tokens"],  base.get("_output_series", []),
            "OUTPUT_TOKEN_ANOMALY",
        ))
        verdicts.append(_agent_range_verdict(
            agent, "latency_p50_ms",
            (cur.get("per_agent_p50_ms") or {}).get(agent, 0.0),
            base["avg_latency_ms"],     base.get("_latency_series", []),
            "LATENCY_ANOMALY",
        ))

        # Tool usage rate — keep absolute-delta rule (it's a percentage in [0,1],
        # categorical-ish: "uses tools" vs "doesn't"). The range method doesn't
        # add much for a bimodal signal.
        cur_rate  = (cur.get("tool_usage_rate_per_agent") or {}).get(agent, 0.0)
        base_rate = base["avg_tool_usage_rate"]
        rate_delta = abs(cur_rate - base_rate)
        verdicts.append(AgentDriftVerdict(
            agent_name=agent, signal="tool_usage_rate",
            current=round(cur_rate, 3), baseline=round(base_rate, 3),
            delta_pct=round((cur_rate - base_rate) * 100, 1),
            drifted=rate_delta > t["tool_usage_rate_delta"],
            label="TOOL_USAGE_ANOMALY",
        ))

        result[agent] = verdicts

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_anomaly_report(
    current_metrics:    dict,
    baseline_metrics_list: list[dict],
    baseline_version:   int = 1,
    thresholds:         dict | None = None,
) -> DriftReport:
    t = {**DEFAULTS, **(thresholds or {})}

    if not baseline_metrics_list:
        return AnomalyReport(
            run_id=current_metrics["run_id"],
            task_type=current_metrics.get("task_type", ""),
            prompt_version=current_metrics.get("prompt_version", 1),
            baseline_version=baseline_version,
            baseline_run_count=0,
            verdicts=[],
            per_agent_verdicts={},
        )

    base        = _baseline_run_stats(baseline_metrics_list)
    agent_base  = _baseline_agent_stats(baseline_metrics_list)

    verdicts = [
        _check_turns(current_metrics, base, t),
        _check_route(current_metrics, base),
        _check_tool_failure(current_metrics, base, t),
        _check_termination(current_metrics, base),
    ]

    per_agent = _per_agent_drift(current_metrics, agent_base, t)

    return AnomalyReport(
        run_id=current_metrics["run_id"],
        task_type=current_metrics.get("task_type", ""),
        prompt_version=current_metrics.get("prompt_version", 1),
        baseline_version=baseline_version,
        baseline_run_count=len(baseline_metrics_list),
        verdicts=verdicts,
        per_agent_verdicts=per_agent,
    )


