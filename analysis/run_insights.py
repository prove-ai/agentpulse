"""Layer 4 — insights.

Two independent kinds of insight:

  1. SINGLE-RUN insights  (no baseline needed)
     Judged on the run's own merits — things that are self-evidently worth
     attention regardless of any reference: unexpected loops, an agent running
     too many times, the run not finishing cleanly, tool failures, a bottleneck
     agent, a tool-retry storm. These ALWAYS appear.

  2. DRIFT insights  (baseline needed)
     Comparison vs the baseline average for the same task type: verbosity shift,
     cost change, an agent that stopped using tools, route change. These only
     appear once a baseline exists, because they're meaningless without a
     reference. (Ratios live here — never as single-run judgements.)

All rules reference agents dynamically, so this generalizes to any system.

Severity:  critical 🔴  ·  warning 🟡  ·  improvement 🟢  ·  info ℹ️
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from analysis.run_anomaly import (
    _baseline_agent_stats, _baseline_run_stats, DEFAULTS,
    _normal_range, _outside_range,
)

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "improvement": 2, "info": 3}
_SEVERITY_ICON  = {"critical": "🔴", "warning": "🟡", "improvement": "🟢", "info": "ℹ️"}

# Single-run thresholds
MAX_EXPECTED_CALLS   = 2     # an agent running more than this = stuck pattern
TOOL_RETRY_THRESHOLD = 5     # same tool called this many times in a run = retry storm
BOTTLENECK_SHARE     = 0.50  # one agent taking >50% of wall-clock time


@dataclass
class Insight:
    severity:       str
    category:       str
    title:          str
    evidence:       str
    interpretation: str
    agent:          Optional[str] = None
    kind:           str = "single-run"   # "single-run" | "drift"
    # Optional structured payload for richer rendering (e.g. the route diff),
    # so evidence strings can stay short instead of dumping a wall of text.
    details:        Optional[dict] = None

    @property
    def icon(self) -> str:
        return _SEVERITY_ICON.get(self.severity, "•")


def _rank_key(i: "Insight") -> int:
    """Display order: risks first, then performance bottlenecks, then positives.

      0 critical / 1 warning  → warning / risk
      2 performance category  → performance bottleneck
      3 improvement           → positive completion
      4 everything else       → info
    """
    if i.severity == "critical":
        return 0
    if i.severity == "warning":
        return 1
    if i.category == "performance":
        return 2
    if i.severity == "improvement":
        return 3
    return 4


def _pct_change(cur: float, base: float) -> float:
    return (cur - base) / base * 100 if base else 0.0


# ═══════════════════════════════════════════════════════════════════════
# 1. SINGLE-RUN INSIGHTS  (no baseline)
# ═══════════════════════════════════════════════════════════════════════
def _outcome(m: dict) -> list[Insight]:
    reason = m.get("termination_reason", "unknown")
    if m.get("terminated_cleanly", False):
        return [Insight(
            "improvement", "reliability",
            "Completed cleanly",
            f"Finished in {m.get('total_turns', 0)} turns · ended with APPROVED",
            "The run reached an approved deliverable without intervention.",
        )]
    return [Insight(
        "critical", "reliability",
        "Run did not finish cleanly",
        f"Stopped after {m.get('total_turns', 0)} turns · {reason}",
        "The run ended without an approved result — investigate the cause.",
    )]


def _tool_failures(m: dict) -> list[Insight]:
    n = m.get("tool_failure_count", 0)
    if n > 0:
        return [Insight(
            "critical", "tools",
            f"{n} tool call{'s' if n > 1 else ''} failed",
            f"{n} of {m.get('total_tool_calls', 0)} tool calls returned an error",
            "Failed tools can feed bad data into later steps.",
        )]
    return []


def _loops(m: dict) -> list[Insight]:
    out: list[Insight] = []

    # Unexpected loops (A→B→A where B didn't request it)
    for p in (m.get("reentry_patterns") or []):
        if p.get("flagged"):
            out.append(Insight(
                "warning", "routing",
                f"Unexpected loop: {p['pattern']}",
                f"Occurred at turn {p['turn']}",
                "An agent was re-invoked without explicitly asking for it — likely inefficiency.",
            ))
        else:
            out.append(Insight(
                "info", "routing",
                f"Re-entry: {p['pattern']}",
                f"Occurred at turn {p['turn']}",
                "An agent ran again because a colleague explicitly requested it — expected behaviour.",
            ))

    # Any agent running more than the expected cap. Running multiple times is a
    # legitimate pattern in many designs (iterative / multi-round agents), so this
    # is informational — not an anomaly. A genuine stuck loop surfaces separately
    # as an "Unexpected loop" (re-entry not requested) or as a turn-count anomaly
    # vs the baseline.
    for agent, count in (m.get("agent_turn_counts") or {}).items():
        if count > MAX_EXPECTED_CALLS:
            out.append(Insight(
                "info", "routing",
                f"{agent} ran {count} times",
                f"{agent} was selected {count} times in one run",
                "Expected for iterative agents. Worth a look only if the run also "
                "stalled or shows an unexpected loop.",
                agent=agent,
            ))
    return out


def _tool_retry(m: dict) -> list[Insight]:
    out: list[Insight] = []
    for agent, tools in (m.get("tool_distribution_per_agent") or {}).items():
        for tool, cnt in tools.items():
            if cnt >= TOOL_RETRY_THRESHOLD:
                out.append(Insight(
                    "warning", "tools",
                    f"{agent} called {tool} {cnt} times",
                    f"{tool} invoked {cnt}× in a single run",
                    "A high repeat count on one tool can indicate a retry loop or thrashing.",
                    agent=agent,
                ))
    return out


def _bottleneck(m: dict) -> list[Insight]:
    per_time = m.get("per_agent_total_ms") or {}
    total    = m.get("total_wall_clock_ms", 0)
    if not per_time or total <= 0:
        return []
    agent = max(per_time, key=per_time.get)
    share = per_time[agent] / total
    if share >= BOTTLENECK_SHARE:
        return [Insight(
            "info", "performance",
            f"{agent} was the bottleneck",
            f"{per_time[agent]/1000:.1f}s — {share*100:.0f}% of the run's {total/1000:.1f}s",
            "Most of the wall-clock time was spent in this one agent.",
            agent=agent,
        )]
    return []


def single_run_insights(m: dict) -> list[Insight]:
    out: list[Insight] = []
    out += _outcome(m)
    out += _tool_failures(m)
    out += _loops(m)
    out += _tool_retry(m)
    out += _bottleneck(m)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 2. DRIFT INSIGHTS  (vs baseline — ratios live here)
# ═══════════════════════════════════════════════════════════════════════
def _drift_tools(m, agent_base, baseline_list) -> list[Insight]:
    out: list[Insight] = []
    cur_rates = m.get("tool_usage_rate_per_agent") or {}
    for agent, base in agent_base.items():
        base_rate = base.get("avg_tool_usage_rate", 0.0)
        cur_rate  = cur_rates.get(agent, 0.0)
        if base_rate >= 0.5 and cur_rate < 0.1:
            out.append(Insight(
                "critical", "tools",
                f"{agent} didn't use tools this run",
                f"Usually calls tools in {base_rate*100:.0f}% of turns · this run 0%",
                "Possible fabrication — verify the answer. Investigate if this happens repeatedly.",
                agent=agent, kind="anomaly",
            ))
    return out


def _drift_routing(m, run_base) -> list[Insight]:
    out: list[Insight] = []
    cur_seq  = m.get("agent_sequence") or []
    base_seq = run_base.get("agent_sequence_mode") or []
    if base_seq and cur_seq != base_seq:
        if cur_seq and cur_seq[-1] != base_seq[-1]:
            out.append(Insight(
                "critical", "routing",
                f"This run didn't end with {base_seq[-1]}",
                f"Usual last agent: {base_seq[-1]} · this run: {cur_seq[-1]}",
                "Final quality gate skipped or replaced. Watch for repeated occurrences.",
                kind="anomaly",
            ))
        else:
            added   = [a for a in dict.fromkeys(cur_seq)  if a not in base_seq]
            removed = [a for a in dict.fromkeys(base_seq) if a not in cur_seq]
            na, nr  = len(added), len(removed)
            def _ag(n):  # "agent" / "agents"
                return f"{n} agent{'s' if n != 1 else ''}"
            if na and nr:
                evidence = f"This run used {_ag(na)} not in the usual path and skipped {nr}."
            elif na:
                evidence = f"This run used {_ag(na)} not in the usual path."
            elif nr:
                evidence = f"This run skipped {_ag(nr)} from the usual path."
            else:
                evidence = "This run took a different order than the usual path."
            out.append(Insight(
                "warning", "routing", "Unusual route",
                evidence,
                "Single-run route variance. If repeated across runs it becomes a drift.",
                kind="anomaly",
                details={
                    "usual_steps": len(base_seq),
                    "this_steps":  len(cur_seq),
                    "usual_path":  base_seq,
                    "this_path":   cur_seq,
                    "added":       added,
                    "removed":     removed,
                },
            ))
    return out


def _drift_verbosity(m, agent_base) -> list[Insight]:
    out: list[Insight] = []
    cur_ratios = m.get("per_agent_output_input_ratio") or {}
    cur_inputs = m.get("per_agent_input_tokens") or {}
    for agent, base in agent_base.items():
        base_ratio = base.get("avg_ratio", 0.0)
        cur_ratio  = cur_ratios.get(agent, 0.0)
        if base_ratio <= 0:
            continue
        # Skip agents with no LLM input this run: the output/input ratio is
        # undefined (defaults to 0.0), not genuinely "terse". This avoids a
        # bogus "0.00 verbosity" flag for deterministic agents (e.g. a risk
        # step with no LLM call) or agents simply not used in this run.
        if agent not in cur_ratios or (cur_inputs.get(agent, 0) or 0) <= 0:
            continue
        # Range-based: only flag if this run falls outside the agent's normal spread
        rng = _normal_range(base.get("_ratio_series", []))
        if not _outside_range(cur_ratio, rng):
            continue
        change = _pct_change(cur_ratio, base_ratio)
        more = change > 0
        rng_str = f"{rng[0]:.2f}–{rng[1]:.2f}" if rng else "n/a"
        out.append(Insight(
            "warning", "verbosity",
            f"{agent} unusually {'verbose' if more else 'terse'} this run",
            f"Output-per-input this run {cur_ratio:.2f} · normal range {rng_str}",
            "Single-run anomaly. Becomes drift only if it persists across runs.",
            agent=agent, kind="anomaly",
        ))
    return out


def anomaly_insights(m: dict, baseline_list: list[dict]) -> list[Insight]:
    """Single-run anomalies vs the baseline average — variance, not drift."""
    agent_base = _baseline_agent_stats(baseline_list)
    run_base   = _baseline_run_stats(baseline_list)
    out: list[Insight] = []
    out += _drift_tools(m, agent_base, baseline_list)
    out += _drift_routing(m, run_base)
    out += _drift_verbosity(m, agent_base)
    return out




# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════
def build_insights(
    current_metrics: dict,
    baseline_metrics_list: list[dict] | None = None,
) -> list[Insight]:
    """Within-run + single-run anomaly insights (no version drift here)."""
    insights = single_run_insights(current_metrics)
    if baseline_metrics_list:
        insights += anomaly_insights(current_metrics, baseline_metrics_list)
    insights.sort(key=_rank_key)
    return insights


def severity_counts(insights: list[Insight]) -> dict[str, int]:
    counts = {"critical": 0, "warning": 0, "improvement": 0, "info": 0}
    for i in insights:
        counts[i.severity] = counts.get(i.severity, 0) + 1
    return counts


def rank_insights(insights: list[Insight]) -> list[Insight]:
    """Sort a combined list: risk → performance bottleneck → positive → info."""
    return sorted(insights, key=_rank_key)
