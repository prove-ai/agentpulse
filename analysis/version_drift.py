"""Layer 5 — version-level aggregate drift.

Compares a whole prompt VERSION's behaviour against the baseline version,
not a single run. This is the real drift signal: one run can be noisy, but a
shift that holds across many runs means the prompt change actually changed
behaviour.

Cohort validity (the "compare the right batch" problem):
  - Always scoped to one task_type. v1 csv-analysis vs v2 csv-analysis only.
  - Auto-pairing: if the SAME question text appears in both versions, the
    comparison restricts to those shared questions ("paired" mode) and can
    compare absolute signals (tokens, cost) safely because the input is held
    constant.
  - Otherwise it falls back to the task_type average ("approximate" mode) and
    compares ONLY normalized signals (ratios, rates, route, turns) — never raw
    tokens/cost, which depend on question size.
  - Requires MIN_RUNS per cohort before reporting anything.

Output insights reuse the Layer 4 Insight type and feed the same top card,
tagged kind="version".
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from analysis.run_anomaly import DEFAULTS
from analysis.run_insights import Insight

MIN_RUNS = 3

# A drift is "confirmed" only when at least this fraction of target-cohort
# runs shift in the same direction as the cohort average. Otherwise the shift
# is downgraded to "potential drift" (likely noise / one outlier).
CONSISTENCY_THRESHOLD = 0.70


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def _pct(cur, base):
    return (cur - base) / base * 100 if base else 0.0


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------
@dataclass
class VVerdict:
    scope:     str       # "overall" or an agent name
    signal:    str
    base:      float
    target:    float
    delta_pct: float
    drifted:   bool
    label:     str
    absolute:  bool = False   # True if this is an absolute (paired-only) signal
    # Fraction of target-cohort runs that shifted in the same direction as the
    # cohort average. 1.0 = unanimous, 0.5 = split.
    consistency: float = 1.0
    # Status: "drift" (confirmed), "potential" (shift but not consistent), or "" (no shift)
    status:      str   = ""


@dataclass
class VersionDriftReport:
    task_type:        str
    base_version:     int
    target_version:   int
    base_runs:        int
    target_runs:      int
    mode:             str          # "paired" | "approximate" | "insufficient"
    confidence:       str          # "confident" | "low" | "insufficient"
    paired_questions: int = 0
    overall:          list = field(default_factory=list)
    per_agent:        dict = field(default_factory=dict)
    note:             str = ""

    @property
    def available(self) -> bool:
        return self.mode != "insufficient"


# ---------------------------------------------------------------------------
# Cohort aggregation
# ---------------------------------------------------------------------------
def _route_mode(mlist):
    seqs = [tuple(m.get("agent_sequence") or []) for m in mlist]
    return list(max(set(seqs), key=seqs.count)) if seqs else []


def _agg_overall(mlist) -> dict:
    route = _route_mode(mlist)
    stable = sum(1 for m in mlist if (m.get("agent_sequence") or []) == route) / len(mlist) if mlist else 0
    return {
        "avg_turns":     _mean([m.get("total_turns", 0) for m in mlist]),
        "avg_cost":      _mean([m.get("total_cost_usd", 0.0) for m in mlist]),
        "clean_rate":    _mean([1.0 if m.get("terminated_cleanly") else 0.0 for m in mlist]),
        "route_mode":    route,
        "route_stability": stable,
        # Raw per-run series — for directional consistency checks
        "_turns_series":  [m.get("total_turns", 0) for m in mlist],
        "_cost_series":   [m.get("total_cost_usd", 0.0) for m in mlist],
    }


def _directional_consistency(target_series: list[float], base_avg: float) -> float:
    """Fraction of target runs that shift in the same direction as the cohort avg shift.
    Returns 1.0 if the series is all on the same side of base_avg as the mean,
    0.5 if split half-and-half, etc.
    """
    if not target_series:
        return 0.0
    target_avg = _mean(target_series)
    if target_avg == base_avg:
        return 1.0
    direction = 1 if target_avg > base_avg else -1
    same_dir = sum(
        1 for x in target_series
        if (x > base_avg and direction == 1) or (x < base_avg and direction == -1)
    )
    return same_dir / len(target_series)


def _classify(drifted: bool, consistency: float) -> str:
    """Decide drift status given the raw 'drifted' flag and directional consistency."""
    if not drifted:
        return ""
    return "drift" if consistency >= CONSISTENCY_THRESHOLD else "potential"


def _agg_per_agent(mlist) -> dict:
    data = defaultdict(lambda: defaultdict(list))
    for m in mlist:
        for a in (m.get("unique_agents") or []):
            data[a]["input"].append((m.get("per_agent_input_tokens") or {}).get(a, 0))
            data[a]["output"].append((m.get("per_agent_output_tokens") or {}).get(a, 0))
            data[a]["ratio"].append((m.get("per_agent_output_input_ratio") or {}).get(a, 0.0))
            data[a]["tool"].append((m.get("tool_usage_rate_per_agent") or {}).get(a, 0.0))
            data[a]["lat"].append((m.get("per_agent_p50_ms") or {}).get(a, 0.0))
    out = {}
    for a, d in data.items():
        out[a] = {
            "input":  _mean(d["input"]),
            "output": _mean(d["output"]),
            "ratio":  _mean(d["ratio"]),
            "tool":   _mean(d["tool"]),
            "lat":    _mean(d["lat"]),
            "n":      len(d["input"]),
            # Raw series — for consistency checks
            "_series": {k: list(v) for k, v in d.items()},
        }
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_versions(
    base_list: list[dict],
    target_list: list[dict],
    task_type: str,
    base_version: int,
    target_version: int,
) -> VersionDriftReport:
    if len(base_list) < MIN_RUNS or len(target_list) < MIN_RUNS:
        return VersionDriftReport(
            task_type, base_version, target_version,
            len(base_list), len(target_list),
            mode="insufficient", confidence="insufficient",
            note=f"Need ≥{MIN_RUNS} runs per version (have {len(base_list)} v{base_version}, "
                 f"{len(target_list)} v{target_version}).",
        )

    # Auto-pairing by identical question text
    base_q   = {m.get("task_text") for m in base_list if m.get("task_text")}
    target_q = {m.get("task_text") for m in target_list if m.get("task_text")}
    common   = base_q & target_q

    if common:
        mode = "paired"
        base_used   = [m for m in base_list   if m.get("task_text") in common]
        target_used = [m for m in target_list if m.get("task_text") in common]
        allow_absolute = True
    else:
        mode = "approximate"
        base_used, target_used = base_list, target_list
        allow_absolute = False

    n_min = min(len(base_used), len(target_used))
    confidence = "confident" if n_min >= 5 else "low"

    b_overall = _agg_overall(base_used)
    t_overall = _agg_overall(target_used)
    b_agents  = _agg_per_agent(base_used)
    t_agents  = _agg_per_agent(target_used)

    overall   = _overall_verdicts(b_overall, t_overall, allow_absolute)
    per_agent = _agent_verdicts(b_agents, t_agents, allow_absolute)

    return VersionDriftReport(
        task_type, base_version, target_version,
        len(base_list), len(target_list),
        mode=mode, confidence=confidence,
        paired_questions=len(common),
        overall=overall, per_agent=per_agent,
    )


def _mk(scope, signal, base, target, delta_pct, drifted, label,
        consistency, absolute=False) -> VVerdict:
    return VVerdict(
        scope=scope, signal=signal, base=base, target=target,
        delta_pct=delta_pct, drifted=drifted, label=label,
        absolute=absolute, consistency=round(consistency, 2),
        status=_classify(drifted, consistency),
    )


def _overall_verdicts(b, t, allow_absolute) -> list[VVerdict]:
    out = []

    # turns
    delta = t["avg_turns"] - b["avg_turns"]
    cons  = _directional_consistency(t.get("_turns_series", []), b["avg_turns"])
    out.append(_mk("overall", "avg turns",
                   round(b["avg_turns"], 1), round(t["avg_turns"], 1),
                   round(delta, 1),
                   abs(delta) > DEFAULTS["turn_count_delta"],
                   "TURN_DRIFT", cons))

    # route (categorical — consistency = % of target runs matching the route mode)
    route_changed = b["route_mode"] != t["route_mode"]
    out.append(_mk("overall", "typical route",
                   b["route_mode"], t["route_mode"], 0.0,
                   route_changed, "ROUTE_DRIFT",
                   consistency=t.get("route_stability", 1.0)))

    # clean rate
    cr_delta = (t["clean_rate"] - b["clean_rate"]) * 100
    out.append(_mk("overall", "clean-finish rate",
                   round(b["clean_rate"]*100), round(t["clean_rate"]*100),
                   round(cr_delta, 0), cr_delta < -20,
                   "RELIABILITY_DRIFT", consistency=1.0))

    # cost (absolute — paired only)
    if allow_absolute:
        c_pct = _pct(t["avg_cost"], b["avg_cost"])
        cons  = _directional_consistency(t.get("_cost_series", []), b["avg_cost"])
        out.append(_mk("overall", "avg cost",
                       round(b["avg_cost"], 4), round(t["avg_cost"], 4),
                       round(c_pct, 1),
                       abs(c_pct) > DEFAULTS["cost_pct"]*100,
                       "COST_DRIFT", cons, absolute=True))
    return out


def _agent_verdicts(b_agents, t_agents, allow_absolute) -> dict:
    out = {}
    for agent in t_agents:
        b = b_agents.get(agent)
        if not b:
            continue
        t = t_agents[agent]
        t_series = t.get("_series", {})
        verdicts = []

        # Verbosity ratio
        r_pct = _pct(t["ratio"], b["ratio"])
        cons  = _directional_consistency(t_series.get("ratio", []), b["ratio"])
        verdicts.append(_mk(agent, "output/input ratio",
                            round(b["ratio"], 2), round(t["ratio"], 2),
                            round(r_pct, 1),
                            abs(r_pct) > DEFAULTS["ratio_pct"]*100,
                            "VERBOSITY_DRIFT", cons))

        # Tool usage rate
        tr_delta = t["tool"] - b["tool"]
        cons = _directional_consistency(t_series.get("tool", []), b["tool"])
        verdicts.append(_mk(agent, "tool-usage rate",
                            round(b["tool"]*100), round(t["tool"]*100),
                            round(tr_delta*100, 1),
                            abs(tr_delta) > DEFAULTS["tool_usage_rate_delta"],
                            "TOOL_USAGE_DRIFT", cons))

        # Latency
        l_pct = _pct(t["lat"], b["lat"])
        cons  = _directional_consistency(t_series.get("lat", []), b["lat"])
        verdicts.append(_mk(agent, "latency",
                            round(b["lat"]), round(t["lat"]),
                            round(l_pct, 1),
                            abs(l_pct) > DEFAULTS["latency_pct"]*100,
                            "LATENCY_DRIFT", cons))

        # Tokens (paired only)
        if allow_absolute:
            o_pct = _pct(t["output"], b["output"])
            cons  = _directional_consistency(t_series.get("output", []), b["output"])
            verdicts.append(_mk(agent, "output tokens",
                                round(b["output"]), round(t["output"]),
                                round(o_pct, 1),
                                abs(o_pct) > DEFAULTS["token_pct"]*100,
                                "OUTPUT_TOKEN_DRIFT", cons, absolute=True))

            i_pct = _pct(t["input"], b["input"])
            cons  = _directional_consistency(t_series.get("input", []), b["input"])
            verdicts.append(_mk(agent, "input tokens",
                                round(b["input"]), round(t["input"]),
                                round(i_pct, 1),
                                abs(i_pct) > DEFAULTS["token_pct"]*100,
                                "INPUT_TOKEN_DRIFT", cons, absolute=True))
        out[agent] = verdicts
    return out


# ---------------------------------------------------------------------------
# Insights from a version-drift report (feed the top card)
# ---------------------------------------------------------------------------
def _drift_or_potential(v: VVerdict) -> tuple[str, str]:
    """Return (status_label, severity_modifier) based on directional consistency."""
    if v.status == "drift":
        return ("Drift", "")
    return ("Potential drift", " ⚠ inconsistent direction — may be noise")


def version_insights(r: VersionDriftReport) -> list[Insight]:
    if not r.available:
        return []

    suffix = (f"across {r.target_runs} v{r.target_version} vs {r.base_runs} v{r.base_version} runs"
              f"{' · paired on ' + str(r.paired_questions) + ' question(s)' if r.mode == 'paired' else ' · task-type avg'}")
    conf = "" if r.confidence == "confident" else " (low confidence — few runs)"
    out: list[Insight] = []

    def _add(severity, category, title, evidence, interp, agent=None, status="drift"):
        prefix = "Drift: " if status == "drift" else "Potential drift: "
        # Demote severity one notch for "potential" (not yet confirmed)
        if status == "potential":
            severity = {"critical": "warning", "warning": "info"}.get(severity, severity)
        out.append(Insight(severity, category, prefix + title, evidence, interp,
                           agent=agent, kind="version"))

    # ----- Per-agent -----
    for agent, verdicts in r.per_agent.items():
        for v in verdicts:
            if not v.drifted or not v.status:
                continue
            cons_note = f" · {v.consistency*100:.0f}% of runs in same direction"
            if v.signal == "tool-usage rate" and v.target < 10 and v.base >= 50:
                _add("critical", "tools",
                     f"{agent} stopped using tools",
                     f"Tool usage {v.base:.0f}% → {v.target:.0f}% {suffix}{cons_note}{conf}",
                     "Consistent across runs — likely producing results without computing them."
                     if v.status == "drift" else
                     "Some runs dropped tool use but not all — watch the next few runs.",
                     agent=agent, status=v.status)
            elif v.signal == "output/input ratio":
                more = v.delta_pct > 0
                _add("warning", "verbosity",
                     f"{agent} is {'more' if more else 'less'} verbose",
                     f"Output/input {v.base:.2f} → {v.target:.2f} ({v.delta_pct:+.0f}%) {suffix}{cons_note}{conf}",
                     "Sustained shift in how much it writes per token read."
                     if v.status == "drift" else
                     "Average shifted but runs disagree — may be one outlier.",
                     agent=agent, status=v.status)
            elif v.signal == "output tokens":
                _add("warning", "tokens",
                     f"{agent} writes {'more' if v.delta_pct>0 else 'less'}",
                     f"Avg output {v.base:.0f} → {v.target:.0f} tok ({v.delta_pct:+.0f}%) {suffix}{cons_note}{conf}",
                     "Sustained change in output volume."
                     if v.status == "drift" else
                     "Average shifted but runs disagree — may be one outlier.",
                     agent=agent, status=v.status)

    # ----- Overall -----
    for v in r.overall:
        if not v.drifted or not v.status:
            continue
        cons_note = f" · {v.consistency*100:.0f}% of runs match"
        if v.signal == "typical route":
            _add("warning", "routing", "Typical route changed",
                 f"{' → '.join(v.base)}  ⟶  {' → '.join(v.target)} {suffix}{cons_note}{conf}",
                 "Most v2 runs follow a different path now."
                 if v.status == "drift" else
                 "Route differs from v1 but isn't stable within v2 yet.",
                 status=v.status)
        elif v.signal == "avg turns":
            _add("warning", "routing",
                 f"Runs take {'more' if v.delta_pct>0 else 'fewer'} turns",
                 f"Avg turns {v.base} → {v.target} ({v.delta_pct:+.1f}) {suffix}{cons_note}{conf}",
                 "Sustained change in back-and-forth count."
                 if v.status == "drift" else
                 "Average shifted but runs disagree.",
                 status=v.status)
        elif v.signal == "clean-finish rate":
            _add("critical", "reliability", "More runs failing to finish cleanly",
                 f"Clean-finish rate {v.base:.0f}% → {v.target:.0f}% {suffix}{conf}",
                 "Fewer runs reach an approved result on this version.",
                 status=v.status)
        elif v.signal == "avg cost":
            _add("warning", "cost",
                 f"Cost {'up' if v.delta_pct>0 else 'down'} {abs(v.delta_pct):.0f}%",
                 f"Avg cost ${v.base:.4f} → ${v.target:.4f} {suffix}{cons_note}{conf}",
                 "Sustained cost change on the same questions."
                 if v.status == "drift" else
                 "Average cost shifted but per-run cost varies — may be one outlier.",
                 status=v.status)

    return out
