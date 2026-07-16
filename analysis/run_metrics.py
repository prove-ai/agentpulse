"""Layer 2 — derived metrics.

Computes all signals from raw spans, tool_calls, and handoffs.
Pure functions — no SQLite writes, no side effects.

Categories:
  A. Structural / Topology   (turns, agents, loops, re-entries)
  B. Tool signals             (distribution, counts, rates)
  C. Token signals            (shape, magnitude, distribution, context growth)
  D. Performance / Temporal  (latency, wall clock)
  E. Reliability / Errors    (error spans, tool failures, termination)
  F. Cost                     (per agent, per model, concentration)
  G. Handoff signals          (context ratio, A→B token flow)
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from typing import Any

from storage.sqlite_store import compute_cost


# ---------------------------------------------------------------------------
# A. Structural / Topology
# ---------------------------------------------------------------------------
def structural_metrics(agent_spans: list[dict], run: dict) -> dict[str, Any]:
    sequence = json.loads(run.get("agent_sequence") or "[]")
    unique_agents = list(dict.fromkeys(sequence))  # ordered, deduplicated
    counts = Counter(sequence)
    graph_depth = max((s.get("turn_index") or 0 for s in agent_spans), default=0)

    # Loop / re-entry detection
    # A→B→A is a loop. An agent running again is legitimate in many designs, so
    # we only call a loop "unexpected/inefficient" when the system uses a STATUS
    # protocol and B did NOT request the re-entry (no NEEDS_INFO / CHANGES_REQUESTED
    # pointing back to A). Without that signal (e.g. imported runs, or systems
    # that don't emit STATUS) we can't judge intent, so we don't flag it.
    reentry_patterns: list[dict] = []
    sv_by_turn = {s.get("turn_index", -1): (s.get("status_value") or "") for s in agent_spans}
    has_status_protocol = any(sv.strip() for sv in sv_by_turn.values())

    for i in range(2, len(sequence)):
        a = sequence[i - 2]
        b = sequence[i - 1]
        c = sequence[i]
        if c == a and b != a:
            b_turn_idx = i - 1
            b_status = sv_by_turn.get(b_turn_idx, "")
            requested = (
                ("NEEDS_INFO" in b_status.upper() or "CHANGES_REQUESTED" in b_status.upper())
                and a.upper() in b_status.upper()
            )
            reentry_patterns.append({
                "pattern": f"{a}→{b}→{c}",
                "turn":    i,
                # Only an anomaly when we actually have the signal to judge intent.
                "flagged": has_status_protocol and not requested,
            })

    flagged_loops   = sum(1 for r in reentry_patterns if r["flagged"])
    loop_reentry_count = sum(v - 1 for v in counts.values() if v > 1)

    return {
        "total_turns":          len(sequence),
        "unique_agent_count":   len(unique_agents),
        "agent_sequence":       sequence,
        "unique_agents":        unique_agents,
        "graph_depth":          graph_depth,
        "loop_reentry_count":   loop_reentry_count,
        "reentry_patterns":     reentry_patterns,
        "flagged_loops":        flagged_loops,
        "agent_turn_counts":    dict(counts),
    }


# ---------------------------------------------------------------------------
# B. Tool signals
# ---------------------------------------------------------------------------
def tool_metrics(agent_spans: list[dict], tool_calls: list[dict]) -> dict[str, Any]:
    # span_id → agent_name mapping
    span_agent = {s["span_id"]: s.get("agent_name", "unknown") for s in agent_spans}

    # Per-run tool frequency
    tool_freq = Counter(t["tool_name"] for t in tool_calls)

    # Per-agent: tool distribution + call count + usage rate
    dist_per_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    calls_per_agent: dict[str, int] = defaultdict(int)
    turns_with_tools: dict[str, int] = defaultdict(int)

    for tc in tool_calls:
        agent = span_agent.get(tc.get("span_id", ""), "unknown")
        dist_per_agent[agent][tc["tool_name"]] += 1
        calls_per_agent[agent] += 1

    for s in agent_spans:
        agent = s.get("agent_name", "unknown")
        if (s.get("tool_call_count") or 0) > 0:
            turns_with_tools[agent] += 1

    # Tool usage rate per agent (% of their turns where they called ≥1 tool)
    agent_turn_counts = Counter(s.get("agent_name") for s in agent_spans)
    usage_rate_per_agent = {
        agent: turns_with_tools.get(agent, 0) / count
        for agent, count in agent_turn_counts.items()
        if count > 0
    }

    failed = [t for t in tool_calls if not t.get("success")]

    return {
        "unique_tool_count":          len(tool_freq),
        "unique_tools":               list(tool_freq.keys()),
        "tool_frequency":             dict(tool_freq),
        "total_tool_calls":           len(tool_calls),
        "tool_calls_per_agent":       dict(calls_per_agent),
        "tool_distribution_per_agent": {a: dict(d) for a, d in dist_per_agent.items()},
        "tool_usage_rate_per_agent":  {a: round(r, 3) for a, r in usage_rate_per_agent.items()},
        "tool_failure_count":         len(failed),
        "tool_failure_ratio":         round(len(failed) / len(tool_calls), 3) if tool_calls else 0.0,
    }


# ---------------------------------------------------------------------------
# C. Token signals
# ---------------------------------------------------------------------------
def token_metrics(agent_spans: list[dict]) -> dict[str, Any]:
    total_input  = sum(s.get("input_tokens")  or 0 for s in agent_spans)
    total_output = sum(s.get("output_tokens") or 0 for s in agent_spans)
    total_tokens = total_input + total_output

    # Per-agent totals
    per_inp:   dict[str, int]   = defaultdict(int)
    per_out:   dict[str, int]   = defaultdict(int)
    per_total: dict[str, int]   = defaultdict(int)

    for s in agent_spans:
        name = s.get("agent_name") or "unknown"
        inp  = s.get("input_tokens")  or 0
        out  = s.get("output_tokens") or 0
        per_inp[name]   += inp
        per_out[name]   += out
        per_total[name] += inp + out

    # Shape: output/input ratio
    overall_ratio = round(total_output / total_input, 3) if total_input else 0.0
    per_agent_ratio = {
        name: round(per_out[name] / per_inp[name], 3)
        if per_inp[name] else 0.0
        for name in per_inp
    }

    # Distribution: token share and concentration
    per_agent_share = {
        name: round(per_total[name] / total_tokens * 100, 1)
        for name in per_total
    } if total_tokens else {}

    token_concentration = round(
        max(per_total.values(), default=0) / total_tokens, 3
    ) if total_tokens else 0.0

    # Context growth: input token delta turn-over-turn
    sorted_spans = sorted(agent_spans, key=lambda s: s.get("turn_index") or 0)
    input_per_turn = [s.get("input_tokens") or 0 for s in sorted_spans]

    growth_abs_vals, growth_pct_vals = [], []
    for i in range(1, len(input_per_turn)):
        prev, curr = input_per_turn[i - 1], input_per_turn[i]
        growth_abs_vals.append(curr - prev)
        if prev > 0:
            growth_pct_vals.append((curr - prev) / prev * 100)

    context_growth_abs = round(statistics.mean(growth_abs_vals), 1) if growth_abs_vals else 0.0
    context_growth_pct = round(statistics.mean(growth_pct_vals), 1) if growth_pct_vals else 0.0

    return {
        # Magnitude
        "total_input_tokens":    total_input,
        "total_output_tokens":   total_output,
        "total_tokens":          total_tokens,
        "per_agent_input_tokens":  dict(per_inp),
        "per_agent_output_tokens": dict(per_out),
        # Shape
        "overall_output_input_ratio":    overall_ratio,
        "per_agent_output_input_ratio":  per_agent_ratio,
        # Distribution
        "token_concentration":    token_concentration,
        "per_agent_token_share":  per_agent_share,
        # Context growth
        "context_growth_abs":    context_growth_abs,
        "context_growth_pct":    context_growth_pct,
        "input_tokens_per_turn": input_per_turn,
    }


# ---------------------------------------------------------------------------
# D. Performance / Temporal
# ---------------------------------------------------------------------------
def performance_metrics(agent_spans: list[dict], run: dict) -> dict[str, Any]:
    durations = [s["duration_ms"] for s in agent_spans if s.get("duration_ms")]
    total_wall = run.get("total_duration_ms") or (max(durations, default=0))

    p50 = statistics.median(durations) if durations else 0.0
    p95 = (
        sorted(durations)[int(len(durations) * 0.95)]
        if len(durations) >= 2 else (durations[0] if durations else 0.0)
    )

    per_agent_dur: dict[str, list[float]] = defaultdict(list)
    for s in agent_spans:
        if s.get("duration_ms"):
            per_agent_dur[s.get("agent_name") or "unknown"].append(s["duration_ms"])

    per_p50   = {n: round(statistics.median(v), 1) for n, v in per_agent_dur.items()}
    per_total = {n: round(sum(v), 1) for n, v in per_agent_dur.items()}

    sum_spans = sum(durations)
    parallelism = round(sum_spans / total_wall, 2) if total_wall else 1.0

    # Detect parallel groups: clusters of spans whose time intervals overlap.
    # Group = a maximal set where every span overlaps with at least one other
    # in the same cluster (transitive overlap via sweep-line on start_time).
    groups = _detect_parallel_groups(agent_spans)
    parallel_groups = [g for g in groups if len(g) > 1]
    max_concurrency = max((len(g) for g in groups), default=1) if groups else 1

    return {
        "total_wall_clock_ms":   round(total_wall, 1),
        "sum_of_span_durations": round(sum_spans, 1),
        "parallelism_factor":    parallelism,
        "latency_p50_ms":        round(p50, 1),
        "latency_p95_ms":        round(p95, 1),
        "per_agent_p50_ms":      per_p50,
        "per_agent_total_ms":    per_total,
        # Parallel-group detection
        "parallel_group_count":  len(parallel_groups),
        "parallel_groups":       parallel_groups,   # list of [{agent, turn_index, start_ms, end_ms}, ...]
        "max_concurrency":       max_concurrency,
        "ran_in_parallel":       len(parallel_groups) > 0,
    }


def _detect_parallel_groups(agent_spans: list[dict]) -> list[list[dict]]:
    """Cluster spans by time-interval overlap (sweep-line).

    Two spans A, B are 'parallel' if their [start, end] intervals overlap.
    Groups are connected components of that overlap relation.

    Each member is a small dict with the fields the UI needs.
    Singleton groups (a span overlapping nothing) are also returned.
    """
    spans = [
        {
            "agent":      s.get("agent_name") or "unknown",
            "turn_index": s.get("turn_index"),
            "start_ms":   s.get("start_time_ms") or 0,
            "end_ms":     s.get("end_time_ms")   or 0,
        }
        for s in agent_spans
        if s.get("start_time_ms") is not None and s.get("end_time_ms") is not None
    ]
    if not spans:
        return []

    # Sort by start time
    spans.sort(key=lambda x: x["start_ms"])

    groups: list[list[dict]] = []
    current: list[dict] = [spans[0]]
    current_end = spans[0]["end_ms"]

    for s in spans[1:]:
        # If this span starts before the current group's latest end, it overlaps -> same group
        if s["start_ms"] < current_end:
            current.append(s)
            current_end = max(current_end, s["end_ms"])
        else:
            groups.append(current)
            current = [s]
            current_end = s["end_ms"]
    groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# E. Reliability / Errors
# ---------------------------------------------------------------------------
# Strings that mark a clean finish, regardless of which framework produced them.
# digital-company uses "APPROVED"; LangChain/CrewAI typically use "completed";
# the financial-multiagent system uses "ok" / no reason at all when nothing
# went wrong. The check is case-insensitive substring match.
_CLEAN_TERMINATION_MARKERS = (
    "approved", "completed", "complete", "success", "ok", "done", "finished",
    "chain_complete", "agent_finish",
)
# Strings that explicitly mark a failed/aborted finish.
_FAILED_TERMINATION_MARKERS = (
    "error", "stuck_loop", "max_rounds", "missing_input", "stuck", "timeout",
    "cancelled", "aborted", "failed",
)


def is_clean_termination(reason: str) -> bool:
    """True when this run ended normally. Empty/None counts as clean unless
    a failure marker is present — many frameworks just don't set a reason on
    success."""
    if not reason:
        return True
    low = reason.lower()
    if any(m in low for m in _FAILED_TERMINATION_MARKERS):
        return False
    if any(m in low for m in _CLEAN_TERMINATION_MARKERS):
        return True
    # Unknown reason text → treat as clean (no negative signal).
    return True


def reliability_metrics(agent_spans: list[dict], run: dict) -> dict[str, Any]:
    errors  = [s for s in agent_spans if (s.get("status") or "").upper() == "ERROR"]
    reason  = run.get("termination_reason") or ""
    clean   = is_clean_termination(reason)

    return {
        "error_span_count":   len(errors),
        "error_ratio":        round(len(errors) / len(agent_spans), 3) if agent_spans else 0.0,
        "termination_reason": reason or "completed",
        "terminated_cleanly": clean,
    }


# ---------------------------------------------------------------------------
# F. Cost
# ---------------------------------------------------------------------------
def cost_metrics(agent_spans: list[dict], run: dict) -> dict[str, Any]:
    total_cost = 0.0
    per_agent: dict[str, float] = defaultdict(float)
    per_agent_model: dict[str, str] = {}

    for s in agent_spans:
        name  = s.get("agent_name") or "unknown"
        model = s.get("model") or run.get("model") or ""
        inp   = s.get("input_tokens")  or 0
        out   = s.get("output_tokens") or 0
        cost  = compute_cost(inp, out, model)
        total_cost         += cost
        per_agent[name]    += cost
        per_agent_model[name] = model  # last seen model for this agent

    concentration = round(
        max(per_agent.values(), default=0.0) / total_cost, 3
    ) if total_cost else 0.0

    return {
        "total_cost_usd":     round(total_cost, 6),
        "per_agent_cost_usd": {k: round(v, 6) for k, v in per_agent.items()},
        "cost_concentration": concentration,
        "per_agent_model":    per_agent_model,
    }


# ---------------------------------------------------------------------------
# G. Handoff signals
# ---------------------------------------------------------------------------
def handoff_metrics(handoffs: list[dict]) -> dict[str, Any]:
    if not handoffs:
        return {
            "handoff_count":             0,
            "handoffs":                  [],
            "avg_handoff_context_ratio": 0.0,
            "flagged_handoffs":          0,
        }

    ratios = [h["context_ratio"] for h in handoffs if h.get("context_ratio", 0) > 0]
    avg_ratio = round(statistics.mean(ratios), 3) if ratios else 0.0
    flagged   = sum(1 for h in handoffs if not h.get("was_requested"))

    return {
        "handoff_count":             len(handoffs),
        "handoffs":                  handoffs,
        "avg_handoff_context_ratio": avg_ratio,
        "flagged_handoffs":          flagged,
    }


# ---------------------------------------------------------------------------
# Master: compute all metrics for a single run
# ---------------------------------------------------------------------------
def compute_all(
    run: dict,
    agent_spans: list[dict],
    tool_calls: list[dict],
    handoffs: list[dict] | None = None,
) -> dict[str, Any]:
    return {
        "run_id":         run["run_id"],
        "task_text":      run.get("task_text"),
        "task_type":      run.get("task_type"),
        "prompt_version": run.get("prompt_version"),
        **structural_metrics(agent_spans, run),
        **tool_metrics(agent_spans, tool_calls),
        **token_metrics(agent_spans),
        **performance_metrics(agent_spans, run),
        **reliability_metrics(agent_spans, run),
        **cost_metrics(agent_spans, run),
        **handoff_metrics(handoffs or []),
    }
