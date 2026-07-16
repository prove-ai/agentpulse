"""Drift causal layer — link drifting signals into root-cause → symptom chains.

A single drifting metric is naive. Real drift *propagates*: an upstream BEHAVIOUR
change (often benign-looking on its own — e.g. an agent's output tokens drop)
starves a downstream agent and causes an IMPACT (success falls, retries rise).
The naive view reports two unconnected findings; this layer connects them.

Pipeline:
  1. enrich_drift_signals  — tag each drifting (entity, metric) with direction
                             (up/down) and kind (behaviour | impact).
  2. build_topology        — agent reachability from handoff edges (who feeds whom).
  3. detect_chains         — for each downstream IMPACT, walk UPSTREAM along the
                             topology to a co-timed BEHAVIOUR change = the root;
                             attach the config change that likely triggered it.

Everything is correlational and labelled "likely" — co-timing + topology + a
coinciding config change raise confidence, never proof.
"""
from __future__ import annotations

# Metrics that represent a bad OUTCOME (the symptom). Everything else — tokens,
# latency, payload, context_ratio, tool_calls, … — is BEHAVIOUR (a change in how
# an agent works, which may look harmless in isolation but cause downstream harm).
_IMPACT_METRICS = {"success", "errors", "retries", "retry_rate", "error_rate"}


def _kind(metric: str) -> str:
    return "impact" if metric in _IMPACT_METRICS else "behaviour"


def enrich_drift_signals(drifting: list[dict], series: dict, baseline_runs: int) -> list[dict]:
    """Add `direction` (up/down/flat) and `kind` (behaviour/impact) to each
    drifting signal. `drifting` items are {category, entity, metric, drift_start}."""
    from analysis.metric_series import metric_impact
    out = []
    for d in drifting:
        pts = series.get(d["category"], {}).get(d["entity"], {}).get(d["metric"])
        if not pts:
            continue
        imp = metric_impact(pts, baseline_runs)
        pct = imp.get("pct")
        direction = "flat" if not pct else ("up" if pct > 0 else "down")
        out.append({**d, "direction": direction, "pct": pct,
                    "kind": _kind(d["metric"]),
                    "label": f'{d["entity"]} {d["metric"]} {imp["label"]}'})
    return out


def build_topology(edges: list[tuple[str, str]]) -> dict:
    """From handoff edges (from, to), return reachability maps:
       {"down": {a: set(all agents reachable downstream)},
        "up":   {a: set(all agents that can reach a)},
        "succ": {a: set(direct successors)}}."""
    succ: dict = {}
    nodes = set()
    for a, b in edges:
        succ.setdefault(a, set()).add(b)
        nodes.update((a, b))

    def _reach(start: str) -> set:
        seen, stack = set(), list(succ.get(start, set()))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(succ.get(n, set()))
        return seen

    down = {n: _reach(n) for n in nodes}
    up = {n: {m for m in nodes if n in down.get(m, set())} for n in nodes}
    return {"down": down, "up": up, "succ": succ}


def edges_from_handoff_series(series: dict) -> list[tuple[str, str]]:
    """Recover (from, to) edges from the handoff series keys ("A → B")."""
    edges = []
    for key in series.get("handoffs", {}):
        if " → " in key:
            a, b = key.split(" → ", 1)
            edges.append((a.strip(), b.strip()))
    return edges


def _trigger_for(agent: str, drift_start: int, change_log: list[dict], window: int) -> dict | None:
    """The config change on `agent` (or workflow) nearest to the root's drift_start."""
    cands = [e for e in change_log
             if e.get("scope") in (agent, "workflow")
             and abs((e.get("run_index") or 0) - drift_start) <= window]
    if not cands:
        return None
    return min(cands, key=lambda e: abs((e.get("run_index") or 0) - drift_start))


def detect_chains(signals: list[dict], topology: dict, change_log: list[dict],
                  cotiming: int = 5, trigger_window: int = 5) -> list[dict]:
    """Group enriched drift signals into root → symptom chains.

    For each agent with an IMPACT drift (the symptom), find agents UPSTREAM of it
    whose BEHAVIOUR drifted within `cotiming` runs (the candidate causes); the
    most-upstream candidate is the root. Attach the nearest config change as the
    likely trigger. Returns chains sorted most-confident first.
    """
    up = topology.get("up", {})
    by_agent: dict = {}
    for s in signals:
        by_agent.setdefault(s["entity"], []).append(s)

    impacts = [s for s in signals if s["kind"] == "impact"]
    chains = []
    for sym in impacts:
        sym_agent, t0 = sym["entity"], sym["drift_start"]
        upstream = up.get(sym_agent, set())
        # candidate causes: upstream agents with a co-timed behaviour drift
        causes = [s for s in signals if s["kind"] == "behaviour"
                  and s["entity"] in upstream
                  and abs(s["drift_start"] - t0) <= cotiming]
        if not causes:
            continue
        # root = the most-upstream cause (not downstream of any other cause)
        cause_agents = {c["entity"] for c in causes}
        roots = [c for c in causes
                 if not (up.get(c["entity"], set()) & cause_agents)]
        root = min(roots or causes, key=lambda c: c["drift_start"])
        root_agent = root["entity"]
        trigger = _trigger_for(root_agent, root["drift_start"], change_log, trigger_window)

        confidence = "high" if trigger else "medium"
        root_sigs = [c for c in causes if c["entity"] == root_agent]
        summary = (f"{root_agent} {_join(root_sigs)} → {sym_agent} {sym['metric']} "
                   f"{sym['direction']}"
                   + (f"; likely triggered by {root_agent} {trigger['dimension']} change"
                      if trigger else ""))
        chains.append({
            "root_agent": root_agent,
            "symptom_agent": sym_agent,
            "root_signals": root_sigs,
            "symptom_signal": sym,
            "trigger": trigger,
            "drift_start": root["drift_start"],
            "confidence": confidence,
            "summary": summary,
        })

    # de-dupe identical (root, symptom) pairs; most confident + earliest first
    seen, uniq = set(), []
    chains.sort(key=lambda c: (0 if c["confidence"] == "high" else 1, c["drift_start"]))
    for c in chains:
        key = (c["root_agent"], c["symptom_agent"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _join(sigs: list[dict]) -> str:
    return ", ".join(f'{s["metric"]} {s["direction"]}' for s in sigs)
