"""Change tracking — separate from versioning.

Every run carries a captured config snapshot (config_json: per-agent model /
params / tools / system-message hash, plus a workflow hash). By diffing
consecutive runs we derive a *change log* — when each dimension last changed.

For drift, we surface the changes that happened shortly before the drift window
as **potentially related** (temporal correlation only — never a confirmed cause),
and we also report which dimensions did NOT change, so they can be ruled out.

A recorded change is NOT a version. Versions are explicit user snapshots
(see the versions table / dashboard button).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from analysis.layer1_raw import get_agent_spans

# Per-agent dimensions we track (workflow is handled separately, run-level).
_AGENT_DIMS = ["prompt", "model", "tools", "params"]
ALL_DIMS = ["prompt", "model", "tools", "params", "workflow"]


def _parse_cfg(run: dict):
    raw = run.get("config_json")
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except Exception:
        return None
    return d if isinstance(d, dict) and d.get("agents") is not None else (d if isinstance(d, dict) else None)


def _agent_dim_value(agent_cfg: dict, dim: str):
    if dim == "prompt": return agent_cfg.get("prompt_hash")
    if dim == "model":  return agent_cfg.get("model")
    if dim == "tools":  return tuple(agent_cfg.get("tools") or [])
    if dim == "params": return json.dumps(agent_cfg.get("params") or {}, sort_keys=True)
    return None


def build_change_log(runs: list[dict]) -> list[dict]:
    """`runs` sorted ascending by timestamp. Returns change events:
    {run_index, timestamp, scope (agent name or 'workflow'), dimension, old, new}.
    """
    events: list[dict] = []
    prev_cfg = None
    for idx, r in enumerate(runs):
        cfg = _parse_cfg(r)
        if cfg is None:
            continue
        if prev_cfg is not None:
            # NOTE: we intentionally do NOT diff `workflow_hash` here. It's captured
            # from the SET OF AGENTS a run actually executed (sqlite_store), so in any
            # conditional/router workflow it varies per run by routing — not by any
            # definition change — and would emit a false "workflow changed" almost
            # every run. Real route shifts are handled by Path drift (route_conformance).
            cur_a, prev_a = cfg.get("agents") or {}, prev_cfg.get("agents") or {}
            for agent in sorted(set(cur_a) & set(prev_a)):
                for dim in _AGENT_DIMS:
                    cv = _agent_dim_value(cur_a[agent], dim)
                    pv = _agent_dim_value(prev_a[agent], dim)
                    if cv != pv and (cv or pv):
                        events.append({
                            "run_index": idx, "timestamp": r.get("timestamp", ""),
                            "scope": agent, "dimension": dim, "old": pv, "new": cv,
                        })
        prev_cfg = cfg
    return events


def diff_configs(prev_cfg: dict, cur_cfg: dict) -> list[dict]:
    """Diff two config snapshots → [{scope, dimension, old, new}]. Used for the
    horizontal version comparison (what changed from one version to the next)."""
    prev_cfg, cur_cfg = prev_cfg or {}, cur_cfg or {}
    out = []
    # (workflow_hash intentionally not diffed — see build_change_log note; it tracked
    # the per-run executed agent set, not a real workflow-definition change.)
    cur_a, prev_a = cur_cfg.get("agents") or {}, prev_cfg.get("agents") or {}
    for agent in sorted(set(cur_a) | set(prev_a)):
        for dim in _AGENT_DIMS:
            cv = _agent_dim_value(cur_a.get(agent, {}), dim)
            pv = _agent_dim_value(prev_a.get(agent, {}), dim)
            if cv != pv and (cv or pv):
                out.append({"scope": agent, "dimension": dim,
                            "old": list(pv) if isinstance(pv, tuple) else pv,
                            "new": list(cv) if isinstance(cv, tuple) else cv})
    return out


def _parse_ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def potentially_related_changes(
    runs: list[dict],
    anchor_index: int,
    anchor_ts: str,
    *,
    lookback_runs: int = 20,
    lookback_hours: int = 24,
) -> dict:
    """Changes within `lookback_runs` runs OR `lookback_hours` hours before the
    anchor (start of the recent half). Returns the dimensions that changed
    (with which agents / details) and the ones that stayed the same.

    Correlation only — the caller must present these as *potentially related*,
    never as a confirmed cause.
    """
    events = build_change_log(runs)
    has_config = any(_parse_cfg(r) is not None for r in runs)
    a_ts = _parse_ts(anchor_ts)

    related: list[dict] = []
    for e in events:
        if e["run_index"] > anchor_index:
            continue
        by_runs = (anchor_index - e["run_index"]) <= lookback_runs
        e_ts = _parse_ts(e["timestamp"])
        by_time = (a_ts is not None and e_ts is not None
                   and timedelta(0) <= (a_ts - e_ts) <= timedelta(hours=lookback_hours))
        if by_runs or by_time:
            ev = dict(e)
            ev["runs_before"] = anchor_index - e["run_index"]
            related.append(ev)

    grouped: dict[str, dict] = {}
    for e in related:
        g = grouped.setdefault(e["dimension"], {"dimension": e["dimension"], "scopes": set(), "events": []})
        g["scopes"].add(e["scope"])
        g["events"].append(e)

    changed = []
    for d in ALL_DIMS:
        if d in grouped:
            g = grouped[d]
            g["scopes"] = sorted(s for s in g["scopes"] if s != "workflow")
            changed.append(g)
    unchanged = [d for d in ALL_DIMS if d not in grouped]

    return {
        "has_config":     has_config,
        "anchor_index":   anchor_index,
        "lookback_runs":  lookback_runs,
        "lookback_hours": lookback_hours,
        "changed":        changed,
        "unchanged":      unchanged,
    }


# ---------------------------------------------------------------------------
# Agent-chain (path) change tracking
# ---------------------------------------------------------------------------
# A run's *path* is the shape of its agent chain. We describe that shape as a
# set of directed edges between agents and diff consecutive runs to find when
# the chain was restructured (agents added/removed, or the wiring changed).
#
# The signature is DAG-based: when a run carries DAG fields
# (parent_step_id / branch_id / join_step_id) the edges come straight from the
# DAG — so fan-outs and joins are captured, not just the linear order. Runs
# with no DAG fields (sequential systems) fall back to consecutive turn-order
# edges, which is the linear chain.

def _sequence(spans: list[dict]) -> list[str]:
    """Agent names in turn order (repeats kept, so re-invocation loops show)."""
    return [s.get("agent_name") for s in
            sorted(spans, key=lambda s: s.get("turn_index") or 0)
            if s.get("agent_name")]


def _dag_edges(spans: list[dict]) -> set[tuple[str, str]]:
    """Directed (from_agent → to_agent) edges describing the run's chain.

    Uses the DAG fields when any span has them: parent_step_id gives parent→child
    edges (so a fan-out parent yields one edge per branch), and join_step_id gives
    branch→join edges (the fan-in). Falls back to linear turn-order edges when the
    run has no DAG information.
    """
    by_id = {s.get("span_id"): s for s in spans}

    def name(sid):
        s = by_id.get(sid)
        return s.get("agent_name") if s else None

    has_dag = any(s.get("parent_step_id") or s.get("branch_id") or s.get("join_step_id")
                  for s in spans)
    edges: set[tuple[str, str]] = set()
    if has_dag:
        for s in spans:
            a, b = name(s.get("parent_step_id")), s.get("agent_name")
            if a and b and a != b:
                edges.add((a, b))
            a, b = s.get("agent_name"), name(s.get("join_step_id"))
            if a and b and a != b:
                edges.add((a, b))
    else:
        seq = _sequence(spans)
        for a, b in zip(seq, seq[1:]):
            if a != b:
                edges.add((a, b))
    return edges


def path_signature(spans: list[dict]) -> tuple:
    """Hashable shape of one run's chain: (node set, edge set). Two runs with the
    same signature have the same chain structure."""
    return (frozenset(_sequence(spans)), frozenset(_dag_edges(spans)))


def _summarize(add_a, rem_a, add_e, rem_e, same_nodes) -> str:
    parts: list[str] = []
    if add_a:
        parts.append("Added " + ", ".join(add_a))
    if rem_a:
        parts.append("Removed " + ", ".join(rem_a))
    n_edges = len(add_e) + len(rem_e)
    if n_edges and (same_nodes or not (add_a or rem_a)):
        verb = "Rewired" if same_nodes else "Re-routed"
        parts.append(f"{verb} {n_edges} edge" + ("s" if n_edges != 1 else ""))
    return " · ".join(parts) if parts else "Chain changed"


def current_path(runs: list[dict]) -> dict | None:
    """Shape of the most recent run's chain, for the panel header."""
    runs = sorted(runs, key=lambda r: r.get("timestamp", ""))
    for idx in range(len(runs) - 1, -1, -1):
        spans = get_agent_spans(runs[idx]["run_id"])
        seq = _sequence(spans)
        if seq:
            edges = _dag_edges(spans)
            has_dag = any(s.get("parent_step_id") or s.get("branch_id") or
                          s.get("join_step_id") for s in spans)
            return {
                "run_index": idx,
                "chain":     seq,
                "edges":     sorted([list(e) for e in edges]),
                "has_dag":   has_dag,
            }
    return None


def build_path_change_log(runs: list[dict]) -> list[dict]:
    """`runs` sorted ascending by timestamp. Returns chain-change events — one per
    consecutive run pair whose DAG signature differs:
      {run_index, timestamp, old_chain, new_chain, added_agents, removed_agents,
       added_edges, removed_edges, kind ('structure'|'reorder'), summary}.
    Runs with no spans are skipped (they don't reset the comparison baseline).
    """
    runs = sorted(runs, key=lambda r: r.get("timestamp", ""))
    events: list[dict] = []
    prev = None
    for idx, r in enumerate(runs):
        spans = get_agent_spans(r["run_id"])
        seq = _sequence(spans)
        if not seq:
            continue
        nodes, edges = set(seq), _dag_edges(spans)
        if prev is not None and (prev["nodes"], prev["edges"]) != (nodes, edges):
            added_agents   = sorted(nodes - prev["nodes"])
            removed_agents = sorted(prev["nodes"] - nodes)
            added_edges    = sorted(edges - prev["edges"])
            removed_edges  = sorted(prev["edges"] - edges)
            same_nodes = not added_agents and not removed_agents
            events.append({
                # Indexed by the EARLIER run of the pair: the change is the diff
                # from run `run_index` to run `run_index + 1`.
                "run_index":      prev["idx"],
                "timestamp":      r.get("timestamp", ""),
                "old_chain":      prev["seq"],
                "new_chain":      seq,
                "old_edges":      sorted([list(e) for e in prev["edges"]]),
                "new_edges":      sorted([list(e) for e in edges]),
                "added_agents":   added_agents,
                "removed_agents": removed_agents,
                "added_edges":    [list(e) for e in added_edges],
                "removed_edges":  [list(e) for e in removed_edges],
                "kind":           "reorder" if same_nodes else "structure",
                "summary":        _summarize(added_agents, removed_agents,
                                             added_edges, removed_edges, same_nodes),
            })
        prev = {"idx": idx, "seq": seq, "nodes": nodes, "edges": edges}
    return events



def route_topology(spans: list[dict]):
    """Agent-level DAG from a run's spans, capturing fan-out / parallel branches.
    For each span, the predecessor is its parent_step_id agent (explicit edge), or
    the most recent non-branch 'stem' (parallel sibling), or the previous turn
    (sequential) — mirroring the single-run graph. Returns (nodes, edges, signature)
    where the signature is INVARIANT to the order of parallel siblings (sorted edge
    set + node set), so two runs with the same shape collapse to one route."""
    spans = sorted(spans, key=lambda s: s.get("turn_index") or 0)
    if not spans:
        return [], [], ""
    by_id = {s.get("span_id"): s for s in spans}
    edges, nodes = [], []
    last_stem, prev = None, None
    for s in spans:
        a = s.get("agent_name")
        if a and a not in nodes:
            nodes.append(a)
        pid = s.get("parent_step_id")
        pred = None
        if pid and pid in by_id:
            pred = by_id[pid].get("agent_name")
        elif s.get("branch_id") and last_stem is not None:
            pred = last_stem.get("agent_name")
        elif prev is not None:
            pred = prev.get("agent_name")
        if pred and pred != a:
            edges.append((pred, a))
        if not s.get("branch_id"):
            last_stem = s
        prev = s
    uniq_edges = [list(e) for e in dict.fromkeys(edges)]
    sig = "|".join(f"{x}>{y}" for x, y in sorted(set(edges))) + "#" + ",".join(sorted(set(nodes)))
    return nodes, uniq_edges, sig


def route_key(run: dict):
    """Key used to match/group routes. The run's DAG topology signature when spans
    were attached (`_route_sig`) — invariant to parallel-sibling order — else the
    flat agent_sequence (linear fallback for runs with no captured DAG)."""
    sig = run.get("_route_sig")
    if sig:
        return sig
    return tuple(json.loads(run.get("agent_sequence") or "[]"))


def canonical_route(runs: list[dict]) -> tuple:
    """The canonical route = the most common route across `runs` (by topology when
    available, else flat sequence). Returns the representative NODE LIST for the
    most-common route_key, so the per-run `route_conformance` metric and the route
    displays agree on what 'canonical' means."""
    from collections import Counter
    keyed = [(route_key(r), r) for r in runs]
    keyed = [(k, r) for k, r in keyed if k]
    if not keyed:
        return tuple()
    top_key, _ = Counter(k for k, _ in keyed).most_common(1)[0]
    rep = next(r for k, r in keyed if k == top_key)
    nodes = rep.get("_route_nodes")
    if nodes:
        return tuple(nodes)
    return top_key if isinstance(top_key, tuple) else tuple(json.loads(rep.get("agent_sequence") or "[]"))
