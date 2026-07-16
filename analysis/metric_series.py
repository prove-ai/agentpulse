"""Per-run metric series + generic control-band drift detection.

Powers the Metric Drift View: one value per run for every metric, grouped by
category (agents / handoffs / path / parallel). Drift is flagged generically —
a value outside a baseline band (mean ± k*sigma) — so it works for ANY metric
with no per-metric rules. The first run of a sustained breach is the drift-start
(used to anchor the troubleshooting timeline).
"""

from __future__ import annotations

import json
import statistics

from analysis.layer1_raw import get_agent_spans, get_handoffs
from analysis.run_metrics import is_clean_termination
from analysis.dag import detect_parallel_groups
from storage.sqlite_store import compute_cost


# ---------------------------------------------------------------------------
# Control band — the single, generic "is this drifting?" mechanism
# ---------------------------------------------------------------------------
def control_band(values: list[float], *, baseline_runs: int = 5,
                 k: float = 3.0, consecutive: int = 2) -> dict:
    """Baseline band from the first `baseline_runs` points; flag out-of-band runs.

    Returns band stats, per-point breach flags, and the first index of a
    *sustained* drift (>= `consecutive` consecutive breaches), or None.
    """
    n = len(values)
    if n < baseline_runs + 1:
        return {"available": False, "n": n, "breaches": [False] * n, "drifting": False}

    base = values[:baseline_runs]
    mu = statistics.mean(base)
    sd = statistics.pstdev(base) if len(base) > 1 else 0.0
    if sd == 0:                                  # flat baseline → small tolerance
        sd = max(abs(mu) * 0.02, 1e-9)
    # Floor the band so a coincidentally-tight baseline (small-sample sigma) can't
    # flag ordinary run-to-run noise as drift. The band is always at least ±k*5%.
    sd = max(sd, abs(mu) * 0.05)
    lo, hi = mu - k * sd, mu + k * sd

    # Percentile "normal range" of the baseline (for the shaded chart band).
    sb = sorted(base)
    p10, p90 = _percentile(sb, 0.10), _percentile(sb, 0.90)
    if p90 <= p10:                               # widen a flat band so it's visible
        pad = max(abs(mu) * 0.05, 1e-9)
        p10, p90 = mu - pad, mu + pad

    breaches = [(v < lo or v > hi) for v in values]
    for i in range(baseline_runs):               # baseline defines normal, never a breach
        breaches[i] = False

    drift_start, run = None, 0
    for i, b in enumerate(breaches):
        run = run + 1 if b else 0
        if run >= consecutive:
            drift_start = i - consecutive + 1
            break

    return {
        "available": True, "n": n,
        "mean": round(mu, 4), "low": round(lo, 4), "high": round(hi, 4),
        "p10": round(p10, 4), "p90": round(p90, 4), "baseline_mean": round(mu, 4),
        "breaches": breaches,
        "drift_start": drift_start,
        "drifting": drift_start is not None,
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = q * (len(sorted_vals) - 1)
    lo = int(i)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo)


# ---------------------------------------------------------------------------
# Per-run series, per category. Each returns: entity -> metric -> [points]
# where a point is {x: run_index, y: value, run_id, ts}.
# ---------------------------------------------------------------------------
def _pt(idx, val, r):
    return {"x": idx, "y": val, "run_id": r["run_id"], "ts": r.get("timestamp", "")}


# Categories every run reports, in stable order.
_CATEGORIES = ("agents", "handoffs", "path", "parallel")


def compute_run_metrics(run: dict) -> dict:
    """Layer 2 — the standard metric contract for ONE run, keyed by entity.

    Shape:
        {run_id, ts,
         agents:   {agent_name: {metric: value, ...}},
         handoffs: {"A → B":   {metric: value, ...}},
         path:     {"System":  {metric: value, ...}},
         parallel: {label:     {metric: value, ...}}}

    This is the single source of truth for metric VALUES. `per_run_series` pivots
    these per-run dicts into per-entity time series (drift branch); the single-run
    branch reads one run's dict directly against a baseline cohort. No metric is
    computed anywhere else.
    """
    rid = run["run_id"]
    spans = get_agent_spans(rid)
    clean = 1 if is_clean_termination(run.get("termination_reason") or "") else 0
    seq_spans = [s.get("agent_name") for s in sorted(spans, key=lambda s: s.get("turn_index") or 0)]

    # --- agents: per-agent behaviour & cost ---
    by_agent: dict = {}
    for s in spans:
        by_agent.setdefault(s.get("agent_name"), []).append(s)
    agents: dict = {}
    for agent, sp in by_agent.items():
        if not agent:
            continue
        inp = sum(int(s.get("input_tokens") or 0) for s in sp)
        outp = sum(int(s.get("output_tokens") or 0) for s in sp)
        agents[agent] = {
            "tokens":     inp + outp,
            "latency_s":  round(sum(float(s.get("duration_ms") or 0) for s in sp) / 1000.0, 2),
            "cost_usd":   round(sum(compute_cost(int(s.get("input_tokens") or 0),
                                                 int(s.get("output_tokens") or 0),
                                                 s.get("model") or "") for s in sp), 5),
            "tool_calls": sum(int(s.get("tool_call_count") or 0) for s in sp),
            "errors":     sum(1 for s in sp if s.get("status") == "ERROR"),
            "retries":    sum(int(s.get("retry_count") or 0) for s in sp),
            "reinvoked":  1 if seq_spans.count(agent) > 1 else 0,
            "success":    clean,
        }
    # Token distribution: each agent's share of the run's tokens (is one agent
    # hogging?), plus a run-level concentration = the biggest single share.
    total_tokens = sum(m["tokens"] for m in agents.values())
    for m in agents.values():
        m["token_share"] = round(m["tokens"] / total_tokens * 100, 1) if total_tokens else 0.0
    token_concentration = (round(max((m["tokens"] for m in agents.values()), default=0)
                                 / total_tokens, 3) if total_tokens else 0.0)

    # --- handoffs: per-edge frequency, payload, and wall-clock gap (per-hop latency) ---
    spans_by_turn = {s.get("turn_index"): s for s in spans}
    agg: dict = {}
    for h in get_handoffs(rid):
        a, b = h.get("agent_from"), h.get("agent_to")
        if not a or not b:
            continue
        d = agg.setdefault(f"{a} → {b}", {"count": 0, "payload": 0, "gap_ms": 0.0, "ctx": 0.0})
        d["count"] += 1
        d["payload"] += int(h.get("a_output_tokens") or 0)
        d["ctx"] += float(h.get("context_ratio") or 0)        # receiver-in / sender-out
        sa = spans_by_turn.get(h.get("turn_index_from"))
        sb = spans_by_turn.get(h.get("turn_index_to"))
        if sa and sb and sa.get("end_time_ms") is not None and sb.get("start_time_ms") is not None:
            # Clamp negatives to 0 (overlap/parallel edges have no real gap).
            d["gap_ms"] += max(float(sb["start_time_ms"]) - float(sa["end_time_ms"]), 0.0)
    handoffs = {key: {"frequency": d["count"], "payload_tokens": d["payload"],
                      "hop_latency_s": round(d["gap_ms"] / 1000.0, 2),
                      # context lost/kept across the edge (avg if the edge fired twice)
                      "context_ratio": round(d["ctx"] / d["count"], 3) if d["count"] else 0.0}
                for key, d in agg.items()}

    # --- path ("System"): shape, end-to-end wall-clock, and run-level signals ---
    pseq = json.loads(run.get("agent_sequence") or "[]")
    starts = [float(s["start_time_ms"]) for s in spans if s.get("start_time_ms") is not None]
    ends = [float(s["end_time_ms"]) for s in spans if s.get("end_time_ms") is not None]
    if starts and ends:
        wall = (max(ends) - min(starts)) / 1000.0
    else:                                                # no timing → sum durations
        wall = sum(float(s.get("duration_ms") or 0) for s in spans) / 1000.0

    # context growth: mean turn-over-turn change in INPUT tokens (context ballooning
    # — or starving — as the chain proceeds).
    sorted_sp = sorted(spans, key=lambda s: s.get("turn_index") or 0)
    inp_per_turn = [int(s.get("input_tokens") or 0) for s in sorted_sp]
    growth = [(inp_per_turn[i] - inp_per_turn[i - 1]) / inp_per_turn[i - 1] * 100
              for i in range(1, len(inp_per_turn)) if inp_per_turn[i - 1] > 0]
    context_growth_pct = round(statistics.mean(growth), 1) if growth else 0.0

    # parallelism factor: sum of span durations / wall-clock (1.0 = fully sequential).
    durs = [float(s.get("duration_ms") or 0) for s in spans if s.get("duration_ms")]
    total_wall_ms = run.get("total_duration_ms") or (max(durs, default=0))
    parallelism_factor = round(sum(durs) / total_wall_ms, 2) if total_wall_ms else 1.0

    # re-entry count: agent re-invocations beyond the first (loop pressure).
    reentry_count = sum(c - 1 for c in (seq_spans.count(a) for a in {x for x in seq_spans if x}) if c > 1)

    path = {"System": {"path_length": len(pseq),
                       "loops": len(pseq) - len(set(pseq)),
                       "e2e_latency_s": round(wall, 2),
                       "token_concentration": token_concentration,
                       "context_growth_pct": context_growth_pct,
                       "parallelism_factor": parallelism_factor,
                       "reentry_count": reentry_count}}

    # --- parallel: the dominant fan-out group's bottleneck / wait / balance ---
    parallel: dict = {}
    groups = detect_parallel_groups(spans)
    if groups:
        g = max(groups, key=lambda x: x.wall_clock_ms)
        label = f"{g.parent_agent} →…→ {g.join_agent or '?'}"
        wait = max(g.wall_clock_ms - min((b.duration_ms for b in g.branches), default=0), 0) / 1000.0
        parallel[label] = {"bottleneck_s": round(g.bottleneck.duration_ms / 1000.0, 2),
                           "join_wait_s": round(wait, 2),
                           "efficiency": round(g.efficiency, 3)}

    return {"run_id": rid, "ts": run.get("timestamp", ""),
            "agents": agents, "handoffs": handoffs, "path": path, "parallel": parallel}


def per_run_series(runs: list[dict]) -> dict:
    """Pivot the Layer-2 per-run metrics (`compute_run_metrics`) into per-entity
    time series: category → entity → metric → [{x, y, run_id, ts}].

    Mostly a pure reshape. The one exception is `route_conformance` (path), which
    is *window-relative* — it needs the canonical route across all runs — so it
    can't live in the stateless per-run contract and is computed here.
    An entity/metric only gets a point for the runs where it appears (gaps
    preserved). Runs sorted ascending by timestamp.
    """
    runs = sorted(runs, key=lambda r: r.get("timestamp", ""))
    out: dict = {c: {} for c in _CATEGORIES}
    for idx, r in enumerate(runs):
        m = compute_run_metrics(r)
        for cat in _CATEGORIES:
            cat_out = out[cat]
            for entity, metrics in m[cat].items():
                ent = cat_out.setdefault(entity, {})
                for name, val in metrics.items():
                    ent.setdefault(name, []).append(_pt(idx, val, r))

    # route_conformance: 1.0 if a run took the canonical (most common) route, else
    # 0.0. Matched by route_key — the DAG topology signature when spans were attached
    # (so parallel-sibling reorderings don't count as different routes), else the
    # flat agent_sequence. Window mean = conformance rate; 1 - mean = fallback_rate.
    if runs:
        from collections import Counter
        from analysis.changes import route_key
        keys = [route_key(r) for r in runs]
        nonempty = [k for k in keys if k]
        canon_key = Counter(nonempty).most_common(1)[0][0] if nonempty else None
        sys_m = out["path"].setdefault("System", {})
        conf = sys_m.setdefault("route_conformance", [])
        for idx, (r, k) in enumerate(zip(runs, keys)):
            conf.append(_pt(idx, 1.0 if k == canon_key else 0.0, r))
    return out


# ---------------------------------------------------------------------------
# Chart assembly + "what's drifting" scan
# ---------------------------------------------------------------------------
def metrics_of(category_series: dict) -> list[str]:
    """Distinct metric names available in a category (for the picker)."""
    seen: list[str] = []
    for metrics in category_series.values():
        for m in metrics:
            if m not in seen:
                seen.append(m)
    return seen


def build_charts(category_series: dict, metric: str, band_cfg: dict) -> list[dict]:
    """One chart per entity for the chosen metric: points + band + drift flag."""
    charts = []
    for entity, metrics in sorted(category_series.items()):
        pts = metrics.get(metric)
        if not pts:
            continue
        band = control_band([p["y"] for p in pts], **band_cfg)
        charts.append({"entity": entity, "metric": metric, "points": pts, "band": band})
    # Drifting charts first.
    charts.sort(key=lambda c: (0 if c["band"].get("drifting") else 1, c["entity"]))
    return charts


_TYPE_LABEL = {"handoffs": "Handoff drift", "path": "Path shift",
               "agents": "Agent drift", "parallel": "Parallel imbalance"}
_IMPACT_METRICS = {"success", "errors", "retries", "cost_usd"}


def build_findings(drifting: list[dict], all_series: dict) -> list[dict]:
    """Group drifting metrics into ranked findings (one per category+entity)."""
    from collections import defaultdict
    grouped: dict = defaultdict(list)
    for d in drifting:
        grouped[(d["category"], d["entity"])].append(d)

    findings = []
    for (cat, entity), ds in grouped.items():
        metrics = [d["metric"] for d in ds]
        drift_start = min(d["drift_start"] for d in ds)
        ent_series = all_series.get(cat, {}).get(entity, {})
        runs_seen = max((len(ent_series.get(m, [])) for m in metrics), default=0)
        impacty = any(m in _IMPACT_METRICS or "retry" in m for m in metrics)
        n = len(metrics)
        if "success" in metrics or n >= 3:
            risk = "High"
        elif n == 2 or impacty:
            risk = "Medium"
        else:
            risk = "Watch"
        findings.append({
            "category": cat, "entity": entity, "type": _TYPE_LABEL.get(cat, cat),
            "metrics": metrics, "n_metrics": n, "risk": risk,
            "runs_seen": runs_seen, "drift_start": drift_start,
        })
    # Highest risk first; "Watch" always last. Tie-break by most metrics drifted.
    order = {"High": 0, "Medium": 1, "Watch": 2}
    findings.sort(key=lambda f: (order[f["risk"]], -f["n_metrics"]))
    return findings


def _g(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


def metric_impact(pts: list[dict], baseline_runs: int) -> dict:
    """Impact of a metric: recent-mean vs baseline-mean. Returns a display label
    that falls back to the absolute move when a percentage is undefined
    (baseline of 0, e.g. tool_calls going 0 → 1)."""
    ys = [p["y"] for p in pts]
    base, recent = ys[:baseline_runs], ys[baseline_runs:]
    bm = statistics.mean(base) if base else 0.0
    rm = statistics.mean(recent) if recent else 0.0
    if bm == 0:
        pct = None
        label = "no change" if rm == 0 else f"{_g(bm)} → {_g(rm)}"
    else:
        pct = round((rm - bm) / bm * 100)
        label = f"{pct:+d}%"
    return {"baseline": round(bm, 3), "recent": round(rm, 3), "pct": pct, "label": label}


def drifting_series(all_series: dict, band_cfg: dict) -> list[dict]:
    """Scan every (category, entity, metric) and list those with an active drift."""
    out = []
    for cat, ents in all_series.items():
        for entity, metrics in ents.items():
            for metric, pts in metrics.items():
                band = control_band([p["y"] for p in pts], **band_cfg)
                if band.get("drifting"):
                    out.append({"category": cat, "entity": entity, "metric": metric,
                                "drift_start": band["drift_start"]})
    return out
