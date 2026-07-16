"""Metric drift detection — directional, tiered, corroborated.

The investigation GATE is a Tier-0 (outcome) metric breaching; behaviour metrics
only corroborate. Two clocks, per the design:
  - per-run BREACHES decide WHEN an investigation starts (sustained: N in a row,
    or M of the last N),
  - the RECENT-window mean DELTA decides HOW BAD (lands in watch/drift/high).

Severity escalates by corroboration:
  watch      — 1 low-rank metric moved, no outcome impact
  candidate  — 1 Tier-0 alone, OR 2 related Tier-1 on a component
  drift      — 1 Tier-0 + >=1 supporting metric (same component/window)
  high       — Tier-0 + >=2 supporting + a nearby config/model/prompt/tool change

Output findings feed analysis/drift_chains for the upstream causal walk.
Everything is correlational — a finding is an investigation path, not proof.
"""
from __future__ import annotations

import statistics

# Metrics whose per-run value IS the bad-event signal (no count source needed).
_INHERENT_EVENTS = {"success", "route_conformance"}
_SEV_RANK = {"watch": 0, "candidate": 1, "drift": 2, "high": 3}


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _vals(series: dict, mc: dict, metric: str):
    """Per-run value list for a metric + whether it's an event signal + the RUN
    index of each value (`xs`). For sparse metrics (a handoff/agent absent in some
    runs) the value list is dense but each entry maps to a real run via `xs`, so a
    dense drift-start index can be translated back to the actual run number.
    Rate metrics with a `source` count series map each run to 1.0 if the event
    happened (count > 0) else 0.0."""
    src = mc.get("source", metric)
    pts = series.get(mc["scope"], {}).get(mc["_entity"], {}).get(src)
    if not pts:
        return None, False, []
    ys = [p["y"] for p in pts]
    xs = [p.get("x") for p in pts]
    if "source" in mc:
        return [1.0 if y > 0 else 0.0 for y in ys], True, xs
    return ys, metric in _INHERENT_EVENTS, xs


def _delta(baseline_mean: float, recent_mean: float, mc: dict) -> float:
    """Signed delta in the metric's unit (pp / pct / abs)."""
    unit = mc["unit"]
    if unit == "pp":
        scale = 1.0 if mc.get("scale") == "pct100" else 100.0
        return (recent_mean - baseline_mean) * scale
    if unit == "pct":
        return ((recent_mean - baseline_mean) / baseline_mean * 100.0) if baseline_mean else (
            100.0 if recent_mean else 0.0)
    return recent_mean - baseline_mean                      # abs


def _bad_delta(delta: float, mc: dict) -> float:
    """How far the delta moved in the HARMFUL direction (>=0 = harmful)."""
    d = mc["dir"]
    if d == "up":
        return delta
    if d == "down":
        return -delta
    return abs(delta)                                       # both


_EPS = 1e-9                                                 # baseline "effectively zero"


def _band_from(bad: float, thr: dict):
    """Band from a {watch, drift, high} threshold dict (high optional)."""
    if thr.get("high") is not None and bad >= thr["high"]:
        return "high"
    if bad >= thr["drift"]:
        return "drift"
    if bad >= thr["watch"]:
        return "watch"
    return None


def _band(bad: float, mc: dict):
    return _band_from(bad, mc)


def _per_run_breaches(vals, is_event, baseline_mean, mc, abs_thr=None):
    """Boolean per run: was this run in breach (bad direction past watch)?
    `abs_thr` set = zero-baseline mode: judge the ABSOLUTE change vs `abs_thr`."""
    if is_event:
        bad = (lambda v: v <= 0) if mc["dir"] == "down" else (lambda v: v >= 1)
        return [bad(v) for v in vals]
    out = []
    for v in vals:
        if abs_thr is not None:                            # absolute (~0 baseline)
            out.append(_bad_delta(v - baseline_mean, mc) >= abs_thr["watch"])
        else:
            out.append(_bad_delta(_delta(baseline_mean, v, mc), mc) >= mc["watch"])
    return out


def _sustained(breaches, cfg) -> tuple[bool, int]:
    """Did breaches meet the start rule (N consecutive OR M of last N)? Returns
    (sustained, drift_start_index)."""
    n_con = cfg.get("start_consecutive", 2)
    m, n = cfg.get("start_m_of_n", [3, 5])
    # N consecutive anywhere in the recent region
    run = 0
    first_con = None
    for i, b in enumerate(breaches):
        run = run + 1 if b else 0
        if run >= n_con and first_con is None:
            first_con = i - n_con + 1
    tail = breaches[-n:]
    m_of_n = sum(tail) >= m
    if first_con is not None:
        return True, first_con
    if m_of_n:
        idx = next((len(breaches) - n + j for j, b in enumerate(tail) if b), len(breaches) - 1)
        return True, idx
    return False, len(breaches) - 1


def _adjacent(scope: str, entity: str, all_entities: dict) -> set:
    """Component neighbourhood for corroboration: the entity itself + its
    immediate handoffs (for an agent) / endpoint agents (for a handoff)."""
    members = {(scope, entity)}
    if scope == "agents":
        for h in all_entities.get("handoffs", []):
            a, _, b = h.partition(" → ")
            if entity in (a.strip(), b.strip()):
                members.add(("handoffs", h))
    elif scope == "handoffs":
        a, _, b = entity.partition(" → ")
        for ag in (a.strip(), b.strip()):
            members.add(("agents", ag))
    return members


def metric_breaches(series: dict, cfg: dict) -> list[dict]:
    """Every (entity, metric) whose recent-vs-baseline delta crossed a threshold,
    tagged with tier, band, direction, sustained, and drift_start. The raw signals
    consumed by both classify_drift (grouping) and the causal walk."""
    mcfg = (cfg or {}).get("metrics", {})
    baseline_runs = cfg.get("baseline_runs", 10)
    recent_runs = cfg.get("recent_runs", 10)
    recovery_grace = cfg.get("recovery_grace", 2)          # clean-tail runs => recovered
    breaches = []
    for metric, base in mcfg.items():
        scope = base["scope"]
        for entity in series.get(scope, {}):
            mc = {**base, "_entity": entity}
            vals, is_event, xs = _vals(series, mc, metric)
            if not vals or len(vals) < 3:
                continue
            bvals, rvals = vals[:baseline_runs], vals[-recent_runs:]
            bm, rm = _mean(bvals), _mean(rvals)

            # Zero-baseline guard: a percent metric against a ~0 baseline makes the
            # % blow up to 100% (a handoff/agent absent in the baseline version).
            # Judge by the ABSOLUTE change against the metric's `abs` thresholds
            # instead; with none defined we can't fairly call it drift, so skip.
            abs_thr = base.get("abs") if (mc["unit"] == "pct" and abs(bm) < _EPS) else None
            if mc["unit"] == "pct" and abs(bm) < _EPS and not abs_thr:
                continue
            if abs_thr is not None:
                bad = _bad_delta(rm - bm, mc)              # absolute harmful delta
                band = _band_from(bad, abs_thr)
            else:
                bad = _bad_delta(_delta(bm, rm, mc), mc)
                # Absolute floor: a persistently LOW recent level is a breach on its
                # own (e.g. route_conformance the baseline window absorbed early).
                floor = base.get("floor")
                if floor is not None and rm < floor:
                    scale = 1.0 if base.get("scale") == "pct100" else 100.0
                    bad = max(bad, (floor - rm) * scale)
                band = _band(bad, mc)
            if band is None:
                continue
            pr = _per_run_breaches(vals, is_event, bm, mc, abs_thr)
            # Recovery gate: if the metric has been back within normal for the last
            # `recovery_grace` runs (its last breach is that far from the end), the
            # drift has EXPIRED — don't surface a stale finding for a resolved spike.
            last_breach = max((i for i, b in enumerate(pr) if b), default=-1)
            if last_breach >= 0 and (len(pr) - 1 - last_breach) >= recovery_grace:
                continue
            sustained, ds = _sustained(pr, cfg)
            # ds is a DENSE index into this metric's value list; translate it back to
            # the real RUN number so charts + "since run N" line up, and so co-timing
            # across sparse metrics compares real runs (not mismatched dense indices).
            run_start = xs[ds] if (0 <= ds < len(xs) and xs[ds] is not None) else ds
            breaches.append({
                "scope": scope, "entity": entity, "metric": metric,
                "tier": base["tier"], "band": band, "bad_delta": round(bad, 1),
                "direction": "down" if base["dir"] == "down" else ("up" if base["dir"] == "up"
                             else ("up" if _delta(bm, rm, mc) >= 0 else "down")),
                "sustained": sustained, "drift_start": run_start,
            })
    return breaches


def classify_drift(series: dict, change_log: list[dict], cfg: dict,
                   cotiming: int = 5) -> list[dict]:
    """Return tiered drift findings from the per-run series + config thresholds."""
    exclude = set(cfg.get("exclude_from_corroboration", []))
    all_entities = {cat: list(series.get(cat, {}).keys()) for cat in series}
    breaches = metric_breaches(series, cfg)

    # Severity is driven by the CRITICAL (Tier-0) metric's own strength plus how
    # strongly it's corroborated — not by a raw count of weak signals.
    strong_bands = set(cfg.get("strong_support_bands", ["drift", "high"]))
    change_recur_max = cfg.get("change_recurrence_max", 2)

    # --- 2) group into findings: a Tier-0 trigger + co-timed supporting evidence ---
    # Cause attribution: a config change can only be a drift's CAUSE if it (1) happened
    # AT or BEFORE the drift start, within a short window, and (2) sits on the SAME or
    # an UPSTREAM component (a downstream change can't explain an upstream drift). Same
    # rule as the "Potentially related changes" panel, so escalation and UI agree.
    change_window = cfg.get("related_change_window", 3)
    _DIM_RANK = {"prompt": 0, "model": 1, "tools": 2, "params": 3}
    _edges = []
    for _h in all_entities.get("handoffs", []):
        _a, _, _b = _h.partition(" → ")
        if _a.strip() and _b.strip():
            _edges.append((_a.strip(), _b.strip()))

    def _components(scope, entity):
        # (own agents, upstream-closure) for a finding; upstream=None means system-wide
        if scope == "handoffs":
            a, _, b = entity.partition(" → ")
            own = {a.strip(), b.strip()}
        elif scope == "path":
            return set(), None
        else:
            own = {entity}
        import collections
        rev = collections.defaultdict(set)
        for a, b in _edges:
            rev[b].add(a)
        seen, stack = set(own), list(own)
        while stack:
            for p in rev.get(stack.pop(), ()):
                if p not in seen:
                    seen.add(p); stack.append(p)
        return own, seen

    def _nearby_change(scope, entity, ds):
        own, relevant = _components(scope, entity)
        best, best_key = None, None
        for e in change_log:
            ri = e.get("run_index") or 0
            if not (0 <= ds - ri <= change_window):                 # at/before, within window
                continue
            sc = e.get("scope")
            if sc != "workflow" and relevant is not None and sc not in relevant:
                continue                                            # same/upstream only
            if e.get("dimension") not in _DIM_RANK:
                continue
            key = (0 if sc in own else 1, _DIM_RANK[e["dimension"]], ds - ri)
            if best_key is None or key < best_key:                  # same>upstream, strong dim, recent
                best_key, best = key, e
        return best

    def _meaningful_change(change):
        # A change only corroborates if it's a DISTINCTIVE event, not one that recurs
        # nearly every version (e.g. a prompt-hash bump on each release).
        if not change:
            return False
        key = (change.get("scope"), change.get("dimension"))
        n = sum(1 for e in change_log if (e.get("scope"), e.get("dimension")) == key)
        return n <= change_recur_max

    findings, used = [], set()
    triggers = [b for b in breaches if b["tier"] == 0 and b["sustained"]]
    triggers.sort(key=lambda b: (-_SEV_RANK[b["band"]], b["drift_start"]))
    for t in triggers:
        key = (t["scope"], t["entity"])
        if key in used:
            continue
        members = _adjacent(t["scope"], t["entity"], all_entities)
        support = [b for b in breaches
                   if (b["scope"], b["entity"]) in members
                   and b["metric"] not in exclude
                   and (b["metric"], b["entity"]) != (t["metric"], t["entity"])
                   and abs(b["drift_start"] - t["drift_start"]) <= cotiming]
        change = _nearby_change(t["scope"], t["entity"], t["drift_start"])
        # Only band-strong supporters (drift/high) count toward escalation; watch-band
        # ones are too weak to push severity up. A finding still needs corroboration —
        # a lone Tier-0 metric stays a low-confidence candidate.
        strong = [s for s in support if s["band"] in strong_bands]
        if t["band"] in ("drift", "high") and len(strong) >= 2 and _meaningful_change(change):
            sev = "high"
        elif len(strong) >= 1:
            sev = "drift"
        else:
            sev = "candidate"
        findings.append({
            "scope": t["scope"], "entity": t["entity"], "severity": sev,
            "trigger": t, "supporting": support, "related_change": change,
            "drift_start": t["drift_start"],
            "summary": _summarize(t, support, change),
        })
        used.add(key)

    # 2b) no Tier-0, but 2 related Tier-1 on a component → candidate
    for b in breaches:
        if b["tier"] != 1 or (b["scope"], b["entity"]) in used:
            continue
        members = _adjacent(b["scope"], b["entity"], all_entities)
        peers = [x for x in breaches if x["tier"] == 1
                 and (x["scope"], x["entity"]) in members
                 and abs(x["drift_start"] - b["drift_start"]) <= cotiming]
        if len(peers) >= 2:
            findings.append({
                "scope": b["scope"], "entity": b["entity"], "severity": "candidate",
                "trigger": b, "supporting": [p for p in peers if p is not b],
                "related_change": None, "drift_start": b["drift_start"],
                "summary": _summarize(b, peers, None)})
            used.add((b["scope"], b["entity"]))

    findings.sort(key=lambda f: (-_SEV_RANK[f["severity"]], f["drift_start"]))
    return findings


def _summarize(trigger, support, change) -> str:
    t = f'{trigger["entity"]} {trigger["metric"]} {trigger["direction"]} ({trigger["band"]})'
    if support:
        t += " + " + ", ".join(f'{s["metric"]} {s["direction"]}' for s in support[:3])
    if change:
        t += f'; near {change.get("scope")} {change.get("dimension")} change'
    return t


def investigate(series: dict, change_log: list[dict], cfg: dict,
                cotiming: int = 5) -> dict:
    """Full drift investigation from one set of breach signals:
      - findings : tiered, per-component drift findings (classify_drift)
      - chains   : cross-component root -> symptom causal chains (detect_chains),
                   built by treating Tier-0 outcome breaches as the symptom and
                   walking upstream to a co-timed behaviour breach (the root).
    """
    from analysis.drift_chains import build_topology, edges_from_handoff_series, detect_chains
    findings = classify_drift(series, change_log, cfg, cotiming)
    signals = [{"category": b["scope"], "entity": b["entity"], "metric": b["metric"],
                "drift_start": b["drift_start"], "direction": b["direction"],
                "bad_delta": b["bad_delta"], "band": b["band"], "tier": b["tier"],
                "kind": "impact" if b["tier"] == 0 else "behaviour"}
               for b in metric_breaches(series, cfg)]
    topology = build_topology(edges_from_handoff_series(series))
    chains = detect_chains(signals, topology, change_log, cotiming=cotiming)
    return {"findings": findings, "chains": chains}
