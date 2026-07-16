"""Web dashboard — Flask app to visualise captured runs.

Usage:
    cd observability
    python reporter/dashboard.py
    # Open http://localhost:5001 in your browser.

Routes:
    /          — all runs summary table
    /run/<id>  — full metrics + drift for one run
    /api/runs  — JSON list of all runs (for future use)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_OBS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_OBS_ROOT))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Only sets keys not already in env."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


_load_dotenv(_OBS_ROOT / ".env")

from flask import Flask, render_template, jsonify, abort, request, g

from analysis.layer1_raw import list_runs, get_run, get_agent_spans, get_tool_calls, get_baseline_runs, get_handoffs
from analysis.run_metrics import compute_all, is_clean_termination
from analysis.run_anomaly import build_anomaly_report
from analysis.run_insights import (
    single_run_insights, anomaly_insights, severity_counts, rank_insights,
)
from analysis.version_drift import compare_versions, version_insights
from analysis.dag import detect_parallel_groups, critical_path
from analysis.trends import build_trends
from analysis.changes import potentially_related_changes, diff_configs, _parse_cfg
from storage.sqlite_store import (
    compute_cost, set_active_db_path, resolve_db_path, list_available_dbs, DEFAULT_DB,
    get_versions, create_version_snapshot, latest_run_config,
    get_baseline_version, set_baseline_version,
    get_custom_metrics, save_custom_metrics, get_thresholds, save_thresholds,
    get_events,
)


def _metrics_for_run(run: dict) -> dict:
    """Compute full metrics for a single run (spans + tools + handoffs)."""
    spans = get_agent_spans(run["run_id"])
    tools = get_tool_calls(run["run_id"])
    hoffs = get_handoffs(run["run_id"])
    return compute_all(run, spans, tools, hoffs)


def compute_handoff_leaderboard(early_runs: list[dict], recent_runs: list[dict],
                                top_n: int = 5) -> dict:
    """Common-handoff health — COMPACT cards ranked by total volume.

    Unlike the drift suspects (detailed insight cards), these are intentionally
    lightweight: agent pair, how often it fires, success rate, average payload,
    and average downstream cost. The route layer attaches a small "minor drift"
    badge afterward for any pair that drifted but didn't make the suspect cut.
    """
    from collections import defaultdict

    early_occ  = _collect_handoff_occurrences(early_runs)
    recent_occ = _collect_handoff_occurrences(recent_runs)

    early_by_pair: dict[tuple, list[dict]] = defaultdict(list)
    recent_by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for o in early_occ:
        early_by_pair[(o["from"], o["to"])].append(o)
    for o in recent_occ:
        recent_by_pair[(o["from"], o["to"])].append(o)

    total = len(early_occ) + len(recent_occ)
    all_pairs = set(early_by_pair) | set(recent_by_pair)

    # Rank by total volume across the whole window.
    ranked = sorted(
        all_pairs,
        key=lambda p: -(len(early_by_pair.get(p, [])) + len(recent_by_pair.get(p, []))),
    )[:top_n]

    cards = []
    for pair in ranked:
        a, b = pair
        e_occ = early_by_pair.get(pair, [])
        r_occ = recent_by_pair.get(pair, [])
        all_occ = e_occ + r_occ
        count   = len(all_occ)
        agg     = _aggregate_pair(all_occ)

        # Same Evidence + Downstream metric tables as the drift suspects.
        # Show early→recent deltas when both halves have data; otherwise a
        # snapshot with no delta.
        if e_occ and r_occ:
            early_agg  = _aggregate_pair(e_occ)
            recent_agg = _aggregate_pair(r_occ)
            evidence   = _evidence_rows(early_agg, recent_agg, 0.0, a, b)
            downstream = _downstream_rows(early_agg, recent_agg)
        else:
            primary = _aggregate_pair(r_occ if r_occ else e_occ)
            note    = "(recent only)" if r_occ else "(earlier only)"
            evidence   = _snapshot_rows(primary, pair, label_suffix=note)
            downstream = _downstream_snapshot(primary, label_suffix=note)

        cards.append({
            "from":          a, "to": b,
            "count":         count,
            "pct":           round((count / total * 100), 1) if total else 0.0,
            "success_rate":  round(agg["success_rate"], 1),
            "payload_avg":   int(agg["payload_avg"]),
            "evidence":      evidence,
            "downstream":    downstream,
        })

    return {"total": total, "pairs": cards}


def _collect_handoff_occurrences(runs: list[dict]) -> list[dict]:
    """Walk every run and emit one record per (sender → receiver) edge.

    Each record carries everything the drift rules need: per-occurrence
    payload size, target duration, target re-invocation flag, and the
    run-level signals (clean / failed / remaining duration / remaining tokens
    after this point in the run).
    """
    from analysis.run_metrics import is_clean_termination

    out: list[dict] = []
    for r in runs:
        spans = sorted(
            get_agent_spans(r["run_id"]),
            key=lambda s: s.get("turn_index") or 0,
        )
        if len(spans) < 2:
            continue
        clean = is_clean_termination(r.get("termination_reason") or "")
        # Pre-compute suffix sums so each edge knows what comes downstream.
        n = len(spans)
        tok_suffix = [0] * (n + 1)
        dur_suffix = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            tok = int(spans[i].get("input_tokens") or 0) + int(spans[i].get("output_tokens") or 0)
            dur = float(spans[i].get("duration_ms") or 0)
            tok_suffix[i] = tok_suffix[i + 1] + tok
            dur_suffix[i] = dur_suffix[i + 1] + dur

        # Agent → list of turn indices it ran on (for re-invocation detection)
        turns_by_agent: dict[str, list[int]] = {}
        for s in spans:
            turns_by_agent.setdefault(s.get("agent_name"), []).append(s.get("turn_index") or 0)

        # "Rework" = a span whose agent already ran earlier in the run (a repeat
        # call). Suffix sum so each position knows how much rework comes after it.
        seen: set = set()
        is_repeat = [0] * n
        for j in range(n):
            nm = spans[j].get("agent_name")
            is_repeat[j] = 1 if nm in seen else 0
            seen.add(nm)
        rework_suffix = [0] * (n + 1)
        for j in range(n - 1, -1, -1):
            rework_suffix[j] = rework_suffix[j + 1] + is_repeat[j]

        for i in range(n - 1):
            a, b = spans[i], spans[i + 1]
            a_name = a.get("agent_name")
            b_name = b.get("agent_name")
            if not a_name or not b_name:
                continue
            # Re-invocation: does the receiver's agent appear again strictly
            # later in this run? (counts as a "target re-invocation".)
            b_turn = b.get("turn_index") or 0
            later_turns = [t for t in turns_by_agent.get(b_name, []) if t > b_turn]
            reinvoked = 1 if later_turns else 0

            out.append({
                "from":            a_name,
                "to":              b_name,
                "run_id":          r["run_id"],
                "timestamp":       r.get("timestamp", ""),
                "prompt_version":  r.get("prompt_version"),
                "clean":           clean,
                "payload_tokens":  int(a.get("output_tokens") or 0),
                "target_duration_ms":  float(b.get("duration_ms") or 0),
                "target_output_tokens": int(b.get("output_tokens") or 0),
                "target_reinvoked":    reinvoked,
                # Rework happening AFTER the receiver (repeat agent calls).
                "downstream_rework":   rework_suffix[i + 2],
                "remaining_handoffs":  (n - 1) - (i + 1),
                "remaining_duration_ms": dur_suffix[i + 2],
                "remaining_tokens":      tok_suffix[i + 2],
            })
    return out


# Drift detection is driven entirely by config/drift_rules.yaml.
# See analysis/drift_config.py for the loader + generic evaluator.


def _safe_pct_change(early: float, recent: float) -> float:
    """Signed percentage change. Zero-baseline → 100% if recent non-zero."""
    if early == 0:
        return 0.0 if recent == 0 else 100.0
    return (recent - early) / early * 100.0


def _mean(xs: list[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def _aggregate_pair(occurrences: list[dict]) -> dict:
    """Roll a list of (a→b) occurrences into the six metrics + supporting stats."""
    n = len(occurrences)
    if n == 0:
        return {"count": 0}
    clean_count = sum(1 for o in occurrences if o["clean"])
    return {
        "count":               n,
        "payload_avg":         _mean([o["payload_tokens"] for o in occurrences]),
        "target_duration_avg": _mean([o["target_duration_ms"] for o in occurrences]),
        "target_output_avg":   _mean([o["target_output_tokens"] for o in occurrences]),
        "reinvocation_rate":   sum(o["target_reinvoked"] for o in occurrences) / n * 100,
        "success_rate":        clean_count / n * 100,
        "failure_rate":        (n - clean_count) / n * 100,
        "remaining_handoffs_avg": _mean([o["remaining_handoffs"] for o in occurrences]),
        "remaining_duration_avg": _mean([o["remaining_duration_ms"] for o in occurrences]),
        "remaining_tokens_avg":   _mean([o["remaining_tokens"] for o in occurrences]),
        "downstream_rework_avg":  _mean([o.get("downstream_rework", 0) for o in occurrences]),
    }


def _compute_deltas(early: dict, recent: dict, freq_delta_pct: float) -> dict[str, float]:
    """Build the metric→percent-change dict the config conditions reference.
    Metric names here MUST match the `metric:` values in drift_rules.yaml.
    """
    return {
        "payload_change":            _safe_pct_change(early["payload_avg"],            recent["payload_avg"]),
        "target_duration_change":    _safe_pct_change(early["target_duration_avg"],    recent["target_duration_avg"]),
        "target_output_change":      _safe_pct_change(early["target_output_avg"],      recent["target_output_avg"]),
        "reinvocation_change":       _safe_pct_change(early["reinvocation_rate"],      recent["reinvocation_rate"]),
        "failure_change":            _safe_pct_change(early["failure_rate"],           recent["failure_rate"]),
        "success_change":            _safe_pct_change(early["success_rate"],           recent["success_rate"]),
        "remaining_duration_change": _safe_pct_change(early["remaining_duration_avg"], recent["remaining_duration_avg"]),
        "remaining_tokens_change":   _safe_pct_change(early["remaining_tokens_avg"],   recent["remaining_tokens_avg"]),
        "remaining_handoffs_change": _safe_pct_change(early["remaining_handoffs_avg"], recent["remaining_handoffs_avg"]),
        "frequency_change":          freq_delta_pct,
    }


def _evaluate_rules(early: dict, recent: dict, freq_delta_pct: float, cfg: dict) -> list[str]:
    """Apply every rule in the config. Returns fired rule names, priority-sorted."""
    from analysis.drift_config import evaluate
    deltas = _compute_deltas(early, recent, freq_delta_pct)
    return evaluate(deltas, cfg)


def _severity(early: dict, recent: dict, freq_delta_pct: float, cfg: dict) -> float:
    """Weighted delta-sum. Weights come from config `severity_weights`.

    Directional metrics only count when they move the BAD direction;
    payload + frequency count in either direction (both can hurt).
    """
    w = cfg.get("severity_weights", {})
    succ_drop = max(0.0,  _safe_pct_change(early["success_rate"],         recent["success_rate"]) * -1)
    retry_up  = max(0.0,  _safe_pct_change(early["reinvocation_rate"],    recent["reinvocation_rate"]))
    rd_up     = max(0.0,  _safe_pct_change(early["remaining_duration_avg"], recent["remaining_duration_avg"]))
    rt_up     = max(0.0,  _safe_pct_change(early["remaining_tokens_avg"],   recent["remaining_tokens_avg"]))
    pay_abs   = abs(_safe_pct_change(early["payload_avg"], recent["payload_avg"]))
    freq_abs  = abs(freq_delta_pct)
    return (
        w.get("success_drop", 3)          * succ_drop +
        w.get("reinvocation_up", 2)       * retry_up  +
        w.get("remaining_duration_up", 2) * rd_up     +
        w.get("remaining_tokens_up", 2)   * rt_up     +
        w.get("payload_change_abs", 1)    * pay_abs   +
        w.get("frequency_change_abs", 1)  * freq_abs
    )


def _classify_risk(early: dict, recent: dict, severity: float, cfg: dict) -> str:
    r = cfg.get("risk", {})
    succ_drop = max(0.0, _safe_pct_change(early["success_rate"], recent["success_rate"]) * -1)
    if succ_drop >= r.get("high_success_drop", 10) or severity >= r.get("high_severity", 200):
        return "high"
    if severity < r.get("low_severity", 60):
        return "low"
    return "medium"


def _risk_reason(early: dict, recent: dict, severity: float, rule: str, cfg: dict) -> str:
    """One short clause explaining WHY the risk level was assigned.

    Surfaces the single biggest contributor so the user can see the cause
    without reading the whole evidence table.
    """
    if rule in ("new_route", "vanished_route"):
        return "Structural change in routing — review downstream impact"
    if early is None or recent is None:
        return ""
    high_drop = cfg.get("risk", {}).get("high_success_drop", 10)
    succ_drop = max(0.0, _safe_pct_change(early["success_rate"], recent["success_rate"]) * -1)
    if succ_drop >= high_drop:
        return f"Final success rate fell {succ_drop:.0f}%"
    # Find the largest bad-direction mover for a human-readable reason.
    contributors = [
        ("success rate",        max(0.0, _safe_pct_change(early["success_rate"], recent["success_rate"]) * -1), "fell {:.0f}%"),
        ("re-invocation rate",  max(0.0, _safe_pct_change(early["reinvocation_rate"], recent["reinvocation_rate"])), "rose {:.0f}%"),
        ("remaining duration",  max(0.0, _safe_pct_change(early["remaining_duration_avg"], recent["remaining_duration_avg"])), "rose {:.0f}%"),
        ("remaining tokens",    max(0.0, _safe_pct_change(early["remaining_tokens_avg"], recent["remaining_tokens_avg"])), "rose {:.0f}%"),
        ("payload size",        abs(_safe_pct_change(early["payload_avg"], recent["payload_avg"])), "moved {:.0f}%"),
    ]
    contributors.sort(key=lambda c: -c[1])
    top_name, top_val, top_fmt = contributors[0]
    if top_val <= 0:
        return "Multiple small shifts combined"
    return f"{top_name.capitalize()} {top_fmt.format(top_val)}"


def _rule_meta(rule: str, cfg: dict) -> dict:
    """Look up {title, insight} for a rule from config — checks both the
    metric `rules` section and the `structural_rules` section."""
    rules = cfg.get("rules", {})
    if rule in rules:
        return rules[rule]
    return (cfg.get("structural_rules", {}) or {}).get(rule, {})


def _rule_title(rule: str, cfg: dict) -> str:
    return _rule_meta(rule, cfg).get("title", rule)


def _rule_insight(rule: str, cfg: dict, a: str, b: str) -> str:
    tmpl = _rule_meta(rule, cfg).get("insight", "")
    try:
        return tmpl.format(a=a, b=b)
    except (KeyError, IndexError):
        return tmpl


def _evidence_rows(early: dict, recent: dict, freq_delta_pct: float, a: str, b: str) -> list[dict]:
    """Build the human-readable evidence table for a drift card."""
    def row(label, recent_val, fmt, pct):
        return {"label": label, "recent": fmt.format(recent_val), "pct": round(pct, 1)}
    return [
        row("Payload tokens",          recent["payload_avg"],
            "{:,.0f}", _safe_pct_change(early["payload_avg"], recent["payload_avg"])),
        row("Target re-invocation rate", recent["reinvocation_rate"],
            "{:.0f}%", _safe_pct_change(early["reinvocation_rate"], recent["reinvocation_rate"])),
        row("Final success rate",      recent["success_rate"],
            "{:.0f}%", _safe_pct_change(early["success_rate"], recent["success_rate"])),
    ]


def _snapshot_rows(agg: dict, pair: tuple, label_suffix: str = "") -> list[dict]:
    """Evidence rows for structural drift (new / vanished routes) — no delta,
    just the route's stats at the snapshot (recent for new, early for vanished).
    The template renders a `—` in the delta column when pct is None.
    """
    _, b = pair
    return [
        {"label": "Payload tokens",            "recent": f"{agg['payload_avg']:,.0f}",       "pct": None, "note": label_suffix},
        {"label": "Target re-invocation rate", "recent": f"{agg['reinvocation_rate']:.0f}%", "pct": None, "note": label_suffix},
        {"label": "Final success rate",        "recent": f"{agg['success_rate']:.0f}%",      "pct": None, "note": label_suffix},
    ]


def _downstream_snapshot(agg: dict, label_suffix: str = "") -> list[dict]:
    return [
        {"label": "Target agent duration",  "recent": f"{agg['target_duration_avg']/1000:.1f}s", "pct": None, "note": label_suffix},
        {"label": "Downstream rework count", "recent": f"{agg['downstream_rework_avg']:.1f}",      "pct": None, "note": label_suffix},
    ]


def _downstream_rows(early: dict, recent: dict) -> list[dict]:
    return [
        {"label": "Target agent duration",
         "recent": f"{recent['target_duration_avg']/1000:.1f}s",
         "pct":    round(_safe_pct_change(early["target_duration_avg"], recent["target_duration_avg"]), 1)},
        {"label": "Downstream rework count",
         "recent": f"{recent['downstream_rework_avg']:.1f}",
         "pct":    round(_safe_pct_change(early["downstream_rework_avg"], recent["downstream_rework_avg"]), 1)},
    ]


def _attribution(occurrences_recent: list[dict]) -> str:
    """Honest 'where did this appear?' text — no inflection-point claims for v1."""
    if not occurrences_recent:
        return ""
    # Earliest timestamp in the recent half, plus version range
    ts_first = min((o["timestamp"] for o in occurrences_recent), default="")
    versions = sorted({o["prompt_version"] for o in occurrences_recent if o.get("prompt_version") is not None})
    date_part = ts_first[:10] if ts_first else ""
    if versions:
        if len(versions) == 1:
            v_part = f"version v{versions[0]}"
        else:
            v_part = f"versions v{versions[0]}–v{versions[-1]}"
        return f"Runs starting {date_part} · {v_part}" if date_part else v_part
    return f"Runs starting {date_part}" if date_part else ""


def compute_handoff_drift(early_runs: list[dict], recent_runs: list[dict]) -> dict:
    """Top-3 handoff drift suspects across the early/recent split.

    Returns: {"suspects": [card dicts], "skipped": int}
    Rules, thresholds, weights, and wording all come from
    config/drift_rules.yaml — reloaded here on every call so edits take
    effect without a restart.
    """
    from collections import defaultdict
    from analysis.drift_config import load_drift_config

    cfg = load_drift_config()
    guards = cfg.get("guards", {})
    min_per_half  = guards.get("min_occurrences_per_half", 3)
    unspec_floor  = guards.get("unspecified_floor", 30)
    repl_min_gain = guards.get("replacement_min_share_gain", 15)

    early_occ = _collect_handoff_occurrences(early_runs)
    recent_occ = _collect_handoff_occurrences(recent_runs)

    # Bucket per (sender, receiver)
    early_by_pair: dict[tuple, list[dict]] = defaultdict(list)
    recent_by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for o in early_occ:
        early_by_pair[(o["from"], o["to"])].append(o)
    for o in recent_occ:
        recent_by_pair[(o["from"], o["to"])].append(o)

    all_pairs = set(early_by_pair) | set(recent_by_pair)
    candidates: list[dict] = []

    # Run-level success rates with/without each route — for route impact scoring.
    from analysis.run_metrics import is_clean_termination
    total_recent_runs = len(recent_runs) or 1
    recent_success_overall = (
        sum(1 for r in recent_runs if is_clean_termination(r.get("termination_reason") or ""))
        / total_recent_runs * 100
    )

    # Per-sender outgoing share, so we can detect "traffic moved from b to X".
    def _sender_shares(by_pair: dict, sender: str) -> dict:
        outgoing = {(a, b): len(v) for (a, b), v in by_pair.items() if a == sender}
        tot = sum(outgoing.values()) or 1
        return {b: cnt / tot * 100 for (a, b), cnt in outgoing.items()}

    def _find_replacement(sender: str, vanished_receiver: str) -> str:
        """For a vanished route sender→vanished_receiver, find the route from the
        same sender that GAINED the most share in the recent window.
        Returns a sentence or '' if nothing obvious replaced it."""
        early_share  = _sender_shares(early_by_pair, sender)
        recent_share = _sender_shares(recent_by_pair, sender)
        best_recv, best_gain = None, 0.0
        for recv, r_share in recent_share.items():
            if recv == vanished_receiver:
                continue
            gain = r_share - early_share.get(recv, 0.0)
            if gain > best_gain:
                best_recv, best_gain = recv, gain
        if best_recv and best_gain >= repl_min_gain:   # meaningful replacement only
            e = early_share.get(best_recv, 0.0)
            r = recent_share.get(best_recv, 0.0)
            return f"Likely replacement: {sender} → {best_recv} rose from {e:.0f}% to {r:.0f}% of {sender}'s handoffs."
        return ""

    high_drop = cfg.get("risk", {}).get("high_success_drop", 10)

    for pair in all_pairs:
        e_occ = early_by_pair.get(pair, [])
        r_occ = recent_by_pair.get(pair, [])
        e_n, r_n = len(e_occ), len(r_occ)

        # New route — show what the route looks like right now (no prior to compare)
        if e_n == 0 and r_n >= min_per_half:
            recent_with_route = {o["run_id"] for o in r_occ}
            sr_with = sum(
                1 for r in recent_runs
                if r["run_id"] in recent_with_route and is_clean_termination(r.get("termination_reason") or "")
            ) / max(len(recent_with_route), 1) * 100
            impact = 3 * abs(sr_with - recent_success_overall)
            recent_agg = _aggregate_pair(r_occ)
            new_risk = "high" if (recent_success_overall - sr_with) >= high_drop else "medium"
            # What did this new route pull traffic FROM? (sender's receiver that shrank)
            a_name, _ = pair
            early_share  = _sender_shares(early_by_pair, a_name)
            recent_share = _sender_shares(recent_by_pair, a_name)
            shrunk, drop = None, 0.0
            for recv, e_share in early_share.items():
                d = e_share - recent_share.get(recv, 0.0)
                if d > drop:
                    shrunk, drop = recv, d
            replacement = (
                f"Likely absorbed traffic from {a_name} → {shrunk} (fell {drop:.0f}% of {a_name}'s handoffs)."
                if shrunk and drop >= repl_min_gain else ""
            )
            candidates.append({
                "pair":      pair,
                "rule":      "new_route",
                "rules":     ["new_route"],
                "severity":  impact,
                "risk":      new_risk,
                "risk_reason": _risk_reason(None, None, impact, "new_route", cfg),
                "replacement": replacement,
                "early":     None,
                "recent":    recent_agg,
                "freq_pct":  100.0,
                "evidence":  _snapshot_rows(recent_agg, pair, label_suffix="(new route)"),
                "downstream":_downstream_snapshot(recent_agg, label_suffix="(new route)"),
                "attribution": _attribution(r_occ),
                "observed_label": "Observed in",
                "observed_value": "Recent window only",
            })
            continue

        # Vanished route — show what the route LOOKED LIKE before it disappeared
        if e_n >= min_per_half and r_n == 0:
            early_with_route = {o["run_id"] for o in e_occ}
            sr_was = sum(
                1 for r in early_runs
                if r["run_id"] in early_with_route and is_clean_termination(r.get("termination_reason") or "")
            ) / max(len(early_with_route), 1) * 100
            early_success_overall = (
                sum(1 for r in early_runs if is_clean_termination(r.get("termination_reason") or ""))
                / max(len(early_runs), 1) * 100
            )
            impact = 3 * abs(sr_was - early_success_overall)
            early_agg = _aggregate_pair(e_occ)
            a_name, b_name = pair
            replacement = _find_replacement(a_name, b_name)
            candidates.append({
                "pair":      pair,
                "rule":      "vanished_route",
                "rules":     ["vanished_route"],
                "severity":  impact,
                "risk":      "medium",
                "risk_reason": _risk_reason(None, None, impact, "vanished_route", cfg),
                "replacement": replacement,
                "early":     early_agg,
                "recent":    None,
                "freq_pct":  -100.0,
                "evidence":  _snapshot_rows(early_agg, pair, label_suffix="previous window"),
                "downstream":_downstream_snapshot(early_agg, label_suffix="previous window"),
                "attribution": _attribution(e_occ),
                "observed_label": "Observed in",
                "observed_value": "Previous window only",
            })
            continue

        # Metric drift path — need both halves
        if e_n < min_per_half or r_n < min_per_half:
            continue

        early_agg  = _aggregate_pair(e_occ)
        recent_agg = _aggregate_pair(r_occ)
        # Frequency delta is normalized per-run for honesty (not raw count).
        freq_early  = e_n / max(len(early_runs), 1)
        freq_recent = r_n / max(len(recent_runs), 1)
        freq_pct    = _safe_pct_change(freq_early, freq_recent)

        fired = _evaluate_rules(early_agg, recent_agg, freq_pct, cfg)

        # If nothing fired but SOMETHING is moving past the floor → Unspecified.
        if not fired:
            sig = max(
                abs(_safe_pct_change(early_agg["payload_avg"],         recent_agg["payload_avg"])),
                abs(_safe_pct_change(early_agg["target_duration_avg"], recent_agg["target_duration_avg"])),
                abs(_safe_pct_change(early_agg["reinvocation_rate"],   recent_agg["reinvocation_rate"])),
                abs(_safe_pct_change(early_agg["success_rate"],        recent_agg["success_rate"])),
                abs(freq_pct),
            )
            if sig < unspec_floor:
                continue
            fired = ["unspecified"]

        severity = _severity(early_agg, recent_agg, freq_pct, cfg)
        risk     = _classify_risk(early_agg, recent_agg, severity, cfg)
        # Pick headline rule by configured priority order.
        from analysis.drift_config import priority_order
        order = priority_order(cfg)
        headline = next((r for r in order if r in fired), fired[0])
        also     = [r for r in fired if r != headline]

        a_name, b_name = pair
        candidates.append({
            "pair":      pair,
            "rule":      headline,
            "rules":     fired,
            "also":      also,
            "severity":  round(severity, 1),
            "risk":      risk,
            "risk_reason": _risk_reason(early_agg, recent_agg, severity, headline, cfg),
            "replacement": "",
            "early":     early_agg,
            "recent":    recent_agg,
            "freq_pct":  round(freq_pct, 1),
            "evidence":  _evidence_rows(early_agg, recent_agg, freq_pct, a_name, b_name),
            "downstream":_downstream_rows(early_agg, recent_agg),
            "attribution": _attribution(r_occ),
            "observed_label": "Appeared in",
            "observed_value": None,   # uses attribution instead
        })

    candidates.sort(key=lambda c: -c["severity"])
    suspects = candidates[:3]
    for c in suspects:
        a, b = c["pair"]
        c["from"] = a
        c["to"]   = b
        c["rule_title"]    = _rule_title(c["rule"], cfg)
        c["also_titles"]   = [_rule_title(r, cfg) for r in c.get("also", [])]
        c["insight"]       = _rule_insight(c["rule"], cfg, a, b)

    # Lookup of EVERY drifted pair (not just the top 3) so the Common Handoffs
    # cards can show a small "minor drift" badge for pairs that drifted but
    # didn't make the suspect cut.
    drifted_lookup = {
        c["pair"]: {"risk": c["risk"], "rule_title": _rule_title(c["rule"], cfg)}
        for c in candidates
    }

    return {
        "suspects":       suspects,
        "drifted_lookup": drifted_lookup,
        "evaluated":      len(candidates),
        "min_required":   min_per_half,
    }


# ===========================================================================
# Agent drift — per-agent behaviour change for the Agent health cards.
# ===========================================================================
def _collect_agent_occurrences(runs: list[dict]) -> dict[str, list[dict]]:
    """For every agent, one record per appearance carrying the signals the
    agent rules need: output/input tokens, latency, whether it was re-invoked
    later in the run, the run's clean flag, and remaining run duration."""
    from analysis.run_metrics import is_clean_termination
    from collections import defaultdict

    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        spans = sorted(get_agent_spans(r["run_id"]), key=lambda s: s.get("turn_index") or 0)
        if not spans:
            continue
        clean = is_clean_termination(r.get("termination_reason") or "")
        n = len(spans)
        # Rework suffix: repeat-agent calls after each position.
        seen: set = set()
        is_repeat = [0] * n
        for j in range(n):
            nm = spans[j].get("agent_name")
            is_repeat[j] = 1 if nm in seen else 0
            seen.add(nm)
        rework_suffix = [0] * (n + 1)
        for j in range(n - 1, -1, -1):
            rework_suffix[j] = rework_suffix[j + 1] + is_repeat[j]
        # Downstream-latency suffix: total duration of spans AFTER position i.
        durations = [float(s.get("duration_ms") or 0) for s in spans]
        lat_suffix = [0.0] * (n + 1)
        for j in range(n - 1, -1, -1):
            lat_suffix[j] = lat_suffix[j + 1] + durations[j]
        # Run-level signals shared by every occurrence in this run.
        e2e_ms = float(r.get("total_duration_ms") or lat_suffix[0])
        reason = (r.get("termination_reason") or "").lower()
        is_timeout = 1 if ("timeout" in reason or "max_round" in reason) else 0

        turns_by_agent: dict[str, list[int]] = defaultdict(list)
        for s in spans:
            turns_by_agent[s.get("agent_name")].append(s.get("turn_index") or 0)
        for i, s in enumerate(spans):
            name = s.get("agent_name")
            if not name:
                continue
            t = s.get("turn_index") or 0
            reinvoked = 1 if any(x > t for x in turns_by_agent[name]) else 0
            inp, out = int(s.get("input_tokens") or 0), int(s.get("output_tokens") or 0)
            by_agent[name].append({
                "run_id":        r["run_id"],
                "timestamp":     r.get("timestamp", ""),
                "clean":         clean,
                "output_tokens": out,
                "input_tokens":  inp,
                "latency_ms":    durations[i],
                "reinvoked":     reinvoked,
                # Rework (repeat agent calls) AFTER this agent in the run.
                "downstream_rework": rework_suffix[i + 1],
                # New drift signals.
                "cost":          compute_cost(inp, out, s.get("model") or ""),
                "tool_calls":    int(s.get("tool_call_count") or 0),
                "is_error":      1 if (s.get("status") == "ERROR") else 0,
                "retries":       int(s.get("retry_count") or 0),
                "downstream_latency_ms": lat_suffix[i + 1],
                "e2e_latency_ms": e2e_ms,
                "is_timeout":    is_timeout,
            })
    return by_agent


def _agent_aggregate(occ: list[dict]) -> dict:
    n = len(occ)
    if n == 0:
        return {"count": 0}
    clean = sum(1 for o in occ if o["clean"])
    total_cost = sum(o.get("cost", 0.0) for o in occ)
    return {
        "count":          n,
        "output_avg":     _mean([o["output_tokens"] for o in occ]),
        "input_avg":      _mean([o["input_tokens"] for o in occ]),
        "latency_avg":    _mean([o["latency_ms"] for o in occ]),
        "reinvocation_rate": sum(o["reinvoked"] for o in occ) / n * 100,
        "success_rate":   clean / n * 100,
        "downstream_rework_avg": _mean([o["downstream_rework"] for o in occ]),
        # New signals for the behavior/impact classifier.
        "cost_avg":        _mean([o.get("cost", 0.0) for o in occ]),
        # Cost per successful occurrence (rises when success drops or cost grows).
        "cost_per_success": total_cost / clean if clean else total_cost,
        "tool_calls_avg":  _mean([o.get("tool_calls", 0) for o in occ]),
        "error_rate":      sum(o.get("is_error", 0) for o in occ) / n * 100,
        # % of this agent's occurrences that had at least one LLM-call retry.
        "retry_rate":      sum(1 for o in occ if o.get("retries", 0) > 0) / n * 100,
        "timeout_rate":    sum(o.get("is_timeout", 0) for o in occ) / n * 100,
        "downstream_latency_avg": _mean([o.get("downstream_latency_ms", 0.0) for o in occ]),
        "e2e_latency_avg": _mean([o.get("e2e_latency_ms", 0.0) for o in occ]),
    }


def _agent_deltas(early: dict, recent: dict) -> dict[str, float]:
    return {
        "output_change":             _safe_pct_change(early["output_avg"],     recent["output_avg"]),
        "input_change":              _safe_pct_change(early["input_avg"],      recent["input_avg"]),
        "latency_change":            _safe_pct_change(early["latency_avg"],    recent["latency_avg"]),
        "reinvocation_change":       _safe_pct_change(early["reinvocation_rate"], recent["reinvocation_rate"]),
        "success_change":            _safe_pct_change(early["success_rate"],   recent["success_rate"]),
        "downstream_rework_change":  _safe_pct_change(early["downstream_rework_avg"], recent["downstream_rework_avg"]),
    }


def _concern_tier(metric: str, pct: float, cfg: dict) -> str:
    """Return 'high' / 'medium' / 'none' for a delta — colors by RISK, not
    by raw direction. Driven by the agent_concern section of the config."""
    spec = (cfg.get("agent_concern", {}) or {}).get(metric)
    if not spec:
        return "none"
    direction = spec.get("dir", "up")
    # Only the harmful direction counts (down-metrics: drops are bad).
    if direction == "up":
        mag = pct if pct > 0 else 0
    elif direction == "down":
        mag = -pct if pct < 0 else 0
    else:  # both
        mag = abs(pct)
    if mag >= spec.get("high", 1e9):
        return "high"
    if mag >= spec.get("medium", 1e9):
        return "medium"
    return "none"


_SIGNAL_LABELS = {
    "latency_delta": "latency", "input_tokens_delta": "input tokens",
    "output_tokens_delta": "output tokens", "tool_calls_delta": "tool calls",
    "cost_delta": "cost", "final_success_delta": "final success",
    "re_invocation_delta_pp": "re-invocation", "retry_rate_delta_pp": "retry rate",
    "error_rate_delta_pp": "error rate", "downstream_rework_delta": "downstream rework",
    "downstream_latency_delta": "downstream latency",
    "end_to_end_latency_delta": "end-to-end latency",
    "cost_per_success_delta": "cost per success", "timeout_rate_delta_pp": "timeout rate",
}


def _signal_fires(spec: dict, value: float) -> bool:
    op = spec.get("op", "gte")
    v  = spec.get("value")
    if op == "gte": return value >= v
    if op == "lte": return value <= v
    if op == "gt":  return value >  v
    if op == "lt":  return value <  v
    if op == "out": return value >= spec.get("high", 1e9) or value <= spec.get("low", -1e9)
    return False


def _fired_signals(metrics: dict, group: dict) -> list[str]:
    return [m for m, spec in (group or {}).items() if _signal_fires(spec, metrics.get(m, 0.0))]


def classify_agent_risk(metrics: dict, cfg: dict) -> tuple[str, str]:
    """Behaviour/impact risk classifier (config-driven). Returns (risk, confidence).

    A behavioural change alone tops out at 'Watch'; risk escalates only when
    impact signals fire. Critical is reserved for a success collapse or an
    error/timeout spike. Thresholds live in agent_risk in config/drift_rules.yaml.
    """
    rc   = cfg.get("agent_risk", {}) or {}
    runs = metrics.get("runs", 0)
    if runs < rc.get("min_runs", 5):
        return "Insufficient data", "Low confidence"

    behavior = len(_fired_signals(metrics, rc.get("behavior", {})))
    impact   = len(_fired_signals(metrics, rc.get("impact", {})))
    crit     = rc.get("critical", {}) or {}
    high_min = rc.get("high_min_impact", 2)

    if metrics.get("final_success_delta", 0.0) <= crit.get("success_drop", -0.15):
        risk = "Critical"
    elif (metrics.get("error_rate_delta_pp", 0.0)   >= crit.get("error_spike_pp", 15)
          or metrics.get("timeout_rate_delta_pp", 0.0) >= crit.get("timeout_spike_pp", 15)):
        risk = "Critical"
    elif behavior == 0 and impact == 0:
        risk = "Stable"
    elif behavior > 0 and impact == 0:
        risk = "Watch"
    elif behavior > 0 and impact == 1:
        risk = "Medium"
    elif behavior > 0 and impact >= high_min:
        risk = "High"
    else:
        risk = "Watch"

    # Thin-data cap: under N runs, knock High/Critical to Medium unless severe.
    sc = rc.get("sample_cap", {}) or {}
    if runs < sc.get("runs_below", 20) and risk in ("High", "Critical"):
        if (metrics.get("final_success_delta", 0.0) > sc.get("keep_if_success_below", -0.10)
                and metrics.get("error_rate_delta_pp", 0.0) < sc.get("keep_if_error_pp_above", 10)):
            risk = "Medium"

    conf = rc.get("confidence", {}) or {}
    if runs < conf.get("low_below", 20):
        confidence = "Low confidence"
    elif runs < conf.get("medium_below", 50):
        confidence = "Medium confidence"
    else:
        confidence = "High confidence"
    return risk, confidence


def _classifier_metrics(early: dict, recent: dict, runs: int) -> dict:
    """Build the classifier input with correct units: *_delta = relative fraction,
    *_pp = percentage-point difference, final_success_delta = success-rate point
    change as a fraction (-0.15 = a 15-point drop)."""
    def frac(a, b):
        if a:
            return (b - a) / a
        return 0.0 if not b else 1.0
    def pp(a, b):
        return b - a
    return {
        "runs": runs,
        "latency_delta":            frac(early["latency_avg"],    recent["latency_avg"]),
        "input_tokens_delta":       frac(early["input_avg"],      recent["input_avg"]),
        "output_tokens_delta":      frac(early["output_avg"],     recent["output_avg"]),
        "tool_calls_delta":         frac(early["tool_calls_avg"], recent["tool_calls_avg"]),
        "cost_delta":               frac(early["cost_avg"],       recent["cost_avg"]),
        "final_success_delta":      (recent["success_rate"] - early["success_rate"]) / 100.0,
        "re_invocation_delta_pp":   pp(early["reinvocation_rate"], recent["reinvocation_rate"]),
        "retry_rate_delta_pp":      pp(early["retry_rate"],       recent["retry_rate"]),
        "error_rate_delta_pp":      pp(early["error_rate"],       recent["error_rate"]),
        "timeout_rate_delta_pp":    pp(early["timeout_rate"],     recent["timeout_rate"]),
        "downstream_rework_delta":  frac(early["downstream_rework_avg"], recent["downstream_rework_avg"]),
        "downstream_latency_delta": frac(early["downstream_latency_avg"], recent["downstream_latency_avg"]),
        "end_to_end_latency_delta": frac(early["e2e_latency_avg"], recent["e2e_latency_avg"]),
        "cost_per_success_delta":   frac(early["cost_per_success"], recent["cost_per_success"]),
    }


def compute_agent_drift(early_runs: list[dict], recent_runs: list[dict]) -> dict:
    """Per-agent cards.

    Returns {"drift": [...], "info": [...]}:
      - drift: agents present in BOTH halves with enough data — full drift card.
      - info:  agents present in ONLY one half — lightweight New / Retired card
               (snapshot stats, no drift %), so single-half agents stay visible
               without faking a comparison.
    Rules + concern coloring come from config.
    """
    from analysis.drift_config import load_drift_config, rule_fires

    cfg = load_drift_config()
    agent_rules = cfg.get("agent_rules", {}) or {}
    status_cfg = cfg.get("agent_status", {}) or {}
    # Agent-specific occurrence guard (defaults to 2), independent of the
    # stricter handoff guard so individual agents aren't over-filtered.
    min_per_half = status_cfg.get(
        "min_occurrences_per_half",
        cfg.get("guards", {}).get("min_occurrences_per_half", 3),
    )
    info_min     = status_cfg.get("info_min_occurrences", 2)
    drifting_min = status_cfg.get("drifting_min_rules", 2)

    early_by  = _collect_agent_occurrences(early_runs)
    recent_by = _collect_agent_occurrences(recent_runs)
    agents = sorted(set(early_by) | set(recent_by))

    def _evi(label, recent_val, fmt, pct, metric):
        return {"label": label, "recent": fmt.format(recent_val),
                "pct": round(pct, 1), "concern": _concern_tier(metric, pct, cfg)}

    def _snapshot(agg):
        return [
            {"label": "Output tokens",      "value": f"{agg['output_avg']:,.0f}"},
            {"label": "Input tokens",       "value": f"{agg['input_avg']:,.0f}"},
            {"label": "Latency",            "value": f"{agg['latency_avg']/1000:.1f}s"},
            {"label": "Final success",      "value": f"{agg['success_rate']:.0f}%"},
        ]

    cards: list[dict] = []
    info_cards: list[dict] = []
    for name in agents:
        e_occ, r_occ = early_by.get(name, []), recent_by.get(name, [])
        e_n, r_n = len(e_occ), len(r_occ)

        # Single-half agents → lightweight info card (New / Retired).
        if e_n == 0 and r_n >= info_min:
            agg = _agent_aggregate(r_occ)
            info_cards.append({
                "agent": name, "info_type": "new", "runs_seen": r_n,
                "note": "Appeared in recent runs only — no earlier baseline to compare.",
                "snapshot": _snapshot(agg),
            })
            continue
        if r_n == 0 and e_n >= info_min:
            agg = _agent_aggregate(e_occ)
            info_cards.append({
                "agent": name, "info_type": "retired", "runs_seen": e_n,
                "note": "Seen in earlier runs but not in recent ones.",
                "snapshot": _snapshot(agg),
            })
            continue

        # Real drift needs enough data in BOTH halves.
        if e_n < min_per_half or r_n < min_per_half:
            continue
        early_agg, recent_agg = _agent_aggregate(e_occ), _agent_aggregate(r_occ)
        deltas = _agent_deltas(early_agg, recent_agg)

        evidence = [
            _evi("Output tokens",      recent_agg["output_avg"],  "{:,.0f}", deltas["output_change"],  "output_change"),
            _evi("Input tokens",       recent_agg["input_avg"],   "{:,.0f}", deltas["input_change"],   "input_change"),
            _evi("Latency",            recent_agg["latency_avg"]/1000, "{:.1f}s", deltas["latency_change"], "latency_change"),
            _evi("Re-invocation rate", recent_agg["reinvocation_rate"], "{:.0f}%", deltas["reinvocation_change"], "reinvocation_change"),
        ]
        impact = [
            _evi("Final success",     recent_agg["success_rate"], "{:.0f}%", deltas["success_change"], "success_change"),
            _evi("Downstream rework", recent_agg["downstream_rework_avg"], "{:.1f}", deltas["downstream_rework_change"], "downstream_rework_change"),
        ]
        # --- Behaviour/impact risk classification (replaces the concern rollup) ---
        clf = _classifier_metrics(early_agg, recent_agg, e_n + r_n)
        risk_label, confidence = classify_agent_risk(clf, cfg)
        risk_cfg = cfg.get("agent_risk", {}) or {}
        behavior_fired = _fired_signals(clf, risk_cfg.get("behavior", {}))
        impact_fired   = _fired_signals(clf, risk_cfg.get("impact", {}))
        status = {"Critical": "critical", "High": "high", "Medium": "medium",
                  "Watch": "watch", "Stable": "stable",
                  "Insufficient data": "stable"}.get(risk_label, "stable")
        b_labels = [_SIGNAL_LABELS.get(s, s) for s in behavior_fired]
        i_labels = [_SIGNAL_LABELS.get(s, s) for s in impact_fired]

        # Flag the evidence/impact rows that are abnormal (their classifier signal
        # fired) so the card can highlight just those in red.
        fired_set = set(behavior_fired) | set(impact_fired)
        _row_signal = {
            "Output tokens": "output_tokens_delta", "Input tokens": "input_tokens_delta",
            "Latency": "latency_delta", "Re-invocation rate": "re_invocation_delta_pp",
            "Final success": "final_success_delta", "Downstream rework": "downstream_rework_delta",
        }
        for row in evidence + impact:
            row["abnormal"] = _row_signal.get(row["label"]) in fired_set

        # Insight from the fired signals — behaviour change vs. outcome impact.
        if status == "stable" or (not b_labels and not i_labels):
            insight = f"{name} is behaving consistently with earlier in the window."
        elif i_labels:
            insight = (f"{name} changed ({', '.join(b_labels) or 'behaviour'}) and it shows in "
                       f"outcomes ({', '.join(i_labels)}).")
        else:
            insight = f"{name} changed ({', '.join(b_labels)}) but outcomes held steady."

        main = max(evidence + impact, key=lambda r: abs(r["pct"]))
        cards.append({
            "agent":      name,
            "status":     status,
            "risk_label": risk_label,
            "confidence": confidence,
            "type":       " + ".join((i_labels or b_labels)[:3]) if (i_labels or b_labels) else "—",
            "runs_seen":  e_n + r_n,
            "insight":    insight,
            "main_change": f'{main["label"].lower()} {"+" if main["pct"]>=0 else ""}{main["pct"]:.0f}%',
            "evidence":   evidence,
            "impact":     impact,
            "behavior_fired": b_labels,
            "impact_fired":   i_labels,
            "n_behavior": len(behavior_fired),
            "n_impact":   len(impact_fired),
            "n_fired":    len(behavior_fired) + len(impact_fired),
        })

    # Order: critical → high → medium → watch → stable; most signals first.
    order = {"critical": 0, "high": 1, "medium": 2, "watch": 3, "stable": 4}
    cards.sort(key=lambda c: (order[c["status"]], -c["n_fired"]))
    # Info cards: new agents first, then retired; alphabetical within each.
    info_cards.sort(key=lambda c: (0 if c["info_type"] == "new" else 1, c["agent"]))
    return {"drift": cards, "info": info_cards}


def compute_path_summary(early_runs: list[dict], recent_runs: list[dict]) -> dict:
    """High-level path metrics (no drift rules — just a snapshot + delta).

    A 'path' is the ordered tuple of agent names in a run.
      avg_path_length     — mean agents per run
      unique_paths        — distinct path shapes in the window
      reinvocation_loops  — runs where some agent ran more than once
      most_changed_route  — the path whose frequency moved most early→recent
    """
    from collections import Counter

    def path_of(r):
        spans = sorted(get_agent_spans(r["run_id"]), key=lambda s: s.get("turn_index") or 0)
        return tuple(s.get("agent_name") for s in spans if s.get("agent_name"))

    e_items = [(path_of(r), r) for r in early_runs]
    e_items = [(p, r) for p, r in e_items if p]
    r_items = [(path_of(r), r) for r in recent_runs]
    r_items = [(p, r) for p, r in r_items if p]
    e_paths = [p for p, _ in e_items]
    r_paths = [p for p, _ in r_items]

    def avg_len(paths):
        return (sum(len(p) for p in paths) / len(paths)) if paths else 0.0
    def loops(paths):
        return sum(1 for p in paths if len(p) != len(set(p)))

    e_avg, r_avg     = avg_len(e_paths), avg_len(r_paths)
    e_uniq, r_uniq   = len(set(e_paths)), len(set(r_paths))
    e_loops, r_loops = loops(e_paths), loops(r_paths)

    # Most changed route: biggest |recent − early| occurrence delta.
    e_ctr, r_ctr = Counter(e_paths), Counter(r_paths)
    most_changed, best = None, -1
    for p in set(e_ctr) | set(r_ctr):
        d = abs(r_ctr.get(p, 0) - e_ctr.get(p, 0))
        if d > best:
            best, most_changed = d, p

    route = None
    if most_changed:
        on_early  = [r for p, r in e_items if p == most_changed]
        on_recent = [r for p, r in r_items if p == most_changed]
        ce = _mean([(r.get("total_cost_usd") or 0) for r in on_early])
        cr = _mean([(r.get("total_cost_usd") or 0) for r in on_recent])
        cost_pct = round(_safe_pct_change(ce, cr), 0) if (on_early and on_recent) else None
        route = {
            "from":     most_changed[0],
            "to":       most_changed[-1],
            "agents":   len(most_changed),
            "handoffs": max(len(most_changed) - 1, 0),
            "cost_pct": cost_pct,
            "chain":    list(most_changed),
        }

    return {
        "avg_path_length":    {"value": round(r_avg, 1), "pct": round(_safe_pct_change(e_avg, r_avg), 0)},
        "unique_paths":       {"value": r_uniq,          "pct": round(_safe_pct_change(e_uniq, r_uniq), 0)},
        "reinvocation_loops": {"value": r_loops,         "pct": round(_safe_pct_change(e_loops, r_loops), 0)},
        "most_changed_route": route,
    }


def compute_parallel_health_summary(early_runs: list[dict], recent_runs: list[dict]) -> dict:
    """Parallel-group health, recent half vs early half (same split as the rest
    of System Health). For each run we take its *primary* parallel group (the one
    with the longest wall-clock) and read three metrics, defined exactly as the
    run-detail parallel card shows them so the numbers line up:

      bottleneck_ms — slowest branch's duration (gates the join)
      join_wait_ms  — wall_clock − fastest branch (how long the quick branch idled)
      efficiency    — group.efficiency (1.0 = perfectly balanced fan-out)

    Runs with no parallel group are skipped. If either half ends up with no
    parallel runs, the section reports itself unavailable rather than guessing.
    """
    def primary_group(r):
        groups = detect_parallel_groups(get_agent_spans(r["run_id"]))
        return max(groups, key=lambda g: g.wall_clock_ms) if groups else None

    def collect(runs):
        bott, wait, eff = [], [], []
        for r in runs:
            g = primary_group(r)
            if g is None or not g.branches:
                continue
            bott.append(g.bottleneck.duration_ms)
            wait.append(max(g.wall_clock_ms - min(b.duration_ms for b in g.branches), 0.0))
            eff.append(g.efficiency)
        return bott, wait, eff

    e_bott, e_wait, e_eff = collect(early_runs)
    r_bott, r_wait, r_eff = collect(recent_runs)

    if not (e_bott and r_bott):
        return {"available": False, "n_early": len(e_bott), "n_recent": len(r_bott)}

    def card(early_vals, recent_vals):
        e, r = _mean(early_vals), _mean(recent_vals)
        return {"early": round(e, 3), "recent": round(r, 3),
                "pct": round(_safe_pct_change(e, r), 0)}

    return {
        "available":     True,
        "n_early":       len(e_bott),
        "n_recent":      len(r_bott),
        "bottleneck_ms": card(e_bott, r_bott),
        "join_wait_ms":  card(e_wait, r_wait),
        "efficiency":    card(e_eff, r_eff),
    }


def compute_versions(runs: list[dict]) -> list[dict]:
    """Map runs to manual version snapshots and build the horizontal comparison:
    per-version cohort metrics + the config diff vs the previous version.

    Version 1 is the implicit baseline (runs before the first snapshot). Each
    snapshot in the versions table starts a new version; a run belongs to the
    latest version whose created_at <= run.timestamp.
    """
    versions = get_versions()
    runs_sorted = sorted(runs, key=lambda r: r.get("timestamp", ""))

    def ver_of(r):
        v = 1
        for ver in versions:
            if r.get("timestamp", "") >= (ver["created_at"] or ""):
                v = ver["version_num"]
            else:
                break
        return v

    entries: dict[int, dict] = {1: {"num": 1, "label": "Version 1 (baseline)",
                                    "config": None, "runs": [], "created_at": ""}}
    for ver in versions:
        entries[ver["version_num"]] = {
            "num": ver["version_num"], "label": ver["label"] or f"Version {ver['version_num']}",
            "config": json.loads(ver["config_json"] or "{}"), "runs": [],
            "created_at": ver["created_at"] or "",
        }
    for r in runs_sorted:
        entries.setdefault(ver_of(r), {"num": ver_of(r), "label": f"Version {ver_of(r)}",
                                       "config": None, "runs": []})
        entries[ver_of(r)]["runs"].append(r)

    out = []
    for num in sorted(entries):
        e = entries[num]
        rs = e["runs"]
        cfg = e["config"]
        if not cfg and rs:                       # implicit version: use latest run's config
            cfg = _parse_cfg(rs[-1]) or {}
        n = len(rs)
        out.append({
            "num":   num,
            "label": e["label"],
            "created_at": e.get("created_at", ""),
            "runs":  n,
            "avg_cost":     round(_mean([r.get("total_cost_usd") or 0 for r in rs]), 4) if n else 0,
            "success_rate": round(sum(1 for r in rs if is_clean_termination(
                                r.get("termination_reason") or "")) / n * 100, 0) if n else 0,
            "avg_tokens":   round(_mean([(r.get("total_input_tokens") or 0)
                                + (r.get("total_output_tokens") or 0) for r in rs])) if n else 0,
            "config": cfg or {},
        })
    for i in range(1, len(out)):
        out[i]["diff"] = diff_configs(out[i - 1]["config"], out[i]["config"])
    return out


def _cohort_metrics(task_type: str, version: int, all_runs: list[dict]) -> list[dict]:
    """Metrics for every run of a given task_type + prompt_version."""
    return [
        _metrics_for_run(r) for r in all_runs
        if r.get("task_type") == task_type and r.get("prompt_version") == version
    ]


def build_timeline(agent_spans: list[dict], tool_calls: list[dict] | None = None) -> list[dict]:
    """Build a trace-waterfall: each agent turn as a positioned bar on a shared
    time axis. This is the standard 'what happened' view in LangSmith/Datadog/etc.
    """
    if not agent_spans:
        return []
    tool_calls = tool_calls or []
    span_turn = {s.get("span_id"): s.get("turn_index") for s in agent_spans}
    tools_by_turn: dict = {}
    for tc in tool_calls:
        ti = span_turn.get(tc.get("span_id"))
        if ti is None:
            continue
        d = tools_by_turn.setdefault(ti, {})
        d[tc["tool_name"]] = d.get(tc["tool_name"], 0) + 1

    spans = sorted(agent_spans, key=lambda s: s.get("start_time_ms") or 0)
    t0 = min(s.get("start_time_ms") or 0 for s in spans)
    t_end = max(s.get("end_time_ms") or 0 for s in spans)
    total = (t_end - t0) or 1

    bars = []
    for s in spans:
        start = s.get("start_time_ms") or 0
        dur   = s.get("duration_ms") or 0
        inp   = s.get("input_tokens") or 0
        out   = s.get("output_tokens") or 0
        ti    = s.get("turn_index")
        bars.append({
            "agent":         s.get("agent_name"),
            "turn_index":    ti,
            "offset_pct":    round((start - t0) / total * 100, 2),
            "width_pct":     round(max(dur / total * 100, 2.0), 2),
            "duration_s":    round(dur / 1000, 1),
            "input_tokens":  inp,
            "output_tokens": out,
            "cost":          compute_cost(inp, out, s.get("model") or ""),
            "tool_call_count": s.get("tool_call_count") or 0,
            "tools":         tools_by_turn.get(ti, {}),
            "status_value":  s.get("status_value") or "",
            "model":         s.get("model") or "",
        })
    return bars


_NODE_STYLES = {
    "normal":   {"bg": "#eef2ff", "border": "#3949ab", "bw": 1},
    "approved": {"bg": "#d4edda", "border": "#28a745", "bw": 3},
    "failed":   {"bg": "#f8d7da", "border": "#dc3545", "bw": 3},
    "drift":    {"bg": "#ffffff", "border": "#dc3545", "bw": 4},
    "loop":     {"bg": "#fff3cd", "border": "#fd7e14", "bw": 3},
}


def _node_label(bar: dict, is_last: bool, clean: bool, termination_short: str) -> str:
    """Node label: just the agent name. All detail lives in the hover tooltip."""
    return bar["agent"]


def _node_tooltip(bar: dict, cls: str, retries: int = 0) -> str:
    summary = {
        "approved": "Finished the run — work approved.",
        "failed":   "The run ended here without a clean finish.",
        "drift":    "This agent behaved differently from the baseline.",
        "loop":     "This agent was re-invoked unexpectedly (a loop).",
        "normal":   "Ran once and handed off normally.",
    }.get(cls, "")
    tool_line = (", ".join(f"{t} ×{c}" for t, c in bar["tools"].items())
                 if bar["tools"] else "none")
    in_tok  = int(bar.get("input_tokens")  or 0)
    out_tok = int(bar.get("output_tokens") or 0)
    if in_tok > 0:
        ratio_str = f"{(out_tok / in_tok):.2f}×"
    else:
        ratio_str = "—"
    rows = [
        ("Read (input)",     f"{in_tok:,} tokens"),
        ("Wrote (output)",   f"{out_tok:,} tokens"),
        ("Expansion ratio",  ratio_str),
        ("Took",             f"{bar['duration_s']} seconds"),
    ]
    if retries > 0:
        rows.append(("Retries", f"{retries}"))
    rows.append(("Tools called", tool_line))
    body = "".join(
        f"<tr><td style='color:#9aa0c7;padding:2px 14px 2px 0;white-space:nowrap'>{k}</td>"
        f"<td style='font-weight:600;color:#fff'>{v}</td></tr>"
        for k, v in rows
    )
    return (
        f"<div style='min-width:210px'>"
        f"<div style='font-weight:700;font-size:0.95rem;color:#fff;margin-bottom:3px'>"
        f"{bar['agent']} <span style='color:#9aa0c7;font-weight:400'>· step {bar['turn_index']+1}</span></div>"
        f"<div style='color:#c9c9ff;font-size:0.78rem;margin-bottom:7px'>{summary}</div>"
        f"<table style='font-size:0.82rem;border-collapse:collapse'>{body}</table></div>"
    )


def _edge_tooltip(a_agent: str, b_agent: str, h: dict | None,
                  receiver_outcome: str = "—",
                  downstream_clean: bool | None = None,
                  termination_short: str = "",
                  is_terminal_edge: bool = False) -> str:
    if not h:
        return (f"<div style='min-width:180px'><div style='font-weight:700;color:#fff'>"
                f"{a_agent} → {b_agent}</div>"
                f"<div style='color:#c9c9ff;margin-top:4px;font-size:0.82rem'>"
                f"The same agent kept working — no handoff.</div></div>")

    a_out = int(h.get("a_output_tokens", 0) or 0)
    b_in  = int(h.get("b_input_tokens", 0) or 0)
    kind  = "↩ requested loop-back" if h.get("was_requested") else "→ forward handoff"

    if downstream_clean is True:
        down_html = "<span style='color:#7dd87d'>✓ run completed cleanly</span>"
    elif downstream_clean is False and is_terminal_edge:
        down_html = f"<span style='color:#ff9a9a'>✗ failed here ({termination_short})</span>"
    elif downstream_clean is False:
        down_html = f"<span style='color:#ff9a9a'>✗ failed downstream ({termination_short})</span>"
    else:
        down_html = "<span style='color:#9aa0c7'>—</span>"

    return (
        f"<div style='min-width:240px;max-width:320px'>"
        f"<div style='font-weight:700;font-size:0.92rem;color:#fff;margin-bottom:4px;"
        f"border-bottom:1px solid #3a3a66;padding-bottom:4px'>{a_agent} → {b_agent}</div>"
        f"<div style='color:#9aa0c7;font-size:0.78rem;margin-bottom:5px'>{kind}</div>"
        f"<table style='font-size:0.82rem;border-collapse:collapse'>"
        f"<tr><td style='color:#9aa0c7;padding:2px 14px 2px 0;white-space:nowrap'>Payload size</td>"
        f"<td style='font-weight:600;color:#fff'>{a_out:,} tokens out · {b_in:,} read</td></tr>"
        f"<tr><td style='color:#9aa0c7;padding:2px 14px 2px 0'>Receiver outcome</td>"
        f"<td style='font-weight:600;color:#fff'>{receiver_outcome}</td></tr>"
        f"<tr><td style='color:#9aa0c7;padding:2px 14px 2px 0'>Downstream</td>"
        f"<td style='font-weight:600'>{down_html}</td></tr>"
        f"</table></div>"
    )


# Agent-level node styling for the path-change DAG diff (Drift Investigation
# → path tab). Mirrors the run-explorer palette but the semantics are diff-based:
# green = added in the newer run, red = removed, blue = unchanged.
_PATH_NODE_STYLES = {
    "normal":  {"bg": "#eef2ff", "border": "#3949ab", "bw": 1},
    "added":   {"bg": "#d4edda", "border": "#28a745", "bw": 2},
    "removed": {"bg": "#f8d7da", "border": "#dc3545", "bw": 2},
}


def build_path_graph(nodes, edges, *, added_nodes=(), removed_nodes=(),
                     added_edges=(), removed_edges=()) -> dict:
    """vis-network nodes+edges for an agent-level chain, with optional diff
    highlighting. `nodes` is an iterable of agent names; `edges` an iterable of
    [from, to] pairs. Added/removed sets colour the diff (green/red); removed
    edges are dashed. Used by the path tab to draw chains as DAG flowcharts."""
    added_n, removed_n = set(added_nodes), set(removed_nodes)
    added_e   = {tuple(e) for e in added_edges}
    removed_e = {tuple(e) for e in removed_edges}

    vnodes = []
    for name in nodes:
        state = "added" if name in added_n else ("removed" if name in removed_n else "normal")
        st = _PATH_NODE_STYLES[state]
        vnodes.append({
            "id": name, "label": name,
            "color": {"background": st["bg"], "border": st["border"],
                      "highlight": {"background": st["bg"], "border": st["border"]}},
            "borderWidth": st["bw"],
        })

    vedges = []
    for e in edges:
        a, b = e[0], e[1]
        t = (a, b)
        if t in added_e:
            col, w, dash = "#28a745", 2, False
        elif t in removed_e:
            col, w, dash = "#dc3545", 1, True
        else:
            col, w, dash = "#b0b6c3", 1, False
        vedges.append({"from": a, "to": b, "dashes": dash, "width": w,
                       "color": {"color": col, "highlight": col}})
    return {"nodes": vnodes, "edges": vedges}


def build_graph_data(timeline: list[dict], handoffs: list[dict], drifted_agents: set,
                     loop_turns: set, clean: bool, termination_short: str = "",
                     agent_spans: list[dict] | None = None,
                     critical_span_ids: set | None = None) -> dict:
    """Build vis-network nodes + edges with rich HTML tooltips.

    If `agent_spans` contain DAG fields (`parent_step_id`), edges are drawn
    from the real DAG — supporting fan-out / fan-in / parallel branches.
    Otherwise we fall back to a linear chain from the timeline (existing
    behaviour for sequential systems).

    Nodes on the workflow critical path get a thick red border.
    """
    if not timeline:
        return {"nodes": [], "edges": []}

    agent_spans       = agent_spans or []
    critical_span_ids = critical_span_ids or set()
    # has_dag: any structural info present — parent_step_id (explicit DAG edges)
    # OR branch_id (sibling fan-out marker, used when the orchestrator knows
    # "these ran in parallel" but didn't name the parent step).
    has_dag = any(s.get("parent_step_id") or s.get("branch_id") for s in agent_spans)

    # Map turn_index → span_id so timeline bars can be cross-referenced with DAG spans.
    span_by_turn: dict[int, dict] = {s.get("turn_index"): s for s in agent_spans}

    n = len(timeline)
    nodes = []
    # Pre-compute retries-so-far for each turn: how many prior turns had the
    # SAME agent name. retries = 0 means first invocation, 1 means second, etc.
    seen_agent_counts: dict[str, int] = {}
    retries_by_turn: dict[int, int] = {}
    for bar in sorted(timeline, key=lambda b: b.get("turn_index") or 0):
        name = bar["agent"]
        retries_by_turn[bar["turn_index"]] = seen_agent_counts.get(name, 0)
        seen_agent_counts[name] = seen_agent_counts.get(name, 0) + 1

    # turn_index → reported status string (used by edge tooltips)
    status_by_turn: dict[int, str] = {
        b["turn_index"]: (b.get("status_value") or "—") for b in timeline
    }

    for k, bar in enumerate(timeline):
        i = bar["turn_index"]
        is_last = (k == n - 1)
        span = span_by_turn.get(i, {})
        on_critical = span.get("span_id") in critical_span_ids

        if is_last:
            cls = "approved" if clean else "failed"
        elif bar["agent"] in drifted_agents:
            cls = "drift"
        elif i in loop_turns:
            cls = "loop"
        else:
            cls = "normal"
        st = _NODE_STYLES[cls]

        # Critical-path styling overrides border to red & thick (keep fill from cls).
        if on_critical and has_dag:
            border = "#dc3545"
            bw = max(st["bw"], 4)
        else:
            border = st["border"]
            bw = st["bw"]

        nodes.append({
            "id":    f"T{i}",
            "label": _node_label(bar, is_last, clean, termination_short),
            "title": _node_tooltip(bar, cls, retries_by_turn.get(i, 0)) +
                     ("<div style='margin-top:5px;color:#ff7a7a;font-weight:600;font-size:0.78rem'>● on workflow critical path</div>"
                      if on_critical and has_dag else ""),
            "color": {"background": st["bg"], "border": border,
                      "highlight": {"background": st["bg"], "border": border}},
            "borderWidth": bw,
        })

    edges = []
    if has_dag:
        # Real DAG. Three cases per span when picking its predecessor:
        #   1. parent_step_id is set → use it directly (explicit edge).
        #   2. branch_id is set, no parent → this is a parallel branch of an
        #      implied parent: connect from the most recent non-branch span
        #      (the "stem" before the fan-out). If none exists, the branch
        #      stays a root and renders side-by-side with its siblings.
        #   3. Neither set → sequential span: connect from the immediately
        #      previous span by turn_index (preserves chain semantics for
        #      runs that mix sequential + parallel sections).
        span_id_to_turn  = {s.get("span_id"): s.get("turn_index") for s in agent_spans}
        sorted_by_turn   = sorted(agent_spans, key=lambda s: s.get("turn_index") or 0)
        # Pre-compute, for each index, the most recent "stem" span (no branch_id).
        stem_id_at: list = []
        last_stem_id = None
        for s in sorted_by_turn:
            if not s.get("branch_id"):
                stem_id_at.append(s.get("span_id"))
                last_stem_id = s.get("span_id")
            else:
                stem_id_at.append(last_stem_id)

        # Look up the per-turn bar in `timeline` for payload + downstream data.
        bar_by_turn: dict[int, dict] = {b["turn_index"]: b for b in timeline}
        max_turn = max((b["turn_index"] for b in timeline), default=-1)

        def _edge(from_turn, to_turn, to_span):
            if from_turn is None or to_turn is None or from_turn == to_turn:
                return
            from_critical = any(
                sp.get("turn_index") == from_turn and sp.get("span_id") in critical_span_ids
                for sp in agent_spans
            )
            to_critical = to_span.get("span_id") in critical_span_ids
            both_critical = from_critical and to_critical

            # Build the rich edge tooltip — payload, receiver outcome, downstream.
            from_agent = _lookup_agent(agent_spans, from_turn)
            to_agent   = to_span.get("agent_name", "?")
            from_bar   = bar_by_turn.get(from_turn, {})
            to_bar     = bar_by_turn.get(to_turn, {})
            payload    = int(from_bar.get("output_tokens") or 0)
            recv_in    = int(to_bar.get("input_tokens") or 0)
            recv_out   = status_by_turn.get(to_turn, "—")
            # Downstream success: was the run clean AND did it actually continue
            # past this receiver?  If clean → ✓; if not and the receiver is the
            # last turn → failed here; else failed further down.
            if clean:
                downstream = "<span style='color:#7dd87d'>✓ run completed cleanly</span>"
            elif to_turn == max_turn:
                downstream = f"<span style='color:#ff9a9a'>✗ failed here ({termination_short})</span>"
            else:
                downstream = f"<span style='color:#ff9a9a'>✗ failed downstream ({termination_short})</span>"
            branch_line = (
                f"<div style='color:#c9c9ff;font-size:0.78rem;margin-top:2px'>"
                f"branch <code>{to_span.get('branch_id')}</code></div>"
                if to_span.get("branch_id") else
                ""
            )
            tooltip = (
                f"<div style='min-width:240px'>"
                f"<div style='font-weight:700;color:#fff;border-bottom:1px solid #3a3a66;padding-bottom:4px;margin-bottom:5px'>"
                f"{from_agent} → {to_agent}</div>"
                f"{branch_line}"
                f"<table style='font-size:0.82rem;border-collapse:collapse;margin-top:4px'>"
                f"<tr><td style='color:#9aa0c7;padding:2px 14px 2px 0;white-space:nowrap'>Payload size</td>"
                f"<td style='font-weight:600;color:#fff'>{payload:,} tokens out · {recv_in:,} read</td></tr>"
                f"<tr><td style='color:#9aa0c7;padding:2px 14px 2px 0'>Receiver outcome</td>"
                f"<td style='font-weight:600;color:#fff'>{recv_out}</td></tr>"
                f"<tr><td style='color:#9aa0c7;padding:2px 14px 2px 0'>Downstream</td>"
                f"<td style='font-weight:600'>{downstream}</td></tr>"
                f"</table></div>"
            )

            edges.append({
                "from":  f"T{from_turn}",
                "to":    f"T{to_turn}",
                "label": "",
                "title": tooltip,
                "dashes": False,
                "color":  {"color": "#dc3545" if both_critical else "#b0b6c3",
                           "highlight": "#dc3545" if both_critical else "#3949ab"},
                "width":  3 if both_critical else 1,
            })

        # Are there parent-less parallel branches? If so, add a virtual "Run
        # start" anchor node so vis-network renders them as a fan-out
        # (otherwise isolated nodes stack as a vertical column, hiding the
        # parallel structure).
        rootless_branches = [
            s for idx, s in enumerate(sorted_by_turn)
            if s.get("branch_id")
            and not s.get("parent_step_id")
            and (idx == 0 or not stem_id_at[idx - 1])
        ]
        virtual_root_id = None
        if rootless_branches:
            virtual_root_id = "TROOT"
            nodes.insert(0, {
                "id":    virtual_root_id,
                "label": "▶ Run start",
                "title": "Virtual anchor — the run started here; the agents below "
                         "ran in parallel as independent branches.",
                "color": {"background": "#f1f3f5", "border": "#adb5bd",
                          "highlight": {"background": "#f1f3f5", "border": "#adb5bd"}},
                "borderWidth": 1,
                "font": {"size": 12, "color": "#6c757d",
                         "face": "'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"},
                "shape": "ellipse",
            })

        for idx, s in enumerate(sorted_by_turn):
            to_turn = s.get("turn_index")
            if to_turn is None:
                continue

            parent_id = s.get("parent_step_id")

            # Case 1: explicit DAG edge
            if parent_id:
                _edge(span_id_to_turn.get(parent_id), to_turn, s)
                continue

            # Case 2: branch_id set → connect from the most recent stem span,
            # or from the virtual root if no stem exists.
            if s.get("branch_id"):
                stem = stem_id_at[idx - 1] if idx > 0 else None
                if stem:
                    _edge(span_id_to_turn.get(stem), to_turn, s)
                elif virtual_root_id is not None:
                    # Edge directly from the virtual root, no critical-path styling.
                    edges.append({
                        "from": virtual_root_id, "to": f"T{to_turn}",
                        "label": "", "dashes": True,
                        "color": {"color": "#ced4da", "highlight": "#3949ab"},
                        "width": 1,
                        "title": f"<div style='color:#fff'>parallel branch: "
                                 f"<b>{s.get('agent_name')}</b></div>",
                    })
                continue

            # Case 3: sequential span (no DAG fields). Look at what immediately
            # precedes it — if the preceding run is a parallel group, this span
            # acts as the implicit join: connect from EVERY parallel branch.
            preceding_branches = []
            for j in range(idx - 1, -1, -1):
                prev = sorted_by_turn[j]
                if prev.get("branch_id"):
                    preceding_branches.append(prev)
                else:
                    break
            if preceding_branches:
                for branch in preceding_branches:
                    _edge(branch.get("turn_index"), to_turn, s)
            elif idx > 0:
                _edge(sorted_by_turn[idx - 1].get("turn_index"), to_turn, s)
        # Also draw a virtual edge into the join (from any branch that has join_step_id)
        join_groups: dict = {}
        for s in agent_spans:
            j = s.get("join_step_id")
            if not j:
                continue
            join_groups.setdefault(j, []).append(s)
        span_by_id = {s.get("span_id"): s for s in agent_spans}
        for join_id, branch_heads in join_groups.items():
            to_turn = span_id_to_turn.get(join_id)
            join_span = span_by_id.get(join_id)
            if to_turn is None or join_span is None:
                continue
            # Walk each branch to its deepest terminal (before the join), then
            # draw the edge through the SAME builder as every other edge so the
            # fan-in tooltips carry payload / receiver outcome / downstream too —
            # not a stripped-down "agent → agent (join)" label.
            for head in branch_heads:
                terminals = _branch_terminals(head, agent_spans, join_id)
                for term in terminals:
                    from_turn = span_id_to_turn.get(term["span_id"])
                    _edge(from_turn, to_turn, join_span)
    else:
        # No DAG: fall back to linear handoff edges (existing behavior).
        hlookup = {(h.get("turn_index_from"), h.get("turn_index_to")): h for h in (handoffs or [])}
        last_turn = timeline[-1]["turn_index"] if timeline else -1
        for k in range(1, n):
            a, b = timeline[k - 1], timeline[k]
            ai, bi = a["turn_index"], b["turn_index"]
            h = hlookup.get((ai, bi))
            requested = bool(h and h.get("was_requested"))
            label = "↩ sent back" if requested else ""
            edges.append({
                "from":  f"T{ai}",
                "to":    f"T{bi}",
                "label": label,
                "title": _edge_tooltip(
                    a["agent"], b["agent"], h,
                    receiver_outcome=status_by_turn.get(bi, "—"),
                    downstream_clean=clean,
                    termination_short=termination_short,
                    is_terminal_edge=(bi == last_turn),
                ),
                "dashes": requested,
            })

    return {"nodes": nodes, "edges": edges}


def _lookup_agent(spans: list[dict], turn_index: int) -> str:
    for s in spans:
        if s.get("turn_index") == turn_index:
            return s.get("agent_name") or "?"
    return "?"


def _branch_terminals(branch_head: dict, all_spans: list[dict],
                      stop_at_join: str | None) -> list[dict]:
    """Return the spans inside a branch that have no children (or whose only child is the join)."""
    children_map: dict = {}
    for s in all_spans:
        p = s.get("parent_step_id")
        if p:
            children_map.setdefault(p, []).append(s)

    visited = set()
    queue = [branch_head]
    terminals = []
    while queue:
        node = queue.pop(0)
        sid = node["span_id"]
        if sid in visited:
            continue
        visited.add(sid)
        children = [c for c in children_map.get(sid, []) if c["span_id"] != stop_at_join]
        if not children:
            terminals.append(node)
        else:
            queue.extend(children)
    return terminals

app = Flask(__name__, template_folder="templates")


# ---------------------------------------------------------------------------
# Multi-database support
# ---------------------------------------------------------------------------
# Each request picks which SQLite file to read from. Priority order:
#   1. ?db=<name> query string on the URL                  (explicit)
#   2. obs_db cookie (set by /switch-db/<name>)            (sticky choice)
#   3. DEFAULT_DB                                          (fallback)
#
# All read functions in analysis.layer1_raw go through storage.get_connection,
# which respects the ContextVar set here.

@app.before_request
def _activate_db_for_request():
    # Try in order: ?db=, cookie, the first real DB on disk, then DEFAULT_DB.
    # Drop a stale cookie that points at a DB file that no longer exists.
    available = list_available_dbs()
    candidate = request.args.get("db") or request.cookies.get("obs_db")
    if candidate and candidate not in available:
        candidate = None
    db_name = candidate or (available[0] if available else DEFAULT_DB)
    g.current_db = db_name
    set_active_db_path(resolve_db_path(db_name))


# All stored timestamps are UTC ISO strings; the UI shows US Eastern (EDT/EST).
_DISPLAY_TZ = "America/New_York"


def _to_eastern(ts):
    """Parse a UTC ISO timestamp and return an Eastern-time aware datetime (or None)."""
    if not ts:
        return None
    try:
        import datetime as _dt
        from zoneinfo import ZoneInfo
        d = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d.astimezone(ZoneInfo(_DISPLAY_TZ))
    except Exception:
        return None


# One professional vocabulary for engine severities, shared by the finding badges
# and the Home "Drift Status" (Critical/Warning) so they always read the same.
_SEV_LABEL = {"high": "Critical", "drift": "Warning", "candidate": "Watch", "watch": "Info"}


@app.template_filter("sevlabel")
def _sevlabel(sev):
    return _SEV_LABEL.get(sev, (sev or "").capitalize())


@app.template_filter("et")
def _fmt_et(ts, fmt="%Y-%m-%d %H:%M"):
    """Jinja filter: format a UTC ISO timestamp in US Eastern time."""
    d = _to_eastern(ts)
    if d is None:
        return (str(ts)[:16].replace("T", " ") if ts else "—")
    return d.strftime(fmt)


def _tz_label():
    import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        return _dt.datetime.now(ZoneInfo(_DISPLAY_TZ)).strftime("%Z")    # EDT / EST
    except Exception:
        return "ET"


@app.context_processor
def _inject_db_chrome():
    # Available everywhere in templates: {{ current_db }} and {{ available_dbs }}
    return {
        "current_db":    getattr(g, "current_db", DEFAULT_DB),
        "available_dbs": list_available_dbs(),
        "tz_label":      _tz_label(),
    }


@app.route("/switch-db/<name>")
def switch_db(name: str):
    """Set the active DB cookie and land on the requested view.

    ?next=trends  → System Health     ?next=index (default) → Run history
    Preserves the current view when switching projects instead of always
    bouncing back to the run list.
    """
    from flask import make_response, redirect, url_for, request
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or DEFAULT_DB
    nxt = request.args.get("next", "index")
    _routes = {"trends": "trends", "drift2": "drift_investigation_2",
               "metrics": "drift_investigation_2",        # legacy alias → Drift Investigation
               "timeline": "timeline_view",
               "explore": "explore", "overview": "overview", "changelog": "changelog"}
    target = url_for(_routes[nxt]) if nxt in _routes else url_for("index")
    resp = make_response(redirect(target))
    resp.set_cookie("obs_db", safe, max_age=60 * 60 * 24 * 365)  # 1 year
    return resp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_run_insights(run: dict, metrics: dict, all_runs: list[dict],
                       baseline_metrics: list[dict]):
    """Single source of truth for a run's headline insights.

    Used by BOTH the run-history badge and the single-run page so the two can
    never disagree. Combines within-run insights, version-drift insights, and
    this-run-vs-baseline anomaly insights, deduped by title and ranked by
    severity. Returns (insights, vreport, base_version) — the run page also
    needs vreport/base_version for its version-drift section.
    """
    task_type   = run.get("task_type")
    cur_version = run.get("prompt_version", 1)
    versions = sorted({
        r.get("prompt_version") for r in all_runs if r.get("task_type") == task_type
    })
    base_version = versions[0] if versions else cur_version

    vreport = None
    version_ins = []
    if cur_version != base_version:
        base_cohort   = _cohort_metrics(task_type, base_version, all_runs)
        target_cohort = _cohort_metrics(task_type, cur_version, all_runs)
        vreport = compare_versions(base_cohort, target_cohort, task_type,
                                   base_version, cur_version)
        version_ins = version_insights(vreport)

    combined = (single_run_insights(metrics)
                + version_ins
                + anomaly_insights(metrics, baseline_metrics))
    seen, deduped = set(), []
    for ins in combined:
        if ins.title in seen:
            continue
        seen.add(ins.title)
        deduped.append(ins)
    return rank_insights(deduped), vreport, base_version


def _enrich_run(run: dict) -> dict:
    """Add computed fields to a raw run dict for use in the run-history list."""
    seq = json.loads(run.get("agent_sequence") or "[]")
    clean = is_clean_termination(run.get("termination_reason") or "")

    reason = run.get("termination_reason") or ""
    low = reason.lower()
    if "missing_input" in low:
        short = "MISSING INPUT"
    elif "stuck_loop" in low or "stuck" in low:
        short = "STUCK LOOP"
    elif "max_rounds" in low:
        short = "MAX ROUNDS"
    elif "error" in low:
        short = "ERROR"
    elif clean:
        short = "completed"
    else:
        short = reason[:20] or "unknown"

    return {
        **run,
        "agent_sequence":   seq,
        "total_tokens":     (run.get("total_input_tokens") or 0) + (run.get("total_output_tokens") or 0),
        "clean":            clean,
        "termination_short": short,
    }


def _baseline_metrics_for(run: dict, all_runs: list[dict]) -> list[dict]:
    """Get baseline metrics for the same task_type, excluding the current run."""
    baseline_runs = [
        r for r in all_runs
        if r["run_id"] != run["run_id"]
        and r.get("task_type") == run.get("task_type")
        and r.get("prompt_version") == 1
    ]
    result = []
    for br in baseline_runs:
        spans = get_agent_spans(br["run_id"])
        tools = get_tool_calls(br["run_id"])
        hoffs = get_handoffs(br["run_id"])
        result.append(compute_all(br, spans, tools, hoffs))
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    all_runs = list_runs(limit=200)
    # Chronological run number (1 = earliest), so users can sort by run order.
    for n, i in enumerate(sorted(range(len(all_runs)),
                                 key=lambda i: all_runs[i].get("timestamp", "")), 1):
        all_runs[i]["run_number"] = n
    enriched = [_enrich_run(run) for run in all_runs]

    total = len(enriched)
    clean = sum(1 for r in enriched if r["clean"])
    crashed = sum(1 for r in enriched if not r["clean"])
    avg_cost = sum(r.get("total_cost_usd") or 0 for r in enriched) / total if total else 0

    stats = dict(
        total_runs=total,
        clean_runs=clean,
        crashed_runs=crashed,
        avg_cost=avg_cost,
    )
    return render_template("index.html", runs=enriched, stats=stats)


def _sparkline(vals, w=110, h=28, pad=2, n_pts=34):
    """Normalised polyline points for an inline SVG sparkline."""
    vals = [float(v or 0) for v in vals]
    if len(vals) > n_pts:                            # downsample
        step = len(vals) / n_pts
        vals = [vals[int(i * step)] for i in range(n_pts)]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    out = []
    for i, v in enumerate(vals):
        x = round((i / (n - 1)) * (w - 2 * pad) + pad, 1) if n > 1 else pad
        y = round(h - pad - ((v - lo) / rng) * (h - 2 * pad), 1)
        out.append(f"{x},{y}")
    return " ".join(out)


@app.route("/home")
def home():
    """Landing page — KPIs, every project with a drift sparkline + health, and a
    cross-project recent-activity feed. Each project's drift comes from the SAME
    unified engine as Drift Investigation / Overview (last 30 days), so the Home
    badge and what you see inside a project always agree."""
    import datetime as _dt
    from storage.sqlite_store import resolve_db_path, set_active_db_path
    projects, activity = [], []
    total_runs = drift_alerts = snapshots = 0
    cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=30)).isoformat()

    for name in list_available_dbs():
        try:
            set_active_db_path(resolve_db_path(name))
            runs = list_runs(limit=1000)
        except Exception:
            runs = []
        total = len(runs)
        total_runs += total
        if total == 0:
            projects.append({"name": name, "runs": 0, "last": "", "success": 0,
                             "drift": "empty", "health": "empty", "spark": ""})
            continue

        rs = sorted(runs, key=lambda r: r.get("timestamp", ""))
        last = rs[-1].get("timestamp", "")
        clean = sum(1 for r in rs if is_clean_termination(r.get("termination_reason") or ""))
        success = round(100 * clean / total)
        vals = [(r.get("total_cost_usd") or 0) for r in rs]
        if not any(vals):
            vals = [((r.get("total_input_tokens") or 0) + (r.get("total_output_tokens") or 0)) for r in rs]

        # Drift = the unified engine over the last 30 days (same as Overview): a HIGH
        # finding -> elevated, a DRIFT finding -> moderate, else normal. Candidate /
        # watch (low-confidence) don't raise a project alert, matching the tab.
        recent = [r for r in rs if r.get("timestamp", "") >= cutoff]
        try:
            if recent:
                _vs = get_versions()
                _fv, _tv = _default_from_to(_vs)
                _cruns, _cfg_ov, _ = _assemble_version_compare(recent, _vs, _fv, _tv)
                invs, _s, _c = _build_investigations(_cruns, cfg_override=_cfg_ov)
            else:
                invs = []
        except Exception:
            invs = []
        n_high = sum(1 for x in invs if x["severity"] == "high")
        n_drift = sum(1 for x in invs if x["severity"] == "drift")
        # Same vocabulary as the finding severity badges: High > Drift > (none).
        drift = "high" if n_high else ("drift" if n_drift else "normal")
        if drift != "normal":
            drift_alerts += 1
        health = ("warning" if (drift == "high" or success < 80)
                  else "healthy" if (success >= 95 and drift == "normal") else "good")
        projects.append({"name": name, "runs": total, "last": last, "success": success,
                         "drift": drift, "health": health, "spark": _sparkline(vals)})

        # --- activity ---
        for v in get_versions():
            snapshots += 1
            activity.append({"type": "version", "proj": name, "ts": v["created_at"] or "",
                             "title": "Version snapshot created", "sub": f"{name} · {v['label']}"})
        if drift != "normal":
            _sub = " · ".join(p for p in (f"{n_high} high" if n_high else "",
                                          f"{n_drift} drift" if n_drift else "") if p)
            activity.append({"type": "drift", "proj": name, "ts": last,
                             "title": f"{'Critical' if n_high else 'Warning'} drift in {name}",
                             "sub": f"{name} · {_sub}"})
        activity.append({"type": "runs", "proj": name, "ts": last,
                         "title": "Recent runs", "sub": f"{name} · {total} runs"})

    set_active_db_path(resolve_db_path(getattr(g, "current_db", DEFAULT_DB)))  # restore
    projects.sort(key=lambda p: p["last"], reverse=True)
    activity.sort(key=lambda a: a["ts"], reverse=True)

    stats = {"projects": len(projects), "runs": total_runs,
             "alerts": drift_alerts, "snapshots": snapshots}
    return render_template("home.html", projects=projects, activity=activity[:7], stats=stats)


@app.route("/run/<run_id_prefix>")
def run_detail(run_id_prefix: str):
    all_runs = list_runs(limit=200)
    matched = [r for r in all_runs if r["run_id"].startswith(run_id_prefix)]
    if not matched:
        abort(404)
    run = matched[0]

    agent_spans = get_agent_spans(run["run_id"])
    tool_calls  = get_tool_calls(run["run_id"])
    handoffs    = get_handoffs(run["run_id"])
    metrics     = compute_all(run, agent_spans, tool_calls, handoffs)

    seq   = json.loads(run.get("agent_sequence") or "[]")
    clean = is_clean_termination(run.get("termination_reason") or "")
    reason = run.get("termination_reason") or ""
    if clean:
        short = "completed"
    elif "missing_input" in reason.lower():
        short = "MISSING INPUT"
    elif "stuck" in reason.lower():
        short = "STUCK LOOP"
    elif "error" in reason.lower():
        short = "ERROR"
    else:
        short = reason[:30] or "unknown"

    # ---- This-run-vs-baseline (per-agent across runs) ----
    baseline_metrics = _baseline_metrics_for(run, all_runs)
    bl_version = baseline_metrics[0].get("prompt_version", 1) if baseline_metrics else run.get("prompt_version", 1)
    report = build_anomaly_report(metrics, baseline_metrics, bl_version)

    task_type   = run.get("task_type")
    cur_version = run.get("prompt_version", 1)

    # ---- Top card: within-run + this-run-vs-baseline + version drift (deduped) ----
    # Shared with the run-history badge (build_run_insights) so counts always agree.
    insights, vreport, base_version = build_run_insights(
        run, metrics, all_runs, baseline_metrics)
    ins_counts = severity_counts(insights)

    # Agents flagged by EITHER comparison → graph node styling
    drifted_agents = {
        a for a, vs in (report.per_agent_verdicts or {}).items() if any(v.drifted for v in vs)
    }
    if vreport and vreport.available:
        for a, vs in vreport.per_agent.items():
            if any(v.drifted for v in vs):
                drifted_agents.add(a)

    loop_turns = {p["turn"] for p in (metrics.get("reentry_patterns") or []) if p.get("flagged")}

    # Structural DAG analysis (only fires when DAG fields are populated).
    parallel_groups = detect_parallel_groups(agent_spans)
    critical_span_ids = set(critical_path(agent_spans))

    graph_data = build_graph_data(
        timeline := build_timeline(agent_spans, tool_calls),
        metrics.get("handoffs") or [],
        drifted_agents, loop_turns, clean, short,
        agent_spans=agent_spans,
        critical_span_ids=critical_span_ids,
    )

    return render_template(
        "run.html",
        run_id=run["run_id"],
        task_text=run.get("task_text", ""),
        task_type=task_type or "—",
        prompt_version=cur_version,
        timestamp=run.get("timestamp", ""),
        metrics=metrics,
        clean=clean,
        termination_short=short,
        timeline=timeline,
        insights=insights,
        ins_counts=ins_counts,
        graph_data=json.dumps(graph_data),
        report=report,
        has_baseline=len(baseline_metrics) > 0,
        vreport=vreport,
        base_version=base_version,
        parallel_groups=parallel_groups,
        has_critical_path=len(critical_span_ids) > 0 and any(s.get("parent_step_id") for s in agent_spans),
    )


@app.route("/trends")
def trends():
    """Trend view — gradual drift detection across a window of runs."""
    import datetime as _dt
    from flask import request

    days = int(request.args.get("days", 30))
    task_type_filter = request.args.get("task_type", "") or None
    version_filter   = request.args.get("version", "") or None
    try:
        version_filter_int = int(version_filter) if version_filter else None
    except (TypeError, ValueError):
        version_filter_int = None

    # Pull runs in the window
    all_runs = list_runs(limit=2000)
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)
    runs = [r for r in all_runs if r.get("timestamp", "") >= cutoff.isoformat()]
    if task_type_filter:
        runs = [r for r in runs if r.get("task_type") == task_type_filter]
    if version_filter_int is not None:
        runs = [r for r in runs if r.get("prompt_version") == version_filter_int]

    # All versions seen across the (pre-version-filter) window — drives filter chips
    versions_available = sorted({
        r.get("prompt_version") for r in all_runs
        if r.get("timestamp", "") >= cutoff.isoformat()
        and r.get("prompt_version") is not None
        and (not task_type_filter or r.get("task_type") == task_type_filter)
    })

    # Sort ascending by timestamp (oldest first)
    runs.sort(key=lambda r: r.get("timestamp", ""))

    # Compute metrics per run (the expensive bit — cache could be added later)
    metrics_list = [_metrics_for_run(r) for r in runs]

    # Distinct task types in this window (for filter chips)
    task_types = sorted({r.get("task_type", "unspecified") for r in all_runs})

    # Build trend payload
    trends_data = build_trends(runs, metrics_list)

    # Early/recent split — shared by BOTH common-handoff health and drift.
    mid = len(runs) // 2
    early_runs, recent_runs = runs[:mid], runs[mid:]

    # Per-agent drift cards (replaces the old efficiency-based agent health).
    trends_data["agent_drift"] = compute_agent_drift(early_runs, recent_runs)

    # Path-level summary (no drift rules — just snapshot + delta).
    trends_data["path_summary"] = compute_path_summary(early_runs, recent_runs)

    # Parallel-group health (bottleneck / join wait / efficiency), same split.
    trends_data["parallel_health"] = compute_parallel_health_summary(early_runs, recent_runs)

    # Change tracking: config changes potentially related to the drift (clues, not
    # cause). Anchor = start of the recent half; look back 20 runs OR 24h.
    anchor_ts = recent_runs[0].get("timestamp", "") if recent_runs else ""
    trends_data["related_changes"] = potentially_related_changes(runs, mid, anchor_ts)

    # Manual version snapshots + horizontal comparison.
    trends_data["versions"] = compute_versions(runs)

    # Drift first (it decides what's a "suspect" and what's "minor drift").
    drift = compute_handoff_drift(early_runs, recent_runs)
    trends_data["drift_suspects"] = drift["suspects"]
    drifted_lookup = drift.get("drifted_lookup", {})
    suspect_pairs  = {(s["from"], s["to"]) for s in drift["suspects"]}

    # Common-handoff health (volume-ranked, COMPACT). Exclude the top suspects.
    leaderboard = compute_handoff_leaderboard(early_runs, recent_runs, top_n=8)
    kept = []
    volume_pairs = set()
    for p in leaderboard["pairs"]:
        key = (p["from"], p["to"])
        volume_pairs.add(key)
        if key in suspect_pairs:
            continue          # already shown as a detailed suspect card
        # Attach a small "minor drift" badge if this pair drifted but wasn't a suspect.
        if key in drifted_lookup:
            p["minor_drift"] = drifted_lookup[key]   # {risk, rule_title}
        kept.append(p)
    leaderboard["pairs"] = kept[:5]   # cap the compact list at 5
    trends_data["top_handoffs"] = leaderboard

    # Tag each suspect that's also high-volume so its card shows a badge.
    for s in drift["suspects"]:
        s["also_high_volume"] = (s["from"], s["to"]) in volume_pairs

    # ---- Top-of-page summary numbers ----
    handoff_suspects = len(drift["suspects"])
    agent_suspects = sum(
        1 for c in trends_data["agent_drift"]["drift"]
        if c["status"] in ("critical", "high", "medium")
    )

    def _avg(rows, key, scale=1.0):
        vals = [(r.get(key) or 0) * scale for r in rows]
        return (sum(vals) / len(vals)) if vals else 0.0

    cost_early  = _avg(early_runs, "total_cost_usd")
    cost_recent = _avg(recent_runs, "total_cost_usd")
    wall_early  = _avg(early_runs, "total_duration_ms", 0.001)
    wall_recent = _avg(recent_runs, "total_duration_ms", 0.001)

    trends_data["health_summary"] = {
        "runs_analyzed":    len(runs),
        "drift_total":      handoff_suspects + agent_suspects,
        "drift_handoffs":   handoff_suspects,
        "drift_agents":     agent_suspects,
        "cost_pct":         round(_safe_pct_change(cost_early, cost_recent), 0),
        "wall_pct":         round(_safe_pct_change(wall_early, wall_recent), 0),
    }

    # Brief timestamps: "MM/DD HH:MM" — full ISO is too noisy on the x-axis.
    def _brief_ts(iso: str) -> str:
        # iso like "2025-12-05T14:30:42.123+00:00"
        if not iso or len(iso) < 16:
            return iso or ""
        return f"{iso[5:7]}/{iso[8:10]} {iso[11:16]}"

    # Time-series payload (cost-only chart). Keep tokens off it — users only
    # want the dollar trend at a glance.
    series = [
        {
            "ts":       _brief_ts(r.get("timestamp", "")),
            "cost":     r.get("total_cost_usd", 0) or 0,
            "wall_s":   round((r.get("total_duration_ms", 0) or 0) / 1000, 1),
            "version":  r.get("prompt_version", 1),
            "run_id":   r.get("run_id", "")[:8],
            "task_type": r.get("task_type", ""),
        }
        for r in runs
    ]

    return render_template(
        "trends.html",
        days=days,
        task_type_filter=task_type_filter,
        version_filter=version_filter_int,
        task_types=task_types,
        versions_available=versions_available,
        trends=trends_data,
        series=json.dumps(series),
    )


def _band_cfg() -> dict:
    from analysis.drift_config import load_drift_config
    mb = (load_drift_config().get("metric_band") or {})
    return {"baseline_runs": int(mb.get("baseline_runs", 5)),
            "k": float(mb.get("k", 3.0)),
            "consecutive": int(mb.get("consecutive", 2))}


# Plain-English help for each metric — shown in the (i) hover on every chart.
_METRIC_HELP = {
    "tokens":         "Total tokens (input + output) this agent used per run.",
    "latency_s":      "Wall-clock seconds this agent took per run.",
    "cost_usd":       "Estimated USD cost of this agent's LLM calls per run.",
    "tool_calls":     "Number of tool calls this agent made per run.",
    "errors":         "Count of errored spans for this agent per run.",
    "retries":        "LLM-call retries inside this agent's step per run.",
    "reinvoked":      "1 when this agent ran more than once in a run (a loop), else 0.",
    "success":        "Whether the whole run ended cleanly (1) or failed (0).",
    "frequency":      "How many times this handoff fired per run.",
    "payload_tokens": "Output tokens the sender passed to the receiver on this handoff.",
    "hop_latency_s":  "Wall-clock gap between the sender finishing and the receiver starting (queue/transport time) per run.",
    "path_length":    "Number of steps in the run's agent sequence.",
    "loops":          "Repeated agents in the sequence (length minus unique agents).",
    "e2e_latency_s":  "End-to-end wall-clock seconds for the whole run (first step start to last step finish).",
    "bottleneck_s":   "Duration of the slowest branch in this parallel group.",
    "join_wait_s":    "Time the fan-in waited on the slowest branch.",
    "efficiency":     "Parallel balance: 1.0 = perfectly balanced fan-out, lower = lopsided.",
}


def _ver_of_factory(versions):
    def _ver_of(r):
        v = 1
        for ver in versions:
            if r.get("timestamp", "") >= (ver["created_at"] or ""):
                v = ver["version_num"]
            else:
                break
        return v
    return _ver_of


def _default_from_to(versions):
    """Default Drift-Investigation comparison: the project's set baseline version
    (From) vs every later version (To). Used so Home, Overview, and the Drift
    Investigation default view all analyse the same thing."""
    ver_options = sorted({1} | {v["version_num"] for v in versions})
    base = get_baseline_version()
    if base not in ver_options:
        base = ver_options[0]
    return base, [n for n in ver_options if n > base]


def _assemble_version_compare(runs, versions, from_v, to_vs):
    """Order runs [From baseline …, To version(s) …] and return the drift-config
    window override so the engine measures the WHOLE From version vs the WHOLE To
    version(s). Falls back to the plain timeline when there's nothing to compare.
    Returns (runs, cfg_override, compare_on)."""
    _vof = _ver_of_factory(versions)
    base_runs = sorted((r for r in runs if _vof(r) == from_v), key=lambda r: r.get("timestamp", ""))
    cmp_runs = sorted((r for r in runs if _vof(r) in to_vs), key=lambda r: r.get("timestamp", ""))
    if base_runs and cmp_runs:
        return base_runs + cmp_runs, {"baseline_runs": len(base_runs), "recent_runs": len(cmp_runs)}, True
    if base_runs:
        return base_runs, None, False
    return runs, None, False


def _apply_version_compare(runs, versions, band_cfg, args):
    """Filter runs to the SELECTED version(s) via the multi-value `version` param
    (default: show all). Per-version boundary markers let you read each version on
    the chart — no baseline/compare ('From/To') framing. Returns (runs, band_cfg,
    vcmp) where vcmp drives the Version picker + legend."""
    ver_nums = sorted({1} | {v["version_num"] for v in versions})
    _ver_of = _ver_of_factory(versions)

    getl = (lambda k: args.getlist(k)) if hasattr(args, "getlist") \
        else (lambda k: [x for x in (args.get(k) or "").split(",") if x])
    sel = sorted({int(x) for x in getl("version") if x.isdigit() and int(x) in ver_nums})
    explicit = bool(sel)
    if explicit:
        runs = sorted((r for r in runs if _ver_of(r) in sel), key=lambda r: r.get("timestamp", ""))

    present = [n for n in ver_nums if any(_ver_of(r) == n for r in runs)]
    shown = sel or present
    label = ("All versions" if (not explicit or len(sel) >= len(ver_nums))
             else ", ".join(f"v{n}" for n in sel))
    vcmp = {
        "ver_options": ver_nums, "selected": set(shown), "explicit": explicit,
        "label": label, "available": len(ver_nums) > 1, "seg_versions": present,
    }
    return runs, band_cfg, vcmp


_RANGES = [("15m", "Last 15 Min", 15 * 60), ("1h", "Last Hour", 3600),
           ("12h", "Last 12 Hours", 12 * 3600), ("1d", "Last Day", 86400),
           ("7d", "Last 7 Days", 7 * 86400), ("30d", "Last Month", 30 * 86400)]
_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _browse_context(args, band_cfg):
    """Shared run-loading for the metric pages: time range + task_type + version
    comparison + per-version line colours. Returns one context dict so the
    Drift Investigation and Metrics Explorer pages stay in sync."""
    import datetime as _dt
    import re
    now = _dt.datetime.now(_dt.timezone.utc)
    start, end = args.get("start"), args.get("end")
    rp = args.get("range", "30d")
    rmap = {k: secs for k, _, secs in _RANGES}

    def _delta(s):
        if s in rmap:
            return _dt.timedelta(seconds=rmap[s])
        m = re.match(r"^(\d+)(m|h|d)$", s or "")
        if m:
            n, u = int(m.group(1)), m.group(2)
            return _dt.timedelta(minutes=n) if u == "m" else _dt.timedelta(hours=n) if u == "h" else _dt.timedelta(days=n)
        return _dt.timedelta(days=7)

    def _fmtd(d):
        return f"{_MONTHS[d.month]} {d.day}"

    if start and end:
        lo_dt = _dt.datetime.fromisoformat(start).replace(tzinfo=_dt.timezone.utc)
        hi_dt = _dt.datetime.fromisoformat(end).replace(hour=23, minute=59, second=59, tzinfo=_dt.timezone.utc)
        range_badge, active_range = "custom", ""
    else:
        hi_dt, lo_dt = now, now - _delta(rp)
        range_badge, active_range = rp, rp
    lo, hi = lo_dt.isoformat(), hi_dt.isoformat()       # filtering stays in UTC
    from zoneinfo import ZoneInfo
    lo_e = lo_dt.astimezone(ZoneInfo(_DISPLAY_TZ))       # display in Eastern
    hi_e = hi_dt.astimezone(ZoneInfo(_DISPLAY_TZ))
    if lo_e.date() == hi_e.date():
        range_display = f"{_fmtd(lo_e)}, {lo_e.year}"
    elif lo_e.year == hi_e.year:
        range_display = f"{_fmtd(lo_e)} – {_fmtd(hi_e)}, {hi_e.year}"
    else:
        range_display = f"{_fmtd(lo_e)}, {lo_e.year} – {_fmtd(hi_e)}, {hi_e.year}"

    task_type = args.get("task_type", "")
    all_runs = list_runs(limit=1000)
    runs = [r for r in all_runs if lo <= r.get("timestamp", "") <= hi
            and (not task_type or r.get("task_type") == task_type)]
    runs.sort(key=lambda r: r.get("timestamp", ""))
    task_types = sorted({r.get("task_type", "") for r in all_runs if r.get("task_type")})

    versions = get_versions()
    runs, band_cfg, vcmp = _apply_version_compare(runs, versions, band_cfg, args)
    _vof = _ver_of_factory(versions)
    version_for_x = [_vof(r) for r in sorted(runs, key=lambda r: r.get("timestamp", ""))]
    _PAL = ["#4263eb", "#fa5252", "#f59f00", "#7048e8", "#0ca678", "#e8590c", "#d6336c"]
    ver_color = {n: _PAL[i % len(_PAL)] for i, n in enumerate(vcmp["seg_versions"])}
    vcmp["legend"] = [{"label": f"v{n}", "color": ver_color[n]} for n in vcmp["seg_versions"]]

    return {
        "runs": runs, "band_cfg": band_cfg, "vcmp": vcmp, "lo": lo, "hi": hi,
        "version_for_x": version_for_x, "ver_color": ver_color,
        "task_type": task_type, "task_types": task_types, "versions": versions,
        "rp": rp, "ranges": _RANGES, "range_display": range_display,
        "range_badge": range_badge, "active_range": active_range, "start": start, "end": end,
    }


@app.route("/explore")
def explore():
    """Metrics Explorer — browse every metric by category → entity, with the same
    band/version-comparison visuals. (Custom formulas + thresholds: stage 2.)"""
    from analysis.metric_series import per_run_series, control_band, metric_impact, metrics_of
    ctx = _browse_context(request.args, _band_cfg())
    runs, band_cfg, vcmp = ctx["runs"], ctx["band_cfg"], ctx["vcmp"]
    _attach_routes(runs)
    series = _inject_custom_metrics(per_run_series(runs))

    cats = [c for c in ("agents", "handoffs", "path", "parallel") if series.get(c)]
    _CAT_LABEL = {"agents": "Agents", "handoffs": "Handoffs", "path": "Path", "parallel": "Parallel"}
    category = request.args.get("category") or (cats[0] if cats else "agents")
    cat_series = series.get(category, {})
    entities = sorted(cat_series.keys())
    # Under Path, offer a "Routes" pseudo-entity: the DAG of every distinct route
    # that ran (with count + share), alongside the System metric charts.
    has_routes = any(r.get("agent_sequence") for r in runs)
    if category == "path" and has_routes:
        entities = ["Routes"] + entities
    entity = request.args.get("entity") or (entities[0] if entities else "")

    # Handoffs rail: group "A → B" entities by source agent, plus From/To options.
    handoff_tree, handoff_froms, handoff_tos = [], [], []
    if category == "handoffs":
        from collections import defaultdict
        groups, fset, tset = defaultdict(list), set(), set()
        for e in entities:
            a, _, b = e.partition(" → ")
            a, b = a.strip(), b.strip()
            if not b:
                continue
            groups[a].append({"to": b, "ent": e})
            fset.add(a); tset.add(b)
        handoff_tree = [{"from": a, "targets": sorted(groups[a], key=lambda x: x["to"])}
                        for a in sorted(groups)]
        handoff_froms, handoff_tos = sorted(fset), sorted(tset)

    route_graphs, route_cards, routes_total = {}, [], 0
    if category == "path" and entity == "Routes":
        from collections import Counter
        from analysis.changes import route_key
        # Group runs by topology (route_key) so parallel-sibling reorderings collapse
        # into one route. Each group keeps a representative run for its nodes/edges.
        groups, rep = Counter(), {}
        for r in runs:
            k = route_key(r)
            if not k:
                continue
            groups[k] += 1
            rep.setdefault(k, r)
        routes_total = sum(groups.values())
        canon_key = groups.most_common(1)[0][0] if groups else None
        for i, (k, cnt) in enumerate(groups.most_common()):
            r = rep[k]
            nodes = r.get("_route_nodes") or json.loads(r.get("agent_sequence") or "[]")
            edges = r.get("_route_edges") or [list(e) for e in zip(nodes, nodes[1:])]
            gid = f"rtg{i}"
            route_graphs[gid] = build_path_graph(nodes, edges)
            route_cards.append({"gid": gid, "count": cnt, "length": len(nodes),
                                "pct": round(100 * cnt / routes_total, 1) if routes_total else 0,
                                "canonical": k == canon_key})

    custom_names = _custom_names()
    thresholds = get_thresholds()
    charts = []
    targets = entities if entity in ("", "*") else [entity]
    for ent in targets:
        for metric, pts in cat_series.get(ent, {}).items():
            band = control_band([p["y"] for p in pts], **band_cfg)
            charts.append({"entity": ent, "metric": metric, "category": category, "points": pts,
                           "band": band, "impact": metric_impact(pts, band_cfg["baseline_runs"]),
                           "custom": metric in custom_names,
                           "threshold": thresholds.get(f"{category}|{metric}")})

    # NOTE: the Path category here shows metric charts only (path_length, loops,
    # e2e_latency_s). The agent-chain DAG comparison graphs live in Drift
    # Investigation, surfaced when a path drift is detected — not in the Explorer.

    # Base metrics available in this category (for the formula builder).
    base_metrics = [m for m in metrics_of(cat_series) if m not in custom_names]
    customs_here = [cm for cm in get_custom_metrics() if cm.get("category") == category]

    # Version markers — the vertical dashed lines that label where each version starts.
    versions, _vof = ctx["versions"], _ver_of_factory(ctx["versions"])
    runs_sorted = sorted(runs, key=lambda r: r.get("timestamp", ""))
    present = {_vof(r) for r in runs_sorted}
    markers = []
    for v in versions:
        if v["version_num"] not in present:
            continue
        for i, r in enumerate(runs_sorted):
            if r.get("timestamp", "") >= (v["created_at"] or ""):
                markers.append({"x": i, "label": f"v{v['version_num']}"})
                break

    return render_template(
        "explore.html",
        cats=cats, cat_labels=_CAT_LABEL, category=category, entities=entities, entity=entity,
        handoff_tree=handoff_tree, handoff_froms=handoff_froms, handoff_tos=handoff_tos,
        route_cards=route_cards, routes_total=routes_total, route_graphs_json=json.dumps(route_graphs),
        charts_json=json.dumps(charts), run_count=len(runs), markers_json=json.dumps(markers),
        metric_help_json=json.dumps(_METRIC_HELP), vcmp=vcmp,
        base_metrics=base_metrics, customs_here=customs_here, all_metrics=metrics_of(cat_series),
        rp=ctx["rp"], ranges=ctx["ranges"], active_range=ctx["active_range"],
        range_display=ctx["range_display"], range_badge=ctx["range_badge"],
        start=ctx["start"], end=ctx["end"], task_type=ctx["task_type"], task_types=ctx["task_types"],
    )


_OPS = {"/": lambda a, b: (a / b if b else 0.0), "*": lambda a, b: a * b,
        "+": lambda a, b: a + b, "-": lambda a, b: a - b}


def _inject_custom_metrics(series):
    """Add user-defined derived metrics (metricA op metricB) into the per-run series
    so they render and drift-detect like any built-in metric."""
    for cm in get_custom_metrics():
        cat = cm.get("category")
        cs = series.get(cat)
        if not cs:
            continue
        a, b, op, name = cm.get("a"), cm.get("b"), cm.get("op"), cm.get("name")
        fn = _OPS.get(op)
        if not (a and b and fn and name):
            continue
        ents = list(cs.keys()) if cm.get("entity") in ("", "*", None) else [cm["entity"]]
        for ent in ents:
            em = cs.get(ent, {})
            pa = {p["x"]: p for p in em.get(a, [])}
            pb = {p["x"]: p for p in em.get(b, [])}
            pts = []
            for x in sorted(set(pa) & set(pb)):
                pt = dict(pa[x])
                try:
                    pt["y"] = round(float(fn(pa[x]["y"], pb[x]["y"])), 6)
                except Exception:
                    pt["y"] = 0.0
                pts.append(pt)
            if pts:
                em[name] = pts
                cs[ent] = em
    return series


def _custom_names():
    return {cm["name"] for cm in get_custom_metrics() if cm.get("name")}


@app.route("/custom-metric", methods=["POST"])
def custom_metric_add():
    f = request.form
    name = (f.get("name") or "").strip()
    cm = {"name": name, "category": f.get("category", "agents"),
          "entity": f.get("entity", "*"), "a": f.get("a"), "op": f.get("op"), "b": f.get("b")}
    if name and cm["a"] and cm["b"] and cm["op"] in _OPS:
        metrics = [m for m in get_custom_metrics() if m.get("name") != name]
        metrics.append(cm)
        save_custom_metrics(metrics)
    return jsonify({"ok": True})


@app.route("/custom-metric/delete", methods=["POST"])
def custom_metric_delete():
    name = (request.form.get("name") or request.json.get("name") if request.is_json else request.form.get("name"))
    save_custom_metrics([m for m in get_custom_metrics() if m.get("name") != name])
    return jsonify({"ok": True})


@app.route("/threshold", methods=["POST"])
def threshold_set():
    f = request.form
    key = f"{f.get('category')}|{f.get('metric')}"
    th = get_thresholds()
    pct = (f.get("pct") or "").strip()
    if pct == "":                                    # empty → remove
        th.pop(key, None)
    else:
        try:
            th[key] = {"pct": float(pct), "dir": f.get("dir", "max")}
        except ValueError:
            pass
    save_thresholds(th)
    return jsonify({"ok": True})


_DIM_LABEL = {"model": "Model", "prompt": "Prompt", "tools": "Tools",
              "params": "Parameters", "workflow": "Workflow"}


def _fmt_change(e):
    """Turn a raw change-log entry into a display event with an exact before→after."""
    dim, old, new = e["dimension"], e["old"], e["new"]
    if dim == "prompt":
        detail = "System prompt content changed"          # only a hash is captured, not the text
    elif dim == "tools":
        os_, ns = set(old or ()), set(new or ())
        parts = [f"+ {t}" for t in sorted(ns - os_)] + [f"− {t}" for t in sorted(os_ - ns)]
        detail = ", ".join(parts) if parts else f"{list(old or [])} → {list(new or [])}"
    elif dim == "params":
        detail = " · ".join(f"{k}: {old.get(k)} → {new.get(k)}"
                            for k in (set(old or {}) | set(new or {}))
                            if (old or {}).get(k) != (new or {}).get(k)) \
            if isinstance(old, dict) and isinstance(new, dict) else f"{old} → {new}"
    else:
        detail = f"{old} → {new}"
    return {"dim": dim, "dim_label": _DIM_LABEL.get(dim, dim.title()), "scope": e["scope"],
            "detail": detail, "ts": e.get("timestamp", ""), "run_index": e["run_index"]}


_EVTYPE = {"config_change": "Config change", "version": "Version snapshot", "release": "Release event",
           "drift": "Drift signal", "impact": "Impact signal", "evidence": "Evidence run"}

# A "component" is a part of the workflow whose related files/config changed. We show a
# professional, role-describing label instead of the raw agent codename — users shouldn't
# have to know what "analyst" does internally.
_COMPONENT_INFO = {
    "researcher": ("Researcher agent", "Gathers source material and market data"),
    "analyst": ("Analyst agent", "Generates the trading signal from research"),
    "writer": ("Writer agent", "Drafts the final thesis and output"),
}


def _comp_info(name):
    if not name or name == "—":
        return ("—", "")
    return _COMPONENT_INFO.get(name, (name.replace("_", " ").title() + " agent", "Workflow component"))


@app.route("/changelog")
def changelog():
    """Event Timeline — every captured config/version/release/drift event, with a
    rich per-event detail (who, how, before→after, impact). Drift Investigation
    deep-links here via ?find=dim:scope or ?event=id."""
    import datetime as _dt
    from analysis.changes import build_change_log
    versions = get_versions()

    def _ver_of_ts(ts):
        v = 1
        for ver in versions:
            if ts >= (ver["created_at"] or ""):
                v = ver["version_num"]
            else:
                break
        return v

    # Source events: the rich captured table, or a derived fallback for un-seeded projects.
    events = get_events()
    if not events:
        from storage.sqlite_store import get_prompts
        prompt_text = get_prompts()                  # {hash: full text} for prompt changes
        runs = sorted(list_runs(1000), key=lambda r: r.get("timestamp", ""))
        for e in build_change_log(runs):
            ri = e["run_index"]
            tstamp = runs[ri]["timestamp"] if 0 <= ri < len(runs) else ""
            ev = {"id": f"{ri}:{e['dimension']}:{e['scope']}", "ts": tstamp, "run_index": ri,
                  "type": "config_change", "dim": e["dimension"], "component": e["scope"],
                  "title": f"{e['scope']} — {_DIM_LABEL.get(e['dimension'], e['dimension'])} changed",
                  "before": str(e["old"]), "after": str(e["new"]), "author": "", "source": "auto",
                  "environment": "", "changed_fields": 1, "related_drift": f"agents|{e['scope']}",
                  "impact_json": "[]"}
            if e["dimension"] == "prompt":            # resolve hashes → real before/after text
                ev["hash_before"], ev["hash_after"] = e["old"], e["new"]
                tb, ta = prompt_text.get(e["old"]), prompt_text.get(e["new"])
                # Never show a raw hash to a human; if the full text wasn't captured
                # for this project, say so plainly (the hashes stay in the detail row).
                ev["before"] = tb if tb else "Previous prompt (full text not captured)"
                ev["after"] = ta if ta else "Updated prompt (full text not captured)"
            events.append(ev)
        for v in versions:
            events.append({"id": f"v{v['version_num']}", "ts": v["created_at"] or "", "run_index": 10 ** 6,
                           "type": "version", "component": "—", "dim": None,
                           "title": v["label"] or f"Version {v['version_num']}", "source": "Snapshot",
                           "impact_json": "[]"})
        events.sort(key=lambda x: (x.get("ts") or ""))

    # Filters
    comp = request.args.get("component", "")
    ver_f = request.args.get("version", "")
    rng = request.args.get("range", "")
    lo = ""
    if rng:
        secs = {k: s for k, _, s in _RANGES}.get(rng)
        if secs:
            lo = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=secs)).isoformat()

    def _keep(e):
        if comp and e.get("component") != comp:
            return False
        if ver_f and str(_ver_of_ts(e.get("ts") or "")) != ver_f:
            return False
        if lo and (e.get("ts") or "") < lo:
            return False
        return True

    shown = [e for e in events if _keep(e)]
    for e in shown:
        e["type_label"] = _EVTYPE.get(e.get("type"), "Event")
        e["version"] = _ver_of_ts(e.get("ts") or "")
        e["component_label"], e["component_role"] = _comp_info(e.get("component"))

    # Selection: explicit ?event=id, else ?find=dim:scope (from Drift Investigation), else first.
    sel_id, find = request.args.get("event", ""), request.args.get("find", "")
    selected = next((e for e in shown if e["id"] == sel_id), None) if sel_id else None
    if not selected and find and ":" in find:
        d, s = find.split(":", 1)
        selected = next((e for e in shown if e.get("dim") == d and e.get("component") == s), None) \
            or next((e for e in shown if e["id"] == find), None)
    if not selected and shown:
        selected = shown[0]
    selected_impact = []
    if selected:
        try:
            selected_impact = json.loads(selected.get("impact_json") or "[]")
        except (TypeError, ValueError):
            selected_impact = []

    components = [(c, _comp_info(c)[0]) for c in
                  sorted({e.get("component") for e in events
                          if e.get("component") and e.get("component") != "—"})]
    ver_opts = sorted({1} | {v["version_num"] for v in versions})

    return render_template("changelog.html", events=shown, selected=selected,
                           selected_impact=selected_impact, total=len(shown),
                           components=components, component=comp,
                           ver_opts=ver_opts, version_f=ver_f, ranges=_RANGES, rng=rng)


@app.route("/overview")
def overview():
    """Project landing page — health summary, navigation to the other views, and
    version management (create snapshots, set the comparison baseline)."""
    args = request.args
    if not args.get("range") and not (args.get("start") and args.get("end")):
        from werkzeug.datastructures import MultiDict
        args = MultiDict(request.args)
        args["range"] = "30d"                        # a landing page wants a broad default
    ctx = _browse_context(args, _band_cfg())
    # Same engine AND same default comparison as Drift Investigation (/drift2):
    # baseline version vs every later version, so the summary and the tab agree.
    runs = sorted([r for r in list_runs(limit=1000)
                   if ctx["lo"] <= r.get("timestamp", "") <= ctx["hi"]
                   and (not ctx["task_type"] or r.get("task_type") == ctx["task_type"])],
                  key=lambda r: r.get("timestamp", ""))
    _versions = get_versions()
    _fv, _tv = _default_from_to(_versions)
    runs, _cfg_ov, _ = _assemble_version_compare(runs, _versions, _fv, _tv)
    investigations, series, _clog = _build_investigations(runs, cfg_override=_cfg_ov)
    high = sum(1 for x in investigations if x["severity"] == "high")
    _RISKMAP = {"high": "High", "drift": "Medium", "candidate": "Watch", "watch": "Watch"}
    # Top signals = the significant drift only (high/drift); candidate/watch are too
    # low-confidence to headline (they live behind the dropdown on Drift Investigation).
    top_signals = [{"entity": x["title"], "type": x["kind_label"],
                    "risk": _RISKMAP.get(x["severity"], "Watch"),
                    "metrics": [m["name"] for m in x["metrics"]]}
                   for x in investigations if x["severity"] in ("high", "drift")][:3]
    total = len(runs)
    clean = sum(1 for r in runs if is_clean_termination(r.get("termination_reason") or ""))
    success_rate = round(100 * clean / total) if total else 0
    avg_cost = (sum((r.get("total_cost_usd") or 0) for r in runs) / total) if total else 0
    cat_count = len([c for c in ("agents", "handoffs", "path", "parallel") if series.get(c)])
    from analysis.changes import build_change_log
    _events = get_events()
    event_count = len(_events) if _events else len(build_change_log(runs))

    return render_template(
        "overview.html",
        run_count=total, success_rate=success_rate, avg_cost=avg_cost,
        finding_count=sum(1 for x in investigations if x["severity"] in ("high", "drift")),
        high_count=high, top_signals=top_signals,
        event_count=event_count,
        cat_count=cat_count, versions=compute_versions(runs), baseline_v=get_baseline_version(),
        rp=ctx["rp"], ranges=ctx["ranges"], active_range=ctx["active_range"],
        range_display=ctx["range_display"], range_badge=ctx["range_badge"],
        start=ctx["start"], end=ctx["end"],
    )


@app.route("/set-baseline/<int:num>", methods=["POST"])
def set_baseline(num: int):
    set_baseline_version(num)
    return jsonify({"ok": True, "baseline": num})


# Plain-language definitions surfaced as info tooltips next to each signal chart.
_METRIC_HELP = {
    "success": "Share of runs where this agent finished successfully (higher is better).",
    "error_rate": "Share of runs where this agent raised an error (lower is better).",
    "retry_rate": "Share of runs where this agent retried an LLM call (lower is better).",
    "reinvocation_rate": "Share of runs where this agent was invoked more than once.",
    "latency_s": "Wall-clock time this agent took per run, in seconds.",
    "cost_usd": "LLM spend attributed to this agent per run, in US dollars.",
    "payload_tokens": "Tokens handed off out of this agent per run.",
    "prompt_tokens": "Prompt tokens sent by this agent per run.",
    "completion_tokens": "Completion tokens produced by this agent per run.",
    "total_tokens": "Prompt + completion tokens for this agent per run.",
    "token_share": "This agent's share of the whole run's token usage.",
    "context_ratio": "Input context size relative to this agent's baseline.",
    "hop_latency_s": "Time spent on this handoff between two agents, in seconds.",
    "route_conformance": "Share of runs that followed the usual agent route (higher is better).",
    "e2e_latency_s": "End-to-end latency of the whole run, in seconds.",
    "reentry_count": "How many times the workflow re-entered an agent in a run.",
}


def _build_chain_cards(chains: list[dict]) -> list[dict]:
    """Turn causal chains into investigation cards, headlined at the ROOT:
    What changed / Potentially related event / Why it matters (templated) /
    Next check (templated; LLM-refinable)."""
    from analysis.drift_config import load_drift_config
    _units = {m: cfg.get("unit") for m, cfg in
              ((load_drift_config().get("metric_drift") or {}).get("metrics", {})).items()}

    cards = []
    for c in chains:
        def _sig(s):
            # signed delta in the metric's own unit, so the number is unambiguous:
            #   pct -> "%", rate -> " pp" (percentage points), counts/seconds -> raw
            val = s.get("bad_delta") or 0
            signed = val if s.get("direction") == "up" else -val
            unit = _units.get(s["metric"])
            suffix = "%" if unit == "pct" else (" pp" if unit == "pp" else "")
            return f'{s["entity"]} {s["metric"]} {signed:+g}{suffix}'
        what = [_sig(s) for s in c["root_signals"]] + [_sig(c["symptom_signal"])]
        trig = c.get("trigger")
        related = (f'{trig["scope"]} {trig.get("dimension", "config")} changed near run '
                   f'{trig.get("run_index")}') if trig else None
        why = (f'A change in {c["root_agent"]} propagated downstream to '
               f'{c["symptom_agent"]}, whose outcome ({c["symptom_signal"]["metric"]}) '
               f'worsened. The route is unchanged — the regression came from '
               f'{c["root_agent"]}'
               + (f', likely the {trig.get("dimension")} change.' if trig
                  else ' (no nearby config change — correlation only).'))
        nexts = []
        if trig:
            nexts.append(f'Compare {c["root_agent"]} output before vs after the '
                         f'{trig.get("dimension")} change.')
        nexts.append(f'Open a representative {c["symptom_agent"]} run from the drift '
                     f'window (~run {c["drift_start"]}).')
        nexts.append(f'Trace the {c["root_agent"]} → {c["symptom_agent"]} handoffs for '
                     f'payload / context shifts.')
        cards.append({
            "headline": f'{c["root_agent"]} drift',
            "path": f'{c["root_agent"]} → … → {c["symptom_agent"]}',
            "confidence": c["confidence"], "what_changed": what,
            "related_event": related, "why": why, "next_checks": nexts,
        })
    return cards


def _attach_routes(runs):
    """Attach DAG-topology route info (_route_sig / _route_nodes / _route_edges) to
    each run from its spans, so route matching collapses parallel-sibling reorderings
    and the real (possibly parallel) shape is available for rendering. No-op for runs
    whose spans carry no DAG structure — they keep the flat-sequence fallback."""
    from analysis.layer1_raw import get_spans_for_runs
    from analysis.changes import route_topology
    spans_by_run = get_spans_for_runs([r.get("run_id") for r in runs if r.get("run_id")])
    for r in runs:
        sp = spans_by_run.get(r.get("run_id"))
        if not sp or not any(s.get("parent_step_id") or s.get("branch_id") for s in sp):
            continue
        r["_route_nodes"], r["_route_edges"], r["_route_sig"] = route_topology(sp)
    return runs


def _build_investigations(runs, sort="sev_desc", cfg_override=None):
    """Request-free core of the unified drift engine: run investigate() on `runs`
    and return the ranked investigations (likely-cause chains + tiered component
    findings) plus the series + change log. Shared by /drift2, /drift2/next-checks,
    and the Overview summary so every surface agrees on what 'drift' means.
    `cfg_override` tweaks the drift config (e.g. baseline_runs/recent_runs for a
    version comparison: whole From version vs whole To version)."""
    from analysis.metric_series import per_run_series
    _attach_routes(runs)
    from analysis.changes import build_change_log
    from analysis.drift_detect import investigate
    from analysis.drift_config import load_drift_config

    cfg = load_drift_config().get("metric_drift") or {}
    if cfg_override:
        cfg = {**cfg, **cfg_override}
    series = per_run_series(runs)
    clog = build_change_log(runs)
    inv = investigate(series, clog, cfg)

    _SCOPE = {"agents": "Agent", "handoffs": "Handoff", "path": "Path"}
    # What kind of drift the card announces (the user wants this stated plainly).
    _KIND = {"agents": "Agent behaviour", "handoffs": "Handoff drift", "path": "Path drift"}

    def _metric_dirs(signals):
        # distinct, order-preserving {name, dir} for the card ("metrics influenced" + which way)
        out, seen = [], set()
        for s in signals:
            if s["metric"] in seen:
                continue
            seen.add(s["metric"])
            out.append({"name": s["metric"], "dir": s.get("direction")})
        return out

    _mcfg = cfg
    _units = {m: cc.get("unit") for m, cc in _mcfg.get("metrics", {}).items()}

    def _signed(s):
        # signed delta in the metric's own unit (pct -> %, rate -> pp, else raw)
        val = s.get("bad_delta") or 0
        signed = val if s.get("direction") == "up" else -val
        u = _units.get(s["metric"])
        suf = "%" if u == "pct" else (" pp" if u == "pp" else "")
        return f'{s["entity"]} {s["metric"]} {signed:+g}{suf}'

    def _route_change(drift_start=None):
        # the usual route vs the new route(s) in the DRIFT WINDOW (from drift_start on).
        # Matched by route_key (DAG topology when available) so parallel reorderings
        # aren't counted as new routes; each route carries its real nodes + edges.
        from analysis.changes import route_key
        from collections import Counter
        if not runs:
            return None
        keyed = [(route_key(r), r) for r in runs]
        keyed = [(k, r) for k, r in keyed if k]
        if not keyed:
            return None
        canon_key, _ = Counter(k for k, _ in keyed).most_common(1)[0]
        if drift_start is not None and 0 <= drift_start < len(keyed):
            start_i = drift_start
        else:                                          # fallback: last recent_runs
            start_i = max(0, len(keyed) - _mcfg.get("recent_runs", 10))
        window = keyed[start_i:]
        n_win = len(window)
        off = [(k, r) for k, r in window if k != canon_key]
        if not off or not n_win:
            return None

        def _shape(r):
            nodes = r.get("_route_nodes") or json.loads(r.get("agent_sequence") or "[]")
            edges = r.get("_route_edges") or [list(e) for e in zip(nodes, nodes[1:])]
            return nodes, edges

        canon_nodes, canon_edges = _shape(next(r for k, r in keyed if k == canon_key))
        variants = []
        for k, cnt in Counter(k for k, _ in off).most_common(3):
            nodes, edges = _shape(next(r for kk, r in window if kk == k))
            first = next((i for i, (kk, _) in enumerate(keyed) if kk == k and i >= start_i), None)
            variants.append({"nodes": nodes, "edges": edges,
                             "pct": round(100 * cnt / n_win), "first_run": first})
        return {"canon_nodes": canon_nodes, "canon_edges": canon_edges,
                "off_rate_pct": round(100 * len(off) / n_win),
                "window_start": start_i, "variants": variants}

    investigations = []
    # An entity explained by a causal chain (its root OR its symptom) must NOT also
    # appear as a standalone component finding — that's the SAME drift shown twice.
    # The chain is the richer, unified view, so it wins; collect what it covers.
    chain_covered: set = set()
    for i, c in enumerate(inv["chains"]):                       # likely-cause chains first
        card = _build_chain_cards([c])[0]
        involved = [(s["category"], s["entity"], s["metric"]) for s in c["root_signals"]]
        sm = c["symptom_signal"]
        involved.append((sm["category"], sm["entity"], sm["metric"]))
        chain_covered.update((cat, ent) for cat, ent, _ in involved)
        rcat = c["root_signals"][0]["category"] if c.get("root_signals") else sm["category"]
        # severity = the SYMPTOM's impact severity (how bad); confidence is separate.
        investigations.append({"id": f"chain{i}", "kind": "chain", "title": c["root_agent"],
                               "type_label": "Likely cause", "kind_label": _KIND.get(rcat, _SCOPE.get(rcat, rcat) + " drift"),
                               "metrics": _metric_dirs(list(c["root_signals"]) + [sm]),
                               "severity": sm.get("band", "drift"),
                               "confidence": c["confidence"],
                               "subtitle": f'Scope: {c["root_agent"]} · downstream impact on {c["symptom_agent"]}',
                               "card": card, "involved": involved, "drift_start": c["drift_start"],
                               "next_checks": card["next_checks"], "related_event": card.get("related_event")})
    for j, f in enumerate(inv["findings"]):                     # component findings
        if (f["scope"], f["entity"]) in chain_covered:          # already told by a chain
            continue
        involved = [(f["trigger"]["scope"], f["entity"], f["trigger"]["metric"])]
        involved += [(s["scope"], s["entity"], s["metric"]) for s in f["supporting"]]
        route_change = _route_change(f["drift_start"]) if f["scope"] == "path" else None
        # An independent finding still has a probable cause (a co-timed config change)
        # and next steps — same right-column treatment as a chain, minus the causal walk.
        ch = f.get("related_change")
        ent, ds, trig = f["entity"], f["drift_start"], f["trigger"]["metric"]
        related_event = (f'{ch.get("scope")} {ch.get("dimension", "config")} changed near run '
                         f'{ch.get("run_index")}') if ch else None
        next_checks = []
        if ch:
            next_checks.append(f'Compare {ent} before vs after the {ch.get("dimension")} '
                               f'change near run {ch.get("run_index")}.')
        next_checks.append(f'Open a representative run from the drift window (~run {ds}) '
                           f'and inspect {ent} · {trig}.')
        sup = [s["metric"] for s in f["supporting"][:3]]
        next_checks.append((f'Check whether {", ".join(sup)} moved together with {trig} — '
                            f'is this {ent}-local or systemic?') if sup
                           else f'Check {ent}\'s upstream/downstream neighbours for a co-timed change.')
        investigations.append({"id": f"find{j}", "kind": "finding", "title": f["entity"],
                               "type_label": _SCOPE.get(f["scope"], f["scope"]) + " drift",
                               "kind_label": _KIND.get(f["scope"], _SCOPE.get(f["scope"], f["scope"]) + " drift"),
                               "metrics": _metric_dirs([f["trigger"]] + f["supporting"]),
                               "what_changed": [_signed(s) for s in [f["trigger"]] + f["supporting"]],
                               "route_change": route_change,
                               "severity": f["severity"], "confidence": None,
                               "subtitle": f'Scope: {f["entity"]}', "involved": involved,
                               "drift_start": f["drift_start"],
                               "next_checks": next_checks, "related_event": related_event})

    _rank = {"high": 0, "drift": 1, "candidate": 2, "watch": 3}
    sort = (sort or "sev_desc").lower()
    if sort not in ("sev_desc", "sev_asc"):
        sort = "sev_desc"
    sign = -1 if sort == "sev_asc" else 1          # desc = most severe first
    investigations.sort(key=lambda x: (sign * _rank.get(x["severity"], 4), -(x["drift_start"] or 0)))
    return investigations, series, clog


def investigations_for(*, start=None, end=None, range_param="30d", task_type="",
                       base=None, compared=False, to=None, sort="sev_desc"):
    """Request-free core of the Drift Investigation loader: parse the range, load
    and filter the real-timeline runs, apply optional version cohorting, run
    investigate(), and rank the investigations (likely-cause chains + tiered
    component findings). Shared by /drift2 (via _drift2_investigations) and the
    AgentPulse MCP server so every surface agrees on what 'drift' means.
    Returns (investigations, ctx)."""
    import datetime as _dt
    import re
    from analysis.metric_series import per_run_series
    from analysis.changes import build_change_log
    from analysis.drift_detect import investigate
    from analysis.drift_config import load_drift_config

    band_cfg = _band_cfg()
    now = _dt.datetime.now(_dt.timezone.utc)

    # --- Time range (same parsing as v1) ---
    range_param = range_param or "30d"
    RANGES = [("15m", "Last 15 Min", _dt.timedelta(minutes=15)), ("1h", "Last Hour", _dt.timedelta(hours=1)),
              ("12h", "Last 12 Hours", _dt.timedelta(hours=12)), ("1d", "Last Day", _dt.timedelta(days=1)),
              ("7d", "Last 7 Days", _dt.timedelta(days=7)), ("30d", "Last Month", _dt.timedelta(days=30))]
    rmap = {k: (l, d) for k, l, d in RANGES}
    _MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _fmtd = lambda dt_: f"{_MON[dt_.month]} {dt_.day}"

    def _parse(rp):
        if rp in rmap:
            return rmap[rp]
        m = re.match(r"^(\d+)(m|h|d)$", rp or "")
        if m:
            n, u = int(m.group(1)), m.group(2)
            return (f"Last {n}{u}", _dt.timedelta(minutes=n) if u == "m"
                    else _dt.timedelta(hours=n) if u == "h" else _dt.timedelta(days=n))
        return rmap["7d"]

    if start and end:
        lo_dt = _dt.datetime.fromisoformat(start).replace(tzinfo=_dt.timezone.utc)
        hi_dt = _dt.datetime.fromisoformat(end).replace(hour=23, minute=59, second=59, tzinfo=_dt.timezone.utc)
        range_badge, active_range = "custom", ""
    else:
        _, delta = _parse(range_param)
        hi_dt, lo_dt = now, now - delta
        range_badge, active_range = range_param, range_param
    lo, hi = lo_dt.isoformat(), hi_dt.isoformat()       # filtering stays in UTC
    from zoneinfo import ZoneInfo
    lo_e = lo_dt.astimezone(ZoneInfo(_DISPLAY_TZ))       # display in Eastern
    hi_e = hi_dt.astimezone(ZoneInfo(_DISPLAY_TZ))
    if lo_e.date() == hi_e.date():
        range_display = f"{_fmtd(lo_e)}, {lo_e.year}"
    elif lo_e.year == hi_e.year:
        range_display = f"{_fmtd(lo_e)} – {_fmtd(hi_e)}, {hi_e.year}"
    else:
        range_display = f"{_fmtd(lo_e)}, {lo_e.year} – {_fmtd(hi_e)}, {hi_e.year}"
    rp, task_type = range_param, task_type or ""

    all_runs = list_runs(1000)
    runs = [r for r in all_runs if lo <= r.get("timestamp", "") <= hi
            and (not task_type or r.get("task_type") == task_type)]
    runs.sort(key=lambda r: r.get("timestamp", ""))
    task_types = sorted({r.get("task_type", "") for r in all_runs if r.get("task_type")})
    versions = get_versions()
    _vof = _ver_of_factory(versions)
    ver_options = sorted({1} | {v["version_num"] for v in versions})

    # Version comparison: a single Baseline (From, default = the chosen baseline
    # version) vs one or more Compare-With (To) versions. Default = baseline vs every
    # later version (same default Overview/Home use, so all three agree).
    base_default, _default_to = _default_from_to(versions)
    try:
        from_v = int(base) if base is not None else base_default
    except (TypeError, ValueError):
        from_v = base_default
    if from_v not in ver_options:
        from_v = base_default
    if compared:                                     # form was applied → honour the checkboxes
        to_vs = sorted({int(x) for x in (to or [])
                        if str(x).isdigit() and int(x) in ver_options and int(x) != from_v})
    else:                                            # first load → compare against later versions
        to_vs = [n for n in ver_options if n > from_v]

    runs, cfg_override, compare_on = _assemble_version_compare(runs, versions, from_v, to_vs)

    sort = (sort or "sev_desc").lower()
    if sort not in ("sev_desc", "sev_asc"):
        sort = "sev_desc"
    investigations, series, clog = _build_investigations(runs, sort, cfg_override)

    present = {_vof(r) for r in runs}
    markers = []
    for v in versions:
        if v["version_num"] not in present:
            continue
        for i, r in enumerate(runs):
            if r.get("timestamp", "") >= (v["created_at"] or ""):
                markers.append({"x": i, "label": f"v{v['version_num']}"})
                break

    ctx = {"band_cfg": band_cfg, "series": series, "clog": clog, "runs": runs, "markers": markers,
           "rp": rp, "ranges": RANGES, "range_display": range_display, "range_badge": range_badge,
           "active_range": active_range, "start": start, "end": end, "task_type": task_type,
           "task_types": task_types, "ver_options": ver_options, "baseline_version": base_default,
           "from_v": from_v, "to_vs": to_vs, "compare_on": compare_on, "sort": sort}
    return investigations, ctx


def _drift2_investigations(args):
    """Flask wrapper: pull params off request.args and delegate to investigations_for."""
    return investigations_for(
        start=args.get("start"), end=args.get("end"),
        range_param=args.get("range", "") or "30d",
        task_type=args.get("task_type", ""),
        base=args.get("base"), compared=bool(args.get("compared")),
        to=args.getlist("to"), sort=(args.get("sort") or "sev_desc"))


@app.route("/drift2")
def drift_investigation_2():
    """Drift Investigation — tiered + causal engine in a 3-column shell. Runs
    on the real run timeline (no version-compare reindex, which breaks the causal
    walk); the Version pill scopes the analysis to a single version."""
    import statistics
    investigations, ctx = _drift2_investigations(request.args)
    series, band_cfg = ctx["series"], ctx["band_cfg"]
    sel = next((x for x in investigations if x["id"] == request.args.get("sel")), None)
    if not sel and request.args.get("finding"):       # deep-link from Event Timeline (cat|entity)
        _ent = request.args["finding"].split("|")[-1]
        sel = next((x for x in investigations if x["title"] == _ent), None)
    sel = sel or (investigations[0] if investigations else None)

    def _pctile(sorted_v, q):
        return sorted_v[min(len(sorted_v) - 1, int(q * (len(sorted_v) - 1)))] if sorted_v else 0
    charts, seen = [], set()
    if sel:
        for scope, entity, metric in sel["involved"]:
            if (scope, entity, metric) in seen:
                continue
            seen.add((scope, entity, metric))
            pts = series.get(scope, {}).get(entity, {}).get(metric)
            if not pts:
                continue
            ys = [p["y"] for p in pts]
            base = sorted(ys[:band_cfg["baseline_runs"]] or ys)
            charts.append({"label": f"{entity} · {metric}", "metric": metric,
                           "help": _METRIC_HELP.get(metric, ""),
                           "points": [{"x": p["x"], "y": p["y"]} for p in pts],
                           "band": {"mean": round(statistics.mean(base), 4) if base else 0,
                                    "p10": _pctile(base, 0.1), "p90": _pctile(base, 0.9)}})

    # DAG flowcharts for a Path-drift finding: the common route + each new route,
    # using the real (possibly parallel) topology edges.
    path_graphs, route_variants = {}, []
    rc = sel.get("route_change") if sel else None
    if rc and rc.get("canon_nodes"):
        cn = rc["canon_nodes"]
        ce = [tuple(e) for e in rc["canon_edges"]]
        ceset = set(ce)
        path_graphs["pgPathCanon"] = build_path_graph(cn, [list(e) for e in ce])
        for vi, v in enumerate(rc.get("variants", [])):
            vn = v["nodes"]
            ve = [tuple(e) for e in v["edges"]]
            veset = set(ve)
            gid = f"pgPathVar{vi}"
            path_graphs[gid] = build_path_graph(
                list(dict.fromkeys(list(cn) + list(vn))),
                [list(e) for e in dict.fromkeys(ce + ve)],
                added_nodes=set(vn) - set(cn), removed_nodes=set(cn) - set(vn),
                added_edges=[list(e) for e in ve if e not in ceset],
                removed_edges=[list(e) for e in ce if e not in veset])
            route_variants.append({"gid": gid, "pct": v["pct"], "first_run": v["first_run"]})

    # Resolve prompt hashes → readable text; if the full text wasn't captured, show
    # nothing (a raw hash is meaningless in the timeline).
    from storage.sqlite_store import get_prompts
    _ptext = get_prompts() or {}

    def _ev_val(dim, raw):
        if dim == "prompt":
            t = _ptext.get(raw)
            return (t[:48] + "…") if t and len(t) > 48 else (t or "")
        return str(raw)[:48]

    timeline = sorted([{"run": e["run_index"], "ts": e.get("timestamp", ""), "scope": e["scope"],
                        "dimension": e["dimension"], "old": _ev_val(e["dimension"], e["old"]),
                        "new": _ev_val(e["dimension"], e["new"])} for e in ctx["clog"]], key=lambda t: t["run"])

    ds = sel["drift_start"] if sel else None

    def _cl_link(e):       # → the Event Timeline tab (/changelog), deep-linked to this event
        return f'/changelog?find={e["dimension"]}:{e["scope"]}'

    def _chg_title(e):
        return (f'{e["dimension"].title()} change · {e["scope"]}'
                if e["scope"] != "workflow" else f'{e["dimension"].title()} change')

    # Potentially related changes — RULE-BASED candidate selection (not a symmetric
    # time window). A config change is a plausible cause of the selected drift when:
    #   1. change_run <= drift_start          — the cause precedes the effect
    #   2. within the last N=3 runs before it — recent, not ancient history
    #   3. on the SAME component or an UPSTREAM one (topology from handoff edges)
    #   4. dimension is a real config knob: prompt / model / tools / params
    #   5. the metric drifted AFTER the change — guaranteed by (1): ds >= change_run.
    # Ranked: same-component before upstream, stronger dimension first
    # (prompt > model > tools > params), then most recent. The window + rule are
    # shared with severity escalation (_nearby_change) via the drift config.
    from analysis.drift_config import load_drift_config
    RELATED_N = int((load_drift_config().get("metric_drift") or {}).get("related_change_window", 3))
    _DIM_RANK = {"prompt": 0, "model": 1, "tools": 2, "params": 3}

    # Component graph from observed handoffs (a → b): used to find what's upstream of
    # the drifting component. A change downstream of it can't be its cause.
    _edges = []
    for _key in series.get("handoffs", {}):
        _a, _, _b = _key.partition(" → ")
        if _a.strip() and _b.strip():
            _edges.append((_a.strip(), _b.strip()))

    def _upstream_closure(targets):
        import collections
        rev = collections.defaultdict(set)
        for a, b in _edges:
            rev[b].add(a)
        seen, stack = set(targets), list(targets)
        while stack:
            for p in rev.get(stack.pop(), ()):
                if p not in seen:
                    seen.add(p); stack.append(p)
        return seen

    def _sel_components(s):
        # the agent(s) the finding is ABOUT (its own component)
        if not s:
            return set()
        if s["kind"] == "chain":
            return {s["title"]}                                  # the root agent
        scope = s["involved"][0][0] if s.get("involved") else "agents"
        ent = s["title"]
        if scope == "handoffs":
            a, _, b = ent.partition(" → ")
            return {a.strip(), b.strip()}
        if scope == "path":
            return set()                                         # system-wide: no single component
        return {ent}                                             # an agent

    related_changes = []
    if ds is not None and sel:
        own = _sel_components(sel)
        relevant = _upstream_closure(own) if own else None       # None = system-wide (any agent)
        cands = []
        for e in timeline:
            if not (0 <= ds - e["run"] <= RELATED_N):             # rules 1 + 2
                continue
            if e["dimension"] not in _DIM_RANK:                   # rule 4
                continue
            if relevant is not None and e["scope"] not in relevant:   # rule 3
                continue
            same = bool(own) and e["scope"] in own
            cands.append(((0 if same else 1), _DIM_RANK[e["dimension"]], ds - e["run"], e))
        cands.sort(key=lambda t: t[:3])
        for _same, _dim, _rec, e in cands[:4]:
            related_changes.append({
                "kind": e["dimension"], "run": e["run"],
                "title": (f'{e["scope"].title()} {e["dimension"]} changed near run {e["run"]}'
                          if e["scope"] != "workflow" else f'{e["dimension"].title()} changed near run {e["run"]}'),
                "subtitle": (f'{e["old"]} → {e["new"]}' if (e["old"] or e["new"]) else ""),
                "link": _cl_link(e)})

    # Event timeline — only what happened AROUND the drift window (5 runs before →
    # 3 runs after drift_start), in chronological order. The full history is one
    # click away via "View full timeline". The drift point is shown as a divider.
    WIN_BEFORE, WIN_AFTER = 5, 3

    def _near_drift(run):
        return ds is None or (ds - WIN_BEFORE) <= run <= (ds + WIN_AFTER)

    events_tl = []
    for e in timeline:
        if not _near_drift(e["run"]):
            continue
        events_tl.append({"run": e["run"], "cls": "config", "title": _chg_title(e),
                          "detail": (f'{e["old"]} → {e["new"]}' if (e["old"] or e["new"]) else ""),
                          "when": e["ts"], "link": _cl_link(e)})
    for m in ctx["markers"]:
        if not _near_drift(m["x"]):
            continue
        events_tl.append({"run": m["x"], "cls": "version", "title": "Version change",
                          "detail": f'Deployed {m["label"]}', "when": "",
                          "link": f'/changelog?event={m["label"]}'})
    events_tl.sort(key=lambda x: x["run"])              # chronological (oldest → newest)

    return render_template(
        "drift2.html", investigations=investigations, sel=sel,
        charts_json=json.dumps(charts), markers_json=json.dumps(ctx["markers"]),
        path_graphs_json=json.dumps(path_graphs), route_variants=route_variants,
        timeline=timeline, related_changes=related_changes, events_tl=events_tl,
        run_count=len(ctx["runs"]),
        drift_start=(sel["drift_start"] if sel else None),
        rp=ctx["rp"], ranges=ctx["ranges"], range_display=ctx["range_display"],
        range_badge=ctx["range_badge"], active_range=ctx["active_range"],
        start=ctx["start"], end=ctx["end"], task_type=ctx["task_type"],
        task_types=ctx["task_types"], ver_options=ctx["ver_options"],
        baseline_version=ctx["baseline_version"],
        from_v=ctx["from_v"], to_vs=ctx["to_vs"], compare_on=ctx["compare_on"],
        sort=ctx["sort"])


@app.route("/drift2/next-checks")
def drift2_next_checks():
    """On-demand LLM 'Next check' suggestions for the selected drift2 finding."""
    from analysis.diagnose import suggest_next_checks
    investigations, _ = _drift2_investigations(request.args)
    sel = next((x for x in investigations if x["id"] == request.args.get("sel")), None) \
        or (investigations[0] if investigations else None)
    if not sel:
        return jsonify({"ok": False, "text": "No finding selected."})
    card = sel.get("card") or {}
    _is_chain = sel.get("kind") == "chain"
    context = {
        "instruction": ("You are a senior agent-ops engineer. This is a drift finding "
                        + ("— a LIKELY cause traced upstream from a symptom (correlational, not proven). "
                           if _is_chain else
                           "— a metric that drifted on one component, with a co-timed config change as the "
                           "probable cause (correlational, not proven). ")
                        + "List the 3-5 most efficient NEXT CHECKS to confirm or refute the cause and fix "
                        "it. Be specific and concise; output a plain bullet list, one check per line "
                        "starting with '- '."),
        "finding": sel["title"], "type": sel["type_label"], "severity": sel["severity"],
        "confidence": sel.get("confidence"), "path": card.get("path", sel["subtitle"]),
        "what_changed": card.get("what_changed") or sel.get("what_changed") or [sel["subtitle"]],
        "why_it_matters": card.get("why", ""),
        "potentially_related_event": card.get("related_event") or sel.get("related_event"),
    }
    try:
        return jsonify({"ok": True, "text": suggest_next_checks(context)})
    except Exception as e:
        return jsonify({"ok": False, "text": f"AI suggestion unavailable ({e.__class__.__name__})."})


@app.route("/timeline")
def timeline_view():
    """Event timeline — config changes over time. When reached from a red metric
    (scope + around), highlight the changes near the drift-start that touch the
    same workflow area (potentially related — never a confirmed cause)."""
    import datetime as _dt
    from analysis.changes import build_change_log

    days = int(request.args.get("days", 365))
    scope = request.args.get("scope")          # agent name or "A → B"
    around = request.args.get("around")        # run index of the drift start
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)
    runs = [r for r in list_runs(limit=500) if r.get("timestamp", "") >= cutoff.isoformat()]
    runs.sort(key=lambda r: r.get("timestamp", ""))

    events = build_change_log(runs)
    has_config = any(r.get("config_json") for r in runs)

    # Version snapshots = "big" events that are always visible on the line.
    big_events = []
    for v in get_versions():
        for i, r in enumerate(runs):
            if r.get("timestamp", "") >= (v["created_at"] or ""):
                big_events.append({"x": i, "label": f"v{v['version_num']}",
                                   "detail": f"Version {v['version_num']} snapshot"})
                break

    # Plot points: one per change event (hover for detail; workflow changes are bigger).
    pts = [{"x": e["run_index"], "ts": _fmt_et(e.get("timestamp", "")), "scope": e["scope"],
            "dimension": e["dimension"], "old": str(e["old"]), "new": str(e["new"]),
            "big": e["dimension"] == "workflow"} for e in events]

    anchor = None
    scoped = None
    if scope and around not in (None, "", "None"):
        try:
            anchor = int(around)
        except ValueError:
            anchor = None
        names = {p.strip() for p in scope.replace("→", "|").split("|") if p.strip()}
        rel = [e for e in events
               if (anchor is None or (anchor - 20) <= e["run_index"] <= anchor)
               and (e["scope"] == "workflow" or e["scope"] in names)]
        scoped = {"scope": scope, "anchor": anchor, "events": list(reversed(rel))}

    return render_template(
        "timeline.html",
        events=list(reversed(events)), scoped=scoped, anchor=anchor,
        has_config=has_config, days=days, run_count=len(runs),
        points_json=json.dumps(pts), big_json=json.dumps(big_events),
        x_min=(anchor - 20) if anchor is not None else None,
        x_max=(anchor + 2) if anchor is not None else None,
    )


@app.route("/run/<run_id>/delete", methods=["POST"])
def delete_run(run_id: str):
    """Delete a single run + its spans/handoffs/tool_calls from the active DB."""
    from storage.sqlite_store import get_connection
    conn = get_connection()
    conn.execute("DELETE FROM tool_calls WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM handoffs   WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM spans      WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs       WHERE run_id = ?", (run_id,))
    conn.commit()
    return jsonify({"ok": True, "deleted": run_id})


@app.route("/version/snapshot", methods=["POST"])
def version_snapshot():
    """Save the current settings as the next version (v1 is the implicit baseline)."""
    label = None
    if request.is_json and request.json:
        label = request.json.get("label")
    label = label or request.form.get("label")
    num = create_version_snapshot(label, latest_run_config())
    return jsonify({"ok": True, "version": num})


@app.route("/api/runs")
def api_runs():
    return jsonify(list_runs(limit=200))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    # Render / Fly / Railway / Heroku all set $PORT. Local dev falls back to 5001.
    # Bind to 0.0.0.0 so the platform's load balancer can reach the process.
    port = int(os.environ.get("PORT", 5001))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    # Template auto-reload: pick up .html edits without a process restart.
    # (Python code changes still need a restart — only the templates hot-swap.)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    print(f"Dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host=host, port=port, debug=False)
