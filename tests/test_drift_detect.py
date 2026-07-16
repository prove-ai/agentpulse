#!/usr/bin/env python3
"""Unit test for classify_drift — the tiered, directional, corroborated detector.

Synthetic 20-run series (baseline 0-9, recent 10-19): writer's SUCCESS drops
(Tier-0 outcome), with co-timed supporting evidence (writer latency up, the
analyst->writer handoff payload up) and a nearby writer config change. Expected:
one HIGH finding on writer, trigger=success, supporting includes the others.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.drift_config import load_drift_config       # noqa: E402
from analysis.drift_detect import classify_drift          # noqa: E402


def _pts(vals):
    return [{"x": i, "y": v, "run_id": f"r{i}", "ts": f"t{i:02d}"} for i, v in enumerate(vals)]


def test_escalation_to_high():
    base, rec = 10, 10
    series = {
        "agents": {"writer": {
            "success":   _pts([1.0] * base + [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # ~30% recent
            "latency_s": _pts([2.0] * base + [3.2] * rec),     # +60% magnitude
        }},
        "handoffs": {"analyst → writer": {
            "payload_tokens": _pts([100] * base + [180] * rec),  # +80%
        }},
        "path": {}, "parallel": {},
    }
    # The change must land AT or BEFORE the drift start (run 10) — a change that
    # happens after drift began can't be its cause under the attribution rule.
    change_log = [{"run_index": 10, "scope": "writer", "dimension": "prompt", "old": "h1", "new": "h2"}]
    cfg = load_drift_config().get("metric_drift")

    findings = classify_drift(series, change_log, cfg)
    writer = [f for f in findings if f["entity"] == "writer"]
    assert writer, "expected a writer finding"
    f = writer[0]
    assert f["trigger"]["metric"] == "success", f["trigger"]["metric"]
    assert f["severity"] == "high", f["severity"]
    sup = {s["metric"] for s in f["supporting"]}
    assert "latency_s" in sup and "payload_tokens" in sup, sup
    assert f["related_change"] and f["related_change"]["dimension"] == "prompt"
    print("HIGH finding:", f["summary"])

    # guard: a lone behaviour change (no outcome) must NOT be a Tier-0 drift
    behaviour_only = {"agents": {"x": {"tokens": _pts([100] * base + [160] * rec)}},
                      "handoffs": {}, "path": {}, "parallel": {}}
    f2 = classify_drift(behaviour_only, [], cfg)
    assert all(x["trigger"]["tier"] != 0 for x in f2), "tokens alone should not trigger a Tier-0 finding"


if __name__ == "__main__":
    test_escalation_to_high()
    print("PASS — Tier-0 + corroboration + change => HIGH; behaviour-only stays non-trigger.")
