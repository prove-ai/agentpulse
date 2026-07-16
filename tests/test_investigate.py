#!/usr/bin/env python3
"""End-to-end test: classify_drift -> detect_chains via investigate().

Synthetic 20-run series, pipeline researcher -> analyst -> writer:
  - analyst tokens DROP (upstream behaviour, looks benign)
  - the analyst->writer handoff payload DROPS (the mechanism)
  - writer SUCCESS falls (downstream outcome / symptom)
  - an analyst prompt change happened in the window (the trigger)
investigate() should: (a) surface a writer drift finding, and (b) build a causal
chain whose ROOT is analyst and SYMPTOM is writer, with the trigger attached.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.drift_config import load_drift_config        # noqa: E402
from analysis.drift_detect import investigate              # noqa: E402


def _pts(vals):
    return [{"x": i, "y": v, "run_id": f"r{i}", "ts": f"t{i:02d}"} for i, v in enumerate(vals)]


def test_investigate_links_chain():
    base, rec = 10, 10
    series = {
        "agents": {
            "analyst": {"tokens": _pts([1000] * base + [560] * rec)},        # -44% behaviour
            "writer":  {"success": _pts([1.0] * base + [0, 0, 1, 0, 0, 0, 1, 0, 0, 0])},  # outcome down
        },
        "handoffs": {"analyst → writer": {"payload_tokens": _pts([1000] * base + [560] * rec)}},
        "path": {}, "parallel": {},
    }
    change_log = [{"run_index": 11, "scope": "analyst", "dimension": "prompt", "old": "h1", "new": "h2"}]
    cfg = load_drift_config().get("metric_drift")

    res = investigate(series, change_log, cfg)
    assert any(f["entity"] == "writer" for f in res["findings"]), "expected a writer finding"
    chains = res["chains"]
    assert chains, "expected at least one causal chain"
    c = next((c for c in chains if c["symptom_agent"] == "writer"), None)
    assert c, "expected a chain with writer as the symptom"
    assert c["root_agent"] == "analyst", c["root_agent"]
    assert c["trigger"] and c["trigger"]["dimension"] == "prompt"
    print("finding + chain:", c["summary"])


if __name__ == "__main__":
    test_investigate_links_chain()
    print("PASS — investigate() links a Tier-0 outcome symptom to its upstream root, with trigger.")
