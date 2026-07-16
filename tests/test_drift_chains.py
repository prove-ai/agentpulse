#!/usr/bin/env python3
"""Unit test for the drift causal layer (analysis/drift_chains).

Synthetic scenario (no DB): pipeline researcher → analyst → writer.
  - analyst's output tokens DROP at run 45 (a behaviour change — looks benign)
  - writer's success FALLS at run 47 (the impact / symptom)
  - an analyst prompt change happened at run 45 (the trigger)
Expected: ONE chain, root=analyst (behaviour), symptom=writer (success down),
triggered by the analyst prompt change, high confidence.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.drift_chains import build_topology, detect_chains  # noqa: E402


def test_causal_chain():
    topology = build_topology([("researcher", "analyst"), ("analyst", "writer")])

    # enriched drift signals (as enrich_drift_signals would produce)
    signals = [
        {"category": "agents", "entity": "analyst", "metric": "tokens",
         "drift_start": 45, "direction": "down", "pct": -42, "kind": "behaviour"},
        {"category": "agents", "entity": "writer", "metric": "success",
         "drift_start": 47, "direction": "down", "pct": -15, "kind": "impact"},
        # noise: an unrelated downstream behaviour drift that must NOT become a root
        {"category": "agents", "entity": "writer", "metric": "latency_s",
         "drift_start": 47, "direction": "up", "pct": 30, "kind": "behaviour"},
    ]
    change_log = [{"run_index": 45, "scope": "analyst", "dimension": "prompt",
                   "old": "h1", "new": "h2"}]

    chains = detect_chains(signals, topology, change_log)
    assert len(chains) == 1, f"expected 1 chain, got {len(chains)}"
    c = chains[0]
    assert c["root_agent"] == "analyst", c["root_agent"]
    assert c["symptom_agent"] == "writer", c["symptom_agent"]
    assert c["trigger"] and c["trigger"]["dimension"] == "prompt"
    assert c["confidence"] == "high"
    print("chain:", c["summary"])

    # guard: with no upstream behaviour change, no chain is invented
    none = detect_chains([signals[1]], topology, change_log)
    assert none == [], "should not invent a chain without an upstream cause"


if __name__ == "__main__":
    test_causal_chain()
    print("PASS — causal chain detected (analyst → writer), trigger linked, no false chains.")
