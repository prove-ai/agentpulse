"""Load and apply the handoff-drift rules from config/drift_rules.yaml.

Keeping the rules in YAML means thresholds, rule logic, severity weights,
risk cutoffs, and insight wording can all be tuned WITHOUT touching Python.
The dashboard reloads the file on each request, so edits take effect on the
next page refresh — no restart required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "drift_rules.yaml"

# Built-in fallback so the dashboard still works if the YAML is missing/broken.
_FALLBACK: dict[str, Any] = {
    "rules": {},
    "severity_weights": {
        "success_drop": 3, "reinvocation_up": 2, "remaining_duration_up": 2,
        "remaining_tokens_up": 2, "payload_change_abs": 1, "frequency_change_abs": 1,
    },
    "risk": {"high_success_drop": 10, "high_severity": 200, "low_severity": 60},
    "guards": {
        "min_occurrences_per_half": 3, "unspecified_floor": 30,
        "replacement_min_share_gain": 15, "noise_floor": 10,
    },
}


def load_drift_config() -> dict:
    """Read the YAML fresh. Falls back to built-in defaults on any error."""
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return _FALLBACK
    # Backfill any missing top-level sections from the fallback.
    for key, default in _FALLBACK.items():
        cfg.setdefault(key, default)
    return cfg


# ---------------------------------------------------------------------------
# Condition + rule evaluation
# ---------------------------------------------------------------------------
def _check(deltas: dict[str, float], cond: dict) -> bool:
    """Evaluate a single {metric, op, value} condition against the delta dict."""
    val = deltas.get(cond.get("metric"), 0.0)
    op  = cond.get("op")
    thr = cond.get("value", 0.0)
    if op == "gt":      return val > thr
    if op == "lt":      return val < thr
    if op == "gte":     return val >= thr
    if op == "lte":     return val <= thr
    if op == "abs_gt":  return abs(val) > thr
    if op == "abs_lt":  return abs(val) < thr
    return False


def rule_fires(rule: dict, deltas: dict[str, float]) -> bool:
    """A rule fires when ALL `all` conditions hold AND (any `any` holds, or
    `any` is empty)."""
    all_conds = rule.get("all") or []
    any_conds = rule.get("any") or []
    if not all(_check(deltas, c) for c in all_conds):
        return False
    if any_conds and not any(_check(deltas, c) for c in any_conds):
        return False
    return True


def evaluate(deltas: dict[str, float], cfg: dict) -> list[str]:
    """Return the names of every rule that fires, sorted by priority (asc)."""
    rules = cfg.get("rules", {})
    fired = [name for name, rule in rules.items() if rule_fires(rule, deltas)]
    fired.sort(key=lambda n: rules[n].get("priority", 999))
    return fired


def priority_order(cfg: dict) -> list[str]:
    """Rule names sorted by their configured priority."""
    rules = cfg.get("rules", {})
    return sorted(rules.keys(), key=lambda n: rules[n].get("priority", 999))
