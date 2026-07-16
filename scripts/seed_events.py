#!/usr/bin/env python3
"""Seed the demo project's Event Timeline with realistic rich events.

This is demo data. In production these rows are written by the capture layer
(SDK / UI / CI) — the events table is the home for that. Run:

    python scripts/seed_events.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
random.seed(11)

from analysis.layer1_raw import list_runs                       # noqa: E402
from storage.sqlite_store import (resolve_db_path, set_active_db_path,   # noqa: E402
                                  clear_events, upsert_event)

set_active_db_path(resolve_db_path("demo"))
runs = sorted(list_runs(1000), key=lambda r: r.get("timestamp", ""))


def ts(i):
    return runs[i]["timestamp"] if 0 <= i < len(runs) else (runs[-1]["timestamp"] if runs else "")


def _trend(direction, n=22, start=10.0):
    v, out = start, []
    for _ in range(n):
        v += (1 if direction == "up" else -1) * random.uniform(0.3, 1.3) + random.uniform(-0.5, 0.5)
        out.append(v)
    return out


def spark(direction, w=120, h=26, pad=2):
    vals = _trend(direction)
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    n = len(vals)
    return " ".join(f"{round(i/(n-1)*(w-2*pad)+pad,1)},{round(h-pad-((v-lo)/rng)*(h-2*pad),1)}"
                    for i, v in enumerate(vals))


def imp(items):
    """Each item is either (label, delta_pct, dir) for a relative change, or
    (label, before, after, dir) to show the actual before → after values."""
    out = []
    for it in items:
        if len(it) == 4:
            l, before, after, d = it
            out.append({"label": l, "val": f"{before} → {after}", "dir": d})
        else:
            l, p, d = it
            out.append({"label": l, "pct": p, "dir": d})
    return json.dumps(out)


EVENTS = [
    {"id": "45:prompt:analyst", "ts": ts(45), "run_index": 45, "type": "config_change", "dim": "prompt",
     "component": "analyst", "title": "Analyst prompt changed",
     "before": "Pass a concise signal to the writer.",
     "after": "Pass a detailed thesis with prior context and supporting evidence to the writer.",
     "author": "Alex Chen", "source": "UI change", "environment": "production", "changed_fields": 1,
     "hash_before": "d851f05cbb61", "hash_after": "457c80614e1c", "related_drift": "agents|writer",
     "impact_json": imp([("Writer payload", "+72%", "up"), ("Writer latency", "+31%", "up"),
                         ("Downstream retry rate", "8%", "20%", "up")])},
    {"id": "v2", "ts": ts(45), "run_index": 45, "type": "version", "component": "—",
     "title": "Saved as v2 — analyst prompt", "author": "Alex Chen", "source": "Snapshot",
     "environment": "production", "impact_json": "[]"},
    {"id": "release:46", "ts": ts(46), "run_index": 46, "type": "release", "component": "—",
     "title": "Promoted to production", "author": "Alex Chen", "source": "CI/CD",
     "environment": "production", "impact_json": "[]"},
    {"id": "60:model:writer", "ts": ts(60), "run_index": 60, "type": "config_change", "dim": "model",
     "component": "writer", "title": "Writer model changed", "before": "gpt-4o-mini", "after": "gpt-4o",
     "author": "Sam Lee", "source": "API", "environment": "production", "changed_fields": 1,
     "hash_before": "", "hash_after": "", "related_drift": "agents|writer",
     "impact_json": imp([("Writer cost / run", "+694%", "up"), ("Writer latency", "+44%", "up")])},
    {"id": "drift:61", "ts": ts(61), "run_index": 61, "type": "drift", "component": "writer",
     "title": "Success rate breached control band", "source": "Auto-detected", "environment": "production",
     "related_drift": "agents|writer", "impact_json": "[]"},
    {"id": "impact:66", "ts": ts(66), "run_index": 66, "type": "impact", "component": "writer",
     "title": "Downstream retry rate increased", "source": "Auto-detected", "environment": "production",
     "related_drift": "agents|writer", "impact_json": "[]"},
    {"id": "evidence:66", "ts": ts(66), "run_index": 66, "type": "evidence", "component": "writer",
     "title": "Representative slow run #66", "source": "Auto-detected", "environment": "production",
     "impact_json": "[]"},
    {"id": "75:tools:researcher", "ts": ts(75), "run_index": 75, "type": "config_change", "dim": "tools",
     "component": "researcher", "title": "Researcher tools changed", "before": "(none)", "after": "web_search",
     "author": "Alex Chen", "source": "SDK", "environment": "production", "changed_fields": 1,
     "related_drift": "agents|researcher",
     "impact_json": imp([("Researcher tool calls", "0", "1", "up"), ("Researcher latency", "+18%", "up")])},
    {"id": "v3", "ts": ts(75), "run_index": 75, "type": "version", "component": "—",
     "title": "Saved as v3 — writer gpt-4o + research tool", "author": "Alex Chen", "source": "Snapshot",
     "environment": "production", "impact_json": "[]"},
]

clear_events()
for e in EVENTS:
    upsert_event(e)
print(f"Seeded {len(EVENTS)} events into demo.db")
