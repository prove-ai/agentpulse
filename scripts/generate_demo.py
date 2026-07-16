#!/usr/bin/env python3
"""Generate a 120-run demo DB to explore the Metric Drift View + Event Timeline.

A 4-agent content pipeline (researcher → analyst → writer → critic) that runs
stable for a while, then has four deliberate changes. The first three each cause
a metric drift that has since settled; the fourth is an ONGOING regression at the
very end (writer swapped to a heavier model + rewritten prompt) that is still
elevated at the newest runs — so it surfaces as a live Critical finding.
Three version snapshots. Writes db/demo.db.

    python scripts/generate_demo.py
"""

import datetime
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Start from a clean slate. The importer keys spans/handoffs by a fresh UUID, so
# re-importing into an existing DB ACCUMULATES spans (old + new) instead of
# replacing them — which corrupts every per-agent metric. Delete the file first so
# `generate_demo.py` is idempotent and always produces exactly N runs.
from storage.sqlite_store import resolve_db_path  # noqa: E402

_demo_db = Path(resolve_db_path("demo"))
if _demo_db.exists():
    _demo_db.unlink()

random.seed(7)
N = 120
START = datetime.datetime(2026, 5, 17, 9, 0, tzinfo=datetime.timezone.utc)
SPACING_H = 8  # ~40 days total

# The ongoing (live) regression: the writer is swapped to gpt-4-turbo with a
# rewritten, long-form prompt at run 110 and never recovers through the end.
WRITER_REGRESS = 110
# Runs where the heavier writer actually fails (dense toward the end so the drift
# is still breaching in the final runs and survives the recovery/expiry gate).
WRITER_FAILS = {111, 112, 114, 115, 117, 118, 119}


def jitter(base, pct=0.05):
    return max(0, base * (1 + random.uniform(-pct, pct)))


def ts_of(k):
    return (START + datetime.timedelta(hours=SPACING_H * k)).isoformat()


runs = []
for k in range(N):
    analyst_drift   = k >= 45    # prompt change → bigger/slower analyst
    researcher_tool = k >= 75    # tool added → researcher tool calls up

    writer_regress  = k >= WRITER_REGRESS   # ONGOING: heavier writer, still live
    writer_failed   = k in WRITER_FAILS     # the heavy writer erroring on this run

    # Writer step: stable & cheap (gpt-4o-mini) the whole way, then a single hard
    # regression at run 110 — gpt-4-turbo + a rewritten long-form prompt — that
    # slows it down, spikes its cost/tokens, and makes it start failing. Because
    # the writer has NO earlier drift, this is the only sustained breach on it, so
    # the finding anchors cleanly to run 110 (the live event) instead of a stale one.
    if writer_regress:
        writer_step = {
            "agent": "writer", "model": "gpt-4-turbo",
            "input_tokens": int(jitter(1000)), "output_tokens": int(jitter(1500)),
            "duration_ms": int(jitter(17000)),
            "status": "ERROR" if writer_failed else "OK",
            "config": {"params": {"temperature": 0.7}, "tools": [],
                       "system_prompt": "WRITER v2 (long-form editorial rewrite)"}}
    else:
        writer_step = {
            "agent": "writer", "model": "gpt-4o-mini",
            "input_tokens": int(jitter(1000)), "output_tokens": int(jitter(600)),
            "duration_ms": int(jitter(3500)),
            "config": {"params": {"temperature": 0.7}, "tools": [],
                       "system_prompt": "WRITER v1"}}

    steps = [
        {"agent": "researcher", "model": "gpt-4o-mini",
         "input_tokens": int(jitter(800)), "output_tokens": int(jitter(400)),
         "duration_ms": int(jitter(3000)),
         "tools": ([{"name": "web_search", "success": True, "duration_ms": int(jitter(500))}]
                   if researcher_tool else []),
         "config": {"params": {"temperature": 0.3},
                    "tools": (["web_search"] if researcher_tool else []),
                    "system_prompt": "RESEARCHER v1"}},

        {"agent": "analyst", "model": "gpt-4o",
         "input_tokens": int(jitter(1200)),
         "output_tokens": int(jitter(900 if analyst_drift else 400)),
         "duration_ms": int(jitter(13000 if analyst_drift else 5000)),
         "config": {"params": {"temperature": 0.6}, "tools": [],
                    "system_prompt": "ANALYST v2 (write a detailed thesis)" if analyst_drift
                                     else "ANALYST v1 (give a brief signal)"}},

        writer_step,

        {"agent": "critic", "model": "gpt-4o-mini",
         "input_tokens": int(jitter(700)), "output_tokens": int(jitter(150)),
         "duration_ms": int(jitter(2500)),
         "config": {"params": {"temperature": 0.2}, "tools": [], "system_prompt": "CRITIC v1"}},
    ]

    # The pipeline is clean until the writer regression starts failing runs.
    clean = not writer_failed
    runs.append({
        "run_id": f"demo-{k:03d}", "task_type": "content-pipeline",
        "timestamp": ts_of(k),
        "status": "OK" if clean else "ERROR",
        "termination_reason": "completed" if clean else "error",
        "steps": steps,
    })

tmp = ROOT / "_demo_runs.json"
tmp.write_text(json.dumps({"db_name": "demo", "runs": runs}))
subprocess.run([sys.executable, str(ROOT / "scripts/import_runs.py"), str(tmp)], check=True)
tmp.unlink()

# Two version snapshots at historical points so the boundaries fall mid-history.
from storage.sqlite_store import resolve_db_path, get_connection  # noqa: E402

conn = get_connection(resolve_db_path("demo"))


def cfg_at(run_idx):
    row = conn.execute("SELECT config_json FROM runs ORDER BY timestamp LIMIT 1 OFFSET ?",
                       (run_idx,)).fetchone()
    return row[0] if row else "{}"


conn.execute("DELETE FROM versions")
conn.execute("INSERT INTO versions (version_num,label,created_at,config_json) VALUES (?,?,?,?)",
             (2, "v2 — analyst prompt", ts_of(45), cfg_at(45)))
conn.execute("INSERT INTO versions (version_num,label,created_at,config_json) VALUES (?,?,?,?)",
             (3, "v3 — researcher web_search tool", ts_of(75), cfg_at(75)))
conn.execute("INSERT INTO versions (version_num,label,created_at,config_json) VALUES (?,?,?,?)",
             (4, "v4 — writer gpt-4-turbo + long-form prompt", ts_of(WRITER_REGRESS),
              cfg_at(WRITER_REGRESS)))
conn.commit()

print(f"Generated db/demo.db: {N} runs, 3 changes "
      f"(analyst@45, researcher@75, writer regression@{WRITER_REGRESS} — live Critical), 3 versions.")
print("Open the dashboard, pick 'demo' in the sidebar, and try Metric Drift View + Event Timeline.")
