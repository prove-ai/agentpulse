#!/usr/bin/env python3
"""Seed a path-drift narrative into the demo project's agent chains.

The demo starts as a stable pipeline; in the recent window the chain shape
drifts — first a required step is skipped, then a new agent appears:

    runs 0–69   researcher → analyst → writer → critic     (stable baseline)
    runs 70–84  researcher → analyst → writer              (critic skipped)
    runs 85–99  researcher → analyst → writer → fact_checker  (new agent added)

To keep every derived signal consistent, this rebuilds, per run:
  - spans       (the chain, with each agent's original compute duration preserved
                 and sequential timing + a small realistic handoff gap so per-hop
                 latency is meaningful rather than zero),
  - handoffs    (one row per consecutive pair, with turn indices + tokens),
  - agent_sequence on the run (so path_length / loops match the spans).

The new fact_checker copies the writer's profile. Re-running is idempotent —
timing gaps use a fixed seed, so the same target chains and values are produced.

    python scripts/seed_path_drift.py
"""
import json
import random
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from storage.sqlite_store import resolve_db_path, get_connection  # noqa: E402

BASELINE = ["researcher", "analyst", "writer", "critic"]
SKIP = ["researcher", "analyst", "writer"]
ADD = ["researcher", "analyst", "writer", "fact_checker"]


def target_chain(i: int) -> list[str]:
    if i < 70:
        return BASELINE
    if i < 85:
        return SKIP
    return ADD


def main():
    random.seed(7)
    conn = get_connection(resolve_db_path("demo"))
    run_ids = [r[0] for r in conn.execute(
        "SELECT run_id FROM runs ORDER BY timestamp").fetchall()]

    changed = 0
    for i, rid in enumerate(run_ids):
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM spans WHERE run_id=? AND agent_name IS NOT NULL "
            "ORDER BY turn_index", (rid,)).fetchall()]
        if not rows:
            continue
        tmpl = {r["agent_name"]: r for r in rows}      # attribute template per agent
        chain = target_chain(i)

        conn.execute("DELETE FROM spans WHERE run_id=? AND agent_name IS NOT NULL", (rid,))
        conn.execute("DELETE FROM handoffs WHERE run_id=?", (rid,))

        cursor, prev_id, built = 0.0, None, []
        for t, agent in enumerate(chain):
            src = tmpl.get(agent) or tmpl.get("writer") or rows[-1]
            dur = float(src.get("duration_ms") or 0)   # preserve compute time
            start, end = cursor, cursor + dur
            rec = dict(src)
            rec.update(span_id=str(uuid.uuid4()), run_id=rid, agent_name=agent, turn_index=t,
                       start_time_ms=round(start, 1), end_time_ms=round(end, 1),
                       duration_ms=round(dur, 1), parent_step_id=prev_id,
                       branch_id=None, join_step_id=None)
            cols = list(rec.keys())
            conn.execute(
                f"INSERT INTO spans ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [rec[c] for c in cols])
            built.append({"agent": agent, "turn": t, "start": start, "end": end,
                          "out": int(src.get("output_tokens") or 0),
                          "in": int(src.get("input_tokens") or 0)})
            prev_id = rec["span_id"]
            cursor = end + random.uniform(80, 400)      # realistic handoff gap

        # Rebuild handoffs from consecutive steps (the gap = next.start - prev.end).
        for hidx in range(len(built) - 1):
            a, b = built[hidx], built[hidx + 1]
            conn.execute(
                "INSERT INTO handoffs (handoff_id, run_id, handoff_index, agent_from, "
                "agent_to, turn_index_from, turn_index_to, a_output_tokens, b_input_tokens, "
                "context_ratio, was_requested) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), rid, hidx, a["agent"], b["agent"], a["turn"], b["turn"],
                 a["out"], b["in"], round(b["in"] / a["out"], 3) if a["out"] else 0, 0))

        conn.execute("UPDATE runs SET agent_sequence=? WHERE run_id=?",
                     (json.dumps(chain), rid))
        changed += 1

    conn.commit()
    print(f"Seeded path drift across {changed} demo runs "
          f"(baseline→skip critic @70→add fact_checker @85); "
          f"spans, handoffs and agent_sequence kept consistent.")


if __name__ == "__main__":
    main()
