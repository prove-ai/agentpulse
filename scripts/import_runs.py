#!/usr/bin/env python3
"""Import external multi-agent run data into an AgentPulse dashboard DB.

Give this script a JSON file (see scripts/import_template.json for the exact
format) and it writes a ready-to-browse SQLite DB under db/<name>.db. Open the
dashboard and the new DB appears in the sidebar picker.

    python scripts/import_runs.py path/to/their_data.json
    python scripts/import_runs.py path/to/their_data.json --db partner_system

You (or whoever sends you data) only need to provide what *they* know: the run
metadata and the ordered list of agent steps with tokens/timing. Everything the
dashboard derives — run totals, cost, agent sequence, handoffs, and the parallel
DAG — is computed here from those steps. No need to hand-compute aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Make the repo importable regardless of where this is run from.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from storage.sqlite_store import (resolve_db_path, get_connection, model_price,
                                  record_prompt, set_active_db_path)


def _cost(inp: int, out: int, model: str) -> float:
    pin, pout = model_price(model or "")
    return (inp * pin + out * pout) / 1_000_000


def _iso(ts: str | None) -> str:
    if ts:
        return ts
    return datetime.now(timezone.utc).isoformat()


def _build_spans(steps: list[dict], run_id: str) -> list[dict]:
    """Turn the human-friendly `steps` list into full span rows, computing the
    DAG fields (parent_step_id / branch_id / join_step_id) from `parallel_group`.
    """
    spans: list[dict] = []
    t_cursor = 0.0
    for i, step in enumerate(steps):
        agent = step.get("agent")
        if not agent:
            raise ValueError(
                f"run {run_id}: step {i} is missing the required 'agent' name. "
                "Every step needs an agent; all other fields have safe defaults."
            )
        inp = int(step.get("input_tokens", 0) or 0)
        out = int(step.get("output_tokens", 0) or 0)
        dur = float(step.get("duration_ms", 0) or 0)
        # start_time_ms is optional; if omitted we lay steps end-to-end so the
        # timeline still renders sensibly. Provide real values to show overlap.
        start = step.get("start_time_ms")
        start = float(start) if start is not None else t_cursor
        t_cursor = start + dur
        spans.append({
            "span_id":      str(uuid.uuid4()),
            "run_id":       run_id,
            "agent_name":   agent,
            "turn_index":   i,
            "start_time_ms": start,
            "end_time_ms":  start + dur,
            "duration_ms":  dur,
            "input_tokens": inp,
            "output_tokens": out,
            "model":        step.get("model", "") or "",
            "tool_call_count": len(step.get("tools", []) or []),
            "status":       step.get("status", "OK") or "OK",
            "status_value": step.get("status_value", "") or "",
            "retry_count":  int(step.get("llm_retries", 0) or 0),
            "parent_step_id": None,
            "branch_id":    None,
            "join_step_id": None,
            "_group":       step.get("parallel_group"),
            "_tools":       step.get("tools", []) or [],
        })

    # --- DAG: derive parent / branch / join from consecutive parallel_group runs ---
    last_stem: str | None = None       # span_id of the last sequential (non-branch) step
    i = 0
    n = len(spans)
    while i < n:
        grp = spans[i]["_group"]
        if grp:
            # Collect the whole contiguous group sharing this label.
            j = i
            while j < n and spans[j]["_group"] == grp:
                j += 1
            members = spans[i:j]
            join_span = spans[j] if j < n else None   # first step after the group
            for m in members:
                m["parent_step_id"] = last_stem
                m["branch_id"]      = m["agent_name"]
                m["join_step_id"]   = join_span["span_id"] if join_span else None
            i = j
        else:
            spans[i]["parent_step_id"] = last_stem
            last_stem = spans[i]["span_id"]
            i += 1
    return spans


def _import_run(conn, run: dict) -> None:
    run_id = run.get("run_id") or str(uuid.uuid4())
    steps  = run.get("steps", [])
    if not steps:
        raise ValueError(f"run {run_id} has no steps")

    spans = _build_spans(steps, run_id)

    # ---- per-run config snapshot (for change tracking & versioning) ----
    agent_configs = {}
    for step in steps:
        c   = step.get("config", {}) or {}
        cfg = {}
        if step.get("model"):  cfg["model"]  = step["model"]
        if c.get("params"):    cfg["params"] = c["params"]
        if c.get("tools"):     cfg["tools"]  = sorted(set(c["tools"]))
        if c.get("system_prompt"):
            cfg["prompt_hash"] = hashlib.sha1(c["system_prompt"].encode()).hexdigest()[:12]
            record_prompt(cfg["prompt_hash"], c["system_prompt"])   # full text → prompts table
        elif c.get("prompt_hash"):
            cfg["prompt_hash"] = c["prompt_hash"]
        if cfg:
            agent_configs[step["agent"]] = cfg
    # No workflow_hash — it was the sha1 of the SET OF AGENTS a run executed, which
    # varies per run by routing and caused false "workflow changed" events. Route
    # shifts are detected by Path drift (route_conformance) instead.
    config_blob  = {"agents": agent_configs}

    total_in   = sum(s["input_tokens"]  for s in spans)
    total_out  = sum(s["output_tokens"] for s in spans)
    total_cost = sum(_cost(s["input_tokens"], s["output_tokens"], s["model"]) for s in spans)
    t0   = min(s["start_time_ms"] for s in spans)
    tend = max(s["end_time_ms"]   for s in spans)
    model = next((s["model"] for s in spans if s["model"]), "")
    seq = [s["agent_name"] for s in spans]

    conn.execute(
        """INSERT OR REPLACE INTO runs
           (run_id, task_text, task_type, prompt_version, model, timestamp,
            total_turns, total_duration_ms, total_input_tokens, total_output_tokens,
            total_cost_usd, termination_reason, status, agent_sequence, prompt_hashes,
            config_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, run.get("task_text", ""), run.get("task_type", "imported"),
         int(run.get("prompt_version", 1)), model, _iso(run.get("timestamp")),
         len(spans), round(tend - t0, 1), total_in, total_out, round(total_cost, 6),
         run.get("termination_reason", "completed"), run.get("status", "OK"),
         json.dumps(seq), "{}", json.dumps(config_blob)),
    )

    for s in spans:
        conn.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, run_id, agent_name, turn_index, start_time_ms, end_time_ms,
                duration_ms, input_tokens, output_tokens, model, tool_call_count,
                status, status_value, parent_step_id, branch_id, join_step_id, retry_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (s["span_id"], s["run_id"], s["agent_name"], s["turn_index"],
             s["start_time_ms"], s["end_time_ms"], s["duration_ms"],
             s["input_tokens"], s["output_tokens"], s["model"], s["tool_call_count"],
             s["status"], s["status_value"], s["parent_step_id"], s["branch_id"],
             s["join_step_id"], s["retry_count"]),
        )
        for t in s["_tools"]:
            conn.execute(
                """INSERT OR REPLACE INTO tool_calls
                   (call_id, span_id, run_id, tool_name, success, duration_ms)
                   VALUES (?,?,?,?,?,?)""",
                (str(uuid.uuid4()), s["span_id"], run_id, t.get("name", "tool"),
                 1 if t.get("success", True) else 0, float(t.get("duration_ms", 0) or 0)),
            )

    # Handoffs: one per consecutive turn (sender → receiver).
    for k in range(1, len(spans)):
        a, b = spans[k - 1], spans[k]
        a_out = a["output_tokens"]
        conn.execute(
            """INSERT OR REPLACE INTO handoffs
               (handoff_id, run_id, handoff_index, agent_from, agent_to,
                turn_index_from, turn_index_to, a_output_tokens, b_input_tokens,
                context_ratio, was_requested)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), run_id, k - 1, a["agent_name"], b["agent_name"],
             a["turn_index"], b["turn_index"], a_out, b["input_tokens"],
             round(b["input_tokens"] / a_out, 3) if a_out else 0.0, 0),
        )
    conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser(description="Import external run data into an AgentPulse DB.")
    ap.add_argument("input", help="Path to the JSON file (see scripts/import_template.json).")
    ap.add_argument("--db", help="DB name to write (db/<name>.db). Overrides the file's db_name.")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    db_name = args.db or data.get("db_name") or "imported"
    runs = data.get("runs", [])
    if not runs:
        print("No runs found in input (expected a top-level 'runs' array).")
        sys.exit(1)

    db_path = resolve_db_path(db_name)
    set_active_db_path(db_path)       # so record_prompt() and friends target this db
    conn = get_connection(db_path)   # creates the schema if needed
    for run in runs:
        _import_run(conn, run)
    conn.close()

    print(f"Imported {len(runs)} run(s) into {db_path}")
    print(f"Open the dashboard and pick '{db_name}' in the sidebar.")


if __name__ == "__main__":
    main()
