#!/usr/bin/env python
"""Send runs from an AgentPulse project database to Cohorta as a new batch.

    python scripts/send_to_cohorta.py --project customer_support --last 20
    python scripts/send_to_cohorta.py --project my-system --last 20 \
        --task-type duplicate_charge_simple --name "dup charge sweep"

Reads db/<project>.db and POSTs the selected runs — with their spans, tool
calls, LLM calls (full request/response text) and handoffs — to Cohorta's
ingest endpoint (default http://127.0.0.1:4910). Cohorta creates an immutable
batch; re-sending the same runs makes a new batch. Stdlib only, on purpose.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

DB_DIR = Path(__file__).parent.parent / "db"


def rows(conn, sql: str, *args) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, args)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", default=None,
                    help="AgentPulse project name; reads db/<project>.db "
                         "next to this script's repo root")
    ap.add_argument("--db", default=None, metavar="FILE",
                    help="path to an AgentPulse SQLite database "
                         "(alternative to --project; works from anywhere)")
    ap.add_argument("--last", type=int, default=20,
                    help="send the N most recent runs (default 20)")
    ap.add_argument("--task-type", default=None,
                    help="only runs of this task_type")
    ap.add_argument("--name", default=None, help="batch name override")
    ap.add_argument("--to", default="http://127.0.0.1:4910",
                    help="Cohorta API base URL")
    ap.add_argument("--out", default=None, metavar="FILE",
                    help="write the batch to a JSON file instead of "
                         "sending it; upload it later from Cohorta's "
                         "Batches page")
    args = ap.parse_args()

    if not args.project and not args.db:
        print("give --project (run from the AgentPulse repo) or --db FILE")
        return 1
    db_path = Path(args.db) if args.db else DB_DIR / f"{args.project}.db"
    if not db_path.exists():
        known = ", ".join(sorted(p.stem for p in DB_DIR.glob("*.db"))) or "none"
        print(f"no database at {db_path} (known projects: {known})")
        return 1
    project = args.project or db_path.stem

    conn = sqlite3.connect(db_path)
    where, params = "", []
    if args.task_type:
        where, params = "WHERE task_type = ?", [args.task_type]
    runs = rows(conn, f"SELECT * FROM runs {where} ORDER BY timestamp DESC"
                       " LIMIT ?", *params, args.last)
    if not runs:
        print("no runs matched")
        return 1

    entries = []
    for run in runs:
        rid = run["run_id"]
        entries.append({
            "run": run,
            "spans": rows(conn, "SELECT * FROM spans WHERE run_id = ?"
                                " ORDER BY turn_index", rid),
            "tool_calls": rows(conn, "SELECT * FROM tool_calls WHERE run_id = ?"
                                     " ORDER BY start_time_ms", rid),
            "llm_calls": rows(conn, "SELECT * FROM llm_calls WHERE run_id = ?"
                                    " ORDER BY start_time_ms", rid),
            "handoffs": rows(conn, "SELECT * FROM handoffs WHERE run_id = ?"
                                   " ORDER BY handoff_index", rid),
        })
    conn.close()

    payload = {"source": "agentpulse", "project": project,
               "name": args.name, "runs": entries}
    body = json.dumps(payload).encode()
    if args.out:
        Path(args.out).write_bytes(body)
        print(f"wrote {len(runs)} runs ({len(body) / 1024:.0f} KB) to "
              f"{args.out}")
        print("upload it from Cohorta's Batches page")
        return 0
    req = urllib.request.Request(
        f"{args.to.rstrip('/')}/api/batches", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"could not reach Cohorta at {args.to}: {exc}")
        print("is it running? (cd ~/Documents/cohorta && ./start.sh)")
        return 1

    print(f"sent {result['run_count']} runs "
          f"({len(body) / 1024:.0f} KB) as batch \"{result['name']}\"")
    print(f"open http://localhost:4900/batch/{result['batch_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
