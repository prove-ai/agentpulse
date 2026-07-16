"""Layer 1 — raw span reader.

Reads spans from SQLite exactly as stored. No calculations, no opinions.
Everything in this module returns plain dicts or lists of dicts.

All functions accept an optional `db_path`. When None (the default), the
storage layer's active ContextVar decides which DB file to read — set per
HTTP request by the Flask dashboard, or implicitly by instrument(db_name=...).
This is how multi-DB switching works without changing every call site.
"""

from __future__ import annotations

from pathlib import Path

from storage.sqlite_store import get_connection


def get_run(run_id: str, db_path: Path | None = None) -> dict | None:
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_runs_by_version(prompt_version: int, db_path: Path | None = None) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM runs WHERE prompt_version = ? ORDER BY timestamp",
        (prompt_version,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_baseline_runs(db_path: Path | None = None) -> list[dict]:
    """Return all runs for the lowest (baseline) prompt version."""
    conn = get_connection(db_path)
    row = conn.execute("SELECT MIN(prompt_version) AS v FROM runs").fetchone()
    if not row or row["v"] is None:
        return []
    return get_runs_by_version(row["v"], db_path)


def get_spans(run_id: str, db_path: Path | None = None) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM spans WHERE run_id = ? ORDER BY turn_index, start_time_ms",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_agent_spans(run_id: str, db_path: Path | None = None) -> list[dict]:
    """Return only agent-turn spans (have gen_ai.agent.name)."""
    conn = get_connection(db_path)
    rows = conn.execute(
        """SELECT * FROM spans
           WHERE run_id = ? AND agent_name IS NOT NULL
           ORDER BY turn_index""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_spans_for_runs(run_ids, db_path: Path | None = None) -> dict:
    """All agent spans for many runs in ONE query, grouped by run_id. Used to
    derive each run's route TOPOLOGY (parent_step_id / branch_id) without N+1."""
    run_ids = list(run_ids)
    if not run_ids:
        return {}
    conn = get_connection(db_path)
    out: dict = {}
    CH = 800                                          # stay under SQLite's param limit
    for i in range(0, len(run_ids), CH):
        chunk = run_ids[i:i + CH]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""SELECT run_id, agent_name, turn_index, span_id, parent_step_id, branch_id
                FROM spans WHERE run_id IN ({ph}) AND agent_name IS NOT NULL
                ORDER BY run_id, turn_index""", chunk).fetchall()
        for r in rows:
            out.setdefault(r["run_id"], []).append(dict(r))
    return out


def get_tool_calls(run_id: str, db_path: Path | None = None) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM tool_calls WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_runs(limit: int = 20, db_path: Path | None = None) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_handoffs(run_id: str, db_path: Path | None = None) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM handoffs WHERE run_id = ? ORDER BY handoff_index",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]
