"""SQLite storage — writes a finalised RunSession to the database.

Schema (4 tables):
  runs       one row per completed run
  spans      one row per agent turn
  tool_calls one row per tool invocation
  handoffs   one row per agent-to-agent transition
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextvars import ContextVar
from pathlib import Path

from sdk.session import RunSession

# All DB files live here. Each `db_name` becomes `<DB_DIR>/<db_name>.db`.
DB_DIR     = Path(__file__).parent.parent / "db"
DEFAULT_DB = "runs"
DB_PATH    = DB_DIR / f"{DEFAULT_DB}.db"   # back-compat: legacy callers import DB_PATH

# Per-request / per-session override (set by the Flask app or by instrument()).
# Thread- and async-safe via ContextVar.
_active_db_path: ContextVar[Path | None] = ContextVar("active_db_path", default=None)


def resolve_db_path(db_name: str | None) -> Path:
    """Map a logical db_name to a file path. None / '' / 'runs' → the default DB."""
    if not db_name or db_name == DEFAULT_DB:
        return DB_PATH
    # Strip anything weird; allow only safe characters in db_name.
    safe = "".join(c for c in db_name if c.isalnum() or c in ("-", "_"))
    if not safe:
        return DB_PATH
    return DB_DIR / f"{safe}.db"


def set_active_db_path(path: Path | None) -> None:
    """Set the per-context active DB. Reset with None."""
    _active_db_path.set(path)


def get_active_db_path() -> Path:
    """Return whichever DB the current context wants — falls back to the default."""
    return _active_db_path.get() or DB_PATH


def list_available_dbs() -> list[str]:
    """Return logical names of every .db file actually present under DB_DIR.

    Only files that exist on disk are surfaced — no phantom 'runs' entry
    just because that's the historical default.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted({p.stem for p in DB_DIR.glob("*.db")})
    return names

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    task_text           TEXT,
    task_type           TEXT,
    prompt_version      INTEGER,
    model               TEXT,
    timestamp           TEXT,
    total_turns         INTEGER,
    total_duration_ms   REAL,
    total_input_tokens  INTEGER,
    total_output_tokens INTEGER,
    total_cost_usd      REAL,
    termination_reason  TEXT,
    status              TEXT,
    agent_sequence      TEXT,
    prompt_hashes       TEXT,
    config_json         TEXT   -- captured per-agent config snapshot (model/params/tools/prompt hash) + workflow hash
);

-- Manual, user-created version snapshots (one row per "Save as Version N").
-- A run belongs to the latest version whose created_at <= run.timestamp;
-- runs before the first snapshot are the implicit Version 1.
CREATE TABLE IF NOT EXISTS versions (
    version_num   INTEGER PRIMARY KEY,
    label         TEXT,
    created_at    TEXT,
    config_json   TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id             TEXT PRIMARY KEY,
    ts             TEXT,
    run_index      INTEGER,
    type           TEXT,      -- config_change | version | release | drift | impact | evidence
    dim            TEXT,      -- model | prompt | tools | params (for config changes)
    component      TEXT,      -- affected agent / handoff
    title          TEXT,
    before         TEXT,
    after          TEXT,
    author         TEXT,
    source         TEXT,      -- UI change | API | SDK | CI/CD | auto
    environment    TEXT,
    changed_fields INTEGER,
    hash_before    TEXT,
    hash_after     TEXT,
    related_drift  TEXT,      -- 'category|entity' for the related Drift Investigation finding
    impact_json    TEXT       -- [{label, pct, dir, spark}]
);

-- Deduped store of full prompt text, keyed by the hash recorded in a run's
-- config_json. Lets the Event Timeline show the real before/after prompt text
-- for a prompt change without bloating every run's config with the full text.
CREATE TABLE IF NOT EXISTS prompts (
    hash        TEXT PRIMARY KEY,
    text        TEXT,
    first_seen  TEXT
);

CREATE TABLE IF NOT EXISTS spans (
    span_id             TEXT PRIMARY KEY,
    run_id              TEXT,
    agent_name          TEXT,
    turn_index          INTEGER,
    start_time_ms       REAL,
    end_time_ms         REAL,
    duration_ms         REAL,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    model               TEXT,
    tool_call_count     INTEGER,
    status              TEXT,
    status_value        TEXT,
    -- DAG fields (Layer A: parallel group detection)
    -- Set on branch spans only; the join is identified by being pointed at.
    parent_step_id      TEXT,  -- span_id of the agent step that spawned this one
    branch_id           TEXT,  -- identifier of this branch within its fan-out group
    join_step_id        TEXT,  -- span_id of the join step (NULL if no join)
    retry_count         INTEGER DEFAULT 0,  -- LLM-call retries within this step
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    call_id         TEXT PRIMARY KEY,
    span_id         TEXT,
    run_id          TEXT,
    tool_name       TEXT,
    success         INTEGER,
    duration_ms     REAL,
    FOREIGN KEY (span_id) REFERENCES spans(span_id),
    FOREIGN KEY (run_id)  REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS handoffs (
    handoff_id          TEXT PRIMARY KEY,
    run_id              TEXT,
    handoff_index       INTEGER,
    agent_from          TEXT,
    agent_to            TEXT,
    turn_index_from     INTEGER,
    turn_index_to       INTEGER,
    a_output_tokens     INTEGER,
    b_input_tokens      INTEGER,
    context_ratio       REAL,
    was_requested       INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

# Migrations: columns added after initial schema. Each entry is idempotent
# (we catch OperationalError when the column already exists).
_MIGRATIONS = [
    "ALTER TABLE spans ADD COLUMN status_value   TEXT DEFAULT ''",
    "ALTER TABLE spans ADD COLUMN parent_step_id TEXT",
    "ALTER TABLE spans ADD COLUMN branch_id      TEXT",
    "ALTER TABLE spans ADD COLUMN join_step_id   TEXT",
    "ALTER TABLE spans ADD COLUMN retry_count    INTEGER DEFAULT 0",
    "ALTER TABLE runs  ADD COLUMN config_json    TEXT",
]


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    # If no explicit path is supplied, fall back to the ContextVar (set by the
    # Flask request or the instrument() session). If neither is set, default DB.
    db_path = db_path or get_active_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # Apply any pending migrations safely
    for sql in _MIGRATIONS:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column/table already exists
    return conn


# ---------------------------------------------------------------------------
# Cost lookup
# ---------------------------------------------------------------------------
_PRICES: dict[str, tuple[float, float]] = {
    # Anthropic — Claude
    "claude-sonnet-4-6":          (3.0,  15.0),
    "claude-opus-4-8":            (15.0, 75.0),
    "claude-opus-4-7":            (15.0, 75.0),
    "claude-haiku-4-5-20251001":  (0.8,   4.0),
    "claude-sonnet-4-5-20250929": (3.0,  15.0),
    "claude-opus-4-1-20250805":   (15.0, 75.0),
    # OpenAI — GPT
    # Prefix-matched against the model string. Order matters: longer keys first.
    "gpt-4o-mini":                (0.15,  0.60),
    "gpt-4o":                     (2.50, 10.00),
    "gpt-4.1-mini":               (0.40,  1.60),
    "gpt-4.1-nano":               (0.10,  0.40),
    "gpt-4.1":                    (2.00,  8.00),
    "gpt-4-turbo":                (10.0, 30.00),
    "gpt-4":                      (30.0, 60.00),
    "gpt-3.5-turbo":              (0.50,  1.50),
    "o1-mini":                    (1.10,  4.40),
    "o1":                         (15.0, 60.00),
    "o3-mini":                    (1.10,  4.40),
}
_DEFAULT_PRICE = (3.0, 15.0)


def model_price(model: str) -> tuple[float, float]:
    for key, price in _PRICES.items():
        if key in (model or ""):
            return price
    return _DEFAULT_PRICE


def compute_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    inp_p, out_p = model_price(model)
    return (input_tokens * inp_p + output_tokens * out_p) / 1_000_000


# ---------------------------------------------------------------------------
# Handoff computation
# ---------------------------------------------------------------------------
def _compute_handoffs(session: RunSession) -> list[dict]:
    """Derive handoff records from consecutive turns."""
    handoffs = []
    turns = session.turns
    for i in range(1, len(turns)):
        a, b = turns[i - 1], turns[i]
        if a.agent_name == b.agent_name:
            continue  # same agent, not a handoff

        ratio = (b.input_tokens / a.output_tokens) if a.output_tokens > 0 else 0.0

        # Was this transition explicitly requested?
        # A's STATUS: NEEDS_INFO: B  or  CHANGES_REQUESTED: B
        sv = (a.status_value or "").upper()
        b_upper = b.agent_name.upper()
        was_requested = (
            ("NEEDS_INFO" in sv or "CHANGES_REQUESTED" in sv)
            and b_upper in sv
        )

        handoffs.append({
            "handoff_id":      str(uuid.uuid4()),
            "run_id":          session.run_id,
            "handoff_index":   len(handoffs),
            "agent_from":      a.agent_name,
            "agent_to":        b.agent_name,
            "turn_index_from": a.turn_index,
            "turn_index_to":   b.turn_index,
            "a_output_tokens": a.output_tokens,
            "b_input_tokens":  b.input_tokens,
            "context_ratio":   round(ratio, 3),
            "was_requested":   1 if was_requested else 0,
        })
    return handoffs


# ---------------------------------------------------------------------------
# Main write function
# ---------------------------------------------------------------------------
def get_versions(db_path: Path | None = None) -> list[dict]:
    """Return manual version snapshots, oldest first."""
    conn = get_connection(db_path)
    return [dict(r) for r in conn.execute(
        "SELECT * FROM versions ORDER BY version_num")]


def create_version_snapshot(label: str, config_json: str,
                            db_path: Path | None = None) -> int:
    """Snapshot the current config as the next version. Version 1 is the implicit
    baseline, so the first snapshot becomes Version 2."""
    import datetime
    conn = get_connection(db_path)
    n = conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0]
    if n == 0:
        next_num = 2
    else:
        next_num = conn.execute("SELECT MAX(version_num) FROM versions").fetchone()[0] + 1
    conn.execute(
        "INSERT INTO versions (version_num, label, created_at, config_json) VALUES (?,?,?,?)",
        (next_num, label or f"Version {next_num}",
         datetime.datetime.now(datetime.timezone.utc).isoformat(), config_json),
    )
    conn.commit()
    return next_num


def latest_run_config(db_path: Path | None = None) -> str:
    """config_json of the most recent run — the 'current settings' to snapshot."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT config_json FROM runs ORDER BY timestamp DESC LIMIT 1").fetchone()
    return (row[0] if row and row[0] else "{}")


def get_setting(key: str, default: str = "", db_path: Path | None = None) -> str:
    """Read a per-project key/value setting."""
    conn = get_connection(db_path)
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row and row[0] is not None else default


def set_setting(key: str, value: str, db_path: Path | None = None) -> None:
    """Write a per-project key/value setting."""
    conn = get_connection(db_path)
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))
    conn.commit()


def get_custom_metrics(db_path: Path | None = None) -> list[dict]:
    """User-defined derived metrics (formulas over existing metrics)."""
    try:
        return json.loads(get_setting("custom_metrics", "[]", db_path) or "[]")
    except (TypeError, ValueError):
        return []


def save_custom_metrics(metrics: list[dict], db_path: Path | None = None) -> None:
    set_setting("custom_metrics", json.dumps(metrics), db_path)


def get_thresholds(db_path: Path | None = None) -> dict:
    """Target lines keyed by 'category|metric' → {value, dir}."""
    try:
        return json.loads(get_setting("thresholds", "{}", db_path) or "{}")
    except (TypeError, ValueError):
        return {}


def save_thresholds(thresholds: dict, db_path: Path | None = None) -> None:
    set_setting("thresholds", json.dumps(thresholds), db_path)


_EVENT_COLS = ["id", "ts", "run_index", "type", "dim", "component", "title", "before", "after",
               "author", "source", "environment", "changed_fields", "hash_before", "hash_after",
               "related_drift", "impact_json"]


def get_events(db_path: Path | None = None) -> list[dict]:
    """Rich, captured change/release/drift events (the Event Timeline source)."""
    conn = get_connection(db_path)
    return [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY run_index, ts")]


def upsert_event(ev: dict, db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    conn.execute(
        f"INSERT OR REPLACE INTO events ({','.join(_EVENT_COLS)}) "
        f"VALUES ({','.join('?' * len(_EVENT_COLS))})",
        [ev.get(c) for c in _EVENT_COLS])
    conn.commit()


def clear_events(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    conn.execute("DELETE FROM events")
    conn.commit()


def record_prompt(prompt_hash: str, text: str, db_path: Path | None = None) -> None:
    """Store the full text of a prompt, keyed by its hash (first write wins)."""
    if not prompt_hash or not text:
        return
    import datetime
    conn = get_connection(db_path)
    conn.execute(
        "INSERT OR IGNORE INTO prompts (hash, text, first_seen) VALUES (?,?,?)",
        (prompt_hash, text, datetime.datetime.now(datetime.timezone.utc).isoformat()))
    conn.commit()


def get_prompts(db_path: Path | None = None) -> dict:
    """Return {hash: text} for resolving prompt-change events to real text."""
    conn = get_connection(db_path)
    return {r[0]: r[1] for r in conn.execute("SELECT hash, text FROM prompts")}


def get_baseline_version(db_path: Path | None = None) -> int:
    """The version chosen as the comparison baseline (defaults to v1)."""
    try:
        return int(get_setting("baseline_version", "1", db_path) or 1)
    except (TypeError, ValueError):
        return 1


def set_baseline_version(num: int, db_path: Path | None = None) -> None:
    set_setting("baseline_version", str(int(num)), db_path)


def write_session(
    session: RunSession,
    prompt_hashes: dict | None = None,
    db_path: Path | None = None,
) -> None:
    """Persist a finalised RunSession to all 4 tables.

    Path priority:
      1. Explicit `db_path` arg.
      2. The session's own `db_name` (set by instrument(db_name=...)).
      3. Whatever the active ContextVar holds.
      4. The default DB.
    """
    import datetime
    if db_path is None and getattr(session, "db_name", None):
        db_path = resolve_db_path(session.db_name)
    conn = get_connection(db_path)

    total_input  = sum(t.input_tokens  for t in session.turns)
    total_output = sum(t.output_tokens for t in session.turns)

    # Use per-turn model for cost (more accurate when model varies)
    total_cost = sum(
        compute_cost(t.input_tokens, t.output_tokens, t.model or session.model)
        for t in session.turns
    )

    # ---- per-run config snapshot (for change tracking & versioning) ----
    agent_cfg = dict(getattr(session, "agent_configs", {}) or {})
    # Full prompt text is captured transiently as cfg["prompt_text"]; dedupe it
    # into the prompts table (keyed by hash) and keep only the hash in config_json.
    for _agent, _c in list(agent_cfg.items()):
        if isinstance(_c, dict) and _c.get("prompt_text"):
            _c = dict(_c)
            record_prompt(_c.get("prompt_hash"), _c.pop("prompt_text"), db_path)
            agent_cfg[_agent] = _c
    # NOTE: no workflow_hash. It used to be the sha1 of the SET OF AGENTS a run
    # executed, which varies per run by routing in any conditional/router workflow —
    # producing a false "workflow changed" event on nearly every run. Real route
    # shifts are detected by Path drift (route_conformance) instead.
    config_blob = {"agents": agent_cfg}

    # ---- runs ----
    conn.execute(
        """
        INSERT OR REPLACE INTO runs
          (run_id, task_text, task_type, prompt_version, model, timestamp,
           total_turns, total_duration_ms, total_input_tokens,
           total_output_tokens, total_cost_usd, termination_reason,
           status, agent_sequence, prompt_hashes, config_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            session.run_id,
            session.task_text,
            session.task_type,
            session.prompt_version,
            session.model,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            len(session.turns),
            round(session.total_duration_ms, 1),
            total_input,
            total_output,
            round(total_cost, 6),
            session.termination_reason,
            session.status,
            json.dumps(session.agent_sequence),
            json.dumps(prompt_hashes or {}),
            json.dumps(config_blob),
        ),
    )

    # ---- spans + tool_calls ----
    span_ids: dict[int, str] = {}  # turn_index -> span_id
    for turn in session.turns:
        # Use the turn's own span_id (auto-generated at start; framework adapters
        # may also have referenced it as a parent_step_id by this point).
        span_id  = getattr(turn, "span_id", None) or str(uuid.uuid4())
        span_ids[turn.turn_index] = span_id
        start_ms = turn.start_ns / 1_000_000
        end_ms   = (turn.end_ns or turn.start_ns) / 1_000_000

        conn.execute(
            """
            INSERT OR REPLACE INTO spans
              (span_id, run_id, agent_name, turn_index, start_time_ms,
               end_time_ms, duration_ms, input_tokens, output_tokens,
               model, tool_call_count, status, status_value,
               parent_step_id, branch_id, join_step_id, retry_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                span_id,
                session.run_id,
                turn.agent_name,
                turn.turn_index,
                start_ms,
                end_ms,
                round(turn.duration_ms, 1),
                turn.input_tokens,
                turn.output_tokens,
                turn.model,
                turn.tool_call_count,
                turn.status,
                turn.status_value,
                # DAG fields (None for sessions that don't populate them).
                getattr(turn, "parent_step_id", None),
                getattr(turn, "branch_id",      None),
                getattr(turn, "join_step_id",   None),
                getattr(turn, "retry_count",    0),
            ),
        )

        for tool in turn.tools:
            conn.execute(
                """
                INSERT INTO tool_calls
                  (call_id, span_id, run_id, tool_name, success, duration_ms)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()), span_id, session.run_id,
                    tool.tool_name, 1 if tool.success else 0,
                    round(tool.duration_ms, 1),
                ),
            )

    # ---- handoffs ----
    for h in _compute_handoffs(session):
        conn.execute(
            """
            INSERT OR REPLACE INTO handoffs
              (handoff_id, run_id, handoff_index, agent_from, agent_to,
               turn_index_from, turn_index_to, a_output_tokens, b_input_tokens,
               context_ratio, was_requested)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                h["handoff_id"], h["run_id"], h["handoff_index"],
                h["agent_from"], h["agent_to"],
                h["turn_index_from"], h["turn_index_to"],
                h["a_output_tokens"], h["b_input_tokens"],
                h["context_ratio"], h["was_requested"],
            ),
        )

    conn.commit()
    conn.close()
