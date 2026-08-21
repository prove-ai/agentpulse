"""Export a captured run as an OpenTelemetry (OTLP JSON) trace.

Turns one AgentPulse run — its spans, LLM calls, and tool calls — into the
OTel GenAI span convention (gen_ai.*) that replay tooling ingests:

    workflow <task_type>              (root span; task text + termination)
      invoke_agent <AgentName>        (one per agent turn)
        chat <model>                  (one per LLM API call, verbatim payloads)
        execute_tool <tool>           (one per tool call, args + result)

Content mapping, chosen to round-trip through the replay harness:
  - gen_ai.input.messages is EXACTLY the recorded `messages` kwarg. The
    Anthropic seam matches a replayed request by its message-role sequence,
    and a live Anthropic call carries the system prompt outside `messages` —
    so the system prompt is exported as gen_ai.request.system, never
    prepended as a message.
  - gen_ai.output.messages is [{"role": "assistant", "content": <blocks>}],
    with content blocks (text / tool_use) verbatim from the recorded response.

Usage:
    python3 storage/otel_export.py <run_id | --last [N] | --since WHEN>
        [--db NAME] [-o FILE] [--submit [URL]]

With no -o the export prints to stdout. --submit POSTs it to a replay
server's /api/ingest (default http://localhost:4951).

A run_id exports that run; --last the most recent one. --last N selects
the N most recent runs and --since every run in a period (12h, 90m, 7d,
or an ISO date/datetime; naive values read as local time); both handle
each run separately: one submission per run with --submit, one file per
run into the -o directory, or a dry list of what they found with
neither. The replay server dedupes submissions by content, so
overlapping selections are safe to resend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_OBS_ROOT = Path(__file__).parent.parent
if str(_OBS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OBS_ROOT))

from storage.sqlite_store import get_connection, resolve_db_path

# Sampling/request parameters exported as gen_ai.request.<name>. Content-
# bearing kwargs (messages) are handled separately; system/tools are JSON-
# encoded because OTLP attribute values are scalars or flat arrays.
_PARAM_KEYS = ("temperature", "max_tokens", "max_completion_tokens", "top_p",
               "top_k", "stop_sequences", "stop", "seed", "n",
               "frequency_penalty", "presence_penalty", "reasoning_effort")
_JSON_PARAM_KEYS = ("system", "tools", "tool_choice", "response_format",
                    "thinking", "metadata")


# ---------------------------------------------------------------------------
# OTLP AnyValue encoding
# ---------------------------------------------------------------------------
def _any_value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, list):
        return {"arrayValue": {"values": [_any_value(x) for x in v]}}
    return {"stringValue": json.dumps(v, ensure_ascii=False)}


def _attrs(d: dict) -> list:
    return [{"key": k, "value": _any_value(v)}
            for k, v in d.items() if v is not None and v != ""]


def _span(trace_id, span_id, name, start_ms, end_ms, attrs,
          parent_id=None) -> dict:
    sp = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": str(int((start_ms or 0) * 1_000_000)),
        "endTimeUnixNano": str(int((end_ms or start_ms or 0) * 1_000_000)),
        "attributes": _attrs(attrs),
    }
    if parent_id:
        sp["parentSpanId"] = parent_id
    return sp


def _loads(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Payload → gen_ai.* attribute mapping
# ---------------------------------------------------------------------------
def _llm_attrs(row: dict) -> dict:
    req = _loads(row["request_json"]) or {}
    resp = _loads(row["response_json"]) or {}

    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": req.get("model") or row["model"] or None,
        "gen_ai.usage.input_tokens": row["input_tokens"],
        "gen_ai.usage.output_tokens": row["output_tokens"],
    }
    for k in _PARAM_KEYS:
        if k in req and req[k] is not None:
            attrs[f"gen_ai.request.{k}"] = req[k]
    for k in _JSON_PARAM_KEYS:
        if k in req and req[k] is not None:
            v = req[k]
            attrs[f"gen_ai.request.{k}"] = (
                v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))

    # Input: the exact messages kwarg, verbatim (see module docstring).
    if isinstance(req.get("messages"), list):
        attrs["gen_ai.input.messages"] = json.dumps(
            req["messages"], ensure_ascii=False)
    elif req.get("prompt") is not None:            # legacy completions
        attrs["gen_ai.prompt"] = (
            req["prompt"] if isinstance(req["prompt"], str)
            else json.dumps(req["prompt"], ensure_ascii=False))

    # Output: one assistant message whose content is the recorded blocks.
    out_msg = _output_message(resp)
    if out_msg is not None:
        attrs["gen_ai.output.messages"] = json.dumps(
            [out_msg], ensure_ascii=False)
    stop = _stop_reason(resp)
    if stop:
        attrs["gen_ai.response.finish_reasons"] = [stop]
    return attrs


def _output_message(resp) -> dict | None:
    """Normalize a recorded response (Anthropic model_dump, OpenAI
    model_dump, or streamed-text fallback) to one assistant message."""
    if not isinstance(resp, dict):
        return None
    # Anthropic: {"content": [blocks...], "stop_reason": ...}
    if isinstance(resp.get("content"), list):
        return {"role": "assistant", "content": resp["content"]}
    # OpenAI chat: {"choices": [{"message": {...}}]}
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            out = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                out["tool_calls"] = msg["tool_calls"]
            return out
        text = choices[0].get("text") if isinstance(choices[0], dict) else None
        if isinstance(text, str):
            return {"role": "assistant", "content": text}
    # Streamed OpenAI fallback: {"streamed": true, "content": "..."}
    if isinstance(resp.get("content"), str):
        return {"role": "assistant", "content": resp["content"]}
    return None


def _stop_reason(resp) -> str | None:
    if not isinstance(resp, dict):
        return None
    if resp.get("stop_reason"):
        return str(resp["stop_reason"])
    choices = resp.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict) \
            and choices[0].get("finish_reason"):
        return str(choices[0]["finish_reason"])
    return None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_run(run_id: str, db_path: Path | None = None) -> dict:
    """One run as an OTLP JSON export dict. Raises ValueError if absent."""
    conn = get_connection(db_path)
    run = conn.execute("SELECT * FROM runs WHERE run_id = ?",
                       (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise ValueError(f"run {run_id!r} not found")
    run = dict(run)
    spans = [dict(r) for r in conn.execute(
        "SELECT * FROM spans WHERE run_id = ? ORDER BY turn_index", (run_id,))]
    llm_calls = [dict(r) for r in conn.execute(
        "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY start_time_ms",
        (run_id,))]
    tool_calls = [dict(r) for r in conn.execute(
        "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY rowid", (run_id,))]
    conn.close()

    trace_id = run_id
    root_id = f"root-{run_id[:8]}"
    t0 = min((s["start_time_ms"] for s in spans), default=0)
    t1 = max((s["end_time_ms"] for s in spans), default=t0)

    out_spans = [_span(trace_id, root_id, f"workflow {run['task_type']}",
                       t0, t1, {
                           "agentpulse.run_id": run_id,
                           "agentpulse.task_text": run["task_text"],
                           "agentpulse.task_type": run["task_type"],
                           "agentpulse.termination_reason":
                               run["termination_reason"],
                       })]

    for s in spans:
        out_spans.append(_span(
            trace_id, s["span_id"], f"invoke_agent {s['agent_name']}",
            s["start_time_ms"], s["end_time_ms"], {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": s["agent_name"],
                "agentpulse.status_value": s.get("status_value") or None,
            }, parent_id=root_id))

    for c in llm_calls:
        out_spans.append(_span(
            trace_id, c["call_id"],
            f"chat {c['model'] or 'unknown'}",
            c["start_time_ms"], c["end_time_ms"],
            _llm_attrs(c), parent_id=c["span_id"]))

    # Runs captured before payload capture existed have agent spans with
    # model + tokens but no llm_calls rows. Emit a metrics-only chat span
    # for those turns so the audit reports the true gap (content not
    # captured) instead of "no llm spans found / instrumentation off".
    spans_with_calls = {c["span_id"] for c in llm_calls}
    for s in spans:
        if s["span_id"] in spans_with_calls or not s.get("model"):
            continue
        out_spans.append(_span(
            trace_id, f"{s['span_id']}:legacy", f"chat {s['model']}",
            s["start_time_ms"], s["end_time_ms"], {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": s["model"],
                "gen_ai.usage.input_tokens": s["input_tokens"],
                "gen_ai.usage.output_tokens": s["output_tokens"],
            }, parent_id=s["span_id"]))

    span_start = {s["span_id"]: s["start_time_ms"] for s in spans}
    for t in tool_calls:
        # Legacy rows predate start_time_ms capture; anchor them to their
        # owning agent span so ordering stays sane.
        start = t.get("start_time_ms") or span_start.get(t["span_id"], t0)
        end = start + (t.get("duration_ms") or 0)
        out_spans.append(_span(
            trace_id, t["call_id"], f"execute_tool {t['tool_name']}",
            start, end, {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": t["tool_name"],
                "input.value": t.get("arguments_json") or None,
                "output.value": t.get("result_json") or None,
                "agentpulse.success": bool(t.get("success")),
            }, parent_id=t["span_id"]))

    return {"resourceSpans": [{
        "resource": {"attributes": _attrs(
            {"service.name": f"agentpulse/{run['task_type']}"})},
        "scopeSpans": [{
            "scope": {"name": "agentpulse.export"},
            "spans": out_spans,
        }],
    }]}


def _last_run_id(db_path: Path | None) -> str | None:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT 1").fetchone()
    conn.close()
    return row[0] if row else None


def _since_cutoff(text: str) -> str:
    """A --since value as a UTC ISO cutoff comparable to runs.timestamp.
    Accepts 90m / 12h / 7d shorthand, or an ISO date/datetime (naive
    values are read as local time)."""
    m = re.fullmatch(r"(\d+)([mhd])", text.strip())
    if m:
        n = int(m.group(1))
        delta = {"m": timedelta(minutes=n), "h": timedelta(hours=n),
                 "d": timedelta(days=n)}[m.group(2)]
        return (datetime.now(timezone.utc) - delta).isoformat()
    try:
        dt = datetime.fromisoformat(text.strip())
    except ValueError:
        raise SystemExit(
            f"--since {text!r} is neither a span like 12h / 90m / 7d "
            f"nor an ISO date/datetime like 2026-08-18 or 2026-08-18T14:00")
    if dt.tzinfo is None:
        dt = dt.astimezone()                 # a naive value is local time
    return dt.astimezone(timezone.utc).isoformat()


def _runs_since(db_path: Path | None, cutoff: str) -> list[dict]:
    conn = get_connection(db_path)
    rows = [dict(r) for r in conn.execute(
        "SELECT run_id, timestamp, task_text FROM runs "
        "WHERE timestamp >= ? ORDER BY timestamp", (cutoff,))]
    conn.close()
    return rows


def _last_runs(db_path: Path | None, n: int) -> list[dict]:
    conn = get_connection(db_path)
    rows = [dict(r) for r in conn.execute(
        "SELECT run_id, timestamp, task_text FROM runs "
        "ORDER BY timestamp DESC LIMIT ?", (n,))]
    conn.close()
    return rows[::-1]                        # oldest first, like --since


def _submit(export: dict, url: str) -> str:
    import urllib.request
    req = urllib.request.Request(
        url.rstrip("/") + "/api/ingest",
        data=json.dumps(export).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def _export_many(runs: list[dict], db_path, args, what: str) -> int:
    """Handle a multi-run selection: one submission per run, one file
    per run into the -o directory, or a dry list with neither."""
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for r in runs:
        label = f"{r['run_id']}  {r['timestamp']}"
        if not out_dir and not args.submit:
            print(label)                     # dry list: what a real pass sends
            continue
        export = export_run(r["run_id"], db_path)
        if out_dir:
            f = out_dir / f"{r['run_id']}.json"
            f.write_text(json.dumps(export, indent=2, ensure_ascii=False))
            print(f"wrote {f}", file=sys.stderr)
        if args.submit:
            reply = _submit(export, args.submit)
            print(f"{label} -> {reply}", file=sys.stderr)
    if not out_dir and not args.submit:
        print(f"{len(runs)} run(s) {what}; add --submit to "
              f"send each as its own submission", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id", nargs="?", help="run to export")
    ap.add_argument("--last", nargs="?", const=1, type=int, default=None,
                    metavar="N",
                    help="export the most recent run, or with a count the "
                         "last N runs, each handled as its own export")
    ap.add_argument("--since", default=None, metavar="WHEN",
                    help="every run since WHEN (12h, 90m, 7d, or an ISO "
                         "date/datetime), each handled as its own export")
    ap.add_argument("--db", default=None,
                    help="logical DB name (as passed to instrument(db_name=...))")
    ap.add_argument("-o", "--out", default=None,
                    help="write to FILE (a directory with --since or --last N)")
    ap.add_argument("--submit", nargs="?", const="http://localhost:4951",
                    default=None, metavar="URL",
                    help="POST the export to a replay server's /api/ingest")
    args = ap.parse_args()

    db_path = resolve_db_path(args.db) if args.db else None

    if args.last is not None and args.last < 1:
        ap.error("--last needs a positive count")

    if args.since:
        if args.run_id or args.last is not None:
            ap.error("--since selects the runs itself; drop run_id / --last")
        cutoff = _since_cutoff(args.since)
        runs = _runs_since(db_path, cutoff)
        if not runs:
            print(f"no runs since {cutoff}", file=sys.stderr)
            return 1
        return _export_many(runs, db_path, args, f"since {cutoff}")

    if args.last is not None and args.last > 1:
        if args.run_id:
            ap.error("--last N selects the runs itself; drop run_id")
        runs = _last_runs(db_path, args.last)
        if not runs:
            print("no runs recorded", file=sys.stderr)
            return 1
        return _export_many(runs, db_path, args,
                            f"(the {len(runs)} most recent)")

    run_id = args.run_id
    if args.last is not None and not run_id:
        run_id = _last_run_id(db_path)
    if not run_id:
        ap.error("pass a run_id, --last, or --since")

    export = export_run(run_id, db_path)
    text = json.dumps(export, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.submit:
        reply = _submit(export, args.submit)
        print(f"submitted to {args.submit}: {reply}", file=sys.stderr)
    if not args.out and not args.submit:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
