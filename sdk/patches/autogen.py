"""AutoGen patch — wraps SelectorGroupChat.run_stream.

Turn lifecycle:
  OPEN  → on ThoughtEvent or ToolCallRequestEvent (records real start time)
  CLOSE → on TextMessage or ToolCallSummaryMessage (attaches tokens + STATUS)

Every event is re-yielded unchanged so the caller (Console, async for) sees
no difference. Observability is completely transparent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_OBS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_OBS_ROOT))

from sdk.session import (
    RunSession, get_config,
    set_active_session, clear_active_session,
    parse_status_value,
)
from storage.sqlite_store import write_session

_PATCHED = False


def patch_autogen() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from autogen_agentchat.teams import SelectorGroupChat
    except ImportError:
        return

    original_run_stream = SelectorGroupChat.run_stream

    async def _instrumented_run_stream(self: Any, *, task: Any, **kwargs: Any):
        config = get_config()
        if not config.enabled:
            async for event in original_run_stream(self, task=task, **kwargs):
                yield event
            return

        task_text = task if isinstance(task, str) else str(task)
        session = RunSession(
            task_text=task_text,
            task_type=config.task_type,
            prompt_version=config.prompt_version,
            db_name=config.db_name,
        )
        set_active_session(session)

        try:
            async for event in original_run_stream(self, task=task, **kwargs):
                _process_event(session, event)
                yield event
        finally:
            session.finalise()
            clear_active_session()
            try:
                write_session(session)
            except Exception as e:
                print(f"[observability] warning: could not write session: {e}")

    SelectorGroupChat.run_stream = _instrumented_run_stream
    _PATCHED = True


# ---------------------------------------------------------------------------
# Event dispatcher
# ---------------------------------------------------------------------------
def _process_event(session: RunSession, event: Any) -> None:
    type_name = type(event).__name__
    src = getattr(event, "source", None)

    if not src or src == "user":
        # Handle TaskResult (no source field)
        if type_name == "TaskResult":
            reason = getattr(event, "stop_reason", "") or ""
            session.on_termination(reason)
        return

    # ---- Phase 1: open turn on first agent activity ----
    if type_name in ("ThoughtEvent", "ToolCallRequestEvent"):
        # AutoGen SelectorGroupChat is sequential, so each turn's structural
        # parent = the previous turn's span_id. This enables workflow critical
        # path even though no parallel groups will ever fire (branch_id stays
        # None, which is correct).
        parent_step_id = session.turns[-1].span_id if session.turns else None
        session.on_turn_start(src, parent_step_id=parent_step_id)

    # ---- Tool requests ----
    if type_name == "ToolCallRequestEvent":
        for call in (getattr(event, "content", None) or []):
            call_id   = getattr(call, "id",   "")
            tool_name = getattr(call, "name", "unknown")
            session.on_tool_request(call_id, tool_name)

    # ---- Tool results ----
    elif type_name == "ToolCallExecutionEvent":
        for result in (getattr(event, "content", None) or []):
            call_id  = getattr(result, "call_id",  "")
            is_error = getattr(result, "is_error", False)
            session.on_tool_result(call_id, is_error)

    # ---- Phase 2: close turn on final agent message ----
    elif type_name in ("TextMessage", "ToolCallSummaryMessage"):
        usage  = getattr(event, "models_usage", None)
        inp    = int(getattr(usage, "prompt_tokens",     0) or 0) if usage else 0
        out    = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        model  = str(getattr(usage, "model", "") or "") if usage else ""
        content = getattr(event, "content", "") or ""
        status_value = parse_status_value(content) if isinstance(content, str) else ""

        # If the turn was already opened by a ThoughtEvent, its parent_step_id
        # is already set. If not (agents that go straight to TextMessage),
        # compute it now so on_turn_end's lazy-create path gets it too.
        parent_step_id = None
        if session._current_turn is None or session._current_turn.agent_name != src:
            parent_step_id = session.turns[-1].span_id if session.turns else None

        session.on_turn_end(
            src, inp, out, model, status_value,
            parent_step_id=parent_step_id,
        )

    # ---- Termination ----
    elif type_name == "TaskResult":
        reason = getattr(event, "stop_reason", "") or ""
        session.on_termination(reason)
