"""Anthropic SDK patch — captures LLM call timing, tokens, and full payloads.

Patches both Anthropic.messages.create (sync) and AsyncAnthropic.messages.create
(async) so every API call records an LLMCallRecord: timing, tokens, model, and
the verbatim request/response payloads a replay needs (messages, system, tools,
sampling params in; content blocks, stop_reason, usage out).

The sync surface matters for LangChain `llm.invoke()` callers (e.g. LangGraph
nodes using ChatAnthropic), which go through the synchronous client.

These go into the active session's _pending_api_calls queue.
When the agent's TextMessage arrives in the AutoGen patch,
on_turn_end() claims them and attaches to the correct turn.

This gives:
  - Accurate per-agent tokens (Manager and Reviewer included)
  - Accurate LLM latency (actual API round-trip, not AutoGen overhead)
  - Model name per span (for correct cost calculation)
  - Replayable request/response payloads per call
"""

from __future__ import annotations

import time
from pathlib import Path
import sys

_OBS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_OBS_ROOT))

from sdk.session import LLMCallRecord, get_active_agent, get_active_session, safe_json

_PATCHED = False

# Request kwargs worth recording for replay. Everything the API accepts that
# shapes the completion; auth/transport kwargs are deliberately excluded.
_REQUEST_KEYS = (
    "model", "messages", "system", "tools", "tool_choice", "max_tokens",
    "temperature", "top_p", "top_k", "stop_sequences", "metadata", "thinking",
)


def _request_payload(kwargs: dict) -> str:
    return safe_json({k: kwargs[k] for k in _REQUEST_KEYS if k in kwargs})


def _response_payload(response) -> str:
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        try:
            return safe_json(dump())
        except Exception:
            pass
    return safe_json({
        "content":     getattr(response, "content", None),
        "stop_reason": getattr(response, "stop_reason", None),
        "model":       getattr(response, "model", None),
        "usage":       getattr(response, "usage", None),
    })


def patch_anthropic() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from anthropic.resources.messages.messages import Messages, AsyncMessages
    except ImportError:
        # Anthropic not installed — skip
        return

    def _record(response, start_ns, end_ns, request_json):
        session = get_active_session()
        if session is not None:
            usage = getattr(response, "usage", None)
            inp   = int(getattr(usage, "input_tokens",  0) or 0) if usage else 0
            out   = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
            model = str(getattr(response, "model", "") or "")
            session._pending_api_calls.append(LLMCallRecord(
                start_ns=start_ns, end_ns=end_ns,
                input_tokens=inp, output_tokens=out, model=model,
                request_json=request_json,
                response_json=_response_payload(response),
                agent=get_active_agent(),
            ))

    original_sync_create = Messages.create

    def _patched_sync_create(self, *args, **kwargs):
        start_ns = time.time_ns()
        request_json = _request_payload(kwargs)
        response = original_sync_create(self, *args, **kwargs)
        _record(response, start_ns, time.time_ns(), request_json)
        return response

    original_async_create = AsyncMessages.create

    async def _patched_async_create(self, *args, **kwargs):
        start_ns = time.time_ns()
        request_json = _request_payload(kwargs)
        response = await original_async_create(self, *args, **kwargs)
        _record(response, start_ns, time.time_ns(), request_json)
        return response

    Messages.create      = _patched_sync_create
    AsyncMessages.create = _patched_async_create
    _PATCHED = True
