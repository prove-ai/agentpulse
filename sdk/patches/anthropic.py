"""Anthropic SDK patch — captures LLM call timing and tokens.

Patches both Anthropic.messages.create (sync) and AsyncAnthropic.messages.create
(async) so every API call records:
  (start_ns, end_ns, input_tokens, output_tokens, model)

The sync surface matters for LangChain `llm.invoke()` callers (e.g. LangGraph
nodes using ChatAnthropic), which go through the synchronous client.

These go into the active session's _pending_api_calls queue.
When the agent's TextMessage arrives in the AutoGen patch,
on_turn_end() claims them and attaches to the correct turn.

This gives:
  - Accurate per-agent tokens (Manager and Reviewer included)
  - Accurate LLM latency (actual API round-trip, not AutoGen overhead)
  - Model name per span (for correct cost calculation)
"""

from __future__ import annotations

import time
from pathlib import Path
import sys

_OBS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_OBS_ROOT))

from sdk.session import get_active_session

_PATCHED = False


def patch_anthropic() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from anthropic.resources.messages.messages import Messages, AsyncMessages
    except ImportError:
        # Anthropic not installed — skip
        return

    def _record(response, start_ns, end_ns):
        session = get_active_session()
        if session is not None:
            usage = getattr(response, "usage", None)
            inp   = int(getattr(usage, "input_tokens",  0) or 0) if usage else 0
            out   = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
            model = str(getattr(response, "model", "") or "")
            session._pending_api_calls.append((start_ns, end_ns, inp, out, model))

    original_sync_create = Messages.create

    def _patched_sync_create(self, *args, **kwargs):
        start_ns = time.time_ns()
        response = original_sync_create(self, *args, **kwargs)
        _record(response, start_ns, time.time_ns())
        return response

    original_async_create = AsyncMessages.create

    async def _patched_async_create(self, *args, **kwargs):
        start_ns = time.time_ns()
        response = await original_async_create(self, *args, **kwargs)
        _record(response, start_ns, time.time_ns())
        return response

    Messages.create      = _patched_sync_create
    AsyncMessages.create = _patched_async_create
    _PATCHED = True
