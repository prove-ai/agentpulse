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

import json
import time

from agentpulse.sdk.session import LLMCallRecord, get_active_agent, get_active_session, safe_json

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


# ---------------------------------------------------------------------------
# Streaming capture — create(stream=True) returns a Stream of events, not a
# Message (this is how LangChain's ChatAnthropic calls the SDK). We wrap the
# stream, accumulate the events back into a Message-shaped payload, and record
# the call when the stream is exhausted or closed.
# ---------------------------------------------------------------------------
class _StreamAccumulator:
    def __init__(self) -> None:
        self.blocks: list[dict] = []
        self.model = ""
        self.stop_reason = None
        self.input_tokens = 0
        self.output_tokens = 0

    def absorb(self, event) -> None:
        try:
            et = getattr(event, "type", "")
            if et == "message_start":
                msg = getattr(event, "message", None)
                self.model = str(getattr(msg, "model", "") or "")
                usage = getattr(msg, "usage", None)
                self.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            elif et == "content_block_start":
                cb = getattr(event, "content_block", None)
                btype = getattr(cb, "type", "unknown")
                block: dict = {"type": btype}
                if btype == "text":
                    block["text"] = getattr(cb, "text", "") or ""
                elif btype == "tool_use":
                    block["id"] = getattr(cb, "id", "")
                    block["name"] = getattr(cb, "name", "")
                    block["_partial_json"] = ""
                elif btype == "thinking":
                    block["thinking"] = getattr(cb, "thinking", "") or ""
                self.blocks.append(block)
            elif et == "content_block_delta":
                delta = getattr(event, "delta", None)
                dtype = getattr(delta, "type", "")
                idx = getattr(event, "index", None)
                if not self.blocks:
                    return
                block = self.blocks[idx] if isinstance(idx, int) and 0 <= idx < len(self.blocks) \
                    else self.blocks[-1]
                if dtype == "text_delta":
                    block["text"] = block.get("text", "") + (getattr(delta, "text", "") or "")
                elif dtype == "input_json_delta":
                    block["_partial_json"] = block.get("_partial_json", "") + \
                        (getattr(delta, "partial_json", "") or "")
                elif dtype == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + \
                        (getattr(delta, "thinking", "") or "")
            elif et == "message_delta":
                delta = getattr(event, "delta", None)
                sr = getattr(delta, "stop_reason", None)
                if sr:
                    self.stop_reason = sr
                usage = getattr(event, "usage", None)
                if usage is not None:
                    self.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        except Exception:
            pass  # capture must never break the user's stream

    def response_json(self) -> str:
        blocks = []
        for b in self.blocks:
            b = dict(b)
            pj = b.pop("_partial_json", None)
            if pj is not None:
                try:
                    b["input"] = json.loads(pj) if pj else {}
                except Exception:
                    b["input"] = pj
            blocks.append(b)
        return safe_json({
            "content": blocks, "stop_reason": self.stop_reason, "model": self.model,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
        })


def _record_stream(acc: _StreamAccumulator, start_ns: int, end_ns: int,
                   request_json: str, agent) -> None:
    session = get_active_session()
    if session is not None:
        session._pending_api_calls.append(LLMCallRecord(
            start_ns=start_ns, end_ns=end_ns,
            input_tokens=acc.input_tokens, output_tokens=acc.output_tokens,
            model=acc.model, request_json=request_json,
            response_json=acc.response_json(), agent=agent,
        ))


class _RecordingStream:
    """Wraps a sync anthropic Stream: forwards everything, records on finish."""

    def __init__(self, inner, start_ns: int, request_json: str) -> None:
        self._inner = inner
        self._acc = _StreamAccumulator()
        self._start_ns = start_ns
        self._request_json = request_json
        self._agent = get_active_agent()
        self._recorded = False

    def _finish(self) -> None:
        if not self._recorded:
            self._recorded = True
            _record_stream(self._acc, self._start_ns, time.time_ns(),
                           self._request_json, self._agent)

    def __iter__(self):
        try:
            for event in self._inner:
                self._acc.absorb(event)
                yield event
        finally:
            self._finish()

    def __enter__(self):
        enter = getattr(self._inner, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, *exc):
        try:
            ex = getattr(self._inner, "__exit__", None)
            return ex(*exc) if callable(ex) else None
        finally:
            self._finish()

    def close(self):
        try:
            close = getattr(self._inner, "close", None)
            if callable(close):
                close()
        finally:
            self._finish()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _AsyncRecordingStream:
    """Async twin of _RecordingStream."""

    def __init__(self, inner, start_ns: int, request_json: str) -> None:
        self._inner = inner
        self._acc = _StreamAccumulator()
        self._start_ns = start_ns
        self._request_json = request_json
        self._agent = get_active_agent()
        self._recorded = False

    def _finish(self) -> None:
        if not self._recorded:
            self._recorded = True
            _record_stream(self._acc, self._start_ns, time.time_ns(),
                           self._request_json, self._agent)

    async def __aiter__(self):
        try:
            async for event in self._inner:
                self._acc.absorb(event)
                yield event
        finally:
            self._finish()

    async def __aenter__(self):
        enter = getattr(self._inner, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, *exc):
        try:
            ex = getattr(self._inner, "__aexit__", None)
            return (await ex(*exc)) if callable(ex) else None
        finally:
            self._finish()

    async def aclose(self):
        try:
            aclose = getattr(self._inner, "aclose", None)
            if callable(aclose):
                await aclose()
        finally:
            self._finish()

    def __getattr__(self, name):
        return getattr(self._inner, name)


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


def _is_raw(response) -> bool:
    # client.messages.with_raw_response.create (how LangChain calls the SDK)
    # returns an APIResponse wrapper, not the parsed Message/Stream.
    return type(response).__name__ in ("APIResponse", "LegacyAPIResponse",
                                       "AsyncAPIResponse", "AsyncLegacyAPIResponse")


def _patch_pair(messages_cls, async_messages_cls):
    original_sync_create = messages_cls.create

    def _patched_sync_create(self, *args, **kwargs):
        start_ns = time.time_ns()
        request_json = _request_payload(kwargs)
        response = original_sync_create(self, *args, **kwargs)
        parsed = response
        if _is_raw(response):
            # parse() caches, so the caller's own parse() gets the same object.
            try:
                parsed = response.parse()
            except Exception:
                parsed = response
        if kwargs.get("stream"):
            # Stream of events, not a Message — record when it's consumed.
            wrapper = _RecordingStream(parsed, start_ns, request_json)
            if _is_raw(response):
                try:
                    response.parse = lambda *a, **k: wrapper
                    return response
                except Exception:
                    return response  # can't intercept; skip recording this call
            return wrapper
        _record(parsed, start_ns, time.time_ns(), request_json)
        return response

    original_async_create = async_messages_cls.create

    async def _patched_async_create(self, *args, **kwargs):
        start_ns = time.time_ns()
        request_json = _request_payload(kwargs)
        response = await original_async_create(self, *args, **kwargs)
        parsed = response
        if _is_raw(response):
            try:
                parsed = await response.parse()
            except Exception:
                parsed = response
        if kwargs.get("stream"):
            wrapper = _AsyncRecordingStream(parsed, start_ns, request_json)
            if _is_raw(response):
                try:
                    async def _parse_override(*a, **k):
                        return wrapper
                    response.parse = _parse_override
                    return response
                except Exception:
                    return response
            return wrapper
        _record(parsed, start_ns, time.time_ns(), request_json)
        return response

    messages_cls.create       = _patched_sync_create
    async_messages_cls.create = _patched_async_create

def patch_anthropic() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from anthropic.resources.messages.messages import Messages, AsyncMessages
    except ImportError:
        # Anthropic not installed — skip
        return

    _patch_pair(Messages, AsyncMessages)
    # LangChain routes through client.beta.messages.create when beta features
    # are in play — patch that surface too when it exists.
    try:
        from anthropic.resources.beta.messages.messages import (
            Messages as BetaMessages, AsyncMessages as AsyncBetaMessages,
        )
        _patch_pair(BetaMessages, AsyncBetaMessages)
    except ImportError:
        pass
    _PATCHED = True
