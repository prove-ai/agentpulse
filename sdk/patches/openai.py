"""OpenAI SDK patch — captures LLM call timing and tokens.

Patches both sync and async, both chat-completions and legacy completions.
Each captured call records (start_ns, end_ns, input_tokens, output_tokens, model)
into the active session's _pending_api_calls queue.

Handles three response shapes:
  1. Non-streaming sync/async   → response.usage is present directly.
  2. Streaming async iterator   → wrapped in a transparent async generator that
                                  watches each chunk for chunk.usage (emitted
                                  when stream_options={"include_usage": True})
                                  AND falls back to counting delta tokens by
                                  length if usage is never emitted.
  3. Streaming sync iterator    → same as (2) but synchronous.

The wrapper is fully transparent — the caller iterates as normal and gets the
same chunks in the same order. Tokens are recorded when the stream is exhausted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_OBS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_OBS_ROOT))

from sdk.session import get_active_session, get_active_agent

_PATCHED = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _record(start_ns: int, end_ns: int,
            input_tokens: int, output_tokens: int, model: str) -> None:
    """Push a finished LLM call into the active session's queue.

    Tags the call with the active agent (a ContextVar — per-asyncio-task).
    This is how parallel asyncio.gather agents stay correctly attributed.
    Legacy untagged tuples remain 5 elements; agent-tagged tuples are 6.
    """
    session = get_active_session()
    if session is None:
        return
    agent = get_active_agent()
    session._pending_api_calls.append(
        (start_ns, end_ns,
         int(input_tokens or 0), int(output_tokens or 0),
         str(model or ""), agent)
    )


def _record_from_response(start_ns: int, end_ns: int, response) -> None:
    """Non-streaming responses carry .usage directly."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    _record(
        start_ns, end_ns,
        getattr(usage, "prompt_tokens",     0),
        getattr(usage, "completion_tokens", 0),
        getattr(response, "model", ""),
    )


def _is_stream(response) -> bool:
    """True when the response is a stream (async or sync iterator), not a finished object."""
    if hasattr(response, "usage"):
        return False
    if hasattr(response, "__aiter__") or hasattr(response, "__anext__"):
        return True
    if hasattr(response, "__iter__") and not isinstance(response, (str, bytes, dict)):
        # Sync stream: has __iter__ but no .usage (and isn't a basic container)
        # OpenAI's Stream class also exposes __next__.
        return hasattr(response, "__next__") or hasattr(response, "_iterator")
    return False


# ---------------------------------------------------------------------------
# Streaming wrappers — transparent passthrough that records on stream exhaustion
# ---------------------------------------------------------------------------
class _AsyncStreamWrapper:
    """Wrap an OpenAI async stream so we can sniff usage as chunks fly by.

    Records tokens eagerly the moment a usage chunk arrives, instead of waiting
    for StopAsyncIteration. This is critical: many callers iterate `choices[0]`
    on every chunk and crash on the final usage chunk (which has `choices=[]`).
    If we waited for StopAsyncIteration to record, those crashes would lose
    the call entirely. Recording on usage-chunk arrival makes the wrapper
    resilient to caller exception handling.
    """

    def __init__(self, stream, start_ns: int):
        self._stream      = stream
        self._start_ns    = start_ns
        self._model       = ""
        self._usage_inp   = 0
        self._usage_out   = 0
        self._delta_chars = 0      # fallback if usage is never emitted
        self._finalised   = False  # guard against double-recording

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Pull chunks until we have one safe to hand back to the caller.
        # The OpenAI "include_usage" final chunk has `usage` set AND
        # `choices=[]` — callers often do `chunk.choices[0]` and crash on it.
        # We sniff it for tokens here and swallow it so it never reaches them.
        while True:
            try:
                chunk = await self._stream.__anext__()
            except StopAsyncIteration:
                self._finalise()
                raise
            self._inspect(chunk)
            usage   = getattr(chunk, "usage", None)
            choices = getattr(chunk, "choices", None)
            if usage is not None and not choices:
                # Tokens were already captured inside _inspect → _finalise.
                # Don't yield this chunk — keep it internal to the wrapper.
                continue
            return chunk

    async def aclose(self):
        """Called when an async-for loop exits early (break / exception)."""
        self._finalise()
        underlying_close = getattr(self._stream, "aclose", None)
        if underlying_close is not None:
            await underlying_close()

    # Forward everything else (close, response, etc.) to the underlying stream.
    def __getattr__(self, name):
        return getattr(self._stream, name)

    def _inspect(self, chunk) -> None:
        if not self._model:
            m = getattr(chunk, "model", None)
            if m:
                self._model = str(m)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            # The authoritative "include_usage" chunk has arrived. Record NOW
            # so we don't lose this call if the caller crashes processing it
            # (e.g. doing chunk.choices[0] when choices is empty).
            self._usage_inp = int(getattr(usage, "prompt_tokens",     0) or 0)
            self._usage_out = int(getattr(usage, "completion_tokens", 0) or 0)
            self._finalise()
            return
        # Fallback: accumulate output character count from streamed deltas, so
        # we have *something* if include_usage isn't set on the request.
        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = getattr(choices[0], "delta", None) or getattr(choices[0], "text", None)
            content = getattr(delta, "content", delta if isinstance(delta, str) else None)
            if isinstance(content, str):
                self._delta_chars += len(content)

    def _finalise(self) -> None:
        if self._finalised:
            return
        self._finalised = True
        end_ns = time.time_ns()
        # If the authoritative usage chunk never arrived, estimate output tokens
        # from character count (~4 chars per token is a common rule of thumb).
        out_tokens = self._usage_out or max(1, self._delta_chars // 4)
        _record(self._start_ns, end_ns, self._usage_inp, out_tokens, self._model)


class _SyncStreamWrapper:
    """Same idea, synchronous."""

    def __init__(self, stream, start_ns: int):
        self._stream      = stream
        self._start_ns    = start_ns
        self._model       = ""
        self._usage_inp   = 0
        self._usage_out   = 0
        self._delta_chars = 0
        self._finalised   = False

    def __iter__(self):
        return self

    def __next__(self):
        # Same swallow-the-usage-chunk logic as the async wrapper.
        while True:
            try:
                chunk = next(self._stream)
            except StopIteration:
                self._finalise()
                raise
            self._inspect(chunk)
            usage   = getattr(chunk, "usage", None)
            choices = getattr(chunk, "choices", None)
            if usage is not None and not choices:
                continue
            return chunk

    def __getattr__(self, name):
        return getattr(self._stream, name)

    _inspect  = _AsyncStreamWrapper._inspect
    _finalise = _AsyncStreamWrapper._finalise


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def patch_openai() -> None:
    """Patch every OpenAI create-call surface we know about.

    Safe to call multiple times (no-ops after the first). If the openai
    package isn't installed, returns silently — same convention as the
    Anthropic patch.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        import openai  # noqa: F401  (presence check)
    except ImportError:
        return

    _patch_chat_async()
    _patch_chat_sync()
    _patch_legacy_async()
    _patch_legacy_sync()

    _PATCHED = True


# ---------------------------------------------------------------------------
# Chat completions — the modern endpoint (most users)
# ---------------------------------------------------------------------------
def _wrap_async(response, start_ns):
    """If streaming, wrap in async passthrough; else record non-streaming usage."""
    if _is_stream(response):
        return _AsyncStreamWrapper(response, start_ns)
    _record_from_response(start_ns, time.time_ns(), response)
    return response


def _wrap_sync(response, start_ns):
    if _is_stream(response):
        return _SyncStreamWrapper(response, start_ns)
    _record_from_response(start_ns, time.time_ns(), response)
    return response


def _force_usage(kwargs: dict) -> dict:
    """Inject stream_options={'include_usage': True} when stream=True so the
    OpenAI server sends an authoritative usage chunk at the end of the stream.
    Without this we can only estimate output tokens from delta character count,
    and input tokens stay 0.

    Non-destructive: merges with whatever stream_options the caller already set.
    """
    if not kwargs.get("stream"):
        return kwargs
    so = dict(kwargs.get("stream_options") or {})
    so.setdefault("include_usage", True)
    kwargs["stream_options"] = so
    return kwargs


def _patch_chat_async() -> None:
    try:
        from openai.resources.chat.completions import AsyncCompletions
    except ImportError:
        return
    original = AsyncCompletions.create

    async def _instrumented(self, *args, **kwargs):
        start_ns = time.time_ns()
        kwargs = _force_usage(kwargs)
        response = await original(self, *args, **kwargs)
        return _wrap_async(response, start_ns)

    AsyncCompletions.create = _instrumented


def _patch_chat_sync() -> None:
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        return
    original = Completions.create

    def _instrumented(self, *args, **kwargs):
        start_ns = time.time_ns()
        kwargs = _force_usage(kwargs)
        response = original(self, *args, **kwargs)
        return _wrap_sync(response, start_ns)

    Completions.create = _instrumented


# ---------------------------------------------------------------------------
# Legacy completions endpoint — older code paths
# ---------------------------------------------------------------------------
def _patch_legacy_async() -> None:
    try:
        from openai.resources.completions import AsyncCompletions  # legacy module
    except ImportError:
        return
    original = AsyncCompletions.create

    async def _instrumented(self, *args, **kwargs):
        start_ns = time.time_ns()
        response = await original(self, *args, **kwargs)
        return _wrap_async(response, start_ns)

    AsyncCompletions.create = _instrumented


def _patch_legacy_sync() -> None:
    try:
        from openai.resources.completions import Completions
    except ImportError:
        return
    original = Completions.create

    def _instrumented(self, *args, **kwargs):
        start_ns = time.time_ns()
        response = original(self, *args, **kwargs)
        return _wrap_sync(response, start_ns)

    Completions.create = _instrumented
