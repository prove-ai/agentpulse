"""Capture regression tests: streaming Anthropic responses and LangGraph
node output extraction (the two paths real LangChain/LangGraph apps hit)."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentpulse.sdk.session import RunSession, set_active_session, clear_active_session  # noqa: E402
from agentpulse.sdk.patches.anthropic import _RecordingStream, _AsyncRecordingStream    # noqa: E402
from agentpulse.sdk.patches.langchain import _text_from_outputs                          # noqa: E402


def _fake_events():
    """The event sequence anthropic yields for create(stream=True)."""
    return [
        NS(type="message_start",
           message=NS(model="claude-opus-5", usage=NS(input_tokens=120))),
        NS(type="content_block_start", index=0,
           content_block=NS(type="text", text="")),
        NS(type="content_block_delta", index=0,
           delta=NS(type="text_delta", text="Hello ")),
        NS(type="content_block_delta", index=0,
           delta=NS(type="text_delta", text="world.")),
        NS(type="content_block_stop", index=0),
        NS(type="content_block_start", index=1,
           content_block=NS(type="tool_use", id="tu_1", name="lookup", input={})),
        NS(type="content_block_delta", index=1,
           delta=NS(type="input_json_delta", partial_json='{"q": "who')),
        NS(type="content_block_delta", index=1,
           delta=NS(type="input_json_delta", partial_json='dunit"}')),
        NS(type="content_block_stop", index=1),
        NS(type="message_delta", delta=NS(stop_reason="tool_use"),
           usage=NS(output_tokens=57)),
        NS(type="message_stop"),
    ]


def _session():
    session = RunSession(task_text="t")
    set_active_session(session)
    return session


def test_sync_stream_capture():
    session = _session()
    try:
        wrapped = _RecordingStream(iter(_fake_events()), start_ns=1, request_json="{}")
        events = list(wrapped)
        assert len(events) == 11, "all events must be forwarded untouched"

        assert len(session._pending_api_calls) == 1
        rec = session._pending_api_calls[0]
        assert rec.input_tokens == 120 and rec.output_tokens == 57
        assert rec.model == "claude-opus-5"
        resp = json.loads(rec.response_json)
        assert resp["stop_reason"] == "tool_use"
        assert resp["usage"] == {"input_tokens": 120, "output_tokens": 57}
        assert resp["content"][0] == {"type": "text", "text": "Hello world."}
        assert resp["content"][1]["type"] == "tool_use"
        assert resp["content"][1]["input"] == {"q": "whodunit"}
    finally:
        clear_active_session()


def test_sync_stream_records_once_on_close_and_iter():
    session = _session()
    try:
        wrapped = _RecordingStream(iter(_fake_events()), start_ns=1, request_json="{}")
        list(wrapped)
        wrapped.close()  # second finish path must not double-record
        assert len(session._pending_api_calls) == 1
    finally:
        clear_active_session()


def test_async_stream_capture():
    async def _events():
        for e in _fake_events():
            yield e

    async def _run():
        wrapped = _AsyncRecordingStream(_events(), start_ns=1, request_json="{}")
        return [e async for e in wrapped]

    session = _session()
    try:
        events = asyncio.run(_run())
        assert len(events) == 11
        assert len(session._pending_api_calls) == 1
        rec = session._pending_api_calls[0]
        assert (rec.input_tokens, rec.output_tokens) == (120, 57)
        assert json.loads(rec.response_json)["content"][0]["text"] == "Hello world."
    finally:
        clear_active_session()


class APIResponse:  # the name matters: the patch detects wrappers by type name
    """Fake with_raw_response wrapper: parse() returns the rich object."""
    def __init__(self, parsed):
        self._parsed = parsed

    def parse(self, *a, **k):
        return self._parsed


def _make_patched_pair():
    from agentpulse.sdk.patches.anthropic import _patch_pair

    class Messages:
        def create(self, **kwargs):
            return kwargs["_resp"]

    class AsyncMessages:
        async def create(self, **kwargs):
            return kwargs["_resp"]

    _patch_pair(Messages, AsyncMessages)
    return Messages(), AsyncMessages()


def _fake_message():
    return NS(
        content=[{"type": "text", "text": "parsed!"}], stop_reason="end_turn",
        model="claude-opus-5", usage=NS(input_tokens=11, output_tokens=7),
        model_dump=lambda: {"content": [{"type": "text", "text": "parsed!"}],
                            "stop_reason": "end_turn", "model": "claude-opus-5",
                            "usage": {"input_tokens": 11, "output_tokens": 7}})


def test_raw_response_unwrap_nonstream():
    """LangChain calls messages.with_raw_response.create: the record must come
    from the parsed Message while the caller still gets the raw wrapper."""
    msgs, _ = _make_patched_pair()
    session = _session()
    try:
        raw = APIResponse(_fake_message())
        result = msgs.create(model="m", _resp=raw)
        assert result is raw, "caller must still receive the raw wrapper"
        assert len(session._pending_api_calls) == 1
        rec = session._pending_api_calls[0]
        assert (rec.input_tokens, rec.output_tokens, rec.model) == (11, 7, "claude-opus-5")
        assert json.loads(rec.response_json)["content"][0]["text"] == "parsed!"
    finally:
        clear_active_session()


def test_raw_response_stream():
    """with_raw_response + stream=True: parse() must hand the caller a recording
    wrapper, and consuming it must produce the record."""
    msgs, _ = _make_patched_pair()
    session = _session()
    try:
        raw = APIResponse(iter(_fake_events()))
        result = msgs.create(model="m", stream=True, _resp=raw)
        assert result is raw
        stream = result.parse()
        events = list(stream)
        assert len(events) == 11
        assert len(session._pending_api_calls) == 1
        rec = session._pending_api_calls[0]
        assert (rec.input_tokens, rec.output_tokens) == (120, 57)
        assert json.loads(rec.response_json)["content"][0]["text"] == "Hello world."
    finally:
        clear_active_session()


def test_plain_message_still_recorded():
    """Direct SDK usage (no raw wrapper, no stream) keeps working."""
    msgs, _ = _make_patched_pair()
    session = _session()
    try:
        message = _fake_message()
        result = msgs.create(model="m", _resp=message)
        assert result is message
        assert len(session._pending_api_calls) == 1
        assert session._pending_api_calls[0].input_tokens == 11
    finally:
        clear_active_session()


def test_text_from_outputs_shapes():
    # LangGraph append-reducer shape: last message wins.
    msg = NS(content="The butler did it.")
    assert _text_from_outputs({"messages": [NS(content="earlier"), msg]}) == "The butler did it."
    # Anthropic-style content blocks on the message.
    blocks = NS(content=[{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}])
    assert _text_from_outputs({"messages": [blocks]}) == "part one\npart two"
    # Plain string output and common string fields.
    assert _text_from_outputs("just text") == "just text"
    assert _text_from_outputs({"output": "field text"}) == "field text"
    # Unknown dict falls back to its JSON so nothing is lost.
    assert json.loads(_text_from_outputs({"verdict": {"suspect": "butler"}})) == \
        {"verdict": {"suspect": "butler"}}
    # Never raises.
    assert _text_from_outputs(None) == ""


def test_failed_call_recorded():
    """An API exception still produces a record with the error message."""
    from agentpulse.sdk.patches.anthropic import _patch_pair

    class Messages:
        def create(self, **kwargs):
            raise RuntimeError("overloaded_error: try again later")

    class AsyncMessages:
        async def create(self, **kwargs):
            raise RuntimeError("boom")

    _patch_pair(Messages, AsyncMessages)
    session = _session()
    try:
        try:
            Messages().create(model="m", messages=[])
            assert False, "exception must propagate"
        except RuntimeError:
            pass
        assert len(session._pending_api_calls) == 1
        rec = session._pending_api_calls[0]
        assert rec.error.startswith("RuntimeError: overloaded_error")
        assert rec.request_json  # the request is still captured
    finally:
        clear_active_session()


def test_cache_tokens_captured():
    """Anthropic cache usage fields land on the record (parsed + stream)."""
    msgs, _ = _make_patched_pair()
    session = _session()
    try:
        message = _fake_message()
        message.usage = NS(input_tokens=11, output_tokens=7,
                           cache_read_input_tokens=900, cache_creation_input_tokens=50)
        msgs.create(model="m", _resp=APIResponse(message))
        rec = session._pending_api_calls[0]
        assert (rec.cache_read_tokens, rec.cache_creation_tokens) == (900, 50)
    finally:
        clear_active_session()

    # Stream path: cache usage arrives on message_start.
    from agentpulse.sdk.patches.anthropic import _RecordingStream
    events = _fake_events()
    events[0].message.usage = NS(input_tokens=120, cache_read_input_tokens=777,
                                 cache_creation_input_tokens=33)
    session = _session()
    try:
        list(_RecordingStream(iter(events), start_ns=1, request_json="{}"))
        rec = session._pending_api_calls[0]
        assert (rec.cache_read_tokens, rec.cache_creation_tokens) == (777, 33)
    finally:
        clear_active_session()


def test_cache_aware_cost():
    from agentpulse.storage.sqlite_store import compute_cost, model_price
    inp_p, out_p = model_price("claude-sonnet-4-6")
    base = compute_cost(1000, 100, "claude-sonnet-4-6")
    with_cache = compute_cost(1000, 100, "claude-sonnet-4-6",
                              cache_read_tokens=10_000, cache_creation_tokens=2_000)
    expected_extra = (10_000 * inp_p * 0.1 + 2_000 * inp_p * 1.25) / 1_000_000
    assert abs((with_cache - base) - expected_extra) < 1e-9
    # OpenAI: cached tokens are inside input_tokens, billed at half price.
    gpt_base = compute_cost(1000, 100, "gpt-4o")
    gpt_cached = compute_cost(1000, 100, "gpt-4o", cache_read_tokens=800)
    assert gpt_cached < gpt_base
