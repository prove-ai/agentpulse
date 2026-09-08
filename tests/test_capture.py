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
