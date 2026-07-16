"""LangChain adapter — captures agent + tool events via the callback system.

Strategy: register an `ObservabilityCallback` (a BaseCallbackHandler) as a
*global* callback via LangChain's `register_configure_hook`. This makes the
handler auto-attach to every chain invocation in the process — the user does
not need to pass `callbacks=` anywhere. Same "2-line" UX as the AutoGen patch.

What this unlocks for free
  - LangChain AgentExecutor
  - LangGraph (each node fires a chain_start with the node name)
  - CrewAI and anything else built on LangChain callbacks

Mapping to our RunSession model
  on_chain_start  → on_turn_start(chain_name)    (skipping scaffolding chains)
  on_chain_end    → on_turn_end                  (tokens come from openai/anthropic patches)
  on_tool_start   → on_tool_request(run_id, name)
  on_tool_end     → on_tool_result(run_id, success)
  on_tool_error   → on_tool_result(run_id, error)
  on_agent_finish → on_termination(...)

Session lifecycle: the first chain_start with no parent in the process opens
the RunSession (if instrumentation is enabled and no AutoGen session is
already active). When that same root run_id sees on_chain_end, the session
finalises and writes to SQLite.

Scaffolding chains (RunnableSequence, ChatPromptTemplate, …) are filtered
out so the captured turns are actual agent/node boundaries, not internals.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

_OBS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_OBS_ROOT))

from sdk.session import (
    RunSession, get_config, get_active_session,
    set_active_session, clear_active_session, parse_status_value,
)
from storage.sqlite_store import write_session

_PATCHED = False

# Chain names we ignore — they are LangChain scaffolding, not agent boundaries.
# Anything else with a name is treated as a turn (agent node / executor / custom chain).
_IGNORED_CHAIN_NAMES = {
    "RunnableSequence", "RunnablePassthrough", "RunnableLambda",
    "RunnableParallel", "RunnableMap", "RunnableBranch",
    "RunnableAssign", "RunnableBinding", "RunnableEach",
    "RunnableWithFallbacks", "RunnableRetry",
    "ChatPromptTemplate", "PromptTemplate", "FewShotPromptTemplate",
    "StrOutputParser", "PydanticOutputParser", "JsonOutputParser",
    # Tool-calling / structured-output parsers. These run as nested chains
    # inside an agent node and would otherwise (a) clutter the agent list and
    # (b) steal the LLM token attribution from the agent that owns the call.
    "PydanticToolsParser", "JsonOutputKeyToolsParser", "JsonOutputToolsParser",
    "OpenAIToolsAgentOutputParser", "ToolsAgentOutputParser",
    # LangGraph internals — the compiled-graph wrapper and per-step plumbing
    # that fire chain callbacks but are not agent boundaries. The base-name
    # split in _is_scaffolding() also catches the "ChannelWrite<...>" form.
    "LangGraph", "Pregel", "PregelNode", "RunnableSeq",
    "__start__", "__end__", "ChannelWrite", "ChannelRead",
    "_write", "_route", "_control_branch",
}


def _is_scaffolding(name: str) -> bool:
    """Match scaffolding names robustly, including the generic suffix form
    e.g. 'RunnableParallel<Research,Analyst>' or 'RunnableSequence<...>'.
    """
    if not name:
        return True
    base = name.split("<", 1)[0].strip()
    return base in _IGNORED_CHAIN_NAMES


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def patch_langchain() -> None:
    """Install the observability callback as a global LangChain handler.

    Safe to call multiple times. No-op if langchain-core isn't installed.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
        from langchain_core.tracers.context import register_configure_hook
    except ImportError:
        # LangChain not installed — silent skip, same convention as anthropic/openai patches.
        return

    from contextvars import ContextVar

    handler = _ObservabilityCallback.build(BaseCallbackHandler)()

    # ContextVar holds the handler; register_configure_hook makes LangChain's
    # callback manager auto-attach this var's value to every chain invocation.
    _handler_var: ContextVar = ContextVar("observability_handler", default=None)
    register_configure_hook(_handler_var, inheritable=True)
    _handler_var.set(handler)
    # Keep references alive so they aren't garbage-collected.
    globals()["_handler_var"] = _handler_var
    globals()["_handler"]     = handler

    _PATCHED = True


# ---------------------------------------------------------------------------
# Callback handler — built at patch time so BaseCallbackHandler stays optional
# ---------------------------------------------------------------------------
class _ObservabilityCallback:
    """Wrapper so we can subclass BaseCallbackHandler conditionally."""

    @classmethod
    def build(cls, BaseCallbackHandler):
        class ObservabilityCallback(BaseCallbackHandler):
            # Tell LangChain to keep firing callbacks even when LangSmith etc.
            # are also installed. (Default behaviour anyway, but explicit.)
            ignore_chain = False
            # We listen to LLM/chat-model events to read per-call token usage and
            # attribute it to the issuing agent's span via the run tree — correct
            # even when several agent turns are open at once (parallel branches).
            ignore_llm   = False
            ignore_agent = False
            ignore_chat_model = False
            ignore_retriever = True

            def __init__(self):
                # Track the run_id that opened the current session (top-level chain).
                self._root_run_id: Optional[UUID] = None
                # run_id → agent_name for currently-open turns.
                self._open_turns: dict[UUID, str] = {}
                # DAG tracking
                self._run_to_span: dict[UUID, str] = {}   # run_id → our span_id
                self._parent_of_run: dict[UUID, Optional[UUID]] = {}
                self._parallel_parents: set[UUID] = set()  # run_ids that ARE parallel constructs
                # LLM-call run_id → start_ns, for accurate per-call latency.
                self._llm_starts: dict[UUID, int] = {}
                # span_id → agent_name, so LLM events can attribute captured
                # config to the owning agent.
                self._span_to_agent: dict[str, str] = {}
                # --- LangGraph fan-out/fan-in detection ---
                # Currently-open tracked nodes: run_id → {span_id, parent_run_id, agent_name}.
                # Two nodes open at once under the same graph parent are concurrent branches.
                self._open_tracked: dict[UUID, dict] = {}
                # span_id of the most recent *sequential* node to finish — the fan-out parent.
                self._last_closed_span: Optional[str] = None
                # The parallel group currently being assembled (members + which have closed).
                self._active_group: Optional[dict] = None
                # run_ids that were branch members (so their close doesn't advance the
                # sequential predecessor used as the next fan-out parent).
                self._group_member_runs: set[UUID] = set()

            # ---- DAG helpers --------------------------------------------------
            def _is_parallel_construct(self, name: str, tags) -> bool:
                """A chain that fans out its direct children to run in parallel.

                Recognised:
                  - LangChain RunnableParallel (name contains 'RunnableParallel')
                  - LangChain RunnableMap (legacy parallel)
                  - explicit user 'parallel' tag
                  - LangGraph parallel-step pattern (graph:step:* with parallel sibling)
                """
                if name and ("RunnableParallel" in name or "RunnableMap" in name):
                    return True
                if tags:
                    for t in tags:
                        ts = str(t).lower()
                        if "parallel" in ts:
                            return True
                return False

            def _nearest_tracked_ancestor(self, run_id: Optional[UUID]) -> Optional[str]:
                """Walk up parent_run_id chain to find the nearest tracked span_id.

                Skips scaffolding chains that weren't recorded as turns.
                """
                cur = run_id
                while cur is not None:
                    if cur in self._run_to_span:
                        return self._run_to_span[cur]
                    cur = self._parent_of_run.get(cur)
                return None

            def _branch_id_if_under_parallel(self, name: str, parent_run_id: Optional[UUID]) -> Optional[str]:
                """If any ancestor (up to the nearest tracked one) is a parallel
                construct, return the chain name as the branch_id. Else None.
                """
                cur = parent_run_id
                while cur is not None:
                    if cur in self._parallel_parents:
                        return name  # chain name doubles as the branch identifier
                    # Don't cross a tracked non-parallel ancestor: that would
                    # mean we're deeper than the parallel fan-out itself.
                    if cur in self._run_to_span and cur not in self._parallel_parents:
                        return None
                    cur = self._parent_of_run.get(cur)
                return None

            # -- chain lifecycle ----------------------------------------------------
            def on_chain_start(
                self, serialized, inputs, *,
                run_id, parent_run_id=None, tags=None, metadata=None, **kwargs,
            ):
                if not get_config().enabled:
                    return

                name = _chain_name(serialized, kwargs.get("name"), tags)

                # Record parent relationship for every chain (even scaffolding)
                # so we can walk up the tree later.
                self._parent_of_run[run_id] = parent_run_id

                # First top-level chain in this process → start a session.
                if self._root_run_id is None and get_active_session() is None:
                    cfg = get_config()
                    session = RunSession(
                        task_text=_extract_task(inputs),
                        task_type=cfg.task_type,
                        prompt_version=cfg.prompt_version,
                        db_name=cfg.db_name,
                    )
                    # This adapter attributes LLM tokens per-call via on_llm_end,
                    # so the session must not also claim them in on_turn_end.
                    session.per_call_attribution = True
                    set_active_session(session)
                    self._root_run_id = run_id

                # Mark parallel constructs (even if they're scaffolding-named).
                if self._is_parallel_construct(name, tags):
                    self._parallel_parents.add(run_id)

                # Skip scaffolding — only meaningful chains become turns.
                # Branches reference the GRANDPARENT as parent_step_id (the
                # _nearest_tracked_ancestor walks up past skipped scaffolding).
                if _is_scaffolding(name):
                    return

                session = get_active_session()
                if session is None:
                    return

                # Compute DAG fields BEFORE opening the turn so they go in
                # the TurnData at creation time.
                parent_step_id = self._nearest_tracked_ancestor(parent_run_id)
                if parent_step_id is None:
                    # LangGraph nodes report the graph root as their callback parent
                    # (not the upstream node), so the run tree gives no edge. Chain
                    # to the previous sequential node instead — this also makes the
                    # fan-out parent come out right (siblings share it).
                    parent_step_id = self._last_closed_span
                branch_id      = self._branch_id_if_under_parallel(name, parent_run_id)

                turn = session.on_turn_start(
                    name,
                    parent_step_id=parent_step_id,
                    branch_id=branch_id,
                )
                # Map this run_id to the turn's span_id so descendants can
                # reference it as parent_step_id.
                self._run_to_span[run_id] = turn.span_id
                self._open_turns[run_id]  = name
                self._span_to_agent[turn.span_id] = name

                # --- LangGraph fan-out / fan-in detection (from execution) ---
                # Other tracked nodes already open under the same graph parent are
                # concurrent branches with this one.
                siblings = [rid for rid, info in self._open_tracked.items()
                            if info["parent_run_id"] == parent_run_id]

                # JOIN: this node converges a just-completed parallel group.
                if (self._active_group is not None and not siblings
                        and self._active_group["parent_run_id"] == parent_run_id):
                    members = self._active_group["members"]
                    if self._active_group["closed"] >= {m["run_id"] for m in members}:
                        for m in members:
                            session.set_dag_fields(m["span_id"], join_step_id=turn.span_id)
                        self._active_group = None

                # FAN-OUT: concurrent siblings → a parallel group. Set branch_id on
                # each member, and a shared parent_step_id (the node before the fan).
                if siblings:
                    if self._active_group is None:
                        self._active_group = {
                            "parent_step_id": self._last_closed_span,
                            "parent_run_id":  parent_run_id,
                            "members":        [],
                            "closed":         set(),
                        }
                        for rid in siblings:
                            sib = self._open_tracked[rid]
                            self._active_group["members"].append(
                                {"run_id": rid, "span_id": sib["span_id"]})
                            self._group_member_runs.add(rid)
                            session.set_dag_fields(
                                sib["span_id"],
                                parent_step_id=self._active_group["parent_step_id"],
                                branch_id=sib["agent_name"])
                    self._active_group["members"].append(
                        {"run_id": run_id, "span_id": turn.span_id})
                    self._group_member_runs.add(run_id)
                    session.set_dag_fields(
                        turn.span_id,
                        parent_step_id=self._active_group["parent_step_id"],
                        branch_id=name)

                self._open_tracked[run_id] = {
                    "span_id":       turn.span_id,
                    "parent_run_id": parent_run_id,
                    "agent_name":    name,
                }

            def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
                session = get_active_session()
                agent_name = self._open_turns.pop(run_id, None)
                now = time.time_ns()
                span_id = self._run_to_span.get(run_id)

                # Fan-in bookkeeping: note this branch as closed, and only advance
                # the sequential predecessor for non-branch (sequential) nodes.
                self._open_tracked.pop(run_id, None)
                if self._active_group is not None and run_id in {
                        m["run_id"] for m in self._active_group["members"]}:
                    self._active_group["closed"].add(run_id)
                if run_id not in self._group_member_runs and span_id is not None:
                    self._last_closed_span = span_id

                if session is not None and agent_name:
                    cur = session._current_turn
                    if cur is not None and cur.agent_name == agent_name:
                        status_value = _status_from_outputs(outputs)
                        session.on_turn_end(agent_name, status_value=status_value)
                    elif span_id:
                        # Out-of-order (parallel) close. The turn may have been
                        # closed prematurely when a sibling started — stamp THIS
                        # node's real end time on its span so the duration reflects
                        # actual wall-clock (concurrent branches overlap correctly).
                        sv = _status_from_outputs(outputs)
                        for t in session.turns:
                            if t.span_id == span_id:
                                t.end_ns = now
                                if sv:
                                    t.status_value = sv
                                break

                # Root chain finished → finalise and persist the session.
                if run_id == self._root_run_id and session is not None:
                    if not session.termination_reason:
                        session.on_termination("chain_complete")
                    session.finalise()
                    try:
                        write_session(session)
                    except Exception as e:
                        print(f"[observability] warning: could not write session: {e}")
                    clear_active_session()
                    self._root_run_id = None
                    self._open_turns.clear()
                    self._open_tracked.clear()
                    self._active_group = None
                    self._group_member_runs.clear()
                    self._last_closed_span = None

            def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
                session = get_active_session()
                agent_name = self._open_turns.pop(run_id, None)
                if session is not None and agent_name:
                    if session._current_turn is not None and session._current_turn.agent_name == agent_name:
                        session._current_turn.status = "ERROR"
                    session.on_turn_end(agent_name)

                if run_id == self._root_run_id and session is not None:
                    session.on_termination(f"chain_error: {type(error).__name__}")
                    session.finalise()
                    try:
                        write_session(session)
                    except Exception as e:
                        print(f"[observability] warning: could not write session: {e}")
                    clear_active_session()
                    self._root_run_id = None
                    self._open_turns.clear()

            # -- tool lifecycle ----------------------------------------------------
            def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
                session = get_active_session()
                if session is None:
                    return
                tool_name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
                session.on_tool_request(str(run_id), tool_name)

            def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
                session = get_active_session()
                if session is None:
                    return
                session.on_tool_result(str(run_id), is_error=False)

            def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
                session = get_active_session()
                if session is None:
                    return
                session.on_tool_result(str(run_id), is_error=True)

            # -- agent lifecycle ---------------------------------------------------
            def on_agent_finish(self, finish, *, run_id, parent_run_id=None, **kwargs):
                session = get_active_session()
                if session is None:
                    return
                # Try to extract a STATUS value from the finish output (if the
                # user is using a STATUS protocol). Otherwise use the log line.
                for v in (getattr(finish, "return_values", None) or {}).values():
                    if isinstance(v, str):
                        sv = parse_status_value(v)
                        if sv:
                            session.on_termination(sv)
                            return
                session.on_termination("agent_finish")

            # -- LLM lifecycle (token attribution) ---------------------------------
            def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
                # Record the parent so on_llm_end can walk to the owning agent span,
                # and stamp the start time for accurate per-call latency.
                self._parent_of_run[run_id] = parent_run_id
                self._llm_starts[run_id] = time.time_ns()
                # Capture this agent's config (model/params/tools/prompt hash) for
                # change tracking & versioning.
                session = get_active_session()
                if session is not None:
                    agent = self._span_to_agent.get(self._nearest_tracked_ancestor(parent_run_id))
                    if agent:
                        cfg = _extract_config(serialized, messages, kwargs)
                        if cfg:
                            session.record_agent_config(agent, cfg)

            def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
                self._parent_of_run[run_id] = parent_run_id
                self._llm_starts[run_id] = time.time_ns()

            def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
                session = get_active_session()
                if session is None:
                    return
                inp, out, model = _extract_usage(response)
                if inp == 0 and out == 0:
                    return
                # Attribute to the nearest tracked ancestor — the agent node that
                # issued this call — skipping scaffolding/parser chains in between.
                span_id  = self._nearest_tracked_ancestor(parent_run_id)
                start_ns = self._llm_starts.pop(run_id, 0)
                session.add_llm_usage(span_id, inp, out, model,
                                      start_ns=start_ns, end_ns=time.time_ns())

            def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
                # An LLM call failed → the caller will retry (manual loop or
                # with_retry). Count it against the current agent step. Using only
                # on_llm_error covers both retry styles without double-counting,
                # since a failed call always emits it.
                self._llm_starts.pop(run_id, None)
                session = get_active_session()
                if session is not None:
                    session.on_llm_retry()

        return ObservabilityCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _chain_name(serialized: Optional[dict], explicit_name: Optional[str], tags) -> str:
    """Resolve the most meaningful name for a chain.

    Priority: explicit kwargs `name` > tags that look like a name > serialized
    `name` > last segment of serialized `id`.
    """
    if explicit_name:
        return str(explicit_name)
    if tags:
        # LangGraph passes the node name in tags as e.g. ["graph:step:NodeName"]
        for tag in tags:
            if isinstance(tag, str) and ":" in tag:
                parts = tag.split(":")
                if len(parts) >= 3 and parts[0] in ("graph", "seq"):
                    return parts[-1]
    if serialized:
        name = serialized.get("name")
        if name:
            return str(name)
        id_parts = serialized.get("id") or []
        if isinstance(id_parts, list) and id_parts:
            return str(id_parts[-1])
    return ""


def _extract_config(serialized: Any, messages: Any, kwargs: dict) -> dict:
    """Capture the agent's config from a chat-model start event:
    model, key params, bound tool names, and a hash of the system message.
    Returns {} if nothing useful is available.
    """
    import hashlib

    ip = (kwargs.get("invocation_params") or {}) if isinstance(kwargs, dict) else {}
    model = ip.get("model") or ip.get("model_name") or ""
    if not model and isinstance(serialized, dict):
        model = serialized.get("name") or ""

    params = {}
    for k in ("temperature", "max_tokens", "top_p"):
        if ip.get(k) is not None:
            params[k] = ip.get(k)

    # Tool names (LangChain binds tools as a list of {type, function:{name}} or {name}).
    tools = []
    for t in (ip.get("tools") or []):
        if isinstance(t, dict):
            name = (t.get("function") or {}).get("name") if t.get("function") else t.get("name")
            if name:
                tools.append(name)
    tools = sorted(set(tools))

    # System-message hash (v1: system message only — approximate but stable).
    sys_text = ""
    try:
        first = messages[0] if messages and isinstance(messages[0], list) else messages
        for m in (first or []):
            mtype = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else None)
            if mtype == "system":
                sys_text = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
                break
    except Exception:
        pass
    prompt_hash = hashlib.sha1(str(sys_text).encode()).hexdigest()[:12] if sys_text else ""

    cfg = {}
    if model:       cfg["model"] = str(model)
    if params:      cfg["params"] = params
    if tools:       cfg["tools"] = tools
    if prompt_hash:
        cfg["prompt_hash"] = prompt_hash
        cfg["prompt_text"] = str(sys_text)   # transient — stripped & deduped into prompts table on write
    return cfg


def _extract_usage(response: Any) -> tuple[int, int, str]:
    """Pull (input_tokens, output_tokens, model) out of a LangChain LLMResult.

    Tries the standardised per-message usage_metadata first (works across
    providers), then falls back to the provider-specific llm_output blob.
    """
    inp = out = 0
    model = ""
    try:
        for gen_list in (getattr(response, "generations", None) or []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                um  = getattr(msg, "usage_metadata", None) if msg is not None else None
                if um:
                    inp += int(um.get("input_tokens", 0) or 0)
                    out += int(um.get("output_tokens", 0) or 0)
                if msg is not None and not model:
                    rm = getattr(msg, "response_metadata", None) or {}
                    model = rm.get("model_name") or rm.get("model") or ""
    except Exception:
        pass

    if inp == 0 and out == 0:
        lo = getattr(response, "llm_output", None) or {}
        if isinstance(lo, dict):
            tu = lo.get("token_usage") or lo.get("usage") or {}
            if isinstance(tu, dict):
                inp = int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0)
                out = int(tu.get("completion_tokens") or tu.get("output_tokens") or 0)
            if not model:
                model = lo.get("model_name") or lo.get("model") or ""

    return inp, out, str(model or "")


def _extract_task(inputs: Any) -> str:
    """Best-effort extraction of the user task text from chain inputs."""
    if isinstance(inputs, dict):
        for key in ("input", "question", "query", "task", "prompt", "text"):
            if key in inputs and isinstance(inputs[key], str):
                return inputs[key]
        if "messages" in inputs and isinstance(inputs["messages"], list) and inputs["messages"]:
            last = inputs["messages"][-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
            return str(last)[:500]
        # Fallback: stringify
        return str(inputs)[:500]
    if isinstance(inputs, str):
        return inputs
    return str(inputs)[:500]


def _status_from_outputs(outputs: Any) -> str:
    """Try to find a STATUS line in a chain's outputs (for users following the STATUS protocol)."""
    if not isinstance(outputs, dict):
        return ""
    for v in outputs.values():
        if isinstance(v, str):
            sv = parse_status_value(v)
            if sv:
                return sv
    return ""
