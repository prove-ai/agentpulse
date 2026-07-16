"""Session — in-memory state for one active run.

Turn lifecycle (two-phase):
  Phase 1 — OPEN:  on_turn_start(agent) called when first event from an agent
                   arrives (ThoughtEvent or ToolCallRequestEvent). Records the
                   real start time before any LLM call.
  Phase 2 — CLOSE: on_turn_end(agent, tokens, ...) called when the agent's
                   final TextMessage or ToolCallSummaryMessage arrives. Tokens
                   come from the Anthropic SDK patch (pending_api_calls) or
                   fall back to AutoGen's models_usage field.

This ensures:
  - Tokens are attributed to the correct agent (not the next one)
  - Latency includes the actual LLM call time
  - STATUS values are recorded per turn for loop detection
"""

from __future__ import annotations

import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

_STATUS_RE = re.compile(r"STATUS:\s*([A-Za-z_]+)(?:\s*:\s*([A-Za-z]+))?", re.IGNORECASE)


def parse_status_value(content: str) -> str:
    """Return the last STATUS value from a message, e.g. 'COMPLETE' or 'NEEDS_INFO: DataAnalyst'."""
    if not isinstance(content, str):
        return ""
    matches = _STATUS_RE.findall(content)
    if not matches:
        return ""
    state, role = matches[-1]
    return f"{state.upper()}: {role}" if role else state.upper()


# ---------------------------------------------------------------------------
# Per-tool-call record
# ---------------------------------------------------------------------------
@dataclass
class ToolRecord:
    tool_name:  str
    success:    bool
    start_ns:   int
    end_ns:     int

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000


# ---------------------------------------------------------------------------
# Per-agent-turn record
# ---------------------------------------------------------------------------
@dataclass
class TurnData:
    agent_name:     str
    turn_index:     int
    # Each turn gets its own stable span_id at creation time so framework
    # adapters can reference it as the parent_step_id of subsequent turns
    # (before the run has been written to SQLite).
    span_id:        str  = field(default_factory=lambda: str(uuid.uuid4()))
    start_ns:       int  = field(default_factory=time.time_ns)
    end_ns:         int  = 0
    input_tokens:   int  = 0
    output_tokens:  int  = 0
    model:          str  = ""
    tools:          list[ToolRecord] = field(default_factory=list)
    status:         str  = "OK"         # span status: OK | ERROR
    status_value:   str  = ""           # STATUS line content, e.g. "COMPLETE", "NEEDS_INFO: Writer"
    retry_count:    int  = 0            # LLM-call retries that happened within this turn
    # DAG fields — set by framework adapters that know structural relationships
    # (LangChain via parent_run_id, LangGraph via graph edges, etc.).
    # All None for sequential systems where no DAG information is available.
    parent_step_id: Optional[str] = None  # span_id of parent agent step
    branch_id:      Optional[str] = None  # this branch's identifier within a fan-out group
    join_step_id:   Optional[str] = None  # span_id of the join step (None if no join)

    @property
    def duration_ms(self) -> float:
        end = self.end_ns or time.time_ns()
        return (end - self.start_ns) / 1_000_000

    @property
    def tool_call_count(self) -> int:
        return len(self.tools)

    def close(self, end_ns: Optional[int] = None) -> None:
        self.end_ns = end_ns or time.time_ns()


# ---------------------------------------------------------------------------
# Per-run session
# ---------------------------------------------------------------------------
@dataclass
class RunSession:
    task_text:          str
    task_type:          str  = "unspecified"
    prompt_version:     int  = 1
    # Stamped from InstrumentationConfig at construction so the storage layer
    # can route the run to the correct DB file. None → default DB.
    db_name:            Optional[str] = None
    run_id:             str  = field(default_factory=lambda: str(uuid.uuid4()))
    start_ns:           int  = field(default_factory=time.time_ns)
    end_ns:             int  = 0
    turns:              list[TurnData] = field(default_factory=list)
    termination_reason: str  = ""
    status:             str  = "OK"

    # Anthropic SDK patch deposits (start_ns, end_ns, inp, out, model) here.
    # on_turn_end claims them and attaches to the current turn.
    _pending_api_calls: list = field(default_factory=list, repr=False)
    # When True, a framework adapter attributes each LLM call to its exact span
    # via add_llm_usage() (using the call's run tree). on_turn_end/finalise then
    # skip the single-current-turn pending-claim path, which can't separate
    # concurrently-open turns (parallel branches). Set by the LangChain adapter.
    per_call_attribution: bool = False
    # Pending tool requests: call_id -> (tool_name, start_ns)
    _pending_tools:     dict = field(default_factory=dict, repr=False)
    # The turn currently being built (open phase)
    _current_turn:      Optional[TurnData] = field(default=None, repr=False)
    # agent_name -> captured config {model, params, tools, prompt_hash}, for
    # change tracking & versioning. Last write per agent wins.
    agent_configs:      dict = field(default_factory=dict, repr=False)

    # -----------------------------------------------------------------------
    # Phase 1 — open a turn
    # -----------------------------------------------------------------------
    def on_turn_start(
        self, agent_name: str,
        *,
        parent_step_id: Optional[str] = None,
        branch_id:      Optional[str] = None,
        join_step_id:   Optional[str] = None,
        span_id:        Optional[str] = None,
    ) -> TurnData:
        """Called on first event from an agent. Optionally carry DAG fields
        from a framework adapter that knows the structural relationships.

        Returns the newly opened TurnData so the caller can inspect/modify it
        (e.g. to read its auto-generated span_id for later reference).
        """
        now = time.time_ns()

        # Different agent starting — close the previous turn (no tokens yet)
        if self._current_turn is not None:
            if self._current_turn.agent_name != agent_name:
                self._current_turn.close(now)
                self.turns.append(self._current_turn)
                self._current_turn = None
            else:
                return self._current_turn  # same agent, turn already open

        self._current_turn = TurnData(
            agent_name=agent_name,
            turn_index=len(self.turns),
            start_ns=now,
            parent_step_id=parent_step_id,
            branch_id=branch_id,
            join_step_id=join_step_id,
            **({"span_id": span_id} if span_id else {}),
        )
        return self._current_turn

    # -----------------------------------------------------------------------
    # Phase 2 — close a turn with its tokens
    # -----------------------------------------------------------------------
    def on_turn_end(
        self,
        agent_name:     str,
        input_tokens:   int = 0,
        output_tokens:  int = 0,
        model:          str = "",
        status_value:   str = "",
        # Optional DAG fields, used when the framework adapter knows the
        # parent step but the turn was created lazily here (no on_turn_start).
        parent_step_id: Optional[str] = None,
        branch_id:      Optional[str] = None,
        join_step_id:   Optional[str] = None,
    ) -> None:
        """Called when agent's TextMessage/ToolCallSummaryMessage arrives."""
        now = time.time_ns()

        # If turn wasn't opened (no ThoughtEvent before TextMessage), create it now
        if self._current_turn is None or self._current_turn.agent_name != agent_name:
            if self._current_turn is not None:
                # Close wrong-agent turn
                self._current_turn.close(now)
                self.turns.append(self._current_turn)
            self._current_turn = TurnData(
                agent_name=agent_name,
                turn_index=len(self.turns),
                start_ns=now,
                parent_step_id=parent_step_id,
                branch_id=branch_id,
                join_step_id=join_step_id,
            )  # auto-generated span_id is fine here

        # Token attribution. Under per-call attribution the framework adapter has
        # already routed each LLM call to its own span via add_llm_usage(), so the
        # tokens are present on the turn — claiming pending calls here would both
        # double-count and mis-assign across concurrently-open turns.
        if not self.per_call_attribution:
            # Prefer Anthropic SDK tokens (more accurate — captures all API calls in turn)
            if self._pending_api_calls:
                calls = self._pending_api_calls[:]
                self._pending_api_calls.clear()
                self._current_turn.input_tokens  = sum(c[2] for c in calls)
                self._current_turn.output_tokens = sum(c[3] for c in calls)
                self._current_turn.model = calls[-1][4] if calls else model
                # Use actual LLM call start time for more accurate latency
                api_start = min(c[0] for c in calls)
                if api_start < self._current_turn.start_ns:
                    self._current_turn.start_ns = api_start
            else:
                # Fallback: tokens from AutoGen's models_usage
                self._current_turn.input_tokens  = input_tokens
                self._current_turn.output_tokens = output_tokens
                self._current_turn.model = model or self._current_turn.model

        self._current_turn.status_value = status_value
        self._current_turn.close(now)
        self.turns.append(self._current_turn)
        self._current_turn = None

    # -----------------------------------------------------------------------
    # Per-call token attribution (concurrency-safe)
    # -----------------------------------------------------------------------
    def add_llm_usage(
        self,
        span_id:        Optional[str],
        input_tokens:   int,
        output_tokens:  int,
        model:          str = "",
        start_ns:       int = 0,
        end_ns:         int = 0,
    ) -> None:
        """Attribute one LLM call's tokens directly to the span that issued it.

        Used by framework adapters that can identify the owning agent from the
        call's run tree (e.g. LangChain's parent_run_id chain). Unlike the
        single-current-turn claim in on_turn_end, this stays correct when several
        turns are open at once (parallel branches): each call lands on its own
        span_id, even after that turn was displaced from _current_turn.

        Tokens accumulate (+=) so retries / multi-step calls within one turn sum.
        """
        target: Optional[TurnData] = None
        if span_id is not None:
            if self._current_turn is not None and self._current_turn.span_id == span_id:
                target = self._current_turn
            else:
                for t in self.turns:
                    if t.span_id == span_id:
                        target = t
                        break
        if target is None:
            target = self._current_turn  # best-effort: attribute to the open turn
        if target is None:
            return

        target.input_tokens  += int(input_tokens or 0)
        target.output_tokens += int(output_tokens or 0)
        if model:
            target.model = model
        if start_ns and start_ns < target.start_ns:
            target.start_ns = start_ns

    def set_dag_fields(
        self,
        span_id:        str,
        *,
        parent_step_id: Optional[str] = None,
        branch_id:      Optional[str] = None,
        join_step_id:   Optional[str] = None,
    ) -> None:
        """Set DAG fields on an existing span after creation.

        Needed when structure is only known later than the turn's start — e.g.
        a fan-out isn't recognised until a *second* branch opens, and the join
        isn't known until the branches converge. Only non-None fields are set.
        """
        target: Optional[TurnData] = None
        if self._current_turn is not None and self._current_turn.span_id == span_id:
            target = self._current_turn
        else:
            for t in self.turns:
                if t.span_id == span_id:
                    target = t
                    break
        if target is None:
            return
        if parent_step_id is not None:
            target.parent_step_id = parent_step_id
        if branch_id is not None:
            target.branch_id = branch_id
        if join_step_id is not None:
            target.join_step_id = join_step_id

    def record_agent_config(self, agent_name: str, config: dict) -> None:
        """Store the captured config for an agent (model/params/tools/prompt hash).

        Called by framework adapters at LLM-call time. Last non-empty write wins,
        so the config reflects what the agent actually ran with this run.
        """
        if agent_name and config:
            self.agent_configs[agent_name] = config

    def on_llm_retry(self) -> None:
        """Record one LLM-call retry against the currently-open turn.

        Called by framework adapters when an LLM call errors and is retried
        (distinct from re-invoking the whole agent). No-op if no turn is open.
        """
        if self._current_turn is not None:
            self._current_turn.retry_count += 1

    # -----------------------------------------------------------------------
    # Tool events
    # -----------------------------------------------------------------------
    def on_tool_request(self, call_id: str, tool_name: str) -> None:
        self._pending_tools[call_id] = (tool_name, time.time_ns())

    def on_tool_result(self, call_id: str, is_error: bool) -> None:
        if call_id not in self._pending_tools:
            return
        tool_name, start_ns = self._pending_tools.pop(call_id)
        record = ToolRecord(
            tool_name=tool_name,
            success=not is_error,
            start_ns=start_ns,
            end_ns=time.time_ns(),
        )
        if self._current_turn is not None:
            self._current_turn.tools.append(record)

    # -----------------------------------------------------------------------
    # Termination
    # -----------------------------------------------------------------------
    def on_termination(self, reason: str) -> None:
        self.termination_reason = reason

    # -----------------------------------------------------------------------
    # Finalise — close any open turn and mark the session done
    # -----------------------------------------------------------------------
    def finalise(self) -> None:
        now = time.time_ns()
        if self._current_turn is not None:
            # Claim any remaining pending API calls (skip under per-call
            # attribution, where tokens are already on their spans).
            if self._pending_api_calls and not self.per_call_attribution:
                calls = self._pending_api_calls[:]
                self._pending_api_calls.clear()
                self._current_turn.input_tokens  = sum(c[2] for c in calls)
                self._current_turn.output_tokens = sum(c[3] for c in calls)
                if calls:
                    self._current_turn.model = calls[-1][4]
            self._current_turn.close(now)
            self.turns.append(self._current_turn)
            self._current_turn = None
        self.end_ns = now

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------
    @property
    def agent_sequence(self) -> list[str]:
        return [t.agent_name for t in self.turns]

    @property
    def model(self) -> str:
        for t in self.turns:
            if t.model:
                return t.model
        return ""

    @property
    def total_duration_ms(self) -> float:
        end = self.end_ns or time.time_ns()
        return (end - self.start_ns) / 1_000_000


# ---------------------------------------------------------------------------
# Global active session reference — used by the Anthropic SDK patch
# ---------------------------------------------------------------------------
_active_session: Optional[RunSession] = None


def set_active_session(session: RunSession) -> None:
    global _active_session
    _active_session = session


def get_active_session() -> Optional[RunSession]:
    return _active_session


def clear_active_session() -> None:
    global _active_session
    _active_session = None


# ---------------------------------------------------------------------------
# Per-asyncio-task active agent. ContextVar so parallel asyncio.gather tasks
# each have their own current agent without trampling each other.
# Set by framework adapters (or by user hooks like server/_obs.py) at agent
# entry, cleared at agent exit. Read by the OpenAI/Anthropic patches so each
# captured LLM call is tagged with the agent that issued it.
# ---------------------------------------------------------------------------
_active_agent: ContextVar[Optional[str]] = ContextVar("obs_active_agent", default=None)


def set_active_agent(name: Optional[str]) -> None:
    _active_agent.set(name)


def get_active_agent() -> Optional[str]:
    return _active_agent.get()


# ---------------------------------------------------------------------------
# Global instrumentation config
# ---------------------------------------------------------------------------
@dataclass
class InstrumentationConfig:
    task_type:      str  = "unspecified"
    prompt_version: int  = 1
    db_name:        Optional[str] = None   # None → default DB (runs.db)
    enabled:        bool = False


_config = InstrumentationConfig()


def get_config() -> InstrumentationConfig:
    return _config


def set_config(task_type: str, prompt_version: int,
               db_name: Optional[str] = None) -> None:
    _config.task_type      = task_type
    _config.prompt_version = prompt_version
    _config.db_name        = db_name
    _config.enabled        = True
