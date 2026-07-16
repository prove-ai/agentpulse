#!/usr/bin/env python3
"""Live capture smoke test for LangGraph.

Proves the instrument() callback path captures a real LangGraph run end to end —
spans, agent sequence, handoffs, DAG, and per-agent config/prompt — without
spending API credits (uses a fake chat model). To test against a real provider,
swap FakeListChatModel for ChatAnthropic(model="claude-sonnet-4-6") etc.

    python scripts/smoke_langgraph.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 1) Turn on capture BEFORE building/running the graph.
from sdk.instrument import instrument            # noqa: E402
instrument(task_type="smoke", prompt_version=1, db_name="lg_smoke")

from typing import TypedDict                      # noqa: E402
from langchain_core.messages import SystemMessage, HumanMessage  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langgraph.graph import StateGraph, START, END  # noqa: E402


class State(TypedDict):
    topic: str
    notes: str
    analysis: str
    draft: str


# Each node = one agent, each with its own system prompt (so prompt capture fires).
RESEARCH_PROMPT = "Research the topic concisely and list key facts."
ANALYST_PROMPT = "Turn the research notes into a clear signal."
WRITER_PROMPT = "Write a short summary from the analysis."

researcher_llm = FakeListChatModel(responses=["facts: A, B, C"])
analyst_llm = FakeListChatModel(responses=["signal: bullish"])
writer_llm = FakeListChatModel(responses=["summary text"])


def researcher(state: State):
    out = researcher_llm.invoke([SystemMessage(content=RESEARCH_PROMPT),
                                 HumanMessage(content=state["topic"])])
    return {"notes": out.content}


def analyst(state: State):
    out = analyst_llm.invoke([SystemMessage(content=ANALYST_PROMPT),
                              HumanMessage(content=state["notes"])])
    return {"analysis": out.content}


def writer(state: State):
    out = writer_llm.invoke([SystemMessage(content=WRITER_PROMPT),
                             HumanMessage(content=state["analysis"])])
    return {"draft": out.content}


g = StateGraph(State)
g.add_node("researcher", researcher)
g.add_node("analyst", analyst)
g.add_node("writer", writer)
g.add_edge(START, "researcher")
g.add_edge("researcher", "analyst")
g.add_edge("analyst", "writer")
g.add_edge("writer", END)
app = g.compile()

result = app.invoke({"topic": "AAPL"})
print("graph ran, draft =", result.get("draft"))
