# AgentPulse

Observability and drift detection for multi-agent systems. AgentPulse captures every LLM call, agent turn, tool call, and handoff from your agent system, stores them in local SQLite, and shows you which component drifted and why.

Supports AutoGen and LangChain/LangGraph. Runs entirely on your machine.

## Install

```bash
pip install proveai-agentpulse
```

## Use

Run your app with this command:

```bash
agentpulse run python main.py
```

Then open the dashboard:

```bash
agentpulse dashboard
```

## Learn more

- [Documentation and screenshots](https://prove-ai.github.io/agentpulse/)
- [Source, full README, and sample data](https://github.com/prove-ai/agentpulse)
- [Feedback and issues](https://github.com/prove-ai/agentpulse/issues)
