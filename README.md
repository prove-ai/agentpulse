# AgentPulse

[![tests](https://github.com/prove-ai/agentpulse/actions/workflows/tests.yml/badge.svg)](https://github.com/prove-ai/agentpulse/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Traces tell you *what happened*. They don't tell you *where to start investigating*.

AgentPulse is an **open-source reference implementation** of drift investigation for multi-agent systems. It turns raw agent traces — tokens, cost, latency, tool calls, handoffs, DAG structure — into run-level, agent-level, handoff, route, and drift views, and then does the part we think is genuinely unsolved: when an outcome degrades, it traces the drift **upstream through the handoff graph to the component where it originated** and presents the whole causal path.

**Who it's for.** Teams building in-house observability for multi-agent systems, and anyone exploring how agent failures *should* be investigated. Take the ideas, the schema, or the whole thing.

**What it's not.** A production observability platform. If you need managed tracing at scale today, use Langfuse, LangSmith, or their peers. AgentPulse is a working exploration of the layer *above* traces: the investigation.

![Drift Investigation](docs/screenshots/drift-investigation.png)

*The core view: writer drifted (prompt + model change at run 18), critic's success dropped as a consequence — the fix belongs in writer, not critic.*

---

## 60-second demo

```bash
git clone https://github.com/prove-ai/agentpulse.git
cd agentpulse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reporter/dashboard.py
```

Open <http://localhost:5001>. The bundled sample database (`db/demo.db` — a 4-agent content pipeline, 120 runs, 4 prompt versions, one real drift) is pre-loaded, so the Drift Investigation view above is the first thing you can reproduce. Your own `db/*.db` files are gitignored — only the demo sample is part of the repo.

---

## Architecture

One drift engine, three surfaces — the dashboard, the CLI, and the MCP server call the *same* engine functions, so they always agree on what "drift" means.

```mermaid
flowchart LR
    A["your multi-agent app<br/>+ instrument()"] --> B["sdk/<br/>patches openai · anthropic"]
    B --> C[("SQLite<br/>db/&lt;project&gt;.db")]
    C --> D["analysis/<br/>metrics → anomalies → drift → causal chains"]
    D --> E["Flask dashboard"]
    D --> F["today-drift CLI"]
    D --> G["MCP server → Claude"]
```

| Surface | What it's for |
|---|---|
| **Dashboard** (Flask) | Run explorer, timelines, DAGs, trends, and the Drift Investigation view |
| **CLI** (`today-drift`) | Drift findings as readable cards in your terminal — a daily standup check |
| **MCP server** | The same findings inside Claude Code / Claude Desktop, so a coding agent can run the investigation |

Works with OpenAI and Anthropic SDKs. Framework-agnostic — tested with AutoGen, LangChain, and plain `asyncio.gather` orchestrations.

---

## Ideas you can reuse

Even if you never run AgentPulse, these are the design decisions we'd argue for in any in-house build:

1. **Detect agent, handoff, and route drift separately.** An agent getting slower, a handoff payload shrinking, and the execution path changing are different failure classes with different signals — collapsing them into one "anomaly score" hides where to look. (`analysis/drift_detect.py`, `analysis/version_drift.py`)

2. **Severity needs corroboration, not just magnitude.** A Tier-0 (outcome) breach only escalates to critical when co-timed strong supporting signals *and* a plausible change event line up. A lone moving metric stays a low-confidence candidate. (`analysis/drift_detect.py`, thresholds in `config/drift_rules.yaml`)

3. **Trace the symptom to its upstream origination.** Walk the handoff graph upstream from the breached outcome; stop at the component whose *input* is stable but whose *output* drifted. That's where the fix belongs — the downstream agent that "failed" often didn't change at all. (`analysis/drift_chains.py`)

4. **A change can only explain a drift if it could have caused it.** Config/prompt/model changes are attributed only when they happened at-or-before the drift start, on the same or an upstream component. (`_nearby_change` in `analysis/drift_detect.py`)

5. **Give the investigation to a coding agent, not just a dashboard.** The MCP server exposes findings as structured, root-cause-led cards, so Claude can triage drift, compare releases, and propose next checks conversationally. (`agentpulse_mcp.py`, `.claude/skills/`)

6. **Pin your metric engine with a snapshot test.** Every chart value for the sample data is pinned by a fixture; a refactor that silently changes a metric fails CI loudly. (`tests/test_series_snapshot.py`)

---

## Instrumenting your system

Add two lines at the **top of your entrypoint** (before any agent imports):

```python
import sys; sys.path.insert(0, '/path/to/agentpulse')
from sdk import instrument
instrument(task_type='my-system', prompt_version=1, db_name='my-system')
```

That's it. Run your system as usual. Every LLM call, agent turn, tool call, and handoff is captured into `db/my-system.db`. The dashboard auto-discovers any `*.db` under `db/` and shows a project picker in the sidebar; pass a different `db_name` per system to monitor several at once.

### What gets captured

| Layer | What |
|---|---|
| Per-call | start/end timestamps, input/output tokens, model, latency |
| Per-agent turn | aggregated tokens, duration, tool calls, status, parent agent |
| Per-run | total cost, wall-clock, termination reason, prompt version |
| DAG | parent → child edges (when your orchestrator exposes them), parallel branches, join waits |

No API keys are needed to capture data or browse the dashboard — it only reads SQLite. An `ANTHROPIC_API_KEY` is needed only for the optional AI "suggest next checks" feature (see [Configuration](#configuration)).

---

## The dashboard

### Drift Investigation (`/drift2`)
The core view, shown at the top of this README: findings ranked by severity, the causal path, what changed, why it matters, potentially related changes, and suggested next checks.

### Run explorer (`/`)
Every captured run with status, route, and cost; click through to the per-run detail page with an execution timeline (gantt), the interactive agent-chain DAG, anomalies vs the baseline, and parallel-group efficiency.

![Run explorer](docs/screenshots/run-history.png)

### Metrics Explorer (`/explore`)
Chart any metric for any agent, handoff, or the whole system across runs — same baseline bands and version markers as Drift Investigation, plus custom metrics and thresholds.

![Metrics Explorer](docs/screenshots/metrics-explorer.png)

There's also a trend view (`/trends`) with agent health cards and a handoff health leaderboard, and an event timeline (`/timeline`) of prompt/model/tool changes.

---

## CLI — `today-drift`

The drift findings as terminal cards. No server, no Claude:

```bash
python cli.py                          # all projects, active drifts
python cli.py --project demo           # one project
python cli.py --range 7d               # narrower look-back window (default 30d)
python cli.py --min-severity drift     # hide low-signal watches
python cli.py --next demo:chain0       # next investigation checks for one finding
python cli.py --compare --project demo # version comparison (baseline vs newest)
python cli.py --compare --project demo --all   # step through every version pair
```

Sample output:

```
AgentPulse — 1 drift finding  · as of 2026-07-16 17:09 · range 30d · demo

● writer  ·  Agent behaviour  ·  demo
    Severity: Drift    Confidence: High
    Path:    writer → … → critic
    Why:     A change in writer propagated downstream to critic, whose outcome
             (success) worsened. The fix belongs in writer, not critic.
    Trigger: writer prompt changed near run 110
    Metrics changed (3):
        • writer latency_s +51.2%
        • writer cost_usd +1446.9%
        • critic success -9.3 pp
    active 23d, since Jun 23  ·  id demo:chain0
```

Findings carry an `id` (like `demo:chain0`) — pass it to `--next` to get the recommended follow-up checks for that specific finding.

There's also `report.py`, a per-run metrics report for a single project (`python report.py --all --db demo`).

---

## MCP server — let Claude run the investigation

`agentpulse_mcp.py` exposes the drift engine to Claude Code and Claude Desktop as three tools:

| Tool | What it returns |
|---|---|
| `get_todays_finding` | The drift findings active right now, as root-cause-led investigation cards (which component drifted, why, how long it's been active) |
| `get_version_comparison` | Baseline-vs-newest (or any pair, or every consecutive pair) version comparison, leading with the change that broke an outcome |
| `get_next_check_steps` | Recommended next investigation steps for a finding — AI-generated when an `ANTHROPIC_API_KEY` is configured, deterministic checks otherwise |

### Claude Code

The repo ships with a project-scoped [`.mcp.json`](.mcp.json), so if you followed the demo setup (venv at `.venv/`), just open Claude Code inside the repo and approve the server when prompted:

```bash
cd agentpulse
claude
```

Then ask things like *"what drifted today?"*, *"compare versions of the demo project"*, or *"what should I check next for demo:chain0?"*.

To register it from another directory instead:

```bash
claude mcp add agentpulse -- /path/to/agentpulse/.venv/bin/python /path/to/agentpulse/agentpulse_mcp.py
```

The repo also bundles three Claude Code **skills** under [`.claude/skills/`](.claude/skills) that build on these tools: `drift-triage` (daily standup-style triage), `drift-root-cause-report` (a written root-cause report), and `release-regression-check` (did the last release break anything?).

### Claude Desktop

Add to `claude_desktop_config.json` (absolute paths required — Desktop spawns servers without a working directory):

```json
{
  "mcpServers": {
    "agentpulse": {
      "command": "/path/to/agentpulse/.venv/bin/python",
      "args": ["/path/to/agentpulse/agentpulse_mcp.py"]
    }
  }
}
```

---

## Configuration

Copy [`.env.example`](.env.example) to `.env` and set `ANTHROPIC_API_KEY` to enable the AI "suggest next checks" feature (dashboard button, CLI `--next`, MCP `get_next_check_steps`). Everything else works without it — the MCP tool falls back to deterministic checks.

Drift detection thresholds and handoff rules live in [`config/drift_rules.yaml`](config/drift_rules.yaml) — edit and reload the page; no restart needed.

---

## What we're trying to learn

This is an experiment in how agent failures should be investigated, published to be argued with. If you build or operate multi-agent systems, we'd genuinely like to know:

- **What telemetry do you actually collect** for multi-agent systems — and what does AgentPulse's schema miss?
- **Does the agent / handoff / route drift split match how you triage failures?** Or do you cut the problem differently?
- **Where is the drift detector wrong?** Thresholds live in [`config/drift_rules.yaml`](config/drift_rules.yaml) — if it over- or under-fires on your data, that's exactly the feedback we want.
- **Should agent observability stay a dashboard**, or become structured context for a coding agent that performs the investigation? The MCP server is our bet on the second answer — tell us where it falls short.

Open a [GitHub issue](https://github.com/prove-ai/agentpulse/issues) with your take, your telemetry schema, or a war story about an agent failure that was hard to localize. Design disagreements are as welcome as bug reports.

---

## Project layout

```
agentpulse/
├── sdk/                Patches for OpenAI/Anthropic + the instrument() entry point
├── storage/            SQLite store (multi-DB aware via ContextVar)
├── analysis/           Metric engine: raw → derived → anomalies → trends → drift → DAG
├── reporter/           Flask dashboard + Jinja templates
├── cli.py              today-drift terminal CLI
├── agentpulse_mcp.py   MCP server (3 tools over the same engine)
├── report.py           Per-run metrics report CLI
├── config/             Drift rules + default prompt manifests
├── scripts/            Demo generators & import helpers
├── tests/              pytest suite (incl. a metric snapshot guard)
├── .claude/skills/     Claude Code skills built on the MCP tools
└── db/                 demo.db sample (your own project DBs land here, gitignored)
```

## Development

```bash
python -m pytest tests/          # run the test suite
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the snapshot-test workflow and guidelines.

## Requirements

- Python 3.10+
- Flask 3.x (dashboard), `mcp` (MCP server) — both in `requirements.txt`
- The multi-agent system you observe needs `openai` and/or `anthropic` installed in **its** environment; AgentPulse patches whichever it finds (neither is a hard dependency of AgentPulse itself).

## License

[MIT](LICENSE)
