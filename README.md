# AgentPulse

[![tests](https://github.com/prove-ai/agentpulse/actions/workflows/tests.yml/badge.svg)](https://github.com/prove-ai/agentpulse/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Lightweight observability and **drift investigation** for multi-agent systems.

Drop two lines into your code and AgentPulse captures every agent turn — tokens, cost, latency, tool calls, handoffs, and DAG structure — then answers the question that actually matters when behavior degrades: **which component drifted, when, and why.**

One drift engine, three surfaces:

| Surface | What it's for |
|---|---|
| **Dashboard** (Flask) | Explore runs, timelines, DAGs, trends, and the Drift Investigation view |
| **CLI** (`today-drift`) | Drift findings as readable cards in your terminal — great for a daily standup check |
| **MCP server** | The same findings inside Claude Code / Claude Desktop, so you can investigate conversationally |

All three call the *same* engine functions, so they always agree on what "drift" means.

Works with OpenAI and Anthropic SDKs. Framework-agnostic — tested with AutoGen, LangChain, and plain `asyncio.gather` orchestrations.

---

## Quick start

```bash
git clone https://github.com/prove-ai/agentpulse.git
cd agentpulse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reporter/dashboard.py
```

Open <http://localhost:5001>.

The repo ships with a **sample database** (`db/demo.db` — a 4-agent content pipeline with 120 runs across 4 prompt versions, including a real drift) so you can explore the dashboard immediately. When you want your own data, instrument your multi-agent system (next section). Your own `db/*.db` files are gitignored — only the demo sample is part of the repo.

---

## Instrumenting your system

Add two lines at the **top of your entrypoint** (before any agent imports):

```python
import sys; sys.path.insert(0, '/path/to/agentpulse')
from sdk import instrument
instrument(task_type='my-system', prompt_version=1, db_name='my-system')
```

That's it. Run your system as usual. Every LLM call, agent turn, tool call, and handoff is captured into `db/my-system.db`.

The dashboard auto-discovers any `*.db` file under `db/` and shows a database picker in the sidebar so you can switch between systems.

### What gets captured

| Layer | What |
|---|---|
| Per-call | start/end timestamps, input/output tokens, model, latency |
| Per-agent turn | aggregated tokens, duration, tool calls, status, parent agent |
| Per-run | total cost, wall-clock, termination reason, prompt version |
| DAG | parent → child edges (when your orchestrator exposes them), parallel branches, join waits |

No API keys are needed to capture data or browse the dashboard — it only reads SQLite. An `ANTHROPIC_API_KEY` is needed only for the optional AI "suggest next checks" feature (see [Configuration](#configuration)).

### Multiple systems

Pass a different `db_name` from each system — `instrument(task_type='customer-support', db_name='support')` in one, `db_name='finance'` in another. Each writes to its own `db/<name>.db`; every surface (dashboard sidebar, CLI `--project`, MCP `project` argument) can target any of them.

---

## CLI — `today-drift`

A standalone terminal view of the drift findings. No server, no Claude — it prints human-readable cards straight to your shell:

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

There's also `report.py`, a per-run metrics report for a single project:

```bash
python report.py --all --db demo       # table of all runs
python report.py --last --db demo      # last run: metrics + anomalies vs baseline
```

---

## MCP server — use AgentPulse from Claude

`agentpulse_mcp.py` exposes the drift engine to Claude Code and Claude Desktop as three tools:

| Tool | What it returns |
|---|---|
| `get_todays_finding` | The drift findings active right now, as root-cause-led investigation cards (which component drifted, why, how long it's been active) |
| `get_version_comparison` | Baseline-vs-newest (or any pair, or every consecutive pair) version comparison, leading with the change that broke an outcome |
| `get_next_check_steps` | Recommended next investigation steps for a finding — AI-generated when an `ANTHROPIC_API_KEY` is configured, deterministic checks otherwise |

### Claude Code

The repo ships with a project-scoped [`.mcp.json`](.mcp.json), so if you followed the Quick start (venv at `.venv/`), just open Claude Code inside the repo and approve the server when prompted:

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

## Dashboard tour

### Drift Investigation (`/drift2`)
The core view. When an outcome metric breaches, AgentPulse traces the drift **upstream** through the handoff graph to the component where it originated, and presents the whole causal path: *"writer drifted (prompt change near run 110) → critic's success dropped as a result — fix writer, not critic."* Filter by time range, compare versions, and generate AI next-check suggestions per finding.

### Run history (`/`)
Every captured run with status, route, and a sortable run number. Click a row for the detail page.

### Run detail (`/run/<id>`)
- **Execution timeline** — gantt chart of every agent turn
- **Agent chain DAG** — interactive graph with hover tooltips showing payload size, receiver outcome, and downstream success
- **Anomalies vs Average** — this run vs the average of prior runs of the same task type, overall and per agent
- **Version drift** — when you bump `prompt_version`, the new cohort is automatically compared against the old one
- **Parallel groups** — efficiency, bottleneck branch, join wait time

### Trend view (`/trends`)
Agent health cards (efficiency + drift verdict), the handoff health leaderboard (volume, success rate, payload size, join wait, downstream cost), and cost-per-run over time. Filter by window, task type, and prompt version.

### Explorer (`/explore`)
Chart any metric for any agent, handoff, or the whole system across runs, with multi-select version overlays.

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

---

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
