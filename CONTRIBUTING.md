# Contributing to AgentPulse

Thanks for your interest! Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/prove-ai/agentpulse.git
cd agentpulse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the dashboard against the bundled sample database (`db/demo.db`):

```bash
python reporter/dashboard.py     # http://localhost:5001
```

## Running tests

```bash
python -m pytest tests/
```

One test is a **snapshot guard**: `tests/test_series_snapshot.py` pins the exact
chart values produced by the metric engine for the sample database. If you
intentionally change a metric (or add one), regenerate the fixture and include
it in your PR:

```bash
python tests/test_series_snapshot.py --update
```

If it fails and you did *not* intend to change metric values, that's the test
doing its job — investigate before updating.

## Where things live

| Area | Path |
|---|---|
| Instrumentation (SDK patches) | `sdk/` |
| SQLite storage | `storage/` |
| Metric + drift engine | `analysis/` |
| Flask dashboard | `reporter/` |
| Drift CLI (`today-drift`) | `cli.py` |
| MCP server for Claude | `agentpulse_mcp.py` |
| Detection thresholds/rules | `config/drift_rules.yaml` |

## Guidelines

- The drift engine is shared by the dashboard, the CLI, and the MCP server —
  change behaviour in `analysis/` (or `reporter/dashboard.py` helpers), not in
  a single surface, so all three stay in agreement.
- Detection thresholds belong in `config/drift_rules.yaml`, not in code.
- Keep the SDK dependency-light: `openai`/`anthropic` must remain optional
  (patched only if present in the observed system's environment).
- Please run the test suite before opening a PR.

## Reporting bugs

Open a GitHub issue with: what you ran, what you expected, what happened, and
(if it's a detection question) whether the demo database or an exported run
reproduces it.
