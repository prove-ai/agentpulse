"""Where AgentPulse keeps its data on disk.

Everything lives under one home directory (default ~/.agentpulse):

    ~/.agentpulse/
    ├── db/        SQLite run databases (one per project)
    ├── config/    optional user overrides, e.g. drift_rules.yaml
    └── .env       optional KEY=VALUE env file (e.g. ANTHROPIC_API_KEY)

Override the location with the AGENTPULSE_HOME environment variable — e.g.
point it at a repo checkout to keep using databases stored there.
"""

from __future__ import annotations

import os
from pathlib import Path


def home_dir() -> Path:
    """AgentPulse home — $AGENTPULSE_HOME, or ~/.agentpulse."""
    return Path(os.environ.get("AGENTPULSE_HOME") or Path.home() / ".agentpulse").expanduser()


def db_dir() -> Path:
    return home_dir() / "db"


def config_dir() -> Path:
    return home_dir() / "config"


def env_file() -> Path:
    return home_dir() / ".env"


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from the home .env into os.environ.

    Existing environment values always win — never overrides what's set.
    Needed because MCP hosts (Claude Code/Desktop) and `agentpulse dashboard`
    may be launched without a shell environment.
    """
    path = path or env_file()
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass  # no .env is fine — values may already be in the environment
