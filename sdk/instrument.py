"""instrument() — the single function users add to their agent runner.

Usage (add to the top of your main.py, nothing else changes):

    from observability.sdk import instrument
    instrument(task_type="csv-analysis", prompt_version=1)

What it does:
  1. Stores the task_type and prompt_version in a global config.
  2. Patches AutoGen's SelectorGroupChat.run_stream so every run is
     captured automatically — no other code changes needed.

After runs are captured, use the report CLI to see metrics:

    cd observability
    python report.py --last     # last run + drift
    python report.py --all      # all runs table
    python report.py --drift    # only drifted runs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the observability root is on the path regardless of cwd
_OBS_ROOT = Path(__file__).parent.parent
if str(_OBS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OBS_ROOT))


def instrument(
    task_type: str = "unspecified",
    prompt_version: int = 1,
    db_name: str | None = None,
) -> None:
    """Enable observability for any AutoGen / LangChain agent run in this process.

    Call once, near the top of your agent runner, before the agents execute.
    The task_type, prompt_version, and db_name are attached to every run
    captured in this process invocation.

    Args:
        task_type:      Short label grouping similar tasks, e.g. "csv-analysis".
                        Used to compare like-with-like in drift reports.
        prompt_version: Which prompt config version this run uses.
                        Version 1 is the baseline; bump when you change prompts.
        db_name:        Logical name of the SQLite file under observability/db/.
                        None or "runs" → default db/runs.db. Anything else, e.g.
                        "financial", → db/financial.db. Use this to keep
                        different agent systems' runs in separate files.
    """
    from sdk.session import set_config
    from sdk.patches.autogen   import patch_autogen
    from sdk.patches.anthropic import patch_anthropic
    from sdk.patches.openai    import patch_openai
    from sdk.patches.langchain import patch_langchain

    set_config(task_type=task_type, prompt_version=prompt_version, db_name=db_name)
    # LLM-SDK patches first — they capture tokens regardless of which framework
    # the user is running. Each patch is a no-op if the SDK isn't installed.
    patch_anthropic()
    patch_openai()
    # Framework adapters — each is a no-op if its framework isn't installed.
    patch_autogen()
    patch_langchain()
