"""instrument() — enable observability for the current process.

Most users never call this directly: `agentpulse run python main.py` calls it
before your code executes. If you can't change your launch command, add:

    import agentpulse
    agentpulse.instrument()

near the top of your entry point instead.

After runs are captured:

    agentpulse report --last     # last run + drift
    agentpulse report --all      # all runs table
    agentpulse dashboard         # web dashboard
"""

from __future__ import annotations

import os


def instrument(
    task_type: str | None = None,
    prompt_version: int | None = None,
    db_name: str | None = None,
) -> None:
    """Enable observability for any AutoGen / LangChain agent run in this process.

    Call once, before the agents execute (the `agentpulse run` launcher does
    this for you). All arguments are optional; each falls back to an
    environment variable, then to a default:

        task_type      $AGENTPULSE_TASK_TYPE       default "unspecified"
                       Short label grouping similar tasks, e.g. "csv-analysis".
                       Used to compare like-with-like in drift reports.
        prompt_version $AGENTPULSE_PROMPT_VERSION  default 1
                       Bump when you change your prompts; version 1 is baseline.
        db_name        $AGENTPULSE_DB              default "runs"
                       Logical name of the SQLite file under the AgentPulse
                       home's db/ directory. Use to keep different agent
                       systems' runs in separate files.
    """
    from agentpulse.sdk.session import set_config
    from agentpulse.sdk.patches.autogen   import patch_autogen
    from agentpulse.sdk.patches.anthropic import patch_anthropic
    from agentpulse.sdk.patches.openai    import patch_openai
    from agentpulse.sdk.patches.langchain import patch_langchain

    if task_type is None:
        task_type = os.environ.get("AGENTPULSE_TASK_TYPE") or "unspecified"
    if prompt_version is None:
        try:
            prompt_version = int(os.environ.get("AGENTPULSE_PROMPT_VERSION", "1"))
        except ValueError:
            prompt_version = 1
    if db_name is None:
        db_name = os.environ.get("AGENTPULSE_DB") or None

    set_config(task_type=task_type, prompt_version=prompt_version, db_name=db_name)
    # LLM-SDK patches first — they capture tokens regardless of which framework
    # the user is running. Each patch is a no-op if the SDK isn't installed.
    patch_anthropic()
    patch_openai()
    # Framework adapters — each is a no-op if its framework isn't installed.
    patch_autogen()
    patch_langchain()
