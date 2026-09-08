"""`agentpulse run` — launch a Python program with observability enabled.

    agentpulse run python main.py --my-flag
    agentpulse run main.py
    agentpulse run python -m mypackage.app

instrument() is called before the target's code executes, so the OpenAI and
Anthropic SDKs (and AutoGen / LangChain, when present) are patched before the
program imports them. The target then runs in this same process as __main__,
with sys.argv exactly as if it had been launched directly.

This is the same pattern as `opentelemetry-instrument` and `ddtrace-run`:
no code changes to the instrumented program.
"""

from __future__ import annotations

import os
import runpy
import sys


def run(argv: list[str]) -> int:
    """Execute the command line in ``argv`` with instrumentation active."""
    args = list(argv)

    # Allow (and ignore) a leading interpreter token: `agentpulse run python x.py`
    # and `agentpulse run x.py` behave identically.
    if args and (args[0] in ("python", "python3") or args[0] == sys.executable
                 or os.path.basename(args[0]).startswith("python")):
        args = args[1:]

    if not args:
        print("usage: agentpulse run [python] <script.py> [args...]\n"
              "       agentpulse run [python] -m <module> [args...]", file=sys.stderr)
        return 2

    from agentpulse.sdk.instrument import instrument
    instrument()

    if args[0] == "-m":
        if len(args) < 2:
            print("agentpulse run: -m requires a module name", file=sys.stderr)
            return 2
        module, prog_args = args[1], args[2:]
        sys.argv = [module] + prog_args
        # Mirror `python -m`: the current directory leads sys.path.
        sys.path.insert(0, os.getcwd())
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0

    script, prog_args = args[0], args[1:]
    if not os.path.exists(script):
        print(f"agentpulse run: no such file: {script}", file=sys.stderr)
        return 2
    sys.argv = [script] + prog_args
    # Mirror `python script.py`: the script's directory leads sys.path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
    return 0
