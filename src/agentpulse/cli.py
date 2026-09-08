"""The `agentpulse` command.

    agentpulse run python main.py     capture a program's runs (zero code changes)
    agentpulse report [--last|--all]  terminal metrics report
    agentpulse dashboard [--port N]   web dashboard (http://localhost:5001)
    agentpulse drift [...]            drift findings in the terminal
    agentpulse mcp                    MCP server for Claude Code / Desktop
"""

from __future__ import annotations

import sys

_USAGE = __doc__


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    if argv[0] in ("-V", "--version"):
        from agentpulse import __version__
        print(f"agentpulse {__version__}")
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "run":
        from agentpulse.runner import run
        return run(rest)

    if cmd == "report":
        from agentpulse.report import main as report_main
        sys.argv = ["agentpulse report"] + rest
        report_main()
        return 0

    if cmd == "dashboard":
        from agentpulse.reporter.dashboard import main as dashboard_main
        return dashboard_main(rest)

    if cmd == "drift":
        from agentpulse.drift_cli import main as drift_main
        drift_main(rest)
        return 0

    if cmd == "mcp":
        from agentpulse.mcp_server import mcp
        mcp.run()
        return 0

    print(f"agentpulse: unknown command '{cmd}'\n\n{_USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
