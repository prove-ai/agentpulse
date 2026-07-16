#!/usr/bin/env python
"""today-drift — a standalone terminal view of AgentPulse drift findings.

Same engine as the MCP server and the dashboard (it calls the very functions in
agentpulse_mcp), but prints human-readable cards straight to your shell — no
Claude involved. Examples:

    today-drift                          # all projects, active drifts
    today-drift --project demo           # one project
    today-drift --range 7d               # narrower window
    today-drift --min-severity drift     # hide low-signal watches
    today-drift --next demo:chain0       # next checks for one finding
    today-drift --compare --project demo # version comparison
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from agentpulse_mcp import (
    get_todays_finding, get_version_comparison, get_next_check_steps,
)

# --- colour (auto-off when piped) -------------------------------------------
_TTY = sys.stdout.isatty()


def _c(s, code):
    return f"\033[{code}m{s}\033[0m" if _TTY else str(s)


def bold(s):   return _c(s, "1")
def dim(s):    return _c(s, "2")
def red(s):    return _c(s, "91")
def yellow(s): return _c(s, "93")
def cyan(s):   return _c(s, "96")

_SEV_STYLE = {           # dot + label colour per severity
    "high":      (red, "HIGH"),
    "drift":     (red, "DRIFT"),
    "candidate": (yellow, "CANDIDATE"),
    "watch":     (dim, "WATCH"),
}
_ARROW = {"up": "↑", "down": "↓"}
_MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fmt_date(iso):
    if not iso:
        return "?"
    try:
        d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return f"{_MON[d.month]} {d.day}"
    except (ValueError, AttributeError):
        return iso[:10]


def _print_findings(res):
    n = res["drift_findings"]
    scope = f"{len(res['projects_scanned'])} projects" if len(res["projects_scanned"]) > 1 \
        else res["projects_scanned"][0] if res["projects_scanned"] else "no projects"
    head = f"AgentPulse — {n} drift finding{'s' if n != 1 else ''}"
    print(f"\n{bold(head)}  {dim(f'· as of {res['as_of'][:16].replace('T', ' ')} · range {res['range']} · {scope}')}\n")
    if not n and not res.get("secondary_signals"):
        print(dim("  Nothing drifting in this window. 🎉\n"))
        return
    # Primary: which component drifted + why (the headline, root-cause-led).
    for f in res["findings"]:
        colour, _ = _SEV_STYLE.get(f["severity"], (dim, f["severity"].upper()))
        sev = f.get("severity_label") or f["severity"].title()
        conf = f.get("confidence")
        print(f"{colour('●')} {bold(f['root'])}  {dim('·')}  {f['kind']}  {dim('·')}  {cyan(f['project'])}")
        line = f"    {bold('Severity:')} {colour(bold(sev))}"
        if conf:
            line += f"    {bold('Confidence:')} {conf}"
        print(line)
        if f.get("path"):
            print(f"    {dim('Path:')}    {f['path']}")
        if f.get("why"):
            print(f"    {bold('Why:')}     {f['why']}")
        if f.get("potentially_related_event"):
            print(f"    {yellow('Trigger:')} {f['potentially_related_event']}")
        supp = f.get("supporting_metrics") or []
        if supp:
            print(f"    {bold('Metrics changed')} ({len(supp)}):")
            for mline in supp:
                print(f"        {dim('•')} {mline}")
        age = f"active {f['days_active']}d, since {_fmt_date(f['active_since'])}" \
            if f.get("days_active") is not None else "active"
        print(f"    {dim(age + '  ·  id ' + f['finding_id'])}")
        print()
    # Secondary: weaker signals, one line each — context, not headline.
    sec = res.get("secondary_signals") or []
    if sec:
        print(dim(f"  also moving ({len(sec)} lower-severity signal{'s' if len(sec) != 1 else ''}):"))
        for s in sec:
            colour, label = _SEV_STYLE.get(s["severity"], (dim, s["severity"].upper()))
            moved = f" — {s['moved']}" if s.get("moved") else ""
            print(f"    {colour('○')} {s['root']}  {dim('·')}  {s['kind']}  {dim(f'({label.lower()}){moved}')}  {dim(s['finding_id'])}")
        print()


def _print_next(res):
    if not res.get("ok"):
        print(dim(f"\n  {res.get('reason', 'No finding.')}\n"))
        return
    print(f"\n{bold('Next checks')} for {bold(res['finding_id'])} "
          f"{dim(f'({res['severity']} · root {res['root']})')}\n")
    ai = res.get("ai_next_checks")
    if ai:
        print(bold("  AI suggestions (Opus):"))
        for line in ai.splitlines():
            if line.strip():
                print(f"    {line.strip()}")
        print()
    else:
        print(dim(f"  AI suggestions unavailable — {res.get('ai_error', 'n/a')}\n"))
    print(bold("  Deterministic checks:"))
    for step in res.get("deterministic_next_checks", []):
        print(f"    {dim('-')} {step}")
    print()


def _print_pair_body(res, indent="  "):
    """Render one version-pair comparison (used for both the single-pair view and
    each step of the across-all view)."""
    cmp = res.get("comparing", {})
    b, t = cmp.get("base", {}), cmp.get("target", {})
    runs = dim(f"  · {b.get('runs')} vs {t.get('runs')} runs") if b.get("runs") is not None else ""
    print(f"{indent}{bold('v' + str(b.get('version')))} {dim('(' + str(b.get('label')) + ')')}"
          f"  →  {bold('v' + str(t.get('version')))} {dim('(' + str(t.get('label')) + ')')}{runs}")
    if not res.get("ok"):
        print(f"{indent}  {dim(res.get('reason', ''))}\n")
        return
    crit = res.get("critical_drift") or []
    if not crit:
        print(f"{indent}  {dim(res.get('note', 'No critical drift between these versions.'))}")
    for c in crit:
        print(f"{indent}  {red('●')} {red(bold('CRITICAL'))}  {bold(c['root'])}  {dim('· ' + c['kind'])}")
        if c.get("why"):
            print(f"{indent}      {bold('why')}     {c['why']}")
        if c.get("related_event"):
            print(f"{indent}      {yellow('trigger')} {c['related_event']}")
        for m in c.get("metrics_changed", []):
            print(f"{indent}      {dim('changed')} {m}")
        for o in c.get("outcome_impact", []):
            print(f"{indent}      {red('outcome')} {o}")
    if res.get("other_version_changes"):
        print(f"{indent}  {dim('other (intended) changes:')}")
        for o in res["other_version_changes"]:
            print(f"{indent}      {dim('○ ' + o)}")
    if res.get("low_confidence_signals"):
        print(f"{indent}  {dim('low-confidence signals:')}")
        for o in res["low_confidence_signals"]:
            print(f"{indent}      {dim('· ' + o)}")
    print()


def _print_compare(res):
    if not res.get("ok"):
        print(dim(f"\n  {res.get('reason', 'Cannot compare.')}\n"))
        return
    proj, tt = res.get("project", "?"), res.get("task_type")
    allv = res.get("all_versions") or []
    if res.get("mode") == "across_all":
        print(f"\n{bold('Version evolution')} — {cyan(proj)} {dim('· ' + str(tt))}")
        print(dim("  " + " → ".join(f"v{v['version']}" for v in allv)
                  + f"   ({len(allv)} versions)") + "\n")
        for step in res.get("steps", []):
            _print_pair_body(step)
        return
    # single pair
    print(f"\n{bold('Version comparison')} — {cyan(proj)} "
          f"{dim(f'· {tt} · {res.get('confidence')} confidence')}")
    if len(allv) > 2:
        print(dim(f"  ({len(allv)} versions total: " + ", ".join(f"v{v['version']}" for v in allv) + ")"))
    print()
    _print_pair_body(res)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="today-drift", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", help="one project (db name); default = all projects")
    ap.add_argument("--range", default="30d", help="look-back window, e.g. 30d, 7d, 1d (default 30d)")
    ap.add_argument("--min-severity", default="watch",
                    choices=["high", "drift", "candidate", "watch"],
                    help="lowest severity to show (default watch = all)")
    ap.add_argument("--next", dest="next_id", metavar="PROJECT:ID",
                    help="show next-check steps for a finding id from the list")
    ap.add_argument("--compare", action="store_true",
                    help="version comparison for --project instead of the drift list")
    ap.add_argument("--base", type=int, help="base version number (default: baseline)")
    ap.add_argument("--target", type=int, help="target version number (default: newest)")
    ap.add_argument("--all", dest="across_all", action="store_true",
                    help="compare every consecutive version (v1→v2→v3→…)")
    ap.add_argument("--include-intended", action="store_true",
                    help="also show intended version changes that didn't break an outcome")
    ap.add_argument("--include-low-confidence", action="store_true",
                    help="also show signals that moved with low consistency")
    args = ap.parse_args(argv)

    if args.next_id:
        _print_next(get_next_check_steps(finding_id=args.next_id, range=args.range))
    elif args.compare or args.across_all:
        _print_compare(get_version_comparison(
            project=args.project, base_version=args.base, target_version=args.target,
            across_all=args.across_all, include_intended=args.include_intended,
            include_low_confidence=args.include_low_confidence))
    else:
        _print_findings(get_todays_finding(project=args.project, range=args.range,
                                           min_severity=args.min_severity))


if __name__ == "__main__":
    main()
