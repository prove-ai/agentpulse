"""AgentPulse MCP server — exposes the drift-investigation engine to Claude Code.

Thin wrappers over the SAME functions the dashboard uses, so every surface agrees
on what "drift" means:
  - get_todays_finding     -> reporter.dashboard.investigations_for()  (investigate())
  - get_version_comparison -> analysis.version_drift.compare_versions()
  - get_next_check_steps   -> analysis.diagnose.suggest_next_checks()   (Claude Opus)

Projects are separate SQLite DBs under db/; we switch with set_active_db_path().
Findings are recomputed per call (no persistence yet); "active_since" is derived
from the drift-start run's timestamp so an ongoing drift reads as ongoing, not new.
"""

from __future__ import annotations

import os
import sys
import datetime as dt
from collections import Counter
from dataclasses import asdict, is_dataclass

# Run from anywhere (Claude Desktop has no working dir) — make the repo importable.
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)


def _load_env(path: str) -> None:
    """Load KEY=VALUE lines from a .env next to this script into os.environ, so the
    Opus call in get_next_check_steps finds ANTHROPIC_API_KEY regardless of how the
    server was launched (Claude Code CLI or Desktop, both spawn without a shell env).
    Existing environment values win — we never override what's already set."""
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass  # no .env is fine — the key may already be in the environment


_load_env(os.path.join(_REPO, ".env"))

from mcp.server.fastmcp import FastMCP

from storage.sqlite_store import (
    list_available_dbs, resolve_db_path, set_active_db_path, get_baseline_version,
)
from analysis.layer1_raw import list_runs
from analysis.version_drift import compare_versions
from analysis.diagnose import suggest_next_checks
from reporter.dashboard import investigations_for

mcp = FastMCP("agentpulse")

_SEV_RANK = {"high": 0, "drift": 1, "candidate": 2, "watch": 3}
# The top severity is presented as "Critical" (the user-facing word for tier-0 impact).
_SEV_LABEL = {"high": "Critical", "drift": "Drift", "candidate": "Candidate", "watch": "Watch"}


def _use_project(project: str | None) -> str:
    """Point storage at a project's DB. None -> the default DB. Returns the name used."""
    name = project or "runs"
    set_active_db_path(resolve_db_path(name))
    return name


def _active_since(drift_start, runs):
    """Map a drift-start run index to (iso_timestamp, days_active). Handles ongoing
    drifts: the finding is 'active since' the run where the breach began."""
    if drift_start is None or not runs or not (0 <= drift_start < len(runs)):
        return None, None
    ts = runs[drift_start].get("timestamp")
    if not ts:
        return None, None
    try:
        started = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = dt.datetime.now(started.tzinfo) if started.tzinfo else dt.datetime.now()
        return ts, (now - started).days
    except (ValueError, AttributeError):
        return ts, None


def _why_for_finding(x):
    """A short 'why we concluded this is the root' for a standalone (non-chain)
    finding, when there's no causal-chain narrative to borrow."""
    ev = x.get("related_event")
    lead = ("Its input from upstream is stable but its own behaviour/output drifted"
            + (f", co-timed with: {ev}" if ev else " — this is the origination point")) \
        if x.get("kind") == "finding" else ""
    return x.get("card", {}).get("why") or lead


def _symptom_of(x):
    """The downstream agent that a chain's outcome damage surfaces on (or None)."""
    sub = x.get("subtitle", "") or ""
    if "downstream impact on" in sub:
        return sub.split("downstream impact on ")[-1].strip()
    return None


def _causal_explanation(x):
    """Explain WHY this root is the cause. For a chain, spell out explicitly that the
    downstream symptom's outcome fell because of the upstream root — not itself — so
    the reader knows where the fix belongs."""
    why = _why_for_finding(x)
    root, symptom = x["title"], _symptom_of(x)
    if x.get("kind") == "chain" and symptom and symptom.lower() != root.lower():
        why = (why + f" In other words, {symptom} is where the failure surfaces, but {symptom} "
               f"itself did not change — its input from upstream {root} did. The fix belongs in "
               f"{root}, not {symptom}.").strip()
    return why


def _finding_from_investigation(x, runs, project):
    """Reshape one ranked investigation into a root-cause-led finding card that
    mirrors the /drift2 detail panel: WHICH component drifted + WHY we concluded it,
    with the moved metrics as supporting detail (not the headline)."""
    card = x.get("card") or {}
    since_ts, days = _active_since(x.get("drift_start"), runs)
    return {
        # Stable-ish handle for get_next_check_steps (id is deterministic per query).
        # ":" separator (not "|") so the id is shell-safe as a CLI argument.
        "finding_id": f"{project}:{x['id']}",
        "project": project,
        "headline": card.get("headline") or f"{x['title']} drift",
        "root": x["title"],                                  # the agent / handoff / route that drifted
        "kind": x.get("kind_label") or x["type_label"],      # Agent behaviour | Handoff drift | Path drift | Likely cause
        "path": card.get("path"),                            # causal path (chains only)
        "severity": x["severity"],
        "severity_label": _SEV_LABEL.get(x["severity"], x["severity"].title()),  # "Critical" for high
        "confidence": (x.get("confidence") or card.get("confidence") or "").title() or None,
        "downstream_symptom": _symptom_of(x),                # where the outcome damage surfaces
        "why": _causal_explanation(x),                       # explicit upstream→downstream reasoning
        "potentially_related_event": x.get("related_event") or card.get("related_event"),
        "active_since": since_ts,
        "days_active": days,
        "supporting_metrics": x.get("what_changed") or card.get("what_changed") or [],  # all metrics, as dashboard
        "note": "Investigation PATH, not confirmed causality.",
    }


def _secondary_view(f):
    """Compact one-liner for a lower-severity (candidate/watch) signal — enough to
    know it moved, without competing with the primary drift finding."""
    return {
        "finding_id": f["finding_id"],
        "root": f["root"],
        "kind": f["kind"],
        "severity": f["severity"],
        "moved": (f["supporting_metrics"] or [None])[0],     # the single triggering metric
        "days_active": f["days_active"],
    }


@mcp.tool()
def get_todays_finding(project: str | None = None, range: str = "30d",
                       min_severity: str = "watch") -> dict:
    """Return the drift findings that are ACTIVE right now, as investigation cards.

    A drift persists until fixed, so this reports ongoing drifts with how long each
    has been active (active_since) rather than treating every day as brand new.

    Args:
        project: one project (a db name from list_available_dbs), or None for ALL projects.
        range: look-back window for the analysis, e.g. "30d", "7d", "1d". Drift needs
            history, so keep this wide (default 30d) even though findings are "as of today".
        min_severity: floor severity to include — one of high, drift, candidate, watch.
    """
    floor = _SEV_RANK.get(min_severity, 3)
    projects = [project] if project else list_available_dbs()
    primary, secondary = [], []
    for proj in projects:
        _use_project(proj)
        try:
            invs, ctx = investigations_for(range_param=range)
        except Exception:  # a bad/empty project shouldn't kill the whole scan
            continue
        runs = ctx.get("runs", [])
        for x in invs:
            if _SEV_RANK.get(x["severity"], 3) > floor:
                continue
            f = _finding_from_investigation(x, runs, proj)
            # Real drift (high/drift) is a headline finding; candidate/watch is a
            # secondary signal — same split the dashboard shows (one root card on top,
            # weaker signals below), so the answer leads with WHAT drifted and WHY.
            if x["severity"] in ("high", "drift"):
                primary.append(f)
            else:
                secondary.append(_secondary_view(f))

    _key = lambda f: (_SEV_RANK.get(f["severity"], 4), -(f["days_active"] or 0))
    primary.sort(key=_key)
    secondary.sort(key=_key)
    return {
        "as_of": dt.datetime.now().isoformat(timespec="seconds"),
        "range": range,
        "projects_scanned": projects,
        "drift_findings": len(primary),
        "findings": primary,                 # root-cause-led: which component drifted + why
        "secondary_signals": secondary,      # candidate/watch, compact — context, not headline
    }


def _vd(v):
    """Normalize a VVerdict (dataclass) to a plain dict."""
    return asdict(v) if is_dataclass(v) else dict(v)


def _confident(v):
    """A 'critical' signal = it drifted AND most target runs moved the same way."""
    return v.get("drifted") and (v.get("consistency", 0) or 0) >= 0.70


def _sig_line(v):
    return f"{v['signal']} {v['delta_pct']:+g}% ({v['base']} -> {v['target']})"


def _critical_between(labels, base_v, target_v, task_type, ver_options):
    """Attribute critical drift by walking CONSECUTIVE version steps across
    [base..target]. A single lumped base->target investigation mis-roots when
    several versions changed in between (it traces to the most-upstream change);
    stepping version-by-version credits each drift to the release that introduced
    it. Returns [(introduced_in_version, investigation)], deduped by root."""
    steps = [n for n in ver_options if base_v <= n <= target_v]
    seen, out = set(), []
    for a, b in zip(steps, steps[1:]):
        invs, _ = investigations_for(range_param="3650d", task_type=task_type,
                                     base=a, compared=True, to=[b])
        for x in invs:
            if x["severity"] in ("high", "drift") and x["title"] not in seen:
                seen.add(x["title"])
                out.append((b, x))
    return out


def _compare_pair(all_runs, labels, vof, base_v, target_v, task_type, ver_options,
                  include_intended, include_low_confidence):
    """Compare TWO versions: cumulative metric deltas (base window vs target window)
    plus the outcome-breaking (critical) drift, attributed to the release that
    introduced it via consecutive-step walking."""
    from reporter.dashboard import _metrics_for_run

    base_runs = [r for r in all_runs if vof(r) == base_v and r.get("task_type") == task_type]
    tgt_runs = [r for r in all_runs if vof(r) == target_v and r.get("task_type") == task_type]
    comparing = {
        "base": {"version": base_v, "label": labels.get(base_v, f"v{base_v}"), "runs": len(base_runs)},
        "target": {"version": target_v, "label": labels.get(target_v, f"v{target_v}"), "runs": len(tgt_runs)},
    }
    if len(base_runs) < 3 or len(tgt_runs) < 3:
        return {"ok": False, "comparing": comparing,
                "reason": (f"Need >=3 runs per version (have {len(base_runs)} in v{base_v}, "
                           f"{len(tgt_runs)} in v{target_v}).")}

    report = compare_versions([_metrics_for_run(r) for r in base_runs],
                              [_metrics_for_run(r) for r in tgt_runs],
                              task_type, base_v, target_v)
    per_agent = {ag: [_vd(v) for v in vs] for ag, vs in report.per_agent.items()}
    overall = [_vd(v) for v in report.overall]
    outcome_breaches = [f"{v['signal']}: {v['base']} -> {v['target']} ({v['delta_pct']:+g}%)"
                        for v in overall if _confident(v)]

    crit = _critical_between(labels, base_v, target_v, task_type, ver_options)
    crit_roots = {x["title"] for _, x in crit}

    critical_drift = []
    for introduced_in, x in crit:
        root, card = x["title"], (x.get("card") or {})
        sigs = [v for v in per_agent.get(root, []) if _confident(v)]
        critical_drift.append({
            "root": root,
            "kind": x.get("kind_label") or x["type_label"],
            "introduced_in": labels.get(introduced_in, f"v{introduced_in}"),
            "why": card.get("why") or _why_for_finding(x),
            "related_event": x.get("related_event") or card.get("related_event"),
            "metrics_changed": [_sig_line(v) for v in sigs]
                               or (x.get("what_changed") or card.get("what_changed") or []),
            "outcome_impact": outcome_breaches,
        })

    out = {"ok": True, "comparing": comparing, "confidence": report.confidence,
           "critical_drift": critical_drift}
    if not critical_drift:
        out["note"] = "No outcome-breaking drift between these versions."
    if include_intended:            # confident changes that did NOT break an outcome
        out["other_version_changes"] = [f"{ag} {_sig_line(v)}" for ag, vs in per_agent.items()
                                        if ag not in crit_roots for v in vs if _confident(v)]
    if include_low_confidence:
        out["low_confidence_signals"] = [
            f"{ag} {v['signal']} {v['delta_pct']:+g}% (consistency {v['consistency']})"
            for ag, vs in per_agent.items()
            for v in vs if v.get("drifted") and (v.get("consistency", 0) or 0) < 0.70]
    return out


@mcp.tool()
def get_version_comparison(project: str | None = None, base_version: int | None = None,
                           target_version: int | None = None, task_type: str | None = None,
                           across_all: bool = False, include_intended: bool = False,
                           include_low_confidence: bool = False) -> dict:
    """Compare versions of a project and surface the CRITICAL (outcome-breaking) drift.

    Three modes:
      - default (no versions named): BASELINE vs NEWEST.
      - a specific pair: pass base_version + target_version (e.g. 3 and 4).
      - across_all=True: step through every consecutive version (v1->v2->v3->v4),
        showing what each release changed and which one introduced a critical drift.

    Each comparison states which two versions it used, leads with the ONE change
    that broke an outcome (matched to the drift investigation scoped to that pair),
    and hides intended changes / low-confidence signals unless you ask for them.

    Args:
        project: db name (None -> default DB). Comparison is per-project.
        base_version / target_version: version numbers from the version table
            (None -> baseline / newest). Ignored when across_all=True.
        task_type: scope to one task type (None -> the most common one).
        across_all: compare every consecutive version pair instead of one pair.
        include_intended: also list the other (intended) version changes.
        include_low_confidence: also list signals that moved with low consistency.
    """
    from reporter.dashboard import _ver_of_factory
    from storage.sqlite_store import get_versions

    proj = _use_project(project)
    all_runs = list_runs(1000)
    if not all_runs:
        return {"ok": False, "project": proj, "reason": "No runs in this project."}
    if not task_type:
        tt = Counter(r.get("task_type") for r in all_runs if r.get("task_type"))
        task_type = tt.most_common(1)[0][0] if tt else None

    # Versions come from the version table (temporal), same as the dashboard —
    # NOT the raw prompt_version field.
    versions = get_versions()
    labels = {1: "baseline"}
    for v in versions:
        labels[v["version_num"]] = v.get("label") or f"v{v['version_num']}"
    ver_options = sorted({1} | {v["version_num"] for v in versions})
    if len(ver_options) < 2:
        return {"ok": False, "project": proj, "task_type": task_type,
                "versions_present": ver_options,
                "reason": "Only one version exists — nothing to compare. Snapshot a new version, then re-run."}

    vof = _ver_of_factory(versions)
    all_versions = [{"version": n, "label": labels.get(n, f"v{n}")} for n in ver_options]

    if across_all:
        steps = [_compare_pair(all_runs, labels, vof, ver_options[i], ver_options[i + 1],
                               task_type, ver_options, include_intended, include_low_confidence)
                 for i in range(len(ver_options) - 1)]
        return {"ok": True, "project": proj, "task_type": task_type, "mode": "across_all",
                "all_versions": all_versions, "steps": steps}

    if base_version is None:
        bl = get_baseline_version()
        base_version = bl if bl in ver_options else ver_options[0]
    if target_version is None:
        target_version = ver_options[-1]                       # NEWEST

    pair = _compare_pair(all_runs, labels, vof, base_version, target_version,
                         task_type, ver_options, include_intended, include_low_confidence)
    return {"project": proj, "task_type": task_type, "mode": "pair",
            "all_versions": all_versions, **pair}


@mcp.tool()
def get_next_check_steps(finding_id: str | None = None, project: str | None = None,
                         range: str = "30d") -> dict:
    """Generate the recommended next investigation steps for a drift finding.

    Reuses the platform's "Generate AI suggestions" backend (Claude Opus 4.8), so
    the steps match what the dashboard produces. Falls back to deterministic checks
    if the LLM is unavailable (e.g. no ANTHROPIC_API_KEY in this process).

    Args:
        finding_id: a finding_id from get_todays_finding (form "project|id"). If given,
            it selects both the project and the specific finding.
        project: project to look in when finding_id omits it (None -> default DB).
        range: same look-back window used by get_todays_finding, so ids line up.
    """
    proj, fid = project, finding_id
    if finding_id and ":" in finding_id:
        proj, fid = finding_id.split(":", 1)
    proj = _use_project(proj)

    invs, ctx = investigations_for(range_param=range)
    sel = next((x for x in invs if x["id"] == fid), None) or (invs[0] if invs else None)
    if not sel:
        return {"ok": False, "reason": "No active finding to explain in this project/range."}

    card = sel.get("card") or {}
    context = {
        "instruction": ("You are a senior agent-ops engineer. This is a drift finding — a LIKELY "
                        "cause traced upstream from a symptom (correlational, not proven). List the "
                        "3-5 most efficient NEXT CHECKS to confirm or refute the cause and fix it. "
                        "Be specific and concise; output a plain bullet list, one check per line "
                        "starting with '- '."),
        "finding": sel["title"], "type": sel["type_label"], "severity": sel["severity"],
        "confidence": sel.get("confidence"), "path": card.get("path", sel["subtitle"]),
        "what_changed": sel.get("what_changed") or card.get("what_changed", [sel["subtitle"]]),
        "why_it_matters": card.get("why", ""),
        "potentially_related_event": sel.get("related_event") or card.get("related_event"),
    }

    result = {
        "ok": True,
        "finding_id": f"{proj}:{sel['id']}",
        "root": sel["title"],
        "severity": sel["severity"],
        "deterministic_next_checks": sel.get("next_checks", []),
    }
    try:
        result["ai_next_checks"] = suggest_next_checks(context)
    except Exception as e:
        result["ai_next_checks"] = None
        result["ai_error"] = f"AI suggestion unavailable ({e.__class__.__name__})."
    return result


if __name__ == "__main__":
    mcp.run()
