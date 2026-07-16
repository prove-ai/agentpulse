"""LLM-assisted drift triage — "suggest next checks".

Given a detected drift (one finding) plus the changes around it, ask Claude for
the most efficient next troubleshooting steps. The whole point is to save the
engineer time when there are several events or the cause isn't obvious.

The context is rebuilt server-side from the same series/change-log functions the
dashboard uses — the client only names the finding, never the evidence.
"""

from __future__ import annotations

from analysis.metric_series import control_band, metric_impact
from analysis.changes import build_change_log

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You are an observability assistant for LangGraph multi-agent systems. "
    "An automated detector has flagged a metric drift (a sustained move outside a "
    "baseline control band). You are given the drifted agent/handoff, which of its "
    "metrics moved (baseline vs recent) and which stayed stable, the config changes "
    "that happened at or before the drift, and the other drifts in the system.\n\n"
    "Your job is to save the engineer triage time. Be concrete and concise. "
    "Correlation is not causation — a change near the drift is a lead, not a verdict. "
    "Respond in this exact shape and nothing else:\n"
    "First line: one sentence naming the most likely explanation (or 'Unclear — ' "
    "and the single best thing to disambiguate).\n"
    "Then 2-4 lines, each starting with '- ', each a single concrete next check, "
    "ordered most-informative first. Reference specific runs, agents, or config "
    "changes from the context. No preamble, no closing remarks, no markdown headers."
)


def build_context(runs: list[dict], series: dict, finding: dict, band_cfg: dict,
                  findings: list[dict]) -> dict:
    """Compact, high-signal evidence packet for one finding."""
    cat, entity = finding["category"], finding["entity"]
    br = band_cfg["baseline_runs"]
    ent = series.get(cat, {}).get(entity, {})

    drifted, stable = [], []
    for metric, pts in ent.items():
        band = control_band([p["y"] for p in pts], **band_cfg)
        imp = metric_impact(pts, br)
        if band.get("drifting"):
            drifted.append({"metric": metric, "change": imp["label"],
                            "baseline": imp["baseline"], "recent": imp["recent"]})
        else:
            stable.append(metric)

    changes = [{"run": e["run_index"],
                "what": f"{e['dimension']} changed"
                        + ("" if e["scope"] == "workflow" else f" ({e['scope']})"),
                "detail": f"{e['old']} -> {e['new']}"} for e in build_change_log(runs)]
    ds = finding["drift_start"]
    before = [c for c in changes if c["run"] <= ds][-4:]
    after = [c for c in changes if c["run"] > ds][:2]

    others = [f"{f['entity']} ({f['risk']}): {', '.join(f['metrics'])} @ run {f['drift_start']}"
              for f in findings if f"{f['category']}|{f['entity']}" !=
              f"{cat}|{entity}"]

    return {
        "drifted_entity": entity, "kind": cat, "risk": finding["risk"],
        "drift_started_at_run": ds,
        "drifted_metrics": drifted,
        "stable_metrics": stable,
        "config_changes_at_or_before_drift": before,
        "config_changes_just_after_drift": after,
        "other_drifts_in_system": others,
    }


def suggest_next_checks(context: dict) -> str:
    """Call Claude for a short triage suggestion. Raises on SDK/auth errors so the
    route can surface a friendly message."""
    import json
    import ssl
    import anthropic
    import certifi
    import httpx

    # Connect directly with certifi's CA bundle and do NOT inherit ambient env
    # SSL/proxy settings. On some local setups those env settings route httpx
    # through an interceptor or a broken trust store and TLS verification fails
    # (APIConnectionError / CERTIFICATE_VERIFY_FAILED) even though a raw socket to
    # api.anthropic.com verifies fine. trust_env=False + an explicit certifi
    # context makes the connection deterministic. (If you ever must reach the API
    # through a required outbound proxy, drop trust_env=False.)
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    http_client = httpx.Client(verify=ssl_ctx, trust_env=False, timeout=60.0)
    client = anthropic.Anthropic(http_client=http_client)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(context, indent=2)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()
