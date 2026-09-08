"""Trend-window context for the /trends page.

build_trends splits the run window in half (early vs recent) and returns the
window bookkeeping (counts, versions, boundary timestamps) plus the top
handoff pairs. The per-agent and per-handoff drift verdicts shown on the page
are computed in reporter/dashboard.py on top of this window split.
"""

from __future__ import annotations


MIN_RUNS_FOR_TREND = 4    # need at least this many runs in the window


# ---------------------------------------------------------------------------
# Top-level entry — used by the dashboard route
# ---------------------------------------------------------------------------
def build_trends(
    runs:    list[dict],     # sorted ascending by timestamp
    metrics: list[dict],     # parallel to runs
) -> dict:
    """Return everything the /trends template needs."""
    n = len(runs)
    if n < MIN_RUNS_FOR_TREND:
        return {
            "insufficient":  True,
            "runs_count":    n,
            "min_required":  MIN_RUNS_FOR_TREND,
        }

    # Split in half
    mid = n // 2
    early_runs, recent_runs = runs[:mid], runs[mid:]

    # Versions present in this window
    versions = sorted({r.get("prompt_version") for r in runs if r.get("prompt_version") is not None})

    # Top handoff pairs across the window
    top_handoffs = compute_top_handoffs(metrics, top_n=5)

    return {
        "insufficient": False,
        "runs_count":   n,
        "n_early":      len(early_runs),
        "n_recent":     len(recent_runs),
        "versions":     versions,
        "top_handoffs": top_handoffs,
        "early_end_ts": early_runs[-1]["timestamp"] if early_runs else "",
        "recent_start_ts": recent_runs[0]["timestamp"] if recent_runs else "",
    }


# ---------------------------------------------------------------------------
# Top handoffs — most common (sender → receiver) pairs across the window
# ---------------------------------------------------------------------------
def compute_top_handoffs(metrics_list: list[dict], top_n: int = 5) -> dict:
    """Count handoff pairs across all runs in the window.

    Returns: { 'pairs': [{from, to, count, pct}], 'total': int }
    Each metrics dict is expected to have a `handoffs` list of
    {agent_a, agent_b, ...} entries.
    """
    counts: dict[tuple[str, str], int] = {}
    for m in metrics_list:
        for h in (m.get("handoffs") or []):
            a = h.get("agent_a") or h.get("from")
            b = h.get("agent_b") or h.get("to")
            if not a or not b:
                continue
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    pairs = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    return {
        "total": total,
        "pairs": [
            {
                "from":  a, "to": b,
                "count": c,
                "pct":   round((c / total * 100), 1) if total else 0.0,
            }
            for (a, b), c in pairs
        ],
    }
