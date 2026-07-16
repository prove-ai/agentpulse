"""Seed synthetic parallel runs into the observability DB.

Theme: 'Market Research Report' workflow — 5 runs with realistic agent
names so the user can compare strategies (sequential vs balanced parallel
vs imbalanced vs nested vs critical-path-through-side-branch).

Same task_type so they appear side-by-side on the index for comparison.

Usage:
    cd observability
    python scripts/seed_parallel.py
"""

from __future__ import annotations

import datetime
import json
import sys
import uuid
from pathlib import Path

_OBS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_OBS_ROOT))

from storage.sqlite_store import get_connection, compute_cost


def insert_run(
    label: str,
    task_text: str,
    task_type: str,
    spans_spec: list[dict],
) -> str:
    """Insert one synthetic run with DAG-populated spans.

    Each span spec: {name, start_s, end_s, parent, branch_id?, join?,
                     tokens_in, tokens_out, model?}
    """
    conn = get_connection()
    run_id = str(uuid.uuid4())

    span_ids = {s["name"]: str(uuid.uuid4()) for s in spans_spec}

    total_in  = sum(s["tokens_in"]  for s in spans_spec)
    total_out = sum(s["tokens_out"] for s in spans_spec)
    model     = spans_spec[0].get("model", "claude-sonnet-4-6")
    total_cost = sum(
        compute_cost(s["tokens_in"], s["tokens_out"], s.get("model", model))
        for s in spans_spec
    )

    wall_ms = (max(s["end_s"] for s in spans_spec) -
               min(s["start_s"] for s in spans_spec)) * 1000
    sequence = [s["name"] for s in sorted(spans_spec, key=lambda x: x["start_s"])]

    conn.execute(
        """
        INSERT INTO runs
          (run_id, task_text, task_type, prompt_version, model, timestamp,
           total_turns, total_duration_ms, total_input_tokens,
           total_output_tokens, total_cost_usd, termination_reason,
           status, agent_sequence, prompt_hashes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, task_text, task_type, 1, model,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            len(spans_spec), round(wall_ms, 1),
            total_in, total_out, round(total_cost, 6),
            "STATUS: APPROVED", "OK",
            json.dumps(sequence), json.dumps({}),
        ),
    )

    for idx, s in enumerate(sorted(spans_spec, key=lambda x: x["start_s"])):
        start_ms = s["start_s"] * 1000
        end_ms   = s["end_s"]   * 1000
        dur_ms   = end_ms - start_ms
        conn.execute(
            """
            INSERT OR REPLACE INTO spans
              (span_id, run_id, agent_name, turn_index, start_time_ms,
               end_time_ms, duration_ms, input_tokens, output_tokens,
               model, tool_call_count, status, status_value,
               parent_step_id, branch_id, join_step_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                span_ids[s["name"]], run_id, s["name"], idx,
                start_ms, end_ms, round(dur_ms, 1),
                s["tokens_in"], s["tokens_out"],
                s.get("model", model), 0, "OK", "STATUS: COMPLETE",
                span_ids[s["parent"]] if s.get("parent") else None,
                s.get("branch_id"),
                span_ids[s["join"]] if s.get("join") else None,
            ),
        )

    conn.commit()
    conn.close()
    print(f"  ✓ {label:50s} → {run_id[:8]}")
    return run_id


# ---------------------------------------------------------------------------
# Theme: "Market Research Report"
#
# All 5 runs use the same task_type so they're comparable side-by-side on
# the index page. Each demonstrates a different orchestration strategy.
# ---------------------------------------------------------------------------
TASK = "Write a market research report on the global EV battery industry."
TYPE = "market-research-report"


def run_1_sequential():
    """Plain sequential workflow — no parallelism, baseline for comparison.

    ProjectManager → MarketResearcher → DataAnalyst → ReportWriter → QualityReviewer
    Total wall clock: 15s. No parallel groups.
    """
    return insert_run(
        "Run 1 · Sequential baseline (no parallel)",
        TASK, TYPE,
        [
            {"name": "ProjectManager",   "start_s": 0.0, "end_s": 1.0, "parent": None,
             "tokens_in": 520, "tokens_out": 95},
            {"name": "MarketResearcher", "start_s": 1.0, "end_s": 5.0, "parent": "ProjectManager",
             "tokens_in": 1800, "tokens_out": 720},
            {"name": "DataAnalyst",      "start_s": 5.0, "end_s": 8.0, "parent": "MarketResearcher",
             "tokens_in": 2400, "tokens_out": 580},
            {"name": "ReportWriter",     "start_s": 8.0, "end_s": 13.0, "parent": "DataAnalyst",
             "tokens_in": 3100, "tokens_out": 1450},
            {"name": "QualityReviewer",  "start_s": 13.0, "end_s": 15.0, "parent": "ReportWriter",
             "tokens_in": 3900, "tokens_out": 320},
        ],
    )


def run_2_balanced_parallel():
    """Fan-out at research stage. All 3 researchers take similar time.

    ProjectManager → [MarketResearcher, IndustryAnalyst, CompetitorScout] → DataAnalyst → ...
    Expected: ~97% efficiency, well-balanced. Saves ~6s vs sequential.
    """
    return insert_run(
        "Run 2 · Balanced parallel (3 researchers ≈ same time)",
        TASK, TYPE,
        [
            {"name": "ProjectManager",   "start_s": 0.0, "end_s": 1.0, "parent": None,
             "tokens_in": 520, "tokens_out": 95},

            # Parallel research stage — all 3 take roughly the same time
            {"name": "MarketResearcher", "start_s": 1.0, "end_s": 4.0, "parent": "ProjectManager",
             "branch_id": "market",     "join": "DataAnalyst",
             "tokens_in": 1700, "tokens_out": 620},
            {"name": "IndustryAnalyst",  "start_s": 1.0, "end_s": 4.2, "parent": "ProjectManager",
             "branch_id": "industry",   "join": "DataAnalyst",
             "tokens_in": 1900, "tokens_out": 680},
            {"name": "CompetitorScout",  "start_s": 1.0, "end_s": 4.1, "parent": "ProjectManager",
             "branch_id": "competitor", "join": "DataAnalyst",
             "tokens_in": 1750, "tokens_out": 600},

            {"name": "DataAnalyst",      "start_s": 4.2, "end_s": 7.2, "parent": "IndustryAnalyst",
             "tokens_in": 2600, "tokens_out": 720},
            {"name": "ReportWriter",     "start_s": 7.2, "end_s": 12.2, "parent": "DataAnalyst",
             "tokens_in": 3400, "tokens_out": 1500},
            {"name": "QualityReviewer",  "start_s": 12.2, "end_s": 14.2, "parent": "ReportWriter",
             "tokens_in": 4200, "tokens_out": 340},
        ],
    )


def run_3_imbalanced_parallel():
    """Same fan-out shape but one researcher dominates — poor efficiency.

    IndustryAnalyst takes 6s (drilling into a hard dataset). The other two
    finish in 1s and just wait. Wall clock = 6s; efficiency drops to ~45%.
    Demonstrates a tuning opportunity: load-balance the researchers.
    """
    return insert_run(
        "Run 3 · Imbalanced parallel (IndustryAnalyst dominates)",
        TASK, TYPE,
        [
            {"name": "ProjectManager",   "start_s": 0.0, "end_s": 1.0, "parent": None,
             "tokens_in": 520, "tokens_out": 95},

            # MarketResearcher finishes quickly
            {"name": "MarketResearcher", "start_s": 1.0, "end_s": 2.0, "parent": "ProjectManager",
             "branch_id": "market",     "join": "DataAnalyst",
             "tokens_in": 800, "tokens_out": 220},
            # IndustryAnalyst is the bottleneck
            {"name": "IndustryAnalyst",  "start_s": 1.0, "end_s": 7.0, "parent": "ProjectManager",
             "branch_id": "industry",   "join": "DataAnalyst",
             "tokens_in": 3800, "tokens_out": 1400},
            # CompetitorScout finishes quickly
            {"name": "CompetitorScout",  "start_s": 1.0, "end_s": 2.2, "parent": "ProjectManager",
             "branch_id": "competitor", "join": "DataAnalyst",
             "tokens_in": 900, "tokens_out": 250},

            {"name": "DataAnalyst",      "start_s": 7.0, "end_s": 10.0, "parent": "IndustryAnalyst",
             "tokens_in": 2700, "tokens_out": 750},
            {"name": "ReportWriter",     "start_s": 10.0, "end_s": 15.0, "parent": "DataAnalyst",
             "tokens_in": 3500, "tokens_out": 1500},
            {"name": "QualityReviewer",  "start_s": 15.0, "end_s": 17.0, "parent": "ReportWriter",
             "tokens_in": 4300, "tokens_out": 350},
        ],
    )


def run_4_double_fanout():
    """Two parallel groups: research stage AND writing stage.

    ProjectManager → [MarketResearcher, IndustryAnalyst, CompetitorScout] → DataAnalyst
                  → [ReportWriter, ChartCreator] → QualityReviewer

    Demonstrates: two independent parallel groups in one workflow,
    each shown as its own card.
    """
    return insert_run(
        "Run 4 · Double fan-out (research stage AND writing stage)",
        TASK, TYPE,
        [
            {"name": "ProjectManager",   "start_s": 0.0, "end_s": 1.0, "parent": None,
             "tokens_in": 540, "tokens_out": 100},

            # Group 1: research fan-out
            {"name": "MarketResearcher", "start_s": 1.0, "end_s": 4.2, "parent": "ProjectManager",
             "branch_id": "market",     "join": "DataAnalyst",
             "tokens_in": 1800, "tokens_out": 650},
            {"name": "IndustryAnalyst",  "start_s": 1.0, "end_s": 4.0, "parent": "ProjectManager",
             "branch_id": "industry",   "join": "DataAnalyst",
             "tokens_in": 1850, "tokens_out": 640},
            {"name": "CompetitorScout",  "start_s": 1.0, "end_s": 4.5, "parent": "ProjectManager",
             "branch_id": "competitor", "join": "DataAnalyst",
             "tokens_in": 1900, "tokens_out": 680},

            {"name": "DataAnalyst",      "start_s": 4.5, "end_s": 7.5, "parent": "MarketResearcher",
             "tokens_in": 2700, "tokens_out": 760},

            # Group 2: writing fan-out
            {"name": "ReportWriter",     "start_s": 7.5, "end_s": 12.0, "parent": "DataAnalyst",
             "branch_id": "report",     "join": "QualityReviewer",
             "tokens_in": 3500, "tokens_out": 1450},
            {"name": "ChartCreator",     "start_s": 7.5, "end_s": 10.0, "parent": "DataAnalyst",
             "branch_id": "charts",     "join": "QualityReviewer",
             "tokens_in": 2200, "tokens_out": 850},

            {"name": "QualityReviewer",  "start_s": 12.0, "end_s": 14.0, "parent": "ReportWriter",
             "tokens_in": 4400, "tokens_out": 360},
        ],
    )


def run_5_parallel_side_branch():
    """User's A→B→C/D pattern with realistic names.

    The main chain (Market → Data → Writer) is the critical path. A side
    branch (CompetitorScout) runs in parallel to the *whole* chain but
    finishes earlier — it's slack, not critical.

    Critical path = ProjectManager → MarketResearcher → DataAnalyst → ReportWriter → QualityReviewer
    Side branch = CompetitorScout (runs but doesn't determine wall clock)
    """
    return insert_run(
        "Run 5 · Critical path through a long branch; side branch has slack",
        TASK, TYPE,
        [
            {"name": "ProjectManager",   "start_s": 0.0, "end_s": 1.0, "parent": None,
             "tokens_in": 520, "tokens_out": 95},

            # Main chain (branch "main") — runs sequentially
            {"name": "MarketResearcher", "start_s": 1.0, "end_s": 5.0, "parent": "ProjectManager",
             "branch_id": "main",       "join": "QualityReviewer",
             "tokens_in": 1800, "tokens_out": 720},
            {"name": "DataAnalyst",      "start_s": 5.0, "end_s": 8.0, "parent": "MarketResearcher",
             "tokens_in": 2400, "tokens_out": 580},
            {"name": "ReportWriter",     "start_s": 8.0, "end_s": 13.0, "parent": "DataAnalyst",
             "tokens_in": 3100, "tokens_out": 1450},

            # Side branch — runs in parallel with the entire main chain
            {"name": "CompetitorScout",  "start_s": 1.0, "end_s": 9.0, "parent": "ProjectManager",
             "branch_id": "side",       "join": "QualityReviewer",
             "tokens_in": 2100, "tokens_out": 880},

            {"name": "QualityReviewer",  "start_s": 13.0, "end_s": 15.0, "parent": "ReportWriter",
             "tokens_in": 3900, "tokens_out": 320},
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Seeding 5 synthetic 'Market Research Report' runs for comparison…\n")
    run_1_sequential()
    run_2_balanced_parallel()
    run_3_imbalanced_parallel()
    run_4_double_fanout()
    run_5_parallel_side_branch()
    print("\nDone. Open the dashboard, click into each run, and compare:")
    print("  Run 1 — no parallel (15s)")
    print("  Run 2 — balanced parallel: ~97% efficiency (saves time)")
    print("  Run 3 — imbalanced parallel: ~45% efficiency (one slow agent ruins fan-out)")
    print("  Run 4 — two parallel groups in one workflow")
    print("  Run 5 — long main chain with slack side branch")
