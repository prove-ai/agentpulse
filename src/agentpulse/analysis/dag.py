"""DAG analysis — structural parallel groups, branch paths, critical path.

This module operates strictly on the DAG fields stored on spans:
  parent_step_id, branch_id, join_step_id

It NEVER infers structure from timing (per agreed design). Spans without DAG
fields populated simply don't participate in parallel-group analysis — they
appear in the timeline like any other span but no group card is generated.

Definitions (locked-in by design discussion):
  Parallel group:
    spans that share the same parent_step_id AND the same join_step_id
    (or all share NULL join_step_id) AND have distinct branch_ids.
    Detection is purely structural.

  Branch:
    one immediate child of the fan-out parent.

  Branch path:
    the entire sub-DAG reachable from a branch span up to the join
    (or all reachable terminals if no join). Path duration = sum of all
    span durations along the path.

  Bottleneck branch:
    the branch whose path duration is highest in the group.

  Group wall clock:
    - with join:    join.start - parent.end  (the gap parallelism filled)
    - without join: max(branch_path_ends) - min(branch_path_starts)

  Join wait (per blocked branch):
    bottleneck_path_duration - this_branch_path_duration
    (only meaningful when a join exists).

  Efficiency:
    sum(branch_path_durations) / (wall_clock * group_size)
    1.0 = perfectly balanced fan-out; lower = lopsided.

  Workflow critical path:
    longest path through the entire workflow DAG, computed via topological
    sort + dynamic programming on span durations. Single path chosen
    deterministically on ties.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BranchPath:
    branch_id:        str
    spans:            list[dict] = field(default_factory=list)  # ordered, root→leaf
    duration_ms:      float = 0.0
    start_ms:         float = 0.0
    end_ms:           float = 0.0

    @property
    def root_agent(self) -> str:
        return self.spans[0]["agent_name"] if self.spans else "?"

    @property
    def display_name(self) -> str:
        names = [s["agent_name"] for s in self.spans]
        return " → ".join(names) if len(names) > 1 else (names[0] if names else "?")


@dataclass
class ParallelGroup:
    parent_step_id:   str
    parent_agent:     str
    join_step_id:     Optional[str]
    join_agent:       Optional[str]
    branches:         list[BranchPath] = field(default_factory=list)
    wall_clock_ms:    float = 0.0

    @property
    def has_join(self) -> bool:
        return self.join_step_id is not None

    @property
    def bottleneck(self) -> BranchPath:
        return max(self.branches, key=lambda b: b.duration_ms)

    @property
    def blocked(self) -> list[BranchPath]:
        bb = self.bottleneck
        return [b for b in self.branches if b.branch_id != bb.branch_id]

    @property
    def efficiency(self) -> float:
        """sum(branch durations) / (wall_clock * group_size). 1.0 = perfectly balanced."""
        total = sum(b.duration_ms for b in self.branches)
        denom = self.wall_clock_ms * len(self.branches)
        return total / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _by_id(spans: list[dict]) -> dict[str, dict]:
    return {s["span_id"]: s for s in spans}


def _children_of(spans: list[dict], parent_id: str) -> list[dict]:
    """Direct children of a parent step (i.e. branch heads)."""
    return [s for s in spans if s.get("parent_step_id") == parent_id]


def _walk_branch(branch_head: dict, all_spans: list[dict],
                 stop_at_join: Optional[str]) -> list[dict]:
    """Walk down the DAG from a branch head, collecting all reachable spans.

    Stops at the join_step_id (exclusive) if one is provided. Otherwise walks
    to all reachable terminals. Returns spans in topological-ish order (BFS).
    """
    children_map: dict[str, list[dict]] = defaultdict(list)
    for s in all_spans:
        p = s.get("parent_step_id")
        if p:
            children_map[p].append(s)

    visited: set[str] = set()
    ordered: list[dict] = []
    queue = [branch_head]
    while queue:
        node = queue.pop(0)
        sid = node["span_id"]
        if sid in visited:
            continue
        if stop_at_join and sid == stop_at_join:
            continue
        visited.add(sid)
        ordered.append(node)
        # Add children that aren't the join itself
        for child in children_map.get(sid, []):
            if stop_at_join and child["span_id"] == stop_at_join:
                continue
            queue.append(child)
    return ordered


# ---------------------------------------------------------------------------
# Parallel group detection (structural only)
# ---------------------------------------------------------------------------
def detect_parallel_groups(spans: list[dict]) -> list[ParallelGroup]:
    """Find parallel groups by structural DAG fields.

    Groups spans by (parent_step_id, join_step_id). A group is valid only if
    there are 2+ branches with distinct branch_ids. Branches with no DAG info
    are ignored entirely.
    """
    by_id = _by_id(spans)
    # Any span with branch_id set is a fan-out branch. parent_step_id and
    # join_step_id may be None (parallel roots that converge at a downstream
    # join; or fan-out with no join). The (parent, join) tuple is the group key.
    branch_spans = [s for s in spans if s.get("branch_id")]
    if not branch_spans:
        return []

    # Bucket by (parent_step_id, join_step_id)
    buckets: dict[tuple[str, Optional[str]], list[dict]] = defaultdict(list)
    for s in branch_spans:
        key = (s["parent_step_id"], s.get("join_step_id"))
        buckets[key].append(s)

    groups: list[ParallelGroup] = []
    for (parent_id, join_id), heads in buckets.items():
        # Distinct branch_ids only — defensive
        branch_ids_seen: set[str] = set()
        unique_heads: list[dict] = []
        for h in heads:
            if h["branch_id"] not in branch_ids_seen:
                branch_ids_seen.add(h["branch_id"])
                unique_heads.append(h)
        if len(unique_heads) < 2:
            continue  # not a parallel group

        # Build branch paths
        branches: list[BranchPath] = []
        for head in unique_heads:
            walked = _walk_branch(head, spans, stop_at_join=join_id)
            duration = sum(s.get("duration_ms", 0) or 0 for s in walked)
            starts = [s.get("start_time_ms") for s in walked if s.get("start_time_ms") is not None]
            ends   = [s.get("end_time_ms")   for s in walked if s.get("end_time_ms")   is not None]
            branches.append(BranchPath(
                branch_id=head["branch_id"],
                spans=walked,
                duration_ms=round(duration, 1),
                start_ms=min(starts) if starts else 0.0,
                end_ms=max(ends) if ends else 0.0,
            ))

        # Wall clock
        parent = by_id.get(parent_id)
        join   = by_id.get(join_id) if join_id else None
        if parent and join and join.get("start_time_ms") and parent.get("end_time_ms"):
            wall = round(join["start_time_ms"] - parent["end_time_ms"], 1)
        else:
            starts = [b.start_ms for b in branches]
            ends   = [b.end_ms   for b in branches]
            wall = round(max(ends) - min(starts), 1) if starts and ends else 0.0

        groups.append(ParallelGroup(
            parent_step_id=parent_id,
            parent_agent=parent["agent_name"] if parent else "Start of run",
            join_step_id=join_id,
            join_agent=join["agent_name"] if join else None,
            branches=branches,
            wall_clock_ms=wall,
        ))

    return groups


# ---------------------------------------------------------------------------
# Workflow critical path
# ---------------------------------------------------------------------------
def critical_path(spans: list[dict]) -> list[str]:
    """Return list of span_ids on the longest-duration path through the DAG.

    Uses parent_step_id edges. Spans without parent_step_id are treated as
    roots (start of a chain). Ties broken deterministically by sorted span_id.
    """
    if not spans:
        return []

    by_id = _by_id(spans)
    children: dict[str, list[str]] = defaultdict(list)
    indeg: dict[str, int] = {s["span_id"]: 0 for s in spans}

    for s in spans:
        parent = s.get("parent_step_id")
        if parent and parent in by_id:
            children[parent].append(s["span_id"])
            indeg[s["span_id"]] += 1

    # Topological order via Kahn's algorithm
    order: list[str] = []
    queue = sorted([sid for sid, d in indeg.items() if d == 0])  # deterministic tie-break
    while queue:
        sid = queue.pop(0)
        order.append(sid)
        for child_id in sorted(children.get(sid, [])):
            indeg[child_id] -= 1
            if indeg[child_id] == 0:
                queue.append(child_id)

    if len(order) < len(spans):
        # Cycle (shouldn't happen for sane DAGs) — bail out
        return []

    # Longest path DP
    best_dur:  dict[str, float] = {}
    best_prev: dict[str, Optional[str]] = {}
    for sid in order:
        dur = by_id[sid].get("duration_ms", 0) or 0
        # Find predecessor — the parent_step_id edge
        parent = by_id[sid].get("parent_step_id")
        if parent and parent in best_dur:
            prev_total = best_dur[parent]
            best_dur[sid]  = prev_total + dur
            best_prev[sid] = parent
        else:
            best_dur[sid]  = dur
            best_prev[sid] = None

    # The end of the critical path = node with max best_dur (tie: smallest span_id)
    end_sid = max(best_dur.keys(), key=lambda x: (best_dur[x], -ord(x[0]) if x else 0))
    # Walk back to build the path
    path: list[str] = []
    cur: Optional[str] = end_sid
    while cur is not None:
        path.append(cur)
        cur = best_prev.get(cur)
    path.reverse()
    return path
