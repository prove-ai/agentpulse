---
name: release-regression-check
description: Check whether a new version introduced a regression via AgentPulse version comparison. Use when the user asks "did my release break anything", "compare versions", "is v4 safe", "what changed since baseline", or names two versions to compare.
---

# Release regression check

Answer "did this release regress?" and attribute the drift to the release that caused it.

## Steps
1. Call `get_version_comparison`:
   - No versions named → default (baseline → newest).
   - Two named (e.g. "v3 vs v4") → `base_version` + `target_version`.
   - "which release broke it" / "across all versions" → `across_all=True`.
2. Lead with the **critical drift**, stating which **version introduced it** (`introduced_in`).
3. Report the metrics that changed for that root and the **outcome it broke** (e.g. clean-finish rate).
4. Only mention intended changes if asked (`include_intended=True`) — label them intended, not regressions. Hide low-confidence unless asked (`include_low_confidence=True`).
5. If a critical drift is found, offer `get_next_check_steps`.

## Principles
- Always state which two versions were compared.
- "Baseline vs newest" spans multiple releases — attribute to the release that introduced the drift, not the whole span.
- A metric moving ≠ a regression. Only outcome-breaking, high-confidence drift is critical.
