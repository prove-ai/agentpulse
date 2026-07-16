---
name: drift-root-cause-report
description: Produce a full root-cause report for a specific drift — finding, which version introduced it, evidence, and remediation. Use when the user asks to "investigate <agent>", "write up the drift", "root cause report", or drills into a specific finding.
---

# Drift root-cause report

Assemble a complete incident-style report by chaining all three tools.

## Steps
1. `get_todays_finding` (named project) → identify the critical finding + its `finding_id`.
2. `get_version_comparison` with `across_all=True` → pinpoint which version introduced it.
3. `get_next_check_steps` with the `finding_id` → remediation (live Opus + deterministic).
4. Assemble:
   - **Finding**: root component + Severity + Confidence.
   - **Causal path**: upstream root → downstream symptom; state the symptom isn't at fault.
   - **Introduced in**: version + label + related config/model/prompt change.
   - **Evidence**: the metrics that moved + the outcome broken.
   - **Active since / duration**.
   - **Next checks**: numbered, most-informative first.

## Principles
- One root cause per report — don't sprawl across every candidate.
- Keep causal language correlational.
- The report should let an engineer act without opening the dashboard.
