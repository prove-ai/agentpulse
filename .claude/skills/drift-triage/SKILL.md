---
name: drift-triage
description: Triage active agent drift across AgentPulse projects. Use when the user asks "what drifted today", "any drift", "check for regressions", "how are my agents doing", or wants a drift standup. Leads with the critical root cause, not raw metrics.
---

# Drift triage

Produce a prioritized triage — lead with the root cause, explain why, recommend the next step.

## Steps
1. Call `get_todays_finding`. Pass `project` only if the user named one; else scan all. Keep `range` at 30d (drift needs history).
2. Lead with the **critical/drift findings** (the `findings` array) — never the secondary signals. For each:
   - State **Severity** (Critical/Drift) and **Confidence**.
   - Name the **root**: which agent / handoff / route drifted.
   - Give the **causal why**: if a downstream agent's outcome dropped, explain it's the upstream root's fault, not the symptom's (use the `why` field).
   - List the moved metrics as evidence, not as the headline.
   - Note `days_active` ("active since X").
3. If exactly one critical finding, immediately call `get_next_check_steps` with its `finding_id` and include the checks.
4. Mention secondary signals only as a one-line "also moving" footnote.

## Principles
- Which component drifted + why > how many metrics moved.
- Drift persists — say "active since X (N days)", don't re-alert as brand new.
- Findings are correlational — "likely / potentially related", never "caused by".
