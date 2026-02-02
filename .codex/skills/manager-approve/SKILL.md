# manager-approve

Purpose: decide which review issues to fix based on business rules.

Input:
- `review_items.json` + business rules from `AGENTS.md`.

Rules:
- Business alignment is the top priority.
- Do not reject because it takes time.
- Only reject for business conflict, mandatory gate failure, ambiguity (needs-human), or high risk.
- Output must conform to `schemas/agent/manager_decision.schema.json`.

Output:
- `manager_decision.json` with task decisions and rationale.
