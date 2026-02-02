# fixer-implement

Purpose: implement approved fixes safely and verify gates.

Input:
- `manager_decision.json` tasks where decision = DO.

Rules:
- Only execute DO tasks.
- Keep changes minimal and focused on the issue.
- Run `make ci` before handing off.
- Output must conform to `schemas/agent/fixer_report.schema.json`.

Notes:
- In automated review loop, commit/push/comment are handled by the orchestrator.
- In manual use, summarize changes, tests, and any residual risk.
