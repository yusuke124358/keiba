# Tasks bus

Purpose: simple inbox/outbox workflow for multi-agent ops.

## Workflow
- Orchestrator writes tasks to `tasks/inbox/` using templates.
- Workers pick a task, execute, and write a report to `tasks/outbox/`.
- Orchestrator reads reports and updates `context/keiba_current.md`.

## Conventions
- File names: `YYYYMMDD_HHMM_<role>_<short>.md`
- One owner per task; keep tasks small and focused.
- Templates live in `tasks/templates/`.

## Notes
- Outbox files are ignored by git to reduce noise.
