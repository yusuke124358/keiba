# Task: <title>

Required section headers must remain exactly as named; do not rename or remove them.
Routing rules:
- Task id = inbox filename stem.
- Report path must be `tasks/outbox/<task_id>_report.md`.
- Owner / role must be `Guard` or `Reviewer`.
- If Auto-review is enabled, Review report path must be `tasks/outbox/<task_id>_review.md` and Review task path must live under `tasks/inbox/`.

## Purpose
- <why this task exists>

## Inputs
- <paths, links, configs>

## review_scope (base/head)
- Base: <base branch or commit>
- Head: <head branch or commit>

## Auto-review (optional)
- Enable: <yes/no>
- Review task path: <tasks/inbox/...>
- Review report path: tasks/outbox/<task_id>_review.md

## Constraints
- Allowed:
  - <allowed operations>
- Forbidden:
  - <forbidden operations>

## Execution viability
- Can run? <yes/no>
- If no, why: <reason>

## Commands
- <command>

## Artifacts
- <path>

## Pass criteria
- <acceptance criteria>

## Report path
- tasks/outbox/<task_id>_report.md

## Owner / role
- Guard | Reviewer
