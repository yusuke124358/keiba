# Agent recovery playbook

Purpose: fast recovery when a new agent joins or context is lost.

## Quick recovery steps
1) Confirm repo root (pwd) and correct repo.
2) Check current branch: `git status -sb`.
3) Read, in order:
   - `memory.md`
   - `context/keiba_current.md`
   - latest file in `tasks/inbox/`
4) Reconfirm your role rules in `docs/ops/agents_roles.md`.
5) Write a short resync note to `tasks/outbox/` before starting work:
   - What you read
   - What you plan to do next

## If blocked
- Post a one-line question in outbox and wait for orchestrator reply.
