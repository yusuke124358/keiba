---
name: pr-ready
description: Codex skill pr-ready
version: 0.1
---

﻿# pr-ready

Purpose: ensure a PR is small, reproducible, and passes required gates before submission.

When to use:
- Before creating a PR or enabling auto-merge.

Steps:
1) Identify the experiment id and max_diff_size from `experiments/backlog.yml`.
2) Verify the diff is within `max_diff_size` (added + deleted lines).
3) Ensure `docs/experiments/<id>.md` exists and follows `docs/experiments/_template.md`.
4) Run required gates: `scripts/verify.ps1` (Windows) or `scripts/verify.sh` (Linux/CI).
5) Summarize test results and risk level in the PR body.

Commands:
- `git diff --numstat`
- `scripts/verify.ps1`
- `scripts/verify.sh`

Output:
- Short checklist with pass/fail for diff size, experiment log, and gates.
