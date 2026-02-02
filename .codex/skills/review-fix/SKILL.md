---
name: review-fix
description: Codex skill review-fix
version: 0.1
---

﻿# review-fix

Purpose: apply review feedback and stabilize CI gates.

When to use:
- After receiving review comments or failing CI checks.

Steps:
1) Collect failing checks and reviewer comments (include bots like CodeRabbit/Cursor/Codex).
2) Translate feedback into a short fix plan that preserves the hypothesis scope.
3) Apply minimal changes to address issues.
4) Re-run required gates: `scripts/verify.ps1` or `scripts/verify.sh`.
5) Update `docs/experiments/<id>.md` if results changed.
6) Commit with a fix message and update the PR.

Output:
- A clean fix commit with passing gates and updated experiment log if needed.
