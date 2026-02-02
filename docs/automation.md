# Automation

## Review Auto-Fix Loop
- Target PRs: label `auto-fix` only; PRs with `needs-human` are excluded.
- Trigger: GitHub Actions schedule (15 min) or manual dispatch.
- Pipeline: Reviewer → Manager → Fixer → `make ci` → commit → publisher push → PR comment.
- Config: `config/auto_fix.yml` (labels, thresholds, codex version).
- Workflow: `.github/workflows/auto_fix_loop.yml` uses `openai/codex-action@v1`.

## Labels
- `auto-fix`: opt-in for automated review fixes.
- `needs-human`: stop condition (manual decision required).

## Stop Conditions
- Consecutive failures: 3
- Issue recurrence: 2
- Max total attempts: 10

## Secrets
- `OPENAI_API_KEY`: Codex execution.
- `AUTO_FIX_PUSH_TOKEN`: publisher PAT (fine-grained) used only for push.

## Notes
- Codex (interactive/local) does not push. Only the auto-fix workflow publisher may push to PR branches.
- Protected branches (main) are never pushed.
- All fixes must pass `make ci`.
