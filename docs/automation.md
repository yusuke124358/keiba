# Automation

## PR Creation Loop (propose)
- Script: `scripts/agent/propose_pr.py` (one hypothesis per run).
- Flow: implement → reviewer → manager → fixer → `make ci` → commit → publisher push → PR create.
- Labels on PR: `autogen`, `auto-fix`.

## PR Improvement Loop (review)
- Target PRs: label `auto-fix` only; PRs with `needs-human` are excluded.
- Trigger: issue_comment / pull_request_review / workflow_run(ci failure) / schedule.
- Pipeline: Reviewer → Manager → Fixer → `make ci` → commit → publisher push → PR comment.
- Config: `config/auto_fix.yml` (labels, thresholds, codex version).
- Workflow: `.github/workflows/auto_fix_loop.yml` uses `openai/codex-action@v1`.
- Runner: self-hosted (GPU/LLM optional).

## Labels
- `auto-fix`: opt-in for automated review fixes.
- `needs-human`: stop condition (manual decision required).
- `autogen`: auto-generated PR.

## Stop Conditions
- Consecutive failures: 3
- Issue recurrence: 2
- Max total attempts: 10

## Secrets
- `OPENAI_API_KEY`: Codex execution.
- `AUTO_FIX_PUSH_TOKEN`: publisher PAT (fine-grained) used only for push.

## Human Commands
- When `needs-human` is added, the bot posts a Human Packet with `/human` commands.
- `/human approve` or `/human clarify` removes `needs-human` and resumes auto-fix.
- `/human reject` keeps `needs-human` and removes `auto-fix`.

## Workflows
- `auto_fix_loop.yml`: runs reviewer/manager/fixer loop.
- `needs_human_notify.yml`: posts Human Packet and assigns the PR author.
- `human_command.yml`: processes `/human` commands and updates labels.

## Notes
- Codex (interactive/local) does not push. Only the auto-fix workflow publisher may push to PR branches.
- Protected branches (main) are never pushed.
- All fixes must pass `make ci`.
