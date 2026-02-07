# Automation

## PR Creation Loop (propose)
- Script: `scripts/agent/propose_pr.py` (one hypothesis per run).
- Flow: implement -> reviewer -> manager -> fixer -> `make ci` -> commit -> publisher push -> PR create.
- Labels on PR: `autogen`, `auto-fix`.

## PR Improvement Loop (review)
- Target PRs: label `auto-fix` only; PRs with `needs-human` are excluded.
- Trigger: issue_comment / pull_request_review / schedule / workflow_dispatch.
- Pipeline: Reviewer -> Manager -> Fixer -> `make ci` -> commit -> publisher push -> PR comment.
- Config: `config/auto_fix.yml` (labels, thresholds).
- Workflow: `.github/workflows/agent_auto_fix.yml`.
- Runner: self-hosted (Codex CLI login, no API key). See `docs/self_hosted_runner.md`.

## Labels
- `auto-fix`: opt-in for automated review fixes.
- `needs-human`: stop condition (manual decision required).
- `autogen`: auto-generated PR.

## Stop Conditions
- Consecutive failures: 3
- Issue recurrence: 2
- Max total attempts: 10

## Secrets
- `AUTO_FIX_PUSH_TOKEN`: publisher PAT (fine-grained) used only for push/PR operations.

## Evaluation Gate
- `scripts/eval_gate.py` runs as part of `make ci`.
- If experiment logs are changed, metrics fields must not be N/A (infra logs are exempt).

## Human Commands
- When `needs-human` is added, the bot posts a Human Packet with `/human` commands.
- `/human approve` or `/human clarify` removes `needs-human` and resumes auto-fix.
- `/human reject` keeps `needs-human` and removes `auto-fix`.

## Workflows
- `agent_propose_pr.yml`: runs propose loop on a self-hosted runner.
- `agent_auto_fix.yml`: runs reviewer/manager/fixer loop on a self-hosted runner.
- `agent_scientist_loop.yml`: runs scientist loop (plan -> run -> commit -> publisher).
- `needs_human_notify.yml`: posts Human Packet and assigns the PR author.
- `human_command.yml`: processes `/human` commands and updates labels.

## Notes
- Codex (interactive/local) does not push. Only the self-hosted workflow publisher may push to PR branches.
- Protected branches (main) are never pushed.
- All fixes must pass `make ci`.

## Scientist Loop
- Seed hypotheses: `experiments/seed_hypotheses.yaml`
- Runs: `experiments/runs/<run_id>.json` and `docs/experiments/<run_id>.md`
- Stats artifacts (gitignored; uploaded in Actions): `artifacts/experiments/<run_id>/per_bet_pnl.csv`, `artifacts/experiments/<run_id>/summary_stats.json` (day-block bootstrap CI + paired delta)
- Baseline cache (gitignored): `artifacts/baselines/<baseline_key>/` (reused across runs when base commit/test period/config match)
- Checkpoints: every 50 runs: `reports/checkpoints/<checkpoint_id>.md` and `experiments/checkpoints/<checkpoint_id>.json`
- Loop behavior (one action per run):
  - Stop if any PR has `needs-human`
  - If checkpoint due, generate summary and stop (publisher adds `needs-human`)
  - Otherwise generate and run one experiment

## Backlog Batch Runner
- Backlog file: `experiments/backlog.yml` (status: `todo`, `in_progress`, `done`, `failed`).
- Script: `scripts/agent/run_backlog_batch.py`.
- Workflow: `.github/workflows/agent_backlog_batch.yml`.
- Defaults: `--count 5`, stop on first failure (use `--continue-on-failure` to override).
- Behavior:
  - Picks `todo` items only.
  - Marks `in_progress`, runs plan + experiment, then marks `done` or `failed`.
  - Optional: push backlog status commits (`--push-base`) and publish PRs (`--publish`).
