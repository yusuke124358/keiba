# Experiment INFRA-014 - Avoid Windows command line length failures for codex exec

## Hypothesis
Passing large Codex prompts via stdin (instead of a command-line argument) prevents "The command line is too long" failures on Windows self-hosted runners and restores `agent_backlog_batch` autonomy.

## Change summary
- Update `scripts/agent/plan_next_experiment.py` and `scripts/agent/checkpoint.py` to pass prompts to `codex exec` via stdin (`-`), avoiding Windows command line length limits.
- Update `scripts/agent/run_backlog_batch.py` to avoid marking backlog items as `failed` when planning fails before the item is marked `in_progress`.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 80

## Commands
- `py64_analysis\.venv\Scripts\python.exe -m pytest py64_analysis/tests --basetemp py64_analysis/tests/_tmp/pytest`
- `py64_analysis\.venv\Scripts\python.exe py64_analysis/scripts/check_system_status.py`

## Metrics (required)
- ROI: N/A
- Total stake: N/A
- n_bets: N/A
- Test period: N/A
- Max drawdown: N/A
- ROI definition: ROI = profit / stake, profit = return - stake.
- Rolling: no
- Design window: N/A
- Eval window: N/A
- Paired delta vs baseline: N/A
- Pooled vs step14 sign mismatch: no
- Preferred ROI for decisions: step14

## Artifacts
- metrics.json: N/A
- comparison.json: N/A
- report: N/A

## Decision
- status: pass
- next_action: merge

