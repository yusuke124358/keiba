# Experiment INFRA-013 - Avoid propose_pr branch push collisions

## Hypothesis
Generating a unique branch name for backlog items without an explicit `branch`, and reusing an existing remote branch when `branch` is specified, prevents non-fast-forward push failures in `Agent Propose PR`.

## Change summary
Update `scripts/agent/propose_pr.py` to avoid branch collisions by:
- Creating unique `agent/{exp_id}-{slug}-{timestamp}` branches when the backlog item does not specify `branch`.
- Checking out an existing remote branch (when present) when the backlog item specifies `branch`, to keep pushes fast-forward.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 120

## Commands
- `py64_analysis\.venv\Scripts\python.exe -m pytest py64_analysis/tests`
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

