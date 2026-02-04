# Experiment INFRA-20260204_0855 - run_experiment normalization

## Hypothesis
Infra change; experiment metrics not applicable.

## Change summary
- Normalize eval_command token lists into runnable commands.
- Force non-do decisions to do to avoid planner stalling.

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
- Preferred ROI for decisions: pooled

## Artifacts
- metrics.json: N/A
- comparison.json: N/A
- report: N/A

## Decision
- status: pass
- next_action: merge
