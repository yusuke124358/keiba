# Experiment INFRA-20260204_1830 - publisher label fix

## Hypothesis
Infra change; experiment metrics not applicable.

## Change summary
- Use PR number to apply labels reliably after creation.
- Create missing labels before attempting to apply.

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
