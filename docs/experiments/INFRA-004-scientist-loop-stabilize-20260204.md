# Experiment INFRA-004 - Scientist loop stabilization

## Hypothesis
Infra-only adjustments to the scientist loop will improve automation stability without affecting backtest metrics.

## Change summary
- Auto-generate infra experiment logs on code-only changes.
- Normalize eval command placeholders and fill experiment log fields.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 200

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
- Preferred ROI for decisions: pooled

## Artifacts
- metrics.json: N/A
- comparison.json: N/A
- report: N/A

## Decision
- status: pass
- next_action: merge
