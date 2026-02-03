# Experiment INFRA-010 - inline review items JSON in prompts

## Hypothesis
Embedding review/manager JSON content in the Manager/Fixer prompts prevents file-resolution errors during propose_pr.

## Change summary
Update `scripts/agent/propose_pr.py` to include review_items and manager_decision JSON content directly in prompts.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 20

## Commands
- `scripts/ci.ps1`

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
