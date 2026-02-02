# Experiment infra_propose_branch_reuse_20260202 - Reuse Existing Agent Branch

## Hypothesis
Allowing `propose_pr.py` to reuse an existing agent branch avoids failures on reruns without changing behavior.

## Change summary
Use `git checkout -B` so repeated runs reset the branch instead of failing.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 200

## Commands
- `make ci`

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
