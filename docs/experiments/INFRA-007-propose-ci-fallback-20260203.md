# Experiment INFRA-007 - propose_pr CI fallback on Windows

## Hypothesis
Allowing propose_pr to fall back to `scripts/ci.ps1` when `make` is unavailable prevents Windows runner failures.

## Change summary
Update `scripts/agent/propose_pr.py` to use `make ci` if present, otherwise run `scripts/ci.ps1` or `scripts/verify.ps1`.

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
