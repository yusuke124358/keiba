# Experiment INFRA-011 - gh auth env override

## Hypothesis
Avoiding AUTO_FIX_PUSH_TOKEN as GH_TOKEN prevents gh CLI 401 errors when the PAT lacks API scopes, while still allowing git push via token.

## Change summary
Update `scripts/agent/propose_pr.py` to use `AUTO_FIX_GH_TOKEN` (or existing `GH_TOKEN`) for gh CLI, and otherwise rely on `gh auth` login.

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
