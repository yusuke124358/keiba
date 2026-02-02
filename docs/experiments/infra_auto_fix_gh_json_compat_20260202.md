# Experiment infra_auto_fix_gh_json_compat_20260202 - Handle Older gh JSON Fields

## Hypothesis
Older gh CLI versions lacking `reviewThreads` can still support the auto-fix loop with a fallback.

## Change summary
Add a fallback path in `review_loop.py` when `gh pr view` does not support `reviewThreads`.

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
