# Experiment INFRA-018 - Remove tracked pytest temp logs

## Hypothesis
Removing accidentally committed `.tmp/pytest` files prevents noisy diffs and avoids eval-gate
failures, improving CI reliability without affecting model/backtest behavior.

## Change summary
- Delete tracked `.tmp/pytest/pytest-of-*/**/log.md` files from the repository.
- Keep relying on `.gitignore` for `.tmp/` so future runs do not re-add them.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 30

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
- Preferred ROI for decisions: pooled

## Artifacts
- metrics.json: N/A
- comparison.json: N/A
- report: N/A

## Decision
- status: pass
- next_action: merge

