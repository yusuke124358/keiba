# Experiment INFRA-017 - Defer requeued backlog items

## Hypothesis
If a backlog item times out or is requeued from stale `in_progress`, moving it to the end of the
queue prevents repeated retries from blocking the remaining TODOs, enabling autonomous progress
toward completing the full backlog.

## Change summary
When `run_backlog_batch.py` requeues an item (timeout or stale in-progress), move that item to the
end of `experiments/backlog.yml`.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 40

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

