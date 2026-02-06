# Experiment INFRA-016 - Backlog batch timeout auto-stash

## Hypothesis
When an experiment times out mid-run, the backlog batch runner should be able to stash partial
changes, update backlog state, and allow the next scheduled batch to resume without manual cleanup.

## Change summary
Auto-stash dirty working trees on timeout/exception in `run_backlog_batch.py`, and enable
`--auto-stash` in the scheduled backlog cron workflow for extra robustness.

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

