# Experiment INFRA-015 - Backlog cron + push-base retry

## Hypothesis
Scheduling backlog batches and making base-branch pushes resilient to non-fast-forward races reduces manual intervention and keeps the hypothesis loop progressing.

## Change summary
- Add `agent_backlog_cron` to run backlog batches on a schedule (count=5) and requeue timed-out items.
- Update backlog batch workflow to avoid cancelling in-progress runs and allow stopping/continuing after failures via input.
- Update `scripts/agent/run_backlog_batch.py` to retry base-branch pushes by fetching and rebasing onto the latest remote tip.
- Disable the hourly schedule for `Agent Propose PR` to reduce noise while backlog automation is running.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 200

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
- Preferred ROI for decisions: step14

## Artifacts
- metrics.json: N/A
- comparison.json: N/A
- report: N/A

## Decision
- status: pass
- next_action: merge

