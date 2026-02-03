# Scientist Loop Infrastructure

- Experiment type: infra
- risk_level: low
- max_diff_size: 200
- ROI: N/A
- Total stake: N/A
- n_bets: N/A
- Test period: N/A
- Max drawdown: N/A
- ROI definition: ROI = profit / stake; profit = return - stake
- Rolling: no
- Design window: N/A
- Eval window: N/A
- Paired delta vs baseline: N/A
- Pooled vs step14 sign mismatch: no
- Preferred ROI for decisions: pooled

## Summary
- Added scientist-loop scaffolding (planning, execution, checkpointing, knowledge ingestion).
- Added prompts/schemas for automated planning and summaries.
- Added publisher workflow separation and runner preflight checks.

## Notes
- Infra-only change. No model/backtest logic altered.
