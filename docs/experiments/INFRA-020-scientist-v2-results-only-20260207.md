# Experiment INFRA-020 - Scientist Loop V2 (results-only + scientist-state ledger)

## Hypothesis
Replacing PR-per-experiment with results-only execution plus an append-only central ledger increases exploration throughput while preserving auditability and restartability.

## Change summary
Add Scientist Loop V2 scripts/workflows and a `scientist-state` append-only ledger branch to decouple throwaway experiments from PR creation and reduce PR volume.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 3000

## Commands
- `make ci`

## Metrics (required)
- ROI: N/A (infra)
- Total stake: N/A (infra)
- n_bets: N/A (infra)
- Test period: N/A (infra)
- Max drawdown: N/A (infra)
- ROI definition: ROI = profit / stake, profit = return - stake.
- Rolling: no
- Design window: N/A
- Eval window: N/A
- Paired delta vs baseline: N/A
- Pooled vs step14 sign mismatch: no
- Preferred ROI for decisions: step14

