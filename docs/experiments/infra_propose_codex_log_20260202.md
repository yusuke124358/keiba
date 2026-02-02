# Experiment infra_propose_codex_log_20260202 - Improve Codex Failure Logging

## Hypothesis
Including the Codex log tail in errors speeds up diagnosis without impacting runtime behavior.

## Change summary
When codex exec fails in `propose_pr.py`, append the last log lines to the exception message.

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
