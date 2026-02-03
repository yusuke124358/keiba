# Experiment INFRA-005 - codex output schema required fields

## Hypothesis
Requiring all top-level fields in the codex output schema avoids API validation errors during propose_pr.

## Change summary
Update `scripts/agent/output_schema.json` to require all top-level fields.

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
- Preferred ROI for decisions: step14

## Artifacts
- metrics.json: N/A
- comparison.json: N/A
- report: N/A

## Decision
- status: pass
- next_action: merge
