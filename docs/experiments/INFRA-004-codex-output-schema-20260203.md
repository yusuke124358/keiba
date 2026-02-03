# Experiment INFRA-004 - codex output schema strictness

## Hypothesis
Tightening the codex output schema (disallowing additional properties and requiring all metric fields) prevents API schema validation errors in propose_pr.

## Change summary
Adjust `scripts/agent/output_schema.json` to set `additionalProperties: false` and add required fields for `metrics` and `artifacts`.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 50

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
