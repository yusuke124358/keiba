# Experiment INFRA-006 - review items schema strictness

## Hypothesis
Requiring all properties in `review_items.schema.json` while allowing nulls for optional fields prevents Codex response-format validation errors.

## Change summary
Update `schemas/agent/review_items.schema.json` to require all item fields and allow null for optional properties.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 20

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
