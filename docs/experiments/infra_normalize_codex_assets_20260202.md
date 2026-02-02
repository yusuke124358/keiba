# Experiment infra_normalize_codex_assets_20260202 - Normalize Codex Assets

## Hypothesis
Normalizing skill files and output schema on runner avoids Codex CLI parsing failures caused by BOM or line endings.

## Change summary
Add a small normalization script and run it in agent workflows before Codex executes.

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
