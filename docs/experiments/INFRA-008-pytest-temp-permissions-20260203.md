# Experiment INFRA-008 - pytest temp dir on Windows

## Hypothesis
Setting TMP/TEMP/TMPDIR to a repo-local directory avoids permission errors when pytest creates temp dirs on Windows runners.

## Change summary
Update `scripts/agent/propose_pr.py` to set TMP/TEMP/TMPDIR to `tmp/pytest` before running CI.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 20

## Commands
- `scripts/ci.ps1`

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
