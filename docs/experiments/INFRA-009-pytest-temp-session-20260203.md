# Experiment INFRA-009 - pytest temp session dir

## Hypothesis
Using a per-run pytest temp directory (PYTEST_TMPDIR) avoids Windows permission issues with shared temp roots.

## Change summary
Update `scripts/agent/propose_pr.py` to create a unique temp directory for each CI run and set TMP/TEMP/TMPDIR/PYTEST_TMPDIR.

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
