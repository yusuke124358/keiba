# Experiment INFRA-020 - Scientist V2 Pipeline + Promotion Wiring

## Hypothesis
Adding a stage-chained Scientist V2 pipeline workflow and a promotion (PR creation) workflow improves autonomous throughput without breaking CI gates.

## Change summary
- Add a V2 pipeline workflow to run Stage1 -> Stage2 -> Holdout sequentially per shard.
- Add a V2 promotion workflow/script to recheck holdout on latest base and create PRs only for accepted seeds.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 400

## Commands
- `py64_analysis\\.venv\\Scripts\\python.exe -m pytest py64_analysis/tests`
- `py64_analysis\\.venv\\Scripts\\python.exe py64_analysis/scripts/check_system_status.py`

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
- Preferred ROI for decisions: pooled

## Artifacts
- metrics.json: N/A
- comparison.json: N/A
- report: N/A

## Decision
- status: pass
- next_action: merge

