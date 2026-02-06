# Experiment INFRA-019 - Skip local main merge before publishing backlog PRs

## Hypothesis
Attempting to `git merge main` into an experiment branch inside `run_backlog_batch.py` can frequently
conflict when `main` advances during long experiments, causing the backlog run to crash and block
autonomous progress. Skipping that local merge (and aborting in-progress git ops during cleanup)
keeps the backlog runner resilient while still allowing PRs to be created and reviewed/merged.

## Change summary
- `scripts/agent/run_backlog_batch.py`: abort in-progress git operations in `ensure_clean()` so
  retries can recover from interrupted merges/rebases.
- `scripts/agent/run_backlog_batch.py`: stop merging `main` into the experiment branch before PR
  publication. If the PR has merge conflicts, GitHub will report it and we can resolve separately.

## Risk
- Experiment type: infra
- risk_level: low
- max_diff_size: 80

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
- Preferred ROI for decisions: pooled

## Artifacts
- metrics.json: N/A
- comparison.json: N/A
- report: N/A

## Decision
- status: pass
- next_action: merge

