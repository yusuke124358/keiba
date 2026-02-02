# Experiment <id> - <title>

## Hypothesis
<write the hypothesis>

## Change summary
<small, focused change description>

## Risk
- Experiment type: experiment|infra
- risk_level: <low|medium|high>
- max_diff_size: <int>

## Commands
- `<command 1>`
- `<command 2>`

## Metrics (required)
- ROI: <value>
- Total stake: <value>
- n_bets: <value>
- Test period: YYYY-MM-DD to YYYY-MM-DD
- Max drawdown: <value>
- ROI definition: ROI = profit / stake, profit = return - stake.
- Rolling: yes|no
- Design window: <if rolling, else N/A>
- Eval window: <if rolling, else N/A>
- Paired delta vs baseline: <if rolling, else N/A>
- Pooled vs step14 sign mismatch: yes|no
- Preferred ROI for decisions: step14|pooled

## Artifacts
- metrics.json: <path>
- comparison.json: <path>
- report: <path>

## Decision
- status: pass|fail|inconclusive
- next_action: merge|iterate|abandon
