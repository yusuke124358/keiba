# Experiment infra_codex_cli_flag_compat_20260202 - Codex CLI Flag Compatibility

## Hypothesis
Detecting supported `codex exec` flags at runtime avoids failures across CLI versions.

## Change summary
Probe `codex exec --help` and only pass optional flags that are supported. When output flags
are unavailable, capture stdout and write it to the expected output file.

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
