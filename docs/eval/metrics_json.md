# metrics.json v0.1

This document describes the normalized metrics file used for automated gates.

## Source mapping
- Holdout: `data/holdout_runs/<run_dir>/summary.json`
  - ROI/profit/stake/maxDD: `backtest.roi`, `backtest.total_profit`, `backtest.total_stake`, `backtest.max_drawdown`
  - Split periods: `train.start/end`, `valid.start/end`, `test.start/end`
  - Pred quality: `pred_quality.logloss_*`, `pred_quality.brier_*`
- Rolling: `data/holdout_runs/<group_dir>/summary.csv`
  - ROI/profit/stake/maxDD: `roi`, `total_profit`, `total_stake`, `max_drawdown` (per-window)
  - Aggregation: sum profit/stake/bets, ROI from totals, maxDD = max(window maxDD)
  - Pred quality: `pred_logloss_*`, `pred_brier_*` (normalized to pred_quality.*)

## Schema (top-level)
- `schema_version`: "0.1"
- `run_kind`: "holdout" | "rolling_holdout"
- `run_dir`: relative path to run dir
- `name`, `generated_at`
- `git_commit`
- `config_used_path`, `config_hash_sha256`
- `data_cutoff`:
  - `db_max_race_date` (preferred)
  - `raw_max_mtime` (fallback)
- `split`: train/valid/test with `start`, `end`, `n_races`
- `universe`: track_codes, require_results, require_ts_win, exclude_race_ids_count, exclude_race_ids_hash
- `betting`: buy_t_minus_minutes, closing_odds_multiplier*, slippage_summary, slippage_table, market_prob_method, market_prob_mode
- `prob_variant_used`: "p_hat" | "p_blend" | "unknown"
- `backtest`: n_bets, roi, total_stake, total_profit, max_drawdown
- `pred_quality`: logloss_* / brier_* (market/blend/calibrated)
- `step14`: optional (if a step14 summary is present)
- `incomparable_reasons`: optional warnings from extraction

## Baseline registration
Use `py64_analysis/scripts/set_baseline.py` to register a baseline:

```
py64_analysis\.venv\Scripts\python.exe py64_analysis/scripts/set_baseline.py ^
  --scenario rolling_default --run-dir data/holdout_runs/<group_dir>
```

This writes:
- `baselines/<scenario>/metrics.json`
- `baselines/<scenario>/baseline.json`

## Comparison (gates)
Use `py64_analysis/scripts/compare_metrics_json.py`:

```
py64_analysis\.venv\Scripts\python.exe py64_analysis/scripts/compare_metrics_json.py ^
  --baseline baselines/<scenario>/metrics.json ^
  --candidate data/holdout_runs/<run_dir>/metrics.json ^
  --gates config/gates/default.yaml
```

Exit codes:
- `0`: pass
- `1`: fail (gates)
- `2`: incomparable (definitions differ)
- `3`: error

Comparison prerequisites (incomparable if any mismatch):
- `schema_version` and `run_kind` must match
- `split.train/valid/test` start/end must match
- `betting.buy_t_minus_minutes` and `prob_variant_used` must match
- `data_cutoff.db_max_race_date` must exist on both sides and match (if present on either side)
- `data_cutoff.raw_max_mtime` must exist on both sides and match (only used when db_max_race_date is absent on both sides)

`comparison.json` includes gate inputs (`baseline`, `candidate`, and threshold/tolerance) for auditability.
