# memory

This file is a short, stable reference for the keiba repo. Keep it concise.

## Repo layout / entry points
- `py32_fetcher/`: JV-Link data fetchers (32-bit).
- `py64_analysis/`: feature build, training, evaluation, backtests (64-bit).
- Main evaluation entry points:
  - `py64_analysis/scripts/run_holdout.py`
  - `py64_analysis/scripts/run_rolling_holdout.py`
  - DB ingest/backfill: `py64_analysis/scripts/load_raw_jsonl_to_db.py`, `backfill_*`, `reparse_*`

## run_dir / out_dir conventions (PR-B)
- Evaluation scripts must accept `--run-dir` explicitly (no implicit latest).
- Default outputs go under:
  `run_dir/analysis/<script_slug>/<timestamp>/`
- Fixed paths like `output_staging` or `output for 5.2pro` are not used by scripts.
- Reference: `docs/eval/run_dir_io_conventions.md`

## Config conventions (PR-C)
- Config resolution order:
  1) `KEIBA_CONFIG_PATH`
  2) `--config`
  3) default `config/config.yaml`
- `config_used.yaml` and `config_origin.json` are written to each run_dir.
- Reference: `docs/eval/config_conventions.md`

## Evaluation facts (PR-A)
- Holdout summary: `data/holdout_runs/<run_dir>/summary.json`
  - ROI/profit/stake/maxDD at `backtest.roi`, `backtest.total_profit`,
    `backtest.total_stake`, `backtest.max_drawdown`
- Rolling summary: `data/holdout_runs/<group_dir>/summary.csv`
  - Per-window columns include `roi`, `total_profit`, `total_stake`,
    `max_drawdown`, `n_bets`, `pred_logloss_*`, `pred_brier_*`
- Normalized metrics: `metrics.json` (v0.1) in each run_dir
  - Baseline pointer: `baselines/<scenario>/baseline.json` + `metrics.json`
  - Compare tool: `py64_analysis/scripts/compare_metrics_json.py`
  - Reference: `docs/eval/metrics_json.md`

## DB write lock (PR-D)
- DB write scripts acquire a file lock by default.
- Default lock path: `%TEMP%/keiba_locks/db_write.lock`
- Override with env: `KEIBA_DB_LOCK_PATH` or `KEIBA_LOCK_DIR`
- CLI flags on DB-write scripts:
  - `--lock-timeout-seconds`, `--lock-poll-seconds`, `--no-lock`
- Reference: `docs/ops/db_write_lock.md`

## Windows pytest tmp/permission issues
- Some old `py64_analysis/tests/tmp*` dirs may be unreadable.
- `py64_analysis/pyproject.toml` sets `norecursedirs = ["tmp*"]` to avoid
  test collection errors. Do not rely on those tmp dirs for tests.

## Update policy
- `memory.md` is updated only by the orchestrator or human reviewer.
