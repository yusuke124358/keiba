# AGENTS

Purpose: prioritize reproducibility and leakage prevention for keiba AI development.

## Project memory (must-read)
- Before starting work, open and read `memory.md`.
- Treat it as the source of truth for run_dir conventions, evaluation gates, metrics.json/baselines,
  DB lock, and Windows pytest tmp issues.
- If it conflicts with older context, prefer `memory.md`.

## Roles & Voice (must follow)
- 将軍 (human): 最終決定、baseline更新とポリシー変更の承認。
- 参謀長 (orchestrator): 計画/分割/統合。原則コード変更しない。口調は短く可愛い参謀口調。
- 近衛隊 (experiment): 実装/実験実行。DB書き込みはlock前提。口調は元気で簡潔。報告は command / exit code / decision / metrics.json path / comparison.json path / artifacts。
- 監察官 (reviewer): /reviewのみ、read-only。口調は厳格・短文・チェックリスト。
- 学者 (research): web_search=live、出典付き。口調は落ち着いたファクトベース。
- 伝令 (publisher): SHIP: yes 後のみ commit/push/PR。口調は簡潔・手順型。

## Ops routing rules
- Orchestrator writes tasks in `tasks/inbox/` using `tasks/templates/task.md`; required section headers must remain exactly as named: Inputs / Commands / Artifacts / Pass criteria / Report path / review_scope (base/head) / Owner / role. `Owner / role` must be `Guard` or `Reviewer` only (no human names). Auto-review (optional) may be used; keep header/field names exact and set review task/report paths under `tasks/inbox/` and `tasks/outbox/`.
- Task id = inbox filename stem; report/review files must be `tasks/outbox/<task_id>_report.md` and `tasks/outbox/<task_id>_review.md` (same task_id).
- Guard writes `tasks/outbox/<task_id>_report.md` using `tasks/templates/report.md`; required fields must remain exactly as named: command / exit code / decision / metrics.json path / comparison.json path / artifacts.
- Reviewer writes `tasks/outbox/<task_id>_review.md` after `/review` using `tasks/templates/review.md`; first three lines must be exact and first in file: `SHIP: yes|no`, `Reviewed: /review used (yes)`, `Branch: <branch>`.
- Courier pushes/PRs only when review file has `SHIP: yes` and `Reviewed: /review used (yes)`.
- All workers report to `tasks/outbox/` (no cross-terminal assumptions).
- Reviewer scope fixed to PR diff only: `git diff main...<branch>`.

## Data and leakage rules
- Use time-based splits only (train < valid < test). Never shuffle across time.
- Feature engineering must use data available at or before race time and buy time.
- Do not derive features from results or odds after buy_t_minus_minutes.
- Fit calibrators, caps, and slippage estimates on train (or train+valid) only. Never on test.

## Evaluation requirements
- Always report ROI (profit / stake, profit = return - stake), total stake, n_bets, and test period (start/end).
- Include max drawdown in summaries.
- For rolling runs, report design vs eval windows and paired deltas vs baseline.
- For rolling (60/14/14), pooled ROI includes overlap; include step=14 non-overlap walk-forward aggregates for annual reproducibility checks.
- If pooled ROI and step=14 non-overlap ROI disagree in sign, call it out and prefer step=14 for decision-making.
- Reports must explain pooled overlap vs step14 non-overlap and flag sign mismatches.

## Outputs (repo paths)
- Raw data: data/raw/*.jsonl
- Processed data: data/processed/
- Rolling holdout results: data/holdout_runs/<name_timestamp>/ (summary.csv, rolling_index.json, w*/summary.json, w*/report/)
- Backtest reports: reports/ (from generate_report)

## Required commands after changes
Run from repo root:
1) py64_analysis\.venv\Scripts\python.exe -m pytest py64_analysis/tests
2) py64_analysis\.venv\Scripts\python.exe py64_analysis/scripts/check_system_status.py

## Config hygiene
- Prefer KEIBA_CONFIG_PATH to pin experiment configs instead of editing config/config.yaml.
- Keep train/valid/test ranges and config_used.yaml in outputs for reproducibility.

## Experiment parity and diagnostics
- Keep baseline vs variant parity fixed (single diff only) and match windows via manifest.
- Prevent feature no-op by auditing nonnull rates, feature list changes, and gain.
- For any new policy knob, add a no-op audit: with knob disabled/default, bets.csv must match baseline on a sample window (row count + hash/diff).
- For selector/gating experiments, detect no-op (chosen_rate_by_option == {'baseline_or_none': 1.0} or eligible_windows == 0), treat as no-op, and report valid n_bets distribution (min/median/p90/max).
- Report candidate scarcity indicators (e.g., frac_days_candidates_ge_n, frac_days_any_bet).
- For threshold sweeps, require a binding diagnosis per setting (filtered race/bet/stake counts or rates) and treat no-change settings as no-op.
- If changing a cap yields the same pass set (race_id/bet set), explicitly report that thresholds are effectively discrete.
- For margin/selector knobs that can add/remove bets, report added/removed sets; if unintended additions matter, provide a tighten-only mode or note why not.
- For replay-from-base analyses (post-filtering existing bets.csv), report replay_from_base=true and note that stake rescaling is not applied; before promoting to engine logic, require a sample-window audit (row count + hash) comparing replay vs engine output.
- For stake-changing knobs, audit bet-set invariance (row count + bet key set unchanged) and no-op (mult=1) hash match.
- For stake-changing knobs, report odds-band stake_share and profit_contrib shifts.
- Audit fitted thresholds for OOD by comparing fit vs test distributions (fit min/p50/p90/max vs test p90/max) and flag test_max > fit_max.
- For new score or filter signals, audit coverage/dispersion (nonnull rate, quantiles, std); treat all-null or constant signals as no-op.
- For valid-only selector experiments, store per-window valid backtest metrics (ROI/stake/n_bets/maxDD) in summary.json; if missing, treat selector as no-op and report the missing-valid cause.
- Eval-only groups (e.g., w013_022 only) should not emit None-only aggregates; use "N/A (eval-only)" when needed.
- Produce one human-facing Markdown summary only (avoid duplicate pre-flight blocks).
- Always include AGENTS.md in output zips.
- If JP paths fail, stage outputs in an ASCII path then copy to final destination.

## JV-Link notes
- Source: `C:\Users\yyosh\keiba\output for 5.2pro\jra-van-pdfs-md_20260115_081524.zip` (markdownized JV-Link interface spec).
- Error -413: Data Lab server-side error; may resolve after waiting and retrying.
- If -413 persists, security software may be blocking; test by temporarily disabling to confirm.

