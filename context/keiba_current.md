# Keiba current context

## 1) Objective
- Maintain stable ROI with reproducible evaluation and leakage prevention.

## 2) Current Baseline
- Path: baselines/<scenario>/metrics.json (TBD)
- Data cutoff: TBD

## 3) Data & Split
- Time-based splits only (train < valid < test).
- Rolling: window=60, step=14, gap=0 (when applicable).
- Comparability: single diff only; match windows via manifest.

## 4) Execution Rules
- All eval scripts require explicit `--run-dir`.
- DB write scripts require lock (docs/ops/db_write_lock.md).
- Baselines changes only with Commander approval; set_baseline is restricted.

## 5) Active Experiments
- None listed. See tasks/inbox for current assignments.

## 6) Gates / Definition of Done
- G0: no leakage, correct split order.
- G1: metrics.json created and compare completed.
- G2: ROI/stake/n_bets/maxDD reported with test period.
- G3: step14 vs pooled ROI sign check noted.
- G4: reviewer SHIP: yes before publish.

## 7) Next Actions
- Orchestrator issues tasks in tasks/inbox.
