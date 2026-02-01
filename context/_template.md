# Project context template

## 1) Objective
- <primary objective: ROI / turnover / calibration>

## 2) Current Baseline
- Path: <baselines/.../metrics.json>
- Data cutoff: <YYYY-MM-DD>

## 3) Data & Split
- Train / valid / test ranges:
  - <train>
  - <valid>
  - <test>
- Rolling conditions: <window/step/gap>
- Comparability rule: <single diff + same windows>

## 4) Execution Rules
- run_dir required for eval scripts.
- DB writes require lock (see docs/ops/db_write_lock.md).
- Baselines change only with Commander approval.

## 5) Active Experiments
- <name>: <hypothesis> (owner)

## 6) Gates / Definition of Done
- G0: <data leakage / split validity>
- G1: <metrics.json + compare>
- G2: <ROI/stake/n_bets/maxDD reported>
- G3: <step14 vs pooled sign check>
- G4: <reviewer SHIP: yes>

## 7) Next Actions
- <next inbox task(s)>
