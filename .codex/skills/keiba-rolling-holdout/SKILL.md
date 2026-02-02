---
name: keiba-rolling-holdout
description: Run rolling holdout windows and generate summary.csv.
---

# keiba-rolling-holdout

Prereqs:
- DB running and populated.
- py64_analysis/.venv installed.

Commands:
```bat
cd C:\Users\yyosh\keiba
py64_analysis\.venv\Scripts\activate
python py64_analysis/scripts/run_rolling_holdout.py --name rolling60_s14_lb365_v14 --test-range-start 2025-10-01 --test-range-end 2025-12-28 --test-window-days 60 --step-days 14 --gap-days 0 --train-lookback-days 365 --valid-window-days 14 --estimate-closing-mult --closing-mult-quantile 0.30
```

Outputs:
- data/holdout_runs/<name_timestamp>/summary.csv
- data/holdout_runs/<name_timestamp>/rolling_index.json
