---
name: keiba-backtest
description: Run backtest and generate reports.
---

# keiba-backtest

Prereqs:
- DB running and features available.
- A trained model artifact.
- py64_analysis/.venv installed.

Commands:
```bat
cd C:\Users\yyosh\keiba
py64_analysis\.venv\Scripts\activate
python -c "from keiba.backtest.engine import run_backtest; from keiba.backtest.report import generate_report; from keiba.db.loader import get_session; from keiba.modeling.train import WinProbabilityModel; from pathlib import Path; s=get_session(); model = WinProbabilityModel.load(Path('models/win_model.pkl')); result = run_backtest(s, start_date='2024-01-01', end_date='2024-12-31', model=model); generate_report(result, Path('reports')); s.close()"
```

Outputs:
- reports/ (backtest.md, equity_curve.png, reliability.png)
