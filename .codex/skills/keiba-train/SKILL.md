---
name: keiba-train
description: Train a win probability model and save artifacts.
---

# keiba-train

Prereqs:
- DB running and features available.
- py64_analysis/.venv installed.

Commands:
```bat
cd C:\Users\yyosh\keiba
py64_analysis\.venv\Scripts\activate
python -c "from keiba.modeling.train import train_model; from keiba.db.loader import get_session; from pathlib import Path; s=get_session(); model, metrics = train_model(s, train_start='2020-01-01', train_end='2023-12-31', model_path=Path('models/win_model.pkl')); s.close()"
```

Outputs:
- models/win_model.pkl
