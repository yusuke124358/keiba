---
name: keiba-features
description: Build features for specific race_ids from Postgres.
---

# keiba-features

Prereqs:
- DB running and populated.
- py64_analysis/.venv installed.

Commands:
```bat
cd C:\Users\yyosh\keiba
py64_analysis\.venv\Scripts\activate
python -c "from keiba.features.build_features import build_features; from keiba.db.loader import get_session; s=get_session(); build_features(s, ['2024122801010101']); s.close()"
```

Outputs:
- Postgres table: features
