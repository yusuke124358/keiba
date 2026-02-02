---
name: keiba-data-update
description: Fetch stored JRA-VAN data and load JSONL into Postgres.
---

# keiba-data-update

Prereqs:
- JV-Link installed and JRA-VAN credentials configured.
- DB running (docker compose up -d db).
- Virtualenvs: py32_fetcher/.venv and py64_analysis/.venv.

Commands (fetch):
```bat
cd C:\Users\yyosh\keiba
py32_fetcher\.venv\Scripts\activate
python py32_fetcher/run_fetch.py stored RACE
python py32_fetcher/run_fetch.py stored 0B41
python py32_fetcher/run_fetch.py stored 0B42
```

Commands (load JSONL into DB):
```bat
py64_analysis\.venv\Scripts\activate
python -c "from keiba.config import get_data_path; from keiba.db.loader import load_all_jsonl_files; load_all_jsonl_files(get_data_path('data/raw'))"
```

Outputs:
- data/raw/*.jsonl
- Postgres tables (fact_race, odds_ts_*)
