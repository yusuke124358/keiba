@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

rem ---
rem 1/9（金）に実行: 次週開催（1/10〜1/12）のRA取得→DB投入
rem 実行タイミング: 1/9の任意の時刻（前日までに実行すればOK）
rem ---

cd /d %~dp0\..

set RACE_DATE_MIN=20260110
set FROM_TIME=20260101000000

echo ========================================
echo [Step 1/3] Fetch RACE data (stored)
echo ========================================
py -3.11-32 py32_fetcher\run_fetch.py stored RACE ^
  --option 2 ^
  --from-time %FROM_TIME% ^
  --min-race-date %RACE_DATE_MIN% ^
  --output-dir data\raw

if errorlevel 1 (
  echo ERROR: RACE fetch failed
  exit /b 1
)

echo.
echo ========================================
echo [Step 2/3] Find latest RACE JSONL and load to DB
echo ========================================

rem 最新のRACE JSONLを探す
for /f "delims=" %%F in ('dir /b /o-d data\raw\RACE_*.jsonl 2^>nul') do (
  set LATEST_RACE_JSONL=data\raw\%%F
  goto :found
)
echo ERROR: No RACE JSONL files found
exit /b 1

:found
echo Loading: %LATEST_RACE_JSONL%

set PYTHONPATH=%CD%\py64_analysis\src
python -c "from pathlib import Path; from keiba.db.loader import load_jsonl_file; import sys; fp = Path(r'%LATEST_RACE_JSONL%'); print(f'Loading: {fp}'); res = load_jsonl_file(fp, batch_size=2000); print(f'Result: {res}'); sys.exit(0 if res.get('errors', 0) == 0 else 1)"
set PYTHONPATH=

if errorlevel 1 (
  echo ERROR: DB load failed
  exit /b 1
)

echo.
echo ========================================
echo [Step 3/3] Verify race dates in DB
echo ========================================

set PYTHONPATH=%CD%\py64_analysis\src
python -c "from datetime import date; from sqlalchemy import text; from keiba.db.loader import get_session; s = get_session(); try: rows = s.execute(text('SELECT date, COUNT(*) AS n_races, MIN(start_time) AS first_start, MAX(start_time) AS last_start FROM fact_race WHERE date BETWEEN DATE ''2026-01-10'' AND DATE ''2026-01-12'' GROUP BY date ORDER BY date'), {}).fetchall(); print('Race dates summary:'); [print(f'  {r[0]}: {r[1]} races, {r[2]} - {r[3]}') for r in rows]; all_ok = all(r[1] > 0 for r in rows) and len(rows) >= 3; print(f'\nStatus: {\"OK\" if all_ok else \"WARNING - check dates\"}'); import sys; sys.exit(0 if all_ok else 1); finally: s.close()"
set PYTHONPATH=

if errorlevel 1 (
  echo WARNING: Some race dates may be missing. Check the output above.
  exit /b 1
)

echo.
echo ========================================
echo SUCCESS: Race data prepared for 2026-01-10 to 2026-01-12
echo ========================================
echo Next steps:
echo   1. Set up Task Scheduler for 1/10-1/12:
echo      - scripts\collect_exotics_due_buy.cmd  (09:40-16:30, every minute)
echo      - scripts\collect_exotics_due_close.cmd (09:40-16:30, every minute)
echo   2. On 1/10 morning, verify that due race_ids are generated correctly
echo.

endlocal







