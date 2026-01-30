@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

rem ---
rem 三連系 forward 収集（close）
rem 仕様: 発走-1分で収集（締切直前相当）
rem 最初は 0B35 のみ推奨。問題なければ 0B35,0B36 に変更。
rem ---

cd /d %~dp0\..

set TAG=close
set SPECS=0B35
set T_MINUS_MIN=1
set WINDOW_SEC=90

set RACE_IDS_FILE=data\state\race_ids_due_close.txt

echo [1/2] generate race_id list...
set PYTHONPATH=%CD%\py64_analysis\src
python py64_analysis\scripts\generate_due_race_ids.py --tag %TAG% --t-minus-min %T_MINUS_MIN% --window-sec %WINDOW_SEC% --output %RACE_IDS_FILE%
set PYTHONPATH=

rem race_id が空なら終了
for %%A in (%RACE_IDS_FILE%) do set SIZE=%%~zA
if "%SIZE%"=="0" (
  echo due races: 0 (skip)
  exit /b 0
)

echo [2/2] collect exotics snapshots...
py -3.11-32 py32_fetcher\collect_exotics_snapshots.py ^
  --tag %TAG% ^
  --race-ids-file %RACE_IDS_FILE% ^
  --specs %SPECS% ^
  --resume ^
  --chunk-size 25 ^
  --sleep-sec 0.2

endlocal

