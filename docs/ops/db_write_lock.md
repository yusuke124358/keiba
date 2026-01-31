# DB write lock

## Purpose
- Prevent concurrent DB writes from multiple processes/worktrees.
- Ensure backfill/load scripts do not corrupt shared DB state.

## Default lock location
- `${TEMP}/keiba_locks/db_write.lock` on Windows
- `/tmp/keiba_locks/db_write.lock` on Unix-like systems

## Override paths
- `KEIBA_LOCK_DIR`: directory for lock files
- `KEIBA_DB_LOCK_PATH`: full lock file path (highest priority)

## CLI options (for DB write scripts)
- `--lock-timeout-seconds` (float, default 0)
- `--lock-poll-seconds` (float, default 0.25)
- `--no-lock` (avoid; emergency only)

## Failure behavior
- If lock is held, scripts exit with a non-zero code and message:
  - "DB write lock is held by another process. Retry later or set --lock-timeout-seconds / KEIBA_DB_LOCK_PATH ..."

## Target scripts
- `py64_analysis/scripts/load_raw_jsonl_to_db.py`
- `py64_analysis/scripts/backfill_pace_from_jsonl.py`
- `py64_analysis/scripts/backfill_pace_from_jvd.py`
- `py64_analysis/scripts/reparse_hr_from_raw_jsonl.py`
- `py64_analysis/scripts/load_exotics_snapshots_dir.py`
- `py64_analysis/scripts/dedupe_features.py`
- `py64_analysis/scripts/regenerate_features_c4.py`
