"""
raw JSONL（py32_fetcher出力）を PostgreSQL に投入するCLI。

例:
  python py64_analysis/scripts/load_raw_jsonl_to_db.py --input-dir data/raw/c2_2020_2023
  python py64_analysis/scripts/load_raw_jsonl_to_db.py --input-file data/raw/c2_2020_2023/RACE_20260105_123000.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]  # keiba/
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_import_path()
    from keiba.db.loader import load_all_jsonl_files, load_jsonl_file

    p = argparse.ArgumentParser(description="Load py32_fetcher JSONL into Postgres")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-dir", type=Path)
    g.add_argument("--input-file", type=Path)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--lock-timeout-seconds", type=float, default=0)
    p.add_argument("--lock-poll-seconds", type=float, default=0.25)
    p.add_argument("--no-lock", action="store_true")
    args = p.parse_args()

    def _run() -> None:
        if args.input_dir:
            if not args.input_dir.exists():
                raise SystemExit(f"input_dir not found: {args.input_dir}")
            load_all_jsonl_files(args.input_dir, batch_size=int(args.batch_size))
        else:
            if not args.input_file.exists():
                raise SystemExit(f"input_file not found: {args.input_file}")
            load_jsonl_file(args.input_file, batch_size=int(args.batch_size))

    if args.no_lock:
        print("DB write lock: DISABLED (--no-lock)")
        _run()
        return

    from keiba.utils.locks import db_write_lock, default_db_lock_path

    lock_path = default_db_lock_path()
    print(
        f"Acquiring DB write lock: {lock_path} "
        f"(timeout={args.lock_timeout_seconds}, poll={args.lock_poll_seconds})"
    )
    with db_write_lock(timeout_seconds=args.lock_timeout_seconds, poll_seconds=args.lock_poll_seconds):
        print("Acquired DB write lock")
        try:
            _run()
        finally:
            print("Released DB write lock")


if __name__ == "__main__":
    main()


