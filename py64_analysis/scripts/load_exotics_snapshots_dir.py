"""
data/raw/exotics_snapshots/ 配下に溜まった 0B35/0B36(JSONL) をDBに投入する。

想定ディレクトリ:
  data/raw/exotics_snapshots/<YYYYMMDD>/<tag>/<data_spec>/*.jsonl

使い方（例）:
  cd py64_analysis
  PYTHONPATH=src python scripts/load_exotics_snapshots_dir.py --input-dir ../data/raw/exotics_snapshots --date 20260106
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]  # keiba/
    src = project_root / "py64_analysis" / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_import_path()
    from keiba.db.loader import load_jsonl_file

    p = argparse.ArgumentParser(description="Load exotics snapshots JSONL (0B35/0B36) into DB")
    p.add_argument("--input-dir", type=Path, required=True, help="data/raw/exotics_snapshots ディレクトリ")
    p.add_argument("--date", type=str, default="", help="YYYYMMDD（未指定なら全日）")
    p.add_argument("--tag", choices=["buy", "close"], default="", help="buy/close（未指定なら両方）")
    p.add_argument("--spec", choices=["0B35", "0B36"], default="", help="spec（未指定なら両方）")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--lock-timeout-seconds", type=float, default=0)
    p.add_argument("--lock-poll-seconds", type=float, default=0.25)
    p.add_argument("--no-lock", action="store_true")
    args = p.parse_args()

    if not args.no_lock:
        if args.lock_timeout_seconds < 0:
            raise SystemExit("--lock-timeout-seconds must be >= 0")
        if args.lock_poll_seconds <= 0:
            raise SystemExit("--lock-poll-seconds must be > 0")

    def _run() -> None:
        base = args.input_dir
        if not base.is_absolute():
            base = (Path(__file__).resolve().parents[2] / base).resolve()
        if not base.exists():
            raise SystemExit(f"input-dir not found: {base}")
    
        # 対象ファイルを列挙
        globs: list[str] = []
        if args.date:
            globs.append(args.date)
        else:
            globs.append("*")
    
        tags = [args.tag] if args.tag else ["buy", "close"]
        specs = [args.spec] if args.spec else ["0B35", "0B36"]
    
        files: list[Path] = []
        for d in globs:
            for tag in tags:
                for spec in specs:
                    files.extend(sorted((base / d / tag / spec).glob("*.jsonl")))
    
        print(f"input_dir={base}")
        print(f"files={len(files)}")
        if not files:
            return
    
        total = {"o5": 0, "o6": 0, "errors": 0, "files": 0}
        for i, fp in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {fp}")
            res = load_jsonl_file(fp, batch_size=int(args.batch_size))
            total["o5"] += int(res.get("o5", 0) or 0)
            total["o6"] += int(res.get("o6", 0) or 0)
            total["errors"] += int(res.get("errors", 0) or 0)
            total["files"] += 1
        print("total:", total)
    

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








