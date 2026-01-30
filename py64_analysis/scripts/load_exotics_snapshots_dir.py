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
    args = p.parse_args()

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


if __name__ == "__main__":
    main()








