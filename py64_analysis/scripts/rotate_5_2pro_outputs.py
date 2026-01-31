"""
Rotate *.zip files under an output directory into old/<timestamp>/.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def rotate_zips(output_dir: Path, *, ts: str | None = None, dry_run: bool = False) -> Path:
    if ts is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(output_dir)
    old_dir = output_dir / "old" / ts
    zips = sorted(output_dir.glob("*.zip"))

    if not zips:
        # 退避対象なしでも old/<ts> は作らない（空ディレクトリ増殖を避ける）
        return old_dir

    if not dry_run:
        old_dir.mkdir(parents=True, exist_ok=True)

    for fp in zips:
        dst = old_dir / fp.name
        if dry_run:
            continue
        # 既に同名がある場合は上書きしない（安全側）
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        shutil.move(str(fp), str(dst))

    return old_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Rotate *.zip under output dir into old/<timestamp>/")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory containing *.zip")
    p.add_argument("--timestamp", type=str, default=None, help="timestamp folder name (default: now)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    if not out.exists():
        raise SystemExit(f"output_dir not found: {out}")

    old_dir = rotate_zips(out, ts=args.timestamp, dry_run=bool(args.dry_run))
    print("output_dir:", out)
    print("rotated_to:", old_dir)


if __name__ == "__main__":
    main()


