#!/usr/bin/env python3
"""
Reset a backlog YAML so all items become runnable again.

This is intended for "rerun everything with the current system" while preserving
the previous backlog state by writing an archive copy.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML (expected mapping): {path}")
    return data


def dump_yaml(data: dict) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=False,
    )


def reset_item(item: dict) -> None:
    item["status"] = "todo"
    for key in ("branch", "picked_at", "run_id", "decision"):
        item.pop(key, None)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backlog", default="experiments/backlog.yml")
    p.add_argument(
        "--archive-dir",
        default="experiments/backlog_archives",
        help="Directory to write the pre-reset archive copy into.",
    )
    p.add_argument(
        "--archive-name",
        default="",
        help="Optional explicit archive filename. Default: backlog_<ts>.yml",
    )
    args = p.parse_args()

    root = repo_root()
    backlog_path = root / args.backlog
    if not backlog_path.exists():
        raise RuntimeError(f"Backlog file not found: {backlog_path}")

    data = load_yaml(backlog_path)
    items = data.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Backlog YAML missing 'items' list.")

    archive_dir = root / args.archive_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    name = args.archive_name.strip() or f"backlog_{utc_timestamp()}.yml"
    archive_path = archive_dir / name
    if archive_path.exists():
        raise RuntimeError(f"Archive path already exists: {archive_path}")

    # Archive before mutation.
    archive_path.write_text(backlog_path.read_text(encoding="utf-8"), encoding="utf-8")

    for item in items:
        if isinstance(item, dict):
            reset_item(item)

    backlog_path.write_text(dump_yaml(data), encoding="utf-8")
    print(f"Archived: {archive_path.relative_to(root).as_posix()}")
    print(f"Reset:    {backlog_path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
