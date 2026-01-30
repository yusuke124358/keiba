from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "config" / "experiments" / "odds_window_s1s2_v0_base.yaml"
DEFAULT_S00_CONFIG = PROJECT_ROOT / "config" / "experiments" / "takeout_ev_margin_s00_eval.yaml"

DEFAULT_DATES = {
    "train_start": "2023-01-01",
    "train_end": "2023-02-28",
    "valid_start": "2023-03-01",
    "valid_end": "2023-03-15",
    "test_start": "2023-03-16",
    "test_end": "2023-03-31",
}


def _run_holdout(config_path: Path, out_dir: Path, dates: dict[str, str]) -> None:
    env = os.environ.copy()
    env["KEIBA_CONFIG_PATH"] = str(config_path)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "py64_analysis" / "scripts" / "run_holdout.py"),
        "--train-start",
        dates["train_start"],
        "--train-end",
        dates["train_end"],
        "--valid-start",
        dates["valid_start"],
        "--valid-end",
        dates["valid_end"],
        "--test-start",
        dates["test_start"],
        "--test-end",
        dates["test_end"],
        "--name",
        "audit_takeout_ev_margin_noop",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def _hash_bets(path: Path, keys: list[str]) -> tuple[int, str]:
    rows, _ = _read_rows(path)
    if not rows:
        return 0, "empty"
    rows_sorted = sorted(rows, key=lambda r: tuple(r.get(k, "") or "" for k in keys))
    h = hashlib.sha256()
    for row in rows_sorted:
        payload = "|".join(row.get(k, "") or "" for k in keys)
        h.update(payload.encode("utf-8"))
        h.update(b"\n")
    return len(rows_sorted), h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit takeout_ev_margin no-op behavior")
    ap.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    ap.add_argument("--s00-config", type=Path, default=DEFAULT_S00_CONFIG)
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "data" / "holdout_runs")
    ap.add_argument("--train-start", type=str, default=DEFAULT_DATES["train_start"])
    ap.add_argument("--train-end", type=str, default=DEFAULT_DATES["train_end"])
    ap.add_argument("--valid-start", type=str, default=DEFAULT_DATES["valid_start"])
    ap.add_argument("--valid-end", type=str, default=DEFAULT_DATES["valid_end"])
    ap.add_argument("--test-start", type=str, default=DEFAULT_DATES["test_start"])
    ap.add_argument("--test-end", type=str, default=DEFAULT_DATES["test_end"])
    args = ap.parse_args()

    dates = {
        "train_start": args.train_start,
        "train_end": args.train_end,
        "valid_start": args.valid_start,
        "valid_end": args.valid_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.out_root / f"audit_takeout_ev_margin_noop_{ts}"
    base_dir = root / "base"
    s00_dir = root / "s00"

    _run_holdout(Path(args.base_config), base_dir, dates)
    _run_holdout(Path(args.s00_config), s00_dir, dates)

    base_bets = base_dir / "bets.csv"
    s00_bets = s00_dir / "bets.csv"
    if not base_bets.exists() or not s00_bets.exists():
        raise SystemExit("bets.csv not found for audit runs")

    base_rows, base_fields = _read_rows(base_bets)
    s00_rows, s00_fields = _read_rows(s00_bets)

    preferred_keys = ["race_id", "horse_no", "asof_time", "stake", "payout", "profit"]
    keys = [k for k in preferred_keys if k in base_fields and k in s00_fields]
    if not keys:
        keys = sorted(set(base_fields) & set(s00_fields))

    base_count, base_hash = _hash_bets(base_bets, keys)
    s00_count, s00_hash = _hash_bets(s00_bets, keys)

    passed = (base_count == s00_count) and (base_hash == s00_hash)
    print(f"[noop] takeout_ev_margin_noop_pass={str(passed).lower()}")

    if not passed:
        print(f"base_count={base_count} base_hash={base_hash}")
        print(f"s00_count={s00_count} s00_hash={s00_hash}")
        sys.exit(1)


if __name__ == "__main__":
    main()
