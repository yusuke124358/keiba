"""
Summarize candidate scarcity metrics from rolling holdout runs.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

WIN_RE = re.compile(r"^w(\d{3})_")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_windows(run_dir: Path) -> list[Path]:
    windows = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        if not WIN_RE.match(p.name):
            continue
        if not (p / "summary.json").exists():
            continue
        if not (p / "bets.csv").exists():
            continue
        windows.append(p)
    windows.sort(key=lambda p: p.name)
    return windows


def _date_range(start: str, end: str) -> list[datetime.date]:
    ds = datetime.strptime(start, "%Y-%m-%d").date()
    de = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    cur = ds
    while cur <= de:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _safe_int(val) -> Optional[int]:
    try:
        if val is None:
            return None
        return int(val)
    except Exception:
        return None


def _summarize_run(run_dir: Path) -> dict[str, Any] | None:
    windows = _parse_windows(run_dir)
    if not windows:
        return None

    rows = []
    for w in windows:
        summary = json.loads((w / "summary.json").read_text(encoding="utf-8"))
        test = summary.get("test") or {}
        test_start = test.get("start")
        test_end = test.get("end")
        if not test_start or not test_end:
            continue
        days = _date_range(test_start, test_end)
        total_days = len(days)
        if total_days <= 0:
            continue

        df = pd.read_csv(w / "bets.csv")
        if "asof_time" in df.columns:
            df["asof_time"] = pd.to_datetime(df["asof_time"], errors="coerce")
            df["asof_date"] = df["asof_time"].dt.date
        else:
            df["asof_date"] = pd.NaT

        days_with_bet = int(df["asof_date"].nunique()) if not df.empty else 0
        n_bets = int(len(df))
        rows.append(
            {
                "test_start": test_start,
                "test_end": test_end,
                "total_days": total_days,
                "days_with_bet": days_with_bet,
                "frac_days_any_bet": days_with_bet / total_days if total_days else None,
                "bets_per_day": n_bets / total_days if total_days else None,
                "n_bets": n_bets,
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    year = int(str(df["test_start"].iloc[0])[:4])
    return {
        "year": year,
        "n_windows": int(len(df)),
        "median_frac_days_any_bet": float(df["frac_days_any_bet"].median()),
        "median_bets_per_day": float(df["bets_per_day"].median()),
        "median_total_days": float(df["total_days"].median()),
        "median_days_with_bet": float(df["days_with_bet"].median()),
        "total_bets": int(df["n_bets"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize candidate scarcity for rolling runs")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--variants", type=str, nargs="*", default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    variants = manifest.get("variants", {})
    if not variants:
        raise SystemExit("No variants in manifest")

    target_variants = args.variants if args.variants else list(variants.keys())

    out_rows = []
    project_root = _project_root()

    for variant in target_variants:
        if variant not in variants:
            continue
        run_dirs = variants.get(variant, [])
        if not run_dirs:
            continue

        for run_dir in run_dirs:
            run_path = Path(run_dir)
            if not run_path.is_absolute():
                run_path = (project_root / run_path).resolve()
            if not run_path.exists():
                continue

            summary = _summarize_run(run_path)
            if not summary:
                continue
            out_rows.append(
                {
                    "year": summary["year"],
                    "variant": variant,
                    "group_dir": str(run_dir),
                    "n_windows": summary["n_windows"],
                    "median_frac_days_any_bet": summary["median_frac_days_any_bet"],
                    "median_bets_per_day": summary["median_bets_per_day"],
                    "median_total_days": summary["median_total_days"],
                    "median_days_with_bet": summary["median_days_with_bet"],
                    "total_bets": summary["total_bets"],
                }
            )

    df = pd.DataFrame(out_rows)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "candidate_scarcity.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
