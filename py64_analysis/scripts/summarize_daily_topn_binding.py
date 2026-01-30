"""
Summarize daily top-N binding from rolling holdout runs.
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

WIN_DIR_RE = re.compile(r"^w(\d{3})_")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_window_files(group_dir: Path) -> list[dict[str, Any]]:
    windows = []
    for p in group_dir.iterdir():
        if not p.is_dir():
            continue
        m = WIN_DIR_RE.match(p.name)
        if not m:
            continue
        sj = p / "summary.json"
        bc = p / "bets.csv"
        if not sj.exists() or not bc.exists():
            continue
        windows.append({"idx": int(m.group(1)), "name": p.name, "dir": p, "summary": sj, "bets": bc})
    windows.sort(key=lambda w: w["idx"])
    return windows


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _date_range(start: str, end: str) -> list[datetime]:
    ds = datetime.strptime(start, "%Y-%m-%d").date()
    de = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    cur = ds
    while cur <= de:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_float(val) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return None
        return float(val)
    except Exception:
        return None


def _infer_n(df: pd.DataFrame, default_n: int) -> Optional[int]:
    if "daily_top_n_n" in df.columns:
        vals = pd.to_numeric(df["daily_top_n_n"], errors="coerce").dropna()
        if not vals.empty:
            return int(vals.iloc[0])
    if default_n and int(default_n) > 0:
        return int(default_n)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize daily top-N binding from rolling group_dir")
    ap.add_argument("--group-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--design-max-idx", type=int, default=12)
    ap.add_argument("--default-n", type=int, default=0, help="Fallback N when daily_top_n_n is missing")
    args = ap.parse_args()

    group_dir = args.group_dir
    if not group_dir.is_absolute():
        group_dir = _project_root() / group_dir
    if not group_dir.exists():
        raise SystemExit(f"group_dir not found: {group_dir}")

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = _project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = _parse_window_files(group_dir)
    if not windows:
        raise SystemExit("No completed windows (summary.json + bets.csv) found under group_dir")

    rows = []
    for w in windows:
        summary = _load_summary(w["summary"])
        test_start = (summary.get("test") or {}).get("start")
        test_end = (summary.get("test") or {}).get("end")
        if not test_start or not test_end:
            continue

        idx = int(w["idx"])
        split = "design" if idx <= int(args.design_max_idx) else "eval"
        year = int(str(test_start)[:4])

        df = pd.read_csv(w["bets"])
        if "asof_time" in df.columns:
            df["asof_time"] = pd.to_datetime(df["asof_time"], errors="coerce")
            df["asof_date"] = df["asof_time"].dt.date
        else:
            df["asof_date"] = pd.NaT

        n = _infer_n(df, args.default_n)
        if n is None:
            n = 0

        cand_before_by_day = {}
        cand_after_by_day = {}
        if "daily_candidates_before" in df.columns:
            df["daily_candidates_before"] = _to_num(df["daily_candidates_before"])
            cand_before_by_day = df.groupby("asof_date")["daily_candidates_before"].max().to_dict()
        if "daily_candidates_after_min" in df.columns:
            df["daily_candidates_after_min"] = _to_num(df["daily_candidates_after_min"])
            cand_after_by_day = df.groupby("asof_date")["daily_candidates_after_min"].max().to_dict()
        if not cand_after_by_day:
            if cand_before_by_day:
                cand_after_by_day = cand_before_by_day
            else:
                cand_after_by_day = df.groupby("asof_date").size().to_dict()
        if not cand_before_by_day:
            cand_before_by_day = cand_after_by_day

        if "daily_selected" in df.columns:
            sel = _to_num(df["daily_selected"]).fillna(0)
            df["daily_selected_num"] = (sel > 0).astype(int)
            selected_by_day = df.groupby("asof_date")["daily_selected_num"].sum().to_dict()
        else:
            selected_by_day = df.groupby("asof_date").size().to_dict()

        days = _date_range(test_start, test_end)
        if not days:
            continue

        cand_before_counts = []
        cand_after_counts = []
        sel_counts = []
        for day in days:
            cand_before_counts.append(_safe_float(cand_before_by_day.get(day, 0.0)) or 0.0)
            cand_after_counts.append(_safe_float(cand_after_by_day.get(day, 0.0)) or 0.0)
            sel_counts.append(_safe_float(selected_by_day.get(day, 0.0)) or 0.0)

        cand_before_arr = np.asarray(cand_before_counts, dtype=float)
        cand_after_arr = np.asarray(cand_after_counts, dtype=float)
        sel_arr = np.asarray(sel_counts, dtype=float)
        mean_candidates_before = float(np.mean(cand_before_arr)) if cand_before_arr.size else None
        mean_candidates_after = float(np.mean(cand_after_arr)) if cand_after_arr.size else None
        frac_ge_n = float(np.mean(cand_after_arr >= float(n))) if cand_after_arr.size and n > 0 else None
        mean_selected = float(np.mean(sel_arr)) if sel_arr.size else None

        rows.append(
            {
                "window": w["name"],
                "year": year,
                "split": split,
                "test_start": test_start,
                "test_end": test_end,
                "n": int(n),
                "mean_candidates_per_day": mean_candidates_after,
                "mean_candidates_before_min_per_day": mean_candidates_before,
                "mean_candidates_after_min_per_day": mean_candidates_after,
                "frac_days_candidates_ge_n": frac_ge_n,
                "mean_selected_per_day": mean_selected,
            }
        )

    out_csv = out_dir / "daily_topn_binding.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")

    agg = (
        df_out.groupby(["year", "split", "n"], dropna=False)
        .agg(
            n_windows=("window", "nunique"),
            mean_candidates_per_day=("mean_candidates_per_day", "mean"),
            mean_candidates_before_min_per_day=("mean_candidates_before_min_per_day", "mean"),
            mean_candidates_after_min_per_day=("mean_candidates_after_min_per_day", "mean"),
            frac_days_candidates_ge_n=("frac_days_candidates_ge_n", "mean"),
            mean_selected_per_day=("mean_selected_per_day", "mean"),
        )
        .reset_index()
    )

    def _df_to_md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    out_md = out_dir / "daily_topn_binding.md"
    lines = [
        "# Daily top-N binding summary",
        "",
        f"- group_dir: {group_dir}",
        "",
        "## Aggregated",
        _df_to_md(agg),
        "",
        "## Per window",
        _df_to_md(df_out),
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
