"""
Analyze ROI by takeout/overround buckets from rolling holdout bets.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text


WIN_RE = re.compile(r"^w(\d{3})_")


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _infer_w_idx_offset(group_dir: Path) -> int:
    name = group_dir.name.lower()
    if "w013_022" in name or "w013-022" in name:
        return 12
    return 0


def _parse_window_idx(name: str) -> Optional[int]:
    m = WIN_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _split_from_idx(idx: int) -> str:
    return "design" if idx <= 12 else "eval"


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fetch_overround(session, race_id: str, asof_time: pd.Timestamp) -> tuple[Optional[float], Optional[float]]:
    if asof_time is None or pd.isna(asof_time):
        return None, None
    race_id = str(race_id)
    query = text(
        """
        WITH snap AS (
            SELECT MAX(asof_time) AS t_snap
            FROM odds_ts_win
            WHERE race_id = CAST(:race_id AS text)
              AND asof_time <= :asof_time
              AND odds > 0
        )
        SELECT
            (SELECT SUM(1.0 / odds)
             FROM odds_ts_win
             WHERE race_id = CAST(:race_id AS text)
               AND asof_time = (SELECT t_snap FROM snap)
               AND odds > 0) AS overround_sum_inv
        """
    )
    row = session.execute(query, {"race_id": race_id, "asof_time": asof_time}).fetchone()
    if not row:
        return None, None
    overround = row[0]
    try:
        overround = float(overround) if overround is not None else None
    except Exception:
        overround = None
    if overround is None or not math.isfinite(overround) or overround <= 0:
        return None, None
    takeout = 1.0 - (1.0 / overround)
    return float(overround), float(takeout)


def _assign_buckets(values: pd.Series, quantiles: list[float]) -> pd.Series:
    values = values.astype(float)
    try:
        return pd.qcut(values, q=quantiles, duplicates="drop")
    except Exception:
        if values.empty:
            return pd.Series([], dtype="object")
        vmin = float(values.min())
        vmax = float(values.max())
        return pd.Series([pd.Interval(vmin, vmax, closed="right")] * len(values), index=values.index)


def _bucket_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.copy()
    df = df[pd.notna(df[metric])]
    if df.empty:
        return pd.DataFrame(columns=[
            "metric",
            "bucket",
            "min_value",
            "max_value",
            "n_races",
            "n_bets",
            "total_stake",
            "total_profit",
            "roi",
            "median_maxdd",
        ])

    buckets = _assign_buckets(df[metric], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    df["bucket"] = buckets.astype(str)

    rows = []
    for bucket, sub in df.groupby("bucket", dropna=False):
        total_stake = float(sub["total_stake"].sum())
        total_profit = float(sub["total_profit"].sum())
        roi = (total_profit / total_stake) if total_stake > 0 else None
        rows.append({
            "metric": metric,
            "bucket": bucket,
            "min_value": float(sub[metric].min()),
            "max_value": float(sub[metric].max()),
            "n_races": int(len(sub)),
            "n_bets": int(sub["n_bets"].sum()),
            "total_stake": total_stake,
            "total_profit": total_profit,
            "roi": roi,
            "median_maxdd": None,
        })
    return pd.DataFrame(rows)


def _best_worst(table: pd.DataFrame) -> tuple[Optional[dict], Optional[dict]]:
    if table.empty or "roi" not in table.columns:
        return None, None
    work = table[pd.notna(table["roi"])].copy()
    if work.empty:
        return None, None
    best = work.sort_values("roi", ascending=False).iloc[0].to_dict()
    worst = work.sort_values("roi", ascending=True).iloc[0].to_dict()
    return best, worst


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze takeout/overround buckets from rolling holdout group dir.")
    ap.add_argument("--group-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["design", "eval", "all"], default="all")
    args = ap.parse_args()

    group_dir = args.group_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    offset = _infer_w_idx_offset(group_dir)
    frames = []
    for p in group_dir.iterdir():
        if not p.is_dir():
            continue
        idx_raw = _parse_window_idx(p.name)
        if idx_raw is None:
            continue
        idx = idx_raw + offset
        split = _split_from_idx(idx)
        if args.split != "all" and split != args.split:
            continue

        bets_path = p / "bets.csv"
        if not bets_path.exists():
            continue
        df = pd.read_csv(bets_path)
        if df.empty:
            continue
        df["window_idx"] = idx
        df["split"] = split
        frames.append(df)

    if not frames:
        raise SystemExit("No bets.csv found for the requested split.")

    bets = pd.concat(frames, ignore_index=True)
    bets["stake"] = _to_num(bets.get("stake"))
    bets["profit"] = _to_num(bets.get("profit"))
    bets["asof_time"] = pd.to_datetime(bets.get("asof_time"), errors="coerce")
    if "overround_sum_inv" in bets.columns:
        bets["overround_sum_inv"] = _to_num(bets.get("overround_sum_inv"))
    else:
        bets["overround_sum_inv"] = np.nan
    if "takeout_implied" in bets.columns:
        bets["takeout_implied"] = _to_num(bets.get("takeout_implied"))
    else:
        bets["takeout_implied"] = np.nan

    race_meta = bets.groupby("race_id", as_index=False).agg(
        asof_time=("asof_time", "first"),
        overround_sum_inv=("overround_sum_inv", "first"),
        takeout_implied=("takeout_implied", "first"),
    )

    missing_mask = race_meta["overround_sum_inv"].isna() | race_meta["takeout_implied"].isna()
    if missing_mask.any():
        _ensure_import_path()
        from keiba.db.loader import get_session

        session = get_session()
        for _, row in race_meta[missing_mask].iterrows():
            overround, takeout = _fetch_overround(session, row["race_id"], row["asof_time"])
            race_meta.loc[race_meta["race_id"] == row["race_id"], "overround_sum_inv"] = overround
            race_meta.loc[race_meta["race_id"] == row["race_id"], "takeout_implied"] = takeout

    race_bets = bets.groupby("race_id", as_index=False).agg(
        n_bets=("stake", "count"),
        total_stake=("stake", "sum"),
        total_profit=("profit", "sum"),
    )
    race_level = race_bets.merge(race_meta, on="race_id", how="left")

    takeout_table = _bucket_table(race_level, "takeout_implied")
    overround_table = _bucket_table(race_level, "overround_sum_inv")
    buckets = pd.concat([takeout_table, overround_table], ignore_index=True)
    buckets.to_csv(out_dir / "buckets.csv", index=False, encoding="utf-8")

    best_takeout, worst_takeout = _best_worst(takeout_table)
    best_over, worst_over = _best_worst(overround_table)

    report_lines = [
        f"# Takeout/Overround buckets ({args.split})",
        "",
        f"- races: {len(race_level)}",
        f"- bets: {int(race_level['n_bets'].sum())}",
        f"- stake: {float(race_level['total_stake'].sum())}",
        f"- profit: {float(race_level['total_profit'].sum())}",
        "",
        "## Buckets (takeout_implied)",
        takeout_table.to_string(index=False) if not takeout_table.empty else "_no data_",
        "",
        "## Buckets (overround_sum_inv)",
        overround_table.to_string(index=False) if not overround_table.empty else "_no data_",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    if worst_takeout is not None:
        print(
            f"[takeout-bucket] split={args.split} metric=takeout worst_bucket={worst_takeout['bucket']} "
            f"roi={worst_takeout['roi']} stake={worst_takeout['total_stake']}"
        )
    if best_takeout is not None:
        print(
            f"[takeout-bucket] split={args.split} metric=takeout best_bucket={best_takeout['bucket']} "
            f"roi={best_takeout['roi']} stake={best_takeout['total_stake']}"
        )
    if worst_over is not None:
        print(
            f"[takeout-bucket] split={args.split} metric=overround worst_bucket={worst_over['bucket']} "
            f"roi={worst_over['roi']} stake={worst_over['total_stake']}"
        )
    if best_over is not None:
        print(
            f"[takeout-bucket] split={args.split} metric=overround best_bucket={best_over['bucket']} "
            f"roi={best_over['roi']} stake={best_over['total_stake']}"
        )


if __name__ == "__main__":
    main()
