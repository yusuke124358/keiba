"""
Summarize non-overlap walk-forward segments from rolling run dirs (step-days slicing).
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

WIN_RE = re.compile(r"^w\d{3}_(\d{8})_(\d{8})$")

DATE_COL_CANDIDATES = [
    ("race_date", "date"),
    ("event_date", "date"),
    ("date", "date"),
    ("settle_date", "date"),
    ("asof_date", "date"),
    ("asof_time", "datetime"),
    ("buy_time", "datetime"),
    ("bet_time", "datetime"),
]

STAKE_COL_CANDIDATES = [
    "stake",
    "stake_yen",
    "stake_amount",
    "bet_amount",
    "stake_raw",
]

RETURN_COL_CANDIDATES = [
    "return",
    "return_yen",
    "payout",
    "payout_yen",
]

PROFIT_COL_CANDIDATES = [
    "profit",
    "net_profit",
    "pnl",
]


@dataclass(frozen=True)
class GroupSpec:
    year: int
    strategy: str
    design_dir: Path
    eval_dir: Path


def _parse_window_dates(name: str) -> Optional[tuple[date, date]]:
    m = WIN_RE.match(name)
    if not m:
        return None
    start = datetime.strptime(m.group(1), "%Y%m%d").date()
    end = datetime.strptime(m.group(2), "%Y%m%d").date()
    return start, end


def _detect_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def _detect_date_column(columns: list[str]) -> tuple[str, str]:
    for col, kind in DATE_COL_CANDIDATES:
        if col in columns:
            return col, kind
    raise ValueError(f"Could not detect date column from candidates: {DATE_COL_CANDIDATES}")


def _read_bets_subset(path: Path) -> tuple[pd.DataFrame, str, str, Optional[str]]:
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)

    date_col, date_kind = _detect_date_column(cols)
    stake_col = _detect_column(cols, STAKE_COL_CANDIDATES)
    if stake_col is None:
        raise ValueError(f"Could not detect stake column from candidates: {STAKE_COL_CANDIDATES}")

    return_col = _detect_column(cols, RETURN_COL_CANDIDATES)
    profit_col = _detect_column(cols, PROFIT_COL_CANDIDATES)
    if return_col is None and profit_col is None:
        raise ValueError(
            f"Could not detect return/profit columns from candidates: "
            f"{RETURN_COL_CANDIDATES} / {PROFIT_COL_CANDIDATES}"
        )

    usecols = [date_col, stake_col]
    if return_col is not None:
        usecols.append(return_col)
    if profit_col is not None and profit_col not in usecols:
        usecols.append(profit_col)

    df = pd.read_csv(path, usecols=usecols)
    if date_kind == "date":
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    if df[date_col].isna().all():
        raise ValueError(f"Date column {date_col} could not be parsed to dates.")

    return df, date_col, stake_col, return_col or profit_col


def _segment_date_range(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _segment_metrics(
    df: pd.DataFrame,
    date_col: str,
    stake_col: str,
    value_col: str,
    segment_start: date,
    segment_end: date,
    initial_bankroll: float,
) -> tuple[pd.DataFrame, dict]:
    mask = (df[date_col] >= segment_start) & (df[date_col] <= segment_end)
    seg = df.loc[mask].copy()
    seg["__date__"] = seg[date_col]
    seg["__stake__"] = pd.to_numeric(seg[stake_col], errors="coerce").fillna(0.0)

    if value_col in RETURN_COL_CANDIDATES:
        seg["__profit__"] = pd.to_numeric(seg[value_col], errors="coerce").fillna(0.0) - seg["__stake__"]
    else:
        seg["__profit__"] = pd.to_numeric(seg[value_col], errors="coerce").fillna(0.0)

    daily = (
        seg.groupby("__date__", dropna=False)
        .agg(stake=("__stake__", "sum"), profit=("__profit__", "sum"), n_bets=("__stake__", "size"))
        .reset_index()
        .rename(columns={"__date__": "date"})
    )

    full_dates = pd.DataFrame({"date": _segment_date_range(segment_start, segment_end)})
    daily = full_dates.merge(daily, on="date", how="left").fillna({"stake": 0.0, "profit": 0.0, "n_bets": 0})
    daily["n_bets"] = daily["n_bets"].astype(int)
    daily = daily.sort_values("date").reset_index(drop=True)

    daily["cum_profit"] = daily["profit"].cumsum()
    daily["equity"] = initial_bankroll + daily["cum_profit"]
    daily["peak"] = daily["equity"].cummax()
    daily["drawdown_pct"] = (daily["peak"] - daily["equity"]) / daily["peak"]
    daily["drawdown_yen"] = daily["peak"] - daily["equity"]

    total_stake = float(daily["stake"].sum())
    total_profit = float(daily["profit"].sum())
    n_bets = int(daily["n_bets"].sum())
    roi = (total_profit / total_stake) if total_stake > 0 else 0.0
    days_total = int(len(daily))
    days_with_bet = int((daily["stake"] > 0).sum())
    frac_days_any_bet = (days_with_bet / days_total) if days_total > 0 else 0.0
    bets_per_day = (n_bets / days_total) if days_total > 0 else 0.0
    max_dd = float(daily["drawdown_pct"].max()) if not daily.empty else 0.0

    metrics = {
        "segment_start": segment_start,
        "segment_end": segment_end,
        "n_bets": n_bets,
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": roi,
        "max_drawdown": max_dd,
        "total_days": days_total,
        "days_with_bet": days_with_bet,
        "frac_days_any_bet": frac_days_any_bet,
        "bets_per_day": bets_per_day,
    }
    return daily, metrics


def _collect_segments(
    run_dir: Path,
    year: int,
    strategy: str,
    split: str,
    step_days: int,
    initial_bankroll: float,
) -> tuple[list[dict], list[dict]]:
    windows = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = _parse_window_dates(p.name)
        if not parsed:
            continue
        windows.append((parsed[0], parsed[1], p))

    windows.sort(key=lambda x: x[0])
    if not windows:
        raise SystemExit(f"No windows found under {run_dir}")

    segments = []
    daily_rows = []
    for test_start, test_end, window_dir in windows:
        segment_start = test_start
        segment_end = test_start + timedelta(days=step_days - 1)
        if segment_end > test_end:
            segment_end = test_end

        bets_path = window_dir / "bets.csv"
        if not bets_path.exists():
            raise SystemExit(f"bets.csv not found: {bets_path}")

        df, date_col, stake_col, value_col = _read_bets_subset(bets_path)
        daily, metrics = _segment_metrics(
            df,
            date_col=date_col,
            stake_col=stake_col,
            value_col=value_col,
            segment_start=segment_start,
            segment_end=segment_end,
            initial_bankroll=initial_bankroll,
        )

        metrics.update(
            {
                "year": year,
                "strategy": strategy,
                "split": split,
                "window_name": window_dir.name,
            }
        )
        segments.append(metrics)

        for _, row in daily.iterrows():
            daily_rows.append(
                {
                    "year": year,
                    "strategy": strategy,
                    "split": split,
                    "date": row["date"],
                    "stake": float(row["stake"]),
                    "profit": float(row["profit"]),
                    "n_bets": int(row["n_bets"]),
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "window_name": window_dir.name,
                }
            )

    return segments, daily_rows


def _daily_aggregate(daily_rows: list[dict], initial_bankroll: float) -> pd.DataFrame:
    if not daily_rows:
        return pd.DataFrame()
    df = pd.DataFrame(daily_rows)
    grouped = (
        df.groupby(["year", "strategy", "split", "date"], dropna=False)
        .agg(
            stake=("stake", "sum"),
            profit=("profit", "sum"),
            n_bets=("n_bets", "sum"),
            segment_overlap_count=("window_name", "nunique"),
        )
        .reset_index()
        .sort_values(["year", "strategy", "split", "date"])
    )

    rows = []
    for (year, strategy, split), sub in grouped.groupby(["year", "strategy", "split"], dropna=False):
        sub = sub.sort_values("date").copy()
        sub["cum_profit"] = sub["profit"].cumsum()
        sub["equity"] = initial_bankroll + sub["cum_profit"]
        sub["peak"] = sub["equity"].cummax()
        sub["drawdown_pct"] = (sub["peak"] - sub["equity"]) / sub["peak"]
        sub["drawdown_yen"] = sub["peak"] - sub["equity"]
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _summary_from_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    rows = []
    for (year, strategy, split), sub in daily_df.groupby(["year", "strategy", "split"], dropna=False):
        total_stake = float(sub["stake"].sum())
        total_profit = float(sub["profit"].sum())
        n_bets = int(sub["n_bets"].sum())
        roi = (total_profit / total_stake) if total_stake > 0 else 0.0
        total_days = int(len(sub))
        days_with_bet = int((sub["stake"] > 0).sum())
        frac_days_any_bet = (days_with_bet / total_days) if total_days > 0 else 0.0
        bets_per_day = (n_bets / total_days) if total_days > 0 else 0.0
        max_dd = float(sub["drawdown_pct"].max()) if not sub.empty else 0.0
        rows.append(
            {
                "year": int(year),
                "strategy": strategy,
                "split": split,
                "n_bets": n_bets,
                "total_stake": total_stake,
                "total_profit": total_profit,
                "roi": roi,
                "max_drawdown": max_dd,
                "total_days": total_days,
                "days_with_bet": days_with_bet,
                "frac_days_any_bet": frac_days_any_bet,
                "bets_per_day": bets_per_day,
            }
        )
    return pd.DataFrame(rows)


def _df_to_md(df: pd.DataFrame, digits: int = 4) -> str:
    def fmt(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{digits}f}"
        return str(val)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize step-days walk-forward segments from rolling run dirs")
    ap.add_argument("--year", action="append", type=int, required=True)
    ap.add_argument("--strategy", action="append", required=True)
    ap.add_argument("--design-dir", action="append", required=True)
    ap.add_argument("--eval-dir", action="append", required=True)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--initial-bankroll", type=float, default=1_000_000.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if not (len(args.year) == len(args.strategy) == len(args.design_dir) == len(args.eval_dir)):
        raise SystemExit("Mismatched --year/--strategy/--design-dir/--eval-dir counts.")

    specs = []
    for year, strategy, design_dir, eval_dir in zip(
        args.year, args.strategy, args.design_dir, args.eval_dir, strict=False
    ):
        specs.append(
            GroupSpec(
                year=int(year),
                strategy=str(strategy),
                design_dir=Path(design_dir),
                eval_dir=Path(eval_dir),
            )
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    segments_all = []
    daily_rows_all = []

    for spec in specs:
        for split, run_dir in [("design", spec.design_dir), ("eval", spec.eval_dir)]:
            if not run_dir.exists():
                raise SystemExit(f"run_dir not found: {run_dir}")
            segments, daily_rows = _collect_segments(
                run_dir=run_dir,
                year=spec.year,
                strategy=spec.strategy,
                split=split,
                step_days=args.step_days,
                initial_bankroll=args.initial_bankroll,
            )
            segments_all.extend(segments)
            daily_rows_all.extend(daily_rows)

    segments_df = pd.DataFrame(segments_all)
    segments_df.to_csv(out_dir / "walkforward_step14_segments.csv", index=False)

    daily_df = _daily_aggregate(daily_rows_all, initial_bankroll=args.initial_bankroll)
    daily_df.to_csv(out_dir / "walkforward_step14_daily.csv", index=False)

    summary_df = _summary_from_daily(daily_df)
    summary_df.to_csv(out_dir / "walkforward_step14_summary.csv", index=False)

    report_lines = []
    report_lines.append("# Walk-forward step=14 summary (from rolling windows)")
    report_lines.append("")
    report_lines.append("- ROI = profit / stake; profit = return - stake. ROI set to 0 when stake=0.")
    report_lines.append("- bets_per_day = total_bets / total_days within the 14-day segments.")
    report_lines.append("- max_drawdown is computed from daily equity (initial bankroll: {:.0f}).".format(args.initial_bankroll))
    report_lines.append("")
    report_lines.append("## Command example")
    report_lines.append("```bat")
    report_lines.append(
        "py64_analysis\\.venv\\Scripts\\python.exe py64_analysis/scripts/summarize_walkforward_step14_from_rolling.py "
        "--step-days 14 --initial-bankroll 1000000 --out-dir <OUT_DIR> "
        "--year 2024 --strategy baseline --design-dir <DESIGN_DIR> --eval-dir <EVAL_DIR> "
        "--year 2024 --strategy q95 --design-dir <DESIGN_DIR> --eval-dir <EVAL_DIR> "
        "--year 2024 --strategy q97 --design-dir <DESIGN_DIR> --eval-dir <EVAL_DIR> "
        "--year 2025 --strategy baseline --design-dir <DESIGN_DIR> --eval-dir <EVAL_DIR> "
        "--year 2025 --strategy q95 --design-dir <DESIGN_DIR> --eval-dir <EVAL_DIR> "
        "--year 2025 --strategy q97 --design-dir <DESIGN_DIR> --eval-dir <EVAL_DIR>"
    )
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## Summary (year x strategy x split)")
    if summary_df.empty:
        report_lines.append("_no summary rows_")
    else:
        report_lines.append(_df_to_md(summary_df))

    report_lines.append("")
    report_lines.append("## q95 vs q97 differences (eval)")
    diff_lines = []
    for year in sorted(summary_df["year"].unique()) if not summary_df.empty else []:
        eval_rows = summary_df[(summary_df["year"] == year) & (summary_df["split"] == "eval")]
        row_q95 = eval_rows[eval_rows["strategy"] == "q95"]
        row_q97 = eval_rows[eval_rows["strategy"] == "q97"]
        if row_q95.empty or row_q97.empty:
            continue
        r95 = row_q95.iloc[0]
        r97 = row_q97.iloc[0]
        diff_lines.append(
            f"- {year} eval: q97-q95 "
            f"delta_roi={r97['roi'] - r95['roi']:.4f} | "
            f"delta_profit={r97['total_profit'] - r95['total_profit']:.0f} | "
            f"delta_stake={r97['total_stake'] - r95['total_stake']:.0f} | "
            f"delta_frac_days_any_bet={r97['frac_days_any_bet'] - r95['frac_days_any_bet']:.4f}"
        )
    report_lines.extend(diff_lines if diff_lines else ["_no eval comparison rows_"])

    report_lines.append("")
    report_lines.append("## Notes")
    report_lines.append("- segment_overlap_count in daily output indicates overlapping segments (should be 1 for non-overlap).")

    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[walkforward] step_days={args.step_days} | years={','.join(str(s.year) for s in specs)} | "
          f"strategies={','.join(sorted(set(s.strategy for s in specs)))}")


if __name__ == "__main__":
    main()
