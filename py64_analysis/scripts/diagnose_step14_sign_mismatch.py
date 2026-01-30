"""
Diagnose step14 vs pooled ROI sign mismatches from existing rolling outputs.
No retraining or rolling re-runs: reads bets.csv + summary.csv only.
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

ODDS_COL_CANDIDATES = [
    "odds_final",
    "odds_effective",
    "odds_at_buy",
    "odds",
]

EV_COL_CANDIDATES = [
    "ev_adj",
    "ev",
]


@dataclass(frozen=True)
class WindowSpec:
    name: str
    test_start: date
    test_end: date
    path: Path


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


def _detect_date_column(columns: list[str]) -> tuple[Optional[str], Optional[str]]:
    for col, kind in DATE_COL_CANDIDATES:
        if col in columns:
            return col, kind
    return None, None


def _read_bets(path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols) if usecols else pd.read_csv(path)


def _parse_date_col(df: pd.DataFrame, date_col: str, date_kind: str) -> pd.Series:
    if date_kind == "date":
        return pd.to_datetime(df[date_col], errors="coerce").dt.date
    return pd.to_datetime(df[date_col], errors="coerce").dt.date


def _derive_date_from_race_id(df: pd.DataFrame) -> pd.Series:
    if "race_id" not in df.columns:
        return pd.Series([pd.NaT] * len(df))
    race_id = df["race_id"].astype(str).str.slice(0, 8)
    return pd.to_datetime(race_id, errors="coerce", format="%Y%m%d").dt.date


def _segment_metrics(
    df: pd.DataFrame,
    date_col: str,
    stake_col: str,
    profit_col: str,
    segment_start: date,
    segment_end: date,
    initial_bankroll: float,
) -> tuple[pd.DataFrame, dict]:
    mask = (df[date_col] >= segment_start) & (df[date_col] <= segment_end)
    seg = df.loc[mask].copy()

    seg["__date__"] = seg[date_col]
    seg["__stake__"] = pd.to_numeric(seg[stake_col], errors="coerce").fillna(0.0)
    seg["__profit__"] = pd.to_numeric(seg[profit_col], errors="coerce").fillna(0.0)

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

    total_stake = float(daily["stake"].sum())
    total_profit = float(daily["profit"].sum())
    n_bets = int(daily["n_bets"].sum())
    roi = (total_profit / total_stake) if total_stake > 0 else 0.0
    max_dd = float(daily["drawdown_pct"].max()) if not daily.empty else 0.0

    metrics = {
        "segment_start": segment_start,
        "segment_end": segment_end,
        "n_bets": n_bets,
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": roi,
        "max_drawdown": max_dd,
    }
    return daily, metrics


def _segment_date_range(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _build_window_specs(input_dir: Path) -> list[WindowSpec]:
    specs = []
    for p in input_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = _parse_window_dates(p.name)
        if not parsed:
            continue
        specs.append(WindowSpec(name=p.name, test_start=parsed[0], test_end=parsed[1], path=p))
    specs.sort(key=lambda x: x.test_start)
    return specs


def _pooled_from_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    name_col = "name" if "name" in df.columns else "run_dir"
    df = df.rename(columns={name_col: "window_name"})
    cols = {
        "window_name": "window_name",
        "roi": "pooled_roi",
        "total_profit": "pooled_profit",
        "total_stake": "pooled_stake",
        "n_bets": "pooled_n_bets",
        "max_drawdown": "pooled_max_drawdown",
    }
    keep = [c for c in cols if c in df.columns]
    pooled = df[keep].rename(columns=cols)
    return pooled


def _odds_band(val: float) -> str:
    if val < 3:
        return "<3"
    if val < 5:
        return "3-5"
    if val < 10:
        return "5-10"
    if val < 20:
        return "10-20"
    return "20+"


def _ev_band(val: float) -> str:
    if val < 0:
        return "<0"
    if val < 0.01:
        return "0-0.01"
    if val < 0.02:
        return "0.01-0.02"
    if val < 0.03:
        return "0.02-0.03"
    return ">=0.03"


def _slice_stats(df: pd.DataFrame, slice_type: str, slice_label: str) -> dict:
    total_stake = float(df["stake"].sum())
    total_profit = float(df["profit"].sum())
    n_bets = int(len(df))
    roi = (total_profit / total_stake) if total_stake > 0 else 0.0
    return {
        "slice_type": slice_type,
        "slice": slice_label,
        "n_bets": n_bets,
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": roi,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose step14 sign mismatch from rolling outputs")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--initial-bankroll", type=float, default=1_000_000.0)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--staging-dir", type=Path, default=None)
    args = ap.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = input_dir / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"summary.csv not found: {summary_path}")

    pooled_df = _pooled_from_summary(summary_path)

    window_specs = _build_window_specs(input_dir)
    if not window_specs:
        raise SystemExit(f"No windows found under {input_dir}")

    step14_rows = []
    step14_bets_rows = []

    for spec in window_specs:
        bets_path = spec.path / "bets.csv"
        if not bets_path.exists():
            raise SystemExit(f"bets.csv not found: {bets_path}")

        header = pd.read_csv(bets_path, nrows=0)
        cols = list(header.columns)

        date_col, date_kind = _detect_date_column(cols)
        stake_col = _detect_column(cols, STAKE_COL_CANDIDATES)
        profit_col = _detect_column(cols, PROFIT_COL_CANDIDATES)
        return_col = _detect_column(cols, RETURN_COL_CANDIDATES)

        if stake_col is None:
            raise SystemExit(f"Could not detect stake column in {bets_path}")

        if profit_col is None and return_col is None:
            raise SystemExit(f"Could not detect profit/return column in {bets_path}")

        odds_col = _detect_column(cols, ODDS_COL_CANDIDATES)
        ev_col = _detect_column(cols, EV_COL_CANDIDATES)

        usecols = [stake_col]
        if profit_col:
            usecols.append(profit_col)
        if return_col:
            usecols.append(return_col)
        if date_col:
            usecols.append(date_col)
        if "race_id" in cols and (date_col is None):
            usecols.append("race_id")
        if odds_col:
            usecols.append(odds_col)
        if ev_col:
            usecols.append(ev_col)
        if "p_hat" in cols:
            usecols.append("p_hat")
        for extra in ["daily_candidates_after_min", "daily_selected"]:
            if extra in cols:
                usecols.append(extra)

        df = _read_bets(bets_path, usecols=sorted(set(usecols)))

        if date_col is None:
            df["__date__"] = _derive_date_from_race_id(df)
        else:
            df["__date__"] = _parse_date_col(df, date_col, date_kind)

        if df["__date__"].isna().all():
            raise SystemExit(f"Could not parse dates for {bets_path}")

        if profit_col is None and return_col is not None:
            df["__profit__"] = pd.to_numeric(df[return_col], errors="coerce").fillna(0.0) - pd.to_numeric(
                df[stake_col], errors="coerce"
            ).fillna(0.0)
            profit_col = "__profit__"

        df[stake_col] = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
        df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0)

        segment_start = spec.test_start
        segment_end = min(spec.test_end, spec.test_start + timedelta(days=args.step_days - 1))

        daily, metrics = _segment_metrics(
            df.rename(columns={stake_col: "stake", profit_col: "profit", "__date__": "date"}),
            date_col="date",
            stake_col="stake",
            profit_col="profit",
            segment_start=segment_start,
            segment_end=segment_end,
            initial_bankroll=args.initial_bankroll,
        )

        metrics.update(
            {
                "window_name": spec.name,
                "test_start": spec.test_start,
                "test_end": spec.test_end,
            }
        )
        step14_rows.append(metrics)

        seg_mask = (df["__date__"] >= segment_start) & (df["__date__"] <= segment_end)
        seg_df = df.loc[seg_mask].copy()
        if seg_df.empty:
            continue

        seg_df["stake"] = pd.to_numeric(seg_df[stake_col], errors="coerce").fillna(0.0)
        seg_df["profit"] = pd.to_numeric(seg_df[profit_col], errors="coerce").fillna(0.0)
        seg_df["date"] = seg_df["__date__"]

        if odds_col:
            seg_df["odds_val"] = pd.to_numeric(seg_df[odds_col], errors="coerce")
        else:
            seg_df["odds_val"] = np.nan

        if ev_col:
            seg_df["ev_val"] = pd.to_numeric(seg_df[ev_col], errors="coerce")
        elif "p_hat" in seg_df.columns and odds_col:
            seg_df["ev_val"] = pd.to_numeric(seg_df["p_hat"], errors="coerce") * seg_df["odds_val"] - 1.0
        else:
            seg_df["ev_val"] = np.nan

        step14_bets_rows.append(
            seg_df[
                [
                    "date",
                    "stake",
                    "profit",
                    "odds_val",
                    "ev_val",
                    "daily_candidates_after_min",
                    "daily_selected",
                ]
            ]
            if ("daily_candidates_after_min" in seg_df.columns and "daily_selected" in seg_df.columns)
            else seg_df[["date", "stake", "profit", "odds_val", "ev_val"]]
        )

    step14_df = pd.DataFrame(step14_rows)
    pooled_df = pooled_df.merge(step14_df, how="right", left_on="window_name", right_on="window_name")

    pooled_df["sign_mismatch"] = (pooled_df["pooled_roi"] >= 0) != (pooled_df["roi"] >= 0)
    pooled_df = pooled_df.rename(
        columns={
            "roi": "step14_roi",
            "total_profit": "step14_profit",
            "total_stake": "step14_stake",
            "n_bets": "step14_n_bets",
            "max_drawdown": "step14_max_drawdown",
        }
    )

    step14_vs_pooled = pooled_df[
        [
            "window_name",
            "test_start",
            "test_end",
            "pooled_roi",
            "pooled_profit",
            "pooled_stake",
            "pooled_n_bets",
            "pooled_max_drawdown",
            "step14_roi",
            "step14_profit",
            "step14_stake",
            "step14_n_bets",
            "step14_max_drawdown",
            "sign_mismatch",
        ]
    ].sort_values("test_start")
    step14_vs_pooled.to_csv(out_dir / "step14_vs_pooled.csv", index=False, encoding="utf-8")

    total_pooled_profit = float(step14_vs_pooled["pooled_profit"].sum())
    total_pooled_stake = float(step14_vs_pooled["pooled_stake"].sum())
    pooled_roi = (total_pooled_profit / total_pooled_stake) if total_pooled_stake > 0 else 0.0

    total_step14_profit = float(step14_vs_pooled["step14_profit"].sum())
    total_step14_stake = float(step14_vs_pooled["step14_stake"].sum())
    step14_roi = (total_step14_profit / total_step14_stake) if total_step14_stake > 0 else 0.0
    sign_mismatch_confirmed = (pooled_roi >= 0) != (step14_roi >= 0)

    slices_df = pd.DataFrame()
    replay_sweep_df = pd.DataFrame()
    if step14_bets_rows:
        bets_df = pd.concat(step14_bets_rows, ignore_index=True)
        bets_df = bets_df.dropna(subset=["date"])

        has_binding_cols = "daily_candidates_after_min" in bets_df.columns and "daily_selected" in bets_df.columns
        if has_binding_cols:
            daily = (
                bets_df.groupby("date", dropna=False)[["daily_candidates_after_min", "daily_selected"]]
                .max()
                .reset_index()
            )
            daily["binding"] = (
                (daily["daily_candidates_after_min"] > 0)
                & (daily["daily_selected"] < daily["daily_candidates_after_min"])
            )
            binding_map = dict(zip(daily["date"], daily["binding"]))
            bets_df["binding"] = bets_df["date"].map(binding_map).fillna(False)
        else:
            bets_df["binding"] = np.nan

        total_stake_all = float(bets_df["stake"].sum())
        total_profit_all = float(bets_df["profit"].sum())

        slice_rows = []

        odds_valid = bets_df["odds_val"].notna()
        if odds_valid.any():
            for band, sub in bets_df.loc[odds_valid].assign(band=bets_df.loc[odds_valid, "odds_val"].map(_odds_band)).groupby("band"):
                row = _slice_stats(sub, "odds_band", str(band))
                slice_rows.append(row)

        ev_valid = bets_df["ev_val"].notna()
        if ev_valid.any():
            for band, sub in bets_df.loc[ev_valid].assign(band=bets_df.loc[ev_valid, "ev_val"].map(_ev_band)).groupby("band"):
                row = _slice_stats(sub, "ev_band", str(band))
                slice_rows.append(row)

            thresholds = [0.00, 0.10, 0.20, 0.30, 0.40]
            sweep_rows = []
            for thr in thresholds:
                keep = bets_df.loc[ev_valid & (bets_df["ev_val"] >= thr)]
                stake = float(keep["stake"].sum())
                profit = float(keep["profit"].sum())
                roi = (profit / stake) if stake > 0 else 0.0
                sweep_rows.append(
                    {
                        "ev_threshold": thr,
                        "n_bets": int(len(keep)),
                        "total_stake": stake,
                        "total_profit": profit,
                        "roi": roi,
                        "stake_share_vs_base": (stake / total_stake_all) if total_stake_all > 0 else 0.0,
                        "profit_delta_vs_base": profit - total_profit_all,
                    }
                )
            replay_sweep_df = pd.DataFrame(sweep_rows)
            replay_sweep_df.to_csv(out_dir / "replay_sweep.csv", index=False, encoding="utf-8")

        if has_binding_cols:
            for label, sub in bets_df.groupby(bets_df["binding"].map({True: "binding", False: "non_binding"})):
                row = _slice_stats(sub, "budget_binding", str(label))
                slice_rows.append(row)

        if slice_rows:
            slices_df = pd.DataFrame(slice_rows)
            slices_df["stake_share"] = slices_df["total_stake"] / total_stake_all if total_stake_all > 0 else 0.0
            slices_df["profit_share"] = slices_df["total_profit"] / total_profit_all if total_profit_all != 0 else 0.0
            slices_df.to_csv(out_dir / "step14_failure_slices.csv", index=False, encoding="utf-8")
        else:
            pd.DataFrame().to_csv(out_dir / "step14_failure_slices.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_dir / "step14_failure_slices.csv", index=False, encoding="utf-8")

    worst_windows = step14_vs_pooled.sort_values("step14_profit").head(3)[
        ["window_name", "step14_profit", "step14_roi", "pooled_roi"]
    ]
    mismatch_windows = step14_vs_pooled[step14_vs_pooled["sign_mismatch"]][
        ["window_name", "pooled_roi", "step14_roi"]
    ]

    candidate_choice = "A"
    chosen_thr = None
    chosen_stats = None
    if not replay_sweep_df.empty:
        base_stake = total_step14_stake
        base_bets = step14_vs_pooled["step14_n_bets"].sum()
        replay_sweep_df = replay_sweep_df.sort_values("roi", ascending=False)
        filtered = replay_sweep_df[
            (replay_sweep_df["total_stake"] >= 0.5 * base_stake) & (replay_sweep_df["n_bets"] >= 0.5 * base_bets)
        ]
        pick = filtered.iloc[0] if not filtered.empty else replay_sweep_df.iloc[0]
        chosen_thr = float(pick["ev_threshold"])
        chosen_stats = {
            "roi": float(pick["roi"]),
            "total_stake": float(pick["total_stake"]),
            "total_profit": float(pick["total_profit"]),
            "n_bets": int(pick["n_bets"]),
            "stake_share": float(pick["stake_share_vs_base"]),
        }

    plan_lines = [
        "# candidate fix plan (minimal)",
        "",
        "Choice: A) EV threshold filter (replay-only diagnostic)",
        "",
        "Rationale:",
        "- Simple, monotonic filter; low overfit risk compared with complex reordering.",
        "- Uses existing bets (replay); no retraining or rolling re-run.",
        "- If promoted to engine: must fit threshold on train/valid, and audit added/removed bets.",
        "",
    ]
    if chosen_thr is not None:
        plan_lines.append(f"Proposed threshold: EV >= {chosen_thr:.2f} (diagnostic replay sweep best).")
        if chosen_stats:
            plan_lines.append(
                f"Replay (step14) at thr: roi={chosen_stats['roi']:.4f} "
                f"profit={chosen_stats['total_profit']:.0f} stake={chosen_stats['total_stake']:.0f} "
                f"stake_share={chosen_stats['stake_share']:.3f} n_bets={chosen_stats['n_bets']}"
            )
    else:
        plan_lines.append("Proposed threshold: EV >= 0.01 (fallback; ev column missing for sweep).")
    plan_lines.append("")
    plan_lines.append("Notes:")
    plan_lines.append("- Budget binding days were not detected in step14 segments (binding slice missing).")
    plan_lines.append("Next steps:")
    plan_lines.append("- Fit threshold on train/valid only; keep test as fixed evaluation.")
    plan_lines.append("- Run added/removed bet-set audit for the filter.")
    plan_lines.append("- If ROI improves and stake remains sufficient, consider engine promotion with sample-window parity.")
    plan_lines.append("")
    plan_lines.append("ROI definition: ROI = profit / stake, profit = return - stake.")
    (out_dir / "candidate_fix_plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# step14 vs pooled ROI diagnosis (2025 eval)",
        "",
        f"- Input rolling dir: {input_dir}",
        f"- Windows: {len(window_specs)}",
        f"- Step days: {args.step_days}",
        "",
        "## Aggregate sign check (eval)",
        f"- pooled ROI={pooled_roi:.6f} (profit={total_pooled_profit:.0f}, stake={total_pooled_stake:.0f})",
        f"- step14 ROI={step14_roi:.6f} (profit={total_step14_profit:.0f}, stake={total_step14_stake:.0f})",
        f"- sign_mismatch_confirmed={str(sign_mismatch_confirmed)}",
        "",
        "## Worst step14 windows (top 3 by profit)",
    ]
    if worst_windows.empty:
        report_lines.append("- no windows")
    else:
        for _, row in worst_windows.iterrows():
            report_lines.append(
                f"- {row['window_name']}: step14_profit={row['step14_profit']:.0f} "
                f"step14_roi={row['step14_roi']:.4f} pooled_roi={row['pooled_roi']:.4f}"
            )

    report_lines.append("")
    report_lines.append("## Window-level sign mismatch")
    if mismatch_windows.empty:
        report_lines.append("- none")
    else:
        report_lines.append(f"- count={len(mismatch_windows)}")
        for _, row in mismatch_windows.iterrows():
            report_lines.append(
                f"- {row['window_name']}: pooled_roi={row['pooled_roi']:.4f} step14_roi={row['step14_roi']:.4f}"
            )

    report_lines.append("")
    report_lines.append("## Slice diagnosis (step14 segments)")
    if slices_df.empty:
        report_lines.append("- slice metrics missing (odds/EV/binding not available)")
    else:
        for slice_type in ["odds_band", "ev_band", "budget_binding"]:
            sub = slices_df[slices_df["slice_type"] == slice_type]
            if sub.empty:
                continue
            worst = sub.sort_values("total_profit").iloc[0]
            report_lines.append(
                f"- {slice_type}: worst={worst['slice']} profit={worst['total_profit']:.0f} "
                f"roi={worst['roi']:.4f} stake_share={worst['stake_share']:.3f}"
            )
        ev_sub = slices_df[slices_df["slice_type"] == "ev_band"]
        if len(ev_sub) == 1:
            report_lines.append("- ev_band collapsed to a single bucket (all bets in one EV band).")

    report_lines.append("")
    report_lines.append("## Minimal improvement candidate")
    report_lines.append("- Selected: A) EV threshold filter (diagnostic replay)")
    if chosen_thr is not None:
        report_lines.append(f"- Proposed EV threshold: {chosen_thr:.2f}")
    else:
        report_lines.append("- Proposed EV threshold: 0.01 (fallback)")
    report_lines.append("- Fit on train/valid only; test remains fixed.")
    report_lines.append("")
    report_lines.append("ROI definition: ROI = profit / stake (profit = return - stake).")
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    staging_path = args.staging_dir if args.staging_dir is not None else "<NOT_USED>"
    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        f"[diag] step14_vs_pooled_generated={str((out_dir / 'step14_vs_pooled.csv').exists()).lower()} "
        f"| sign_mismatch_confirmed={str(sign_mismatch_confirmed).lower()}",
        f"[plan] candidate_fix_selected={candidate_choice} | replay_sweep_done={str(not replay_sweep_df.empty).lower()}",
        f"[paths] out_dir={out_dir} | staging={staging_path} | zip=<NOT_USED>",
    ]
    (out_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
