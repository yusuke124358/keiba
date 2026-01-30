"""
Diagnose ROI monotonicity vs EV within odds bands using existing rolling outputs.
Reads bets.csv from rolling run windows; uses step14 segments (non-overlap).
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


def _derive_date_from_race_id(df: pd.DataFrame) -> pd.Series:
    if "race_id" not in df.columns:
        return pd.Series([pd.NaT] * len(df))
    race_id = df["race_id"].astype(str).str.slice(0, 8)
    return pd.to_datetime(race_id, errors="coerce", format="%Y%m%d").dt.date


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


def _resolve_run_dir(path: Path) -> Path:
    if path.exists():
        return path
    base = Path(r"C:\Users\yyosh\keiba\data\holdout_runs")
    candidate = base / path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"run_dir not found: {path}")


def _load_window_bets(window: WindowSpec, step_days: int) -> pd.DataFrame:
    bets_path = window.path / "bets.csv"
    if not bets_path.exists():
        raise SystemExit(f"bets.csv not found: {bets_path}")

    header = pd.read_csv(bets_path, nrows=0)
    cols = list(header.columns)

    date_col, date_kind = _detect_date_column(cols)
    stake_col = _detect_column(cols, STAKE_COL_CANDIDATES)
    profit_col = _detect_column(cols, PROFIT_COL_CANDIDATES)
    return_col = _detect_column(cols, RETURN_COL_CANDIDATES)
    odds_col = _detect_column(cols, ODDS_COL_CANDIDATES)
    ev_col = _detect_column(cols, EV_COL_CANDIDATES)

    if stake_col is None:
        raise SystemExit(f"Could not detect stake column in {bets_path}")
    if profit_col is None and return_col is None:
        raise SystemExit(f"Could not detect profit/return column in {bets_path}")

    usecols = [stake_col]
    if profit_col:
        usecols.append(profit_col)
    if return_col:
        usecols.append(return_col)
    if date_col:
        usecols.append(date_col)
    elif "race_id" in cols:
        usecols.append("race_id")
    if odds_col:
        usecols.append(odds_col)
    if ev_col:
        usecols.append(ev_col)
    if "p_hat" in cols:
        usecols.append("p_hat")
    if "is_win" in cols:
        usecols.append("is_win")

    df = pd.read_csv(bets_path, usecols=sorted(set(usecols)))

    if date_col is None:
        df["date"] = _derive_date_from_race_id(df)
    else:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    segment_start = window.test_start
    segment_end = min(window.test_end, window.test_start + timedelta(days=step_days - 1))
    mask = (df["date"] >= segment_start) & (df["date"] <= segment_end)
    df = df.loc[mask].copy()

    df["stake"] = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
    if profit_col:
        df["profit"] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0)
    else:
        df["profit"] = (
            pd.to_numeric(df[return_col], errors="coerce").fillna(0.0) - df["stake"]
        )

    if odds_col:
        df["odds_val"] = pd.to_numeric(df[odds_col], errors="coerce")
    else:
        df["odds_val"] = np.nan

    if ev_col:
        df["ev_val"] = pd.to_numeric(df[ev_col], errors="coerce")
    elif "p_hat" in df.columns and odds_col:
        df["ev_val"] = pd.to_numeric(df["p_hat"], errors="coerce") * df["odds_val"] - 1.0
    else:
        df["ev_val"] = np.nan

    if "is_win" in df.columns:
        df["hit"] = pd.to_numeric(df["is_win"], errors="coerce").fillna(0).astype(int)
    elif return_col is not None and return_col in df.columns:
        df["hit"] = (pd.to_numeric(df[return_col], errors="coerce").fillna(0.0) > 0).astype(int)
    else:
        df["hit"] = (df["profit"] > 0).astype(int)

    df["odds_band"] = df["odds_val"].apply(lambda x: _odds_band(x) if pd.notna(x) else None)
    return df


def _make_ev_bins(series: pd.Series, bins: int) -> tuple[pd.Series, list[str]]:
    series = series.dropna()
    if series.empty:
        return pd.Series(dtype="object"), []
    try:
        labels = [f"q{i+1:02d}" for i in range(bins)]
        binned = pd.qcut(series, q=bins, labels=labels, duplicates="drop")
        return binned, list(binned.cat.categories.astype(str))
    except Exception:
        edges = [-np.inf, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
        labels = ["<0", "0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-1.0", ">=1.0"]
        binned = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
        return binned, labels


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_bets=("stake", "size"),
            total_stake=("stake", "sum"),
            profit=("profit", "sum"),
            hit_rate=("hit", "mean"),
            avg_odds=("odds_val", "mean"),
        )
        .reset_index()
    )
    grouped["roi"] = grouped["profit"] / grouped["total_stake"].where(grouped["total_stake"] > 0, np.nan)
    grouped["roi"] = grouped["roi"].fillna(0.0)
    return grouped


def _monotonicity_by_band(roi_df: pd.DataFrame, ev_order: dict[str, int]) -> list[dict]:
    rows = []
    for band, sub in roi_df.groupby("odds_band", dropna=False):
        if pd.isna(band):
            continue
        sub = sub.copy()
        sub["ev_order"] = sub["ev_bin"].map(ev_order).fillna(-1).astype(int)
        sub = sub.sort_values("ev_order")
        if len(sub) < 2:
            rows.append(
                {
                    "odds_band": band,
                    "monotonic": "insufficient",
                    "nondec_rate": None,
                    "slope": None,
                    "best_bin": None,
                    "best_roi": None,
                    "best_n_bets": None,
                    "best_stake_share": None,
                }
            )
            continue
        diffs = sub["roi"].diff().dropna()
        nondec_rate = float((diffs >= 0).mean()) if not diffs.empty else 0.0
        slope = float(sub["roi"].iloc[-1] - sub["roi"].iloc[0])
        best = sub.sort_values("roi", ascending=False).iloc[0]
        rows.append(
            {
                "odds_band": band,
                "monotonic": "yes" if nondec_rate >= 0.6 and slope >= 0 else "no",
                "nondec_rate": nondec_rate,
                "slope": slope,
                "best_bin": best["ev_bin"],
                "best_roi": float(best["roi"]),
                "best_n_bets": int(best["n_bets"]),
                "best_stake_share": float(best["stake_share"]),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose ROI monotonicity by EV bins and odds bands")
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--out_jp", type=Path, required=True)
    ap.add_argument("--out_ascii", type=Path, required=True)
    ap.add_argument("--step_days", type=int, default=14)
    ap.add_argument("--ev_bins", type=int, default=10)
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    out_ascii = args.out_ascii
    out_jp = args.out_jp
    out_ascii.mkdir(parents=True, exist_ok=True)

    windows = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = _parse_window_dates(p.name)
        if not parsed:
            continue
        windows.append(WindowSpec(name=p.name, test_start=parsed[0], test_end=parsed[1], path=p))
    windows.sort(key=lambda x: x.test_start)
    if not windows:
        raise SystemExit(f"No windows found under {run_dir}")

    all_rows = []
    for w in windows:
        df = _load_window_bets(w, args.step_days)
        if df.empty:
            continue
        df["window_id"] = w.name
        all_rows.append(df)

    if not all_rows:
        raise SystemExit("No bets found in step14 segments.")

    all_df = pd.concat(all_rows, ignore_index=True)
    ev_series = all_df["ev_val"].dropna()
    if ev_series.empty:
        raise SystemExit("EV values missing; cannot bin EV.")

    try:
        all_df["ev_bin"] = pd.qcut(all_df["ev_val"], q=args.ev_bins, duplicates="drop")
    except Exception:
        edges = [-np.inf, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
        labels = ["<0", "0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-1.0", ">=1.0"]
        all_df["ev_bin"] = pd.cut(all_df["ev_val"], bins=edges, labels=labels, include_lowest=True)

    total_stake_all = float(all_df["stake"].sum())

    roi_df = _aggregate(all_df, ["odds_band", "ev_bin"])
    roi_df["stake_share"] = roi_df["total_stake"] / total_stake_all if total_stake_all > 0 else 0.0
    roi_df.to_csv(out_ascii / "roi_by_ev_bin.csv", index=False, encoding="utf-8")

    win_df = _aggregate(all_df, ["window_id", "odds_band", "ev_bin"])
    win_df.to_csv(out_ascii / "roi_by_window_ev_bin.csv", index=False, encoding="utf-8")

    if hasattr(all_df["ev_bin"], "cat"):
        ev_categories = [str(c) for c in all_df["ev_bin"].cat.categories]
        ev_order = {str(c): idx for idx, c in enumerate(ev_categories)}
    else:
        ev_order = {str(label): idx for idx, label in enumerate(sorted(roi_df["ev_bin"].dropna().unique()))}
    roi_df["ev_bin"] = roi_df["ev_bin"].astype(str)
    win_df["ev_bin"] = win_df["ev_bin"].astype(str)
    mono_rows = _monotonicity_by_band(roi_df, ev_order)
    mono_df = pd.DataFrame(mono_rows)

    positive_bins = roi_df[(roi_df["roi"] > 0) & (roi_df["n_bets"] >= 100)]
    has_positive_band = not positive_bins.empty
    any_positive_bins = roi_df[roi_df["roi"] > 0]
    sparse_positive = any_positive_bins[any_positive_bins["n_bets"] < 100]

    window_totals = (
        all_df.groupby("window_id", dropna=False)
        .agg(total_profit=("profit", "sum"), total_stake=("stake", "sum"), n_bets=("stake", "size"))
        .reset_index()
        .sort_values("total_profit")
    )
    worst_windows = window_totals.head(3)

    report_lines = [
        "# ROI monotonicity diagnostic (EV x odds band)",
        "",
        f"- run_dir: {run_dir}",
        f"- windows: {len(windows)}",
        f"- step_days: {args.step_days}",
        f"- ev_bins: {args.ev_bins}",
        "",
        "## Monotonicity by odds band",
    ]
    if mono_df.empty:
        report_lines.append("- no rows")
    else:
        for _, row in mono_df.iterrows():
            report_lines.append(
                f"- odds_band={row['odds_band']} | monotonic={row['monotonic']} "
                f"| nondec_rate={row['nondec_rate'] if row['nondec_rate'] is not None else 'N/A'} "
                f"| slope={row['slope'] if row['slope'] is not None else 'N/A'} "
                f"| best_bin={row['best_bin']} roi={row['best_roi']} "
                f"n_bets={row['best_n_bets']} stake_share={row['best_stake_share']}"
            )

    report_lines.append("")
    report_lines.append("## Conclusion")
    if has_positive_band:
        top = positive_bins.sort_values("roi", ascending=False).iloc[0]
        report_lines.append(
            f"- Positive ROI bin exists: odds_band={top['odds_band']} ev_bin={top['ev_bin']} "
            f"roi={top['roi']:.4f} n_bets={int(top['n_bets'])} stake_share={top['stake_share']:.3f}"
        )
        report_lines.append("- Consider targeting this EV/odds region (train/valid tuning) before engine changes.")
    else:
        report_lines.append("- No positive ROI bin with n_bets>=100; EV monotonicity is weak or absent.")
        if not sparse_positive.empty:
            top_sparse = sparse_positive.sort_values("roi", ascending=False).iloc[0]
            report_lines.append(
                f"- Sparse positive bins exist but small: odds_band={top_sparse['odds_band']} "
                f"ev_bin={top_sparse['ev_bin']} roi={top_sparse['roi']:.4f} n_bets={int(top_sparse['n_bets'])}"
            )
        report_lines.append("- Calibration/model update likely required.")

    report_lines.append("")
    report_lines.append("ROI definition: ROI = profit / stake (profit = return - stake).")
    (out_ascii / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    minimal_lines = [
        "# minimal update for chat - ROI EV monotonicity (2026-01-28)",
        "",
        "## Summary",
    ]
    if has_positive_band:
        minimal_lines.append("- EV/odds band shows positive ROI bin with n_bets>=100 (see report).")
    else:
        minimal_lines.append("- No positive ROI bin with n_bets>=100; EV monotonicity weak.")
        if not sparse_positive.empty:
            top_sparse = sparse_positive.sort_values("roi", ascending=False).iloc[0]
            minimal_lines.append(
                f"- sparse positive bin: {top_sparse['odds_band']} {top_sparse['ev_bin']} "
                f"roi={top_sparse['roi']:.4f} n_bets={int(top_sparse['n_bets'])}"
            )

    minimal_lines.append("")
    minimal_lines.append("## Decision")
    if has_positive_band:
        minimal_lines.append("- Worth testing EV/odds targeting rule (train/valid tuned) on step14.")
    else:
        minimal_lines.append("- Move to calibration/model update (task B).")

    minimal_lines.append("")
    minimal_lines.append("## Paths")
    minimal_lines.append(f"- out_ascii: {out_ascii}")
    minimal_lines.append(f"- out_jp: {out_jp}")
    (out_ascii / "minimal_update_for_chat.md").write_text("\n".join(minimal_lines) + "\n", encoding="utf-8")

    if not has_positive_band:
        plan_lines = [
            "# candidate_fix_plan_v2 (design only)",
            "",
            "## Loss structure summary",
            "- EV monotonicity is weak; no positive ROI bins with n_bets>=100.",
            "- Sparse positives exist but are not reliable for rules-only fixes.",
            "",
            "## Priority fixes (design)",
            "1) Probability calibration (train or train+valid only; never test).",
            "2) Market odds shrink/blend (buy-time odds only; no leakage).",
            "3) Odds-band EV thresholds (stricter for 20+, looser for 10-20) tuned on train/valid.",
            "",
            "## Evaluation plan (no heavy reruns)",
            "- First, replay on worst windows only to check direction.",
        ]
        if not worst_windows.empty:
            for _, row in worst_windows.iterrows():
                plan_lines.append(
                    f"  - {row['window_id']}: profit={row['total_profit']:.0f} stake={row['total_stake']:.0f} "
                    f"n_bets={int(row['n_bets'])}"
                )
        plan_lines += [
            "- If direction improves, run full step14 summary and compare to pooled.",
            "",
            "## AGENTS compliance",
            "- Calibrators/thresholds must be fit on train or train+valid only.",
            "- step14 is the decision signal when pooled differs.",
            "",
            "ROI definition: ROI = profit / stake (profit = return - stake).",
        ]
        (out_ascii / "candidate_fix_plan_v2.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    odds_bands = ",".join(sorted(str(b) for b in roi_df["odds_band"].dropna().unique()))
    ev_bins_str = ",".join(sorted(str(b) for b in roi_df["ev_bin"].dropna().unique()))

    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        f"[diag] roi_ev_monotonicity_done=true | ev_bins={ev_bins_str} | odds_bands={odds_bands}",
        f"[paths] out_dir={out_jp} | staging={out_ascii} | zip=<NOT_USED>",
    ]
    (out_ascii / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
