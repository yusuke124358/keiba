"""
Task B diagnostic: calibration + market-anchored EV replay from existing rolling outputs.
Reads existing bets.csv only (replay_from_base); no retraining or rolling re-runs.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

STAKE_COL_CANDIDATES = ["stake", "stake_yen", "stake_amount", "bet_amount", "stake_raw"]
RETURN_COL_CANDIDATES = ["return", "return_yen", "payout", "payout_yen"]
PROFIT_COL_CANDIDATES = ["profit", "net_profit", "pnl"]
ODDS_COL_CANDIDATES = ["odds_effective", "odds_final", "odds_at_buy", "odds"]
P_MODEL_CANDIDATES = ["p_hat", "p_hat_capped", "p_model", "prob", "p_win"]
P_MKT_CANDIDATES = ["p_mkt", "p_mkt_raw", "p_mkt_race"]
EV_CANDIDATES = ["ev_adj", "ev"]


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
    base = PROJECT_ROOT / "data" / "holdout_runs"
    candidate = base / path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"run_dir not found: {path}")


def _collect_windows(run_dir: Path) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = _parse_window_dates(p.name)
        if not parsed:
            continue
        windows.append(WindowSpec(name=p.name, test_start=parsed[0], test_end=parsed[1], path=p))
    windows.sort(key=lambda x: x.test_start)
    return windows


def _load_window_bets(window: WindowSpec, step_days: int) -> pd.DataFrame:
    bets_path = window.path / "bets.csv"
    if not bets_path.exists():
        raise SystemExit(f"bets.csv not found: {bets_path}")

    header = pd.read_csv(bets_path, nrows=0)
    cols = list(header.columns)

    date_col, _date_kind = _detect_date_column(cols)
    stake_col = _detect_column(cols, STAKE_COL_CANDIDATES)
    profit_col = _detect_column(cols, PROFIT_COL_CANDIDATES)
    return_col = _detect_column(cols, RETURN_COL_CANDIDATES)
    odds_col = _detect_column(cols, ODDS_COL_CANDIDATES)
    p_model_col = _detect_column(cols, P_MODEL_CANDIDATES)
    p_mkt_col = _detect_column(cols, P_MKT_CANDIDATES)
    ev_col = _detect_column(cols, EV_CANDIDATES)

    if stake_col is None:
        raise SystemExit(f"Could not detect stake column in {bets_path}")
    if profit_col is None and return_col is None:
        raise SystemExit(f"Could not detect profit/return column in {bets_path}")
    if odds_col is None:
        raise SystemExit(f"Could not detect odds column in {bets_path}")

    usecols = [stake_col, odds_col]
    if profit_col:
        usecols.append(profit_col)
    if return_col:
        usecols.append(return_col)
    if date_col:
        usecols.append(date_col)
    elif "race_id" in cols:
        usecols.append("race_id")
    if p_model_col:
        usecols.append(p_model_col)
    if p_mkt_col:
        usecols.append(p_mkt_col)
    if ev_col:
        usecols.append(ev_col)
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

    df["odds_val"] = pd.to_numeric(df[odds_col], errors="coerce")
    df["odds_band"] = df["odds_val"].apply(lambda x: _odds_band(x) if pd.notna(x) else None)

    if p_model_col:
        df["p_model"] = pd.to_numeric(df[p_model_col], errors="coerce")
    else:
        df["p_model"] = np.nan

    if p_mkt_col:
        df["p_mkt"] = pd.to_numeric(df[p_mkt_col], errors="coerce")
    else:
        df["p_mkt"] = np.nan

    if ev_col:
        df["ev_raw"] = pd.to_numeric(df[ev_col], errors="coerce")
    else:
        if df["p_model"].notna().any():
            df["ev_raw"] = df["p_model"] * df["odds_val"] - 1.0
        else:
            df["ev_raw"] = np.nan

    if "is_win" in df.columns:
        df["hit"] = pd.to_numeric(df["is_win"], errors="coerce").fillna(0).astype(int)
    elif return_col is not None and return_col in df.columns:
        df["hit"] = (pd.to_numeric(df[return_col], errors="coerce").fillna(0.0) > 0).astype(int)
    else:
        df["hit"] = (df["profit"] > 0).astype(int)

    df["window_id"] = window.name
    df["test_start"] = window.test_start
    df["test_end"] = window.test_end
    return df


def _daily_metrics(df: pd.DataFrame, initial_bankroll: float) -> dict:
    if df.empty:
        return {"max_drawdown": 0.0}
    daily = (
        df.groupby("date", dropna=False)
        .agg(stake=("stake", "sum"), profit=("profit", "sum"))
        .reset_index()
        .sort_values("date")
    )
    daily["cum_profit"] = daily["profit"].cumsum()
    daily["equity"] = initial_bankroll + daily["cum_profit"]
    daily["peak"] = daily["equity"].cummax()
    daily["drawdown_pct"] = (daily["peak"] - daily["equity"]) / daily["peak"]
    return {"max_drawdown": float(daily["drawdown_pct"].max())}


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


def _make_ev_bins(series: pd.Series, bins: int) -> pd.Series:
    try:
        return pd.qcut(series, q=bins, duplicates="drop")
    except Exception:
        edges = [-np.inf, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
        labels = ["<0", "0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-1.0", ">=1.0"]
        return pd.cut(series, bins=edges, labels=labels, include_lowest=True)


def _market_anchor_sweep(
    df: pd.DataFrame,
    baseline_overall: dict,
    threshold_global: list[float],
    band_thresholds: dict[str, float],
) -> pd.DataFrame:
    odds_val = df["odds_val"].copy()
    p_model = df["p_model"].copy()
    p_mkt = df["p_mkt"].copy()
    if p_mkt.isna().all():
        p_mkt = 1.0 / odds_val
    p_mkt = p_mkt.clip(lower=0.0, upper=1.0)

    weights = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
    rows = []

    for w in weights:
        p_final = (w * p_model + (1.0 - w) * p_mkt).clip(lower=0.0, upper=1.0)
        ev_final = p_final * odds_val - 1.0
        base = df.copy()
        base["ev_final"] = ev_final

        for t in threshold_global:
            filt = base[base["ev_final"] >= t]
            metrics = _summarize_candidate(filt, baseline_overall)
            rows.append(
                {
                    "candidate_id": f"w{w:.1f}_global{t:.2f}",
                    "w": w,
                    "threshold_mode": "global",
                    "t_global": t,
                    "t_low": "",
                    "t_mid": "",
                    "t_high": "",
                    **metrics,
                }
            )

        # band thresholds variant
        def _band_thr(band: str) -> float:
            return band_thresholds.get(band, band_thresholds.get("default", 0.0))

        thr = base["odds_band"].map(_band_thr)
        filt = base[base["ev_final"] >= thr]
        metrics = _summarize_candidate(filt, baseline_overall)
        rows.append(
            {
                "candidate_id": f"w{w:.1f}_band_default",
                "w": w,
                "threshold_mode": "band_default",
                "t_global": "",
                "t_low": band_thresholds.get("low", ""),
                "t_mid": band_thresholds.get("mid", ""),
                "t_high": band_thresholds.get("high", ""),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def _summarize_candidate(df: pd.DataFrame, baseline_overall: dict) -> dict:
    total_stake = float(df["stake"].sum())
    total_profit = float(df["profit"].sum())
    n_bets = int(len(df))
    roi = (total_profit / total_stake) if total_stake > 0 else 0.0

    by_win = (
        df.groupby("window_id", dropna=False)
        .agg(total_profit=("profit", "sum"), total_stake=("stake", "sum"))
        .reset_index()
    )
    by_win["roi"] = by_win["total_profit"] / by_win["total_stake"].where(by_win["total_stake"] > 0, np.nan)
    by_win["roi"] = by_win["roi"].fillna(0.0)

    min_roi = float(by_win["roi"].min()) if not by_win.empty else 0.0
    med_roi = float(by_win["roi"].median()) if not by_win.empty else 0.0
    max_roi = float(by_win["roi"].max()) if not by_win.empty else 0.0

    stake_share = (total_stake / baseline_overall["total_stake"]) if baseline_overall["total_stake"] > 0 else 0.0
    roi_delta = roi - baseline_overall["roi"]

    return {
        "n_bets": n_bets,
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": roi,
        "stake_share": stake_share,
        "roi_delta_vs_base": roi_delta,
        "min_roi": min_roi,
        "median_roi": med_roi,
        "max_roi": max_roi,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TaskB calibration/anchor diagnostic (replay only)")
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--out_jp", type=Path, default=None)
    ap.add_argument("--step_days", type=int, default=14)
    ap.add_argument("--ev_bins", type=int, default=10)
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    out_dir = args.out_dir
    out_jp = args.out_jp if args.out_jp is not None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    windows = _collect_windows(run_dir)
    if not windows:
        raise SystemExit(f"No windows found under {run_dir}")

    # detect race_id lists
    race_ids_paths = []
    for w in windows:
        for name in ["race_ids_train.txt", "race_ids_valid.txt", "race_ids_test.txt"]:
            p = w.path / name
            if p.exists():
                race_ids_paths.append(str(p))
                break

    # load step14 bets
    sample_cols = {}
    all_rows = []
    for w in windows:
        df = _load_window_bets(w, args.step_days)
        if not df.empty:
            if not sample_cols:
                sample_cols = {
                    "p_model_col": "p_model" if df["p_model"].notna().any() else "missing",
                    "p_mkt_col": "p_mkt" if df["p_mkt"].notna().any() else "derived_or_missing",
                    "odds_col": "odds_val",
                    "ev_col": "ev_raw" if df["ev_raw"].notna().any() else "missing",
                }
            all_rows.append(df)
    if not all_rows:
        raise SystemExit("No bets found in step14 segments.")

    all_df = pd.concat(all_rows, ignore_index=True)

    # baseline step14 vs pooled
    summary_path = run_dir / "summary.csv"
    pooled_df = pd.read_csv(summary_path)
    name_col = "name" if "name" in pooled_df.columns else "run_dir"
    pooled_df = pooled_df.rename(columns={name_col: "window_id"})

    step14_rows = []
    for win, sub in all_df.groupby("window_id", dropna=False):
        total_stake = float(sub["stake"].sum())
        total_profit = float(sub["profit"].sum())
        n_bets = int(len(sub))
        roi = (total_profit / total_stake) if total_stake > 0 else 0.0
        dd = _daily_metrics(sub, 1_000_000.0)["max_drawdown"]
        step14_rows.append(
            {
                "window_id": win,
                "step14_roi": roi,
                "step14_profit": total_profit,
                "step14_stake": total_stake,
                "step14_n_bets": n_bets,
                "step14_max_drawdown": dd,
            }
        )
    step14_df = pd.DataFrame(step14_rows)

    pooled_keep = pooled_df[
        [
            "window_id",
            "roi",
            "total_profit",
            "total_stake",
            "n_bets",
            "max_drawdown",
        ]
    ].rename(
        columns={
            "roi": "pooled_roi",
            "total_profit": "pooled_profit",
            "total_stake": "pooled_stake",
            "n_bets": "pooled_n_bets",
            "max_drawdown": "pooled_max_drawdown",
        }
    )

    baseline_vs = pooled_keep.merge(step14_df, on="window_id", how="left")
    baseline_vs["sign_mismatch"] = (baseline_vs["pooled_roi"] >= 0) != (baseline_vs["step14_roi"] >= 0)
    baseline_vs.to_csv(tables_dir / "baseline_step14_vs_pooled.csv", index=False, encoding="utf-8")

    baseline_overall = {
        "total_stake": float(all_df["stake"].sum()),
        "total_profit": float(all_df["profit"].sum()),
    }
    baseline_overall["roi"] = (
        baseline_overall["total_profit"] / baseline_overall["total_stake"]
        if baseline_overall["total_stake"] > 0
        else 0.0
    )

    # odds band contribution
    odds_df = _aggregate(all_df, ["odds_band"])
    total_profit_all = float(all_df["profit"].sum())
    odds_df["stake_share"] = odds_df["total_stake"] / baseline_overall["total_stake"]
    odds_df["profit_share"] = (
        odds_df["profit"] / total_profit_all if total_profit_all != 0 else 0.0
    )
    odds_df.to_csv(tables_dir / "roi_by_odds_band.csv", index=False, encoding="utf-8")

    # EV bins (raw)
    ev_series = all_df["ev_raw"].dropna()
    if ev_series.empty:
        raise SystemExit("EV values missing; cannot compute EV bins.")
    all_df["ev_bin"] = _make_ev_bins(all_df["ev_raw"], args.ev_bins)
    all_df["ev_bin"] = all_df["ev_bin"].astype(str)
    roi_ev = _aggregate(all_df, ["odds_band", "ev_bin"])
    roi_ev.to_csv(tables_dir / "roi_by_ev_bin_raw.csv", index=False, encoding="utf-8")

    roi_win_ev = _aggregate(all_df, ["window_id", "odds_band", "ev_bin"])
    roi_win_ev.to_csv(tables_dir / "roi_by_window_ev_bin.csv", index=False, encoding="utf-8")

    # calibration by p_model bin
    calib_gap = None
    if all_df["p_model"].notna().any():
        all_df["p_bin"] = _make_ev_bins(all_df["p_model"], args.ev_bins).astype(str)
        calib_df = (
            all_df.groupby("p_bin", dropna=False)
            .agg(n_bets=("stake", "size"), avg_p=("p_model", "mean"), win_rate=("hit", "mean"))
            .reset_index()
        )
        calib_df["gap"] = (calib_df["avg_p"] - calib_df["win_rate"]).abs()
        calib_gap = float(calib_df["gap"].mean())
        calib_df.to_csv(tables_dir / "calibration_by_p_hat_bin.csv", index=False, encoding="utf-8")

    # market anchor sweep
    threshold_global = [0.0, 0.1, 0.2]
    band_thresholds = {"low": 0.0, "mid": 0.1, "high": 0.2, "20+": 0.2, "5-10": 0.1, "default": 0.0}
    sweep_df = _market_anchor_sweep(all_df, baseline_overall, threshold_global, band_thresholds)
    sweep_df.to_csv(tables_dir / "market_anchor_sweep.csv", index=False, encoding="utf-8")

    # candidate selection
    baseline_median = float(baseline_vs["step14_roi"].median())
    candidates = sweep_df[
        (sweep_df["n_bets"] >= 100)
        & (sweep_df["roi"] >= baseline_overall["roi"] + 0.02)
        & (sweep_df["min_roi"] >= -0.5)
        & (sweep_df["median_roi"] >= baseline_median - 0.05)
    ]
    if not candidates.empty:
        best = candidates.sort_values("roi", ascending=False).iloc[0]
        candidate_id = str(best["candidate_id"])
        decision = "engine_promo"
    else:
        candidate_id = "NONE"
        decision = "model_update_next"

    # report
    pooled_total_profit = float(baseline_vs["pooled_profit"].sum())
    pooled_total_stake = float(baseline_vs["pooled_stake"].sum())
    pooled_roi = (pooled_total_profit / pooled_total_stake) if pooled_total_stake > 0 else 0.0
    step14_roi = baseline_overall["roi"]

    report_lines = [
        "# Task B calibration/anchor diagnostic (replay_from_base=true)",
        "",
        f"- run_dir: {run_dir}",
        f"- windows: {len(windows)}",
        f"- step_days: {args.step_days}",
        f"- replay_from_base: true (existing bets.csv only)",
        f"- detected columns: {sample_cols}",
        "",
        "## Step14 vs pooled (eval)",
        f"- pooled ROI={pooled_roi:.6f} (profit={pooled_total_profit:.0f}, stake={pooled_total_stake:.0f})",
        f"- step14 ROI={step14_roi:.6f} (profit={baseline_overall['total_profit']:.0f}, stake={baseline_overall['total_stake']:.0f})",
        f"- sign_mismatch_windows={int(baseline_vs['sign_mismatch'].sum())}",
        "",
        "## EV monotonicity / calibration",
        "- EV bins computed from ev_raw (ev_adj/ev fallback).",
    ]
    if calib_gap is not None:
        report_lines.append(f"- calibration gap (mean |avg_p - win_rate|)={calib_gap:.4f}")
    else:
        report_lines.append("- calibration gap: N/A (p_model missing)")

    report_lines.append("")
    report_lines.append("## Market-anchor sweep (replay)")
    best_row = sweep_df.sort_values("roi", ascending=False).iloc[0]
    report_lines.append(
        f"- best ROI candidate (any n_bets): {best_row['candidate_id']} roi={best_row['roi']:.4f} "
        f"n_bets={int(best_row['n_bets'])} stake_share={best_row['stake_share']:.3f}"
    )
    report_lines.append(
        f"- selected_candidate={candidate_id} (criteria: n_bets>=100, ROI improve, min/median constraints)"
    )
    report_lines.append(f"- decision={decision}")
    report_lines.append("")
    report_lines.append("## Notes")
    if not race_ids_paths:
        report_lines.append("- race_ids_train/valid/test not found under windows; calibration is diagnostic only.")
    else:
        report_lines.append(f"- race_ids files detected (sample): {race_ids_paths[0]}")
    report_lines.append("- Calibration/EV diagnostics use selected bets only (selection bias); quick diagnostic.")
    report_lines.append("- EV/threshold tuning must be re-fit on train/valid before engine promotion.")
    report_lines.append("")
    report_lines.append("ROI definition: ROI = profit / stake (profit = return - stake).")
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    minimal_lines = [
        "# minimal update for chat - TaskB calib/anchor (2026-01-28)",
        "",
        "## Summary",
        f"- step14 ROI={step14_roi:.4f} (pooled ROI={pooled_roi:.4f})",
        f"- selected_candidate={candidate_id} | decision={decision}",
        "",
        "## Paths",
        f"- out_jp: {out_jp}",
        f"- out_ascii: {out_dir}",
    ]
    (out_dir / "minimal_update_for_chat.md").write_text("\n".join(minimal_lines) + "\n", encoding="utf-8")

    stdout_lines = [
        f"[inputs] run_dir={run_dir} | p_model={sample_cols.get('p_model_col')} | "
        f"p_mkt={sample_cols.get('p_mkt_col')} | odds={sample_cols.get('odds_col')} | "
        f"ev={sample_cols.get('ev_col')} | race_ids_found={str(bool(race_ids_paths)).lower()}",
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        "[diag] taskB_calib_anchor_done=true | replay_from_base=true",
        f"[plan] candidate_selected={candidate_id} | decision={decision}",
        f"[paths] out_dir={out_jp} | staging={out_dir} | zip=<NOT_USED>",
    ]
    (out_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
