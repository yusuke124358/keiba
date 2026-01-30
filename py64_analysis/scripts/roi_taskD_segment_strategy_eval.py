"""
Phase2: Segment strategy evaluation (reselect) using full-field predictions.
Builds candidate bet sets from full-field test data and evaluates step14 ROI.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from keiba.db.loader import get_session


ODDS_CANDIDATES = ["odds", "odds_val", "odds_at_buy", "odds_effective", "odds_final"]
Y_CANDIDATES = ["y_win", "is_win", "winner", "y_true", "y"]
P_MODEL_CANDIDATES = ["p_model", "p_win", "prob", "p_hat"]
P_HAT_CANDIDATES = ["p_hat", "p_cal", "p_blend", "p_used"]
P_MKT_CANDIDATES = ["p_mkt", "p_mkt_race", "p_mkt_raw", "p_market"]


def _detect_column(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


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


def _field_size_bin(val: float) -> str:
    if val <= 10:
        return "<=10"
    if val <= 14:
        return "11-14"
    if val <= 18:
        return "15-18"
    return "19+"


def _distance_bin(val: float) -> str:
    if val < 1400:
        return "<1400"
    if val < 1800:
        return "1400-1799"
    if val < 2200:
        return "1800-2199"
    return "2200+"


def _surface_label(val) -> str:
    if val is None:
        return "unknown"
    s = str(val).strip().lower()
    if s in ("1", "turf", "grass", "shiba", "芝"):
        return "turf"
    if s in ("2", "dirt", "dar", "ダ", "ダート"):
        return "dirt"
    if s in ("3", "jump", "障", "障害"):
        return "jump"
    return "unknown"


def _read_fullfield(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)
    odds_col = _detect_column(cols, ODDS_CANDIDATES)
    y_col = _detect_column(cols, Y_CANDIDATES)
    p_model_col = _detect_column(cols, P_MODEL_CANDIDATES)
    p_hat_col = _detect_column(cols, P_HAT_CANDIDATES)
    p_mkt_col = _detect_column(cols, P_MKT_CANDIDATES)
    window_col = "window_id" if "window_id" in cols else None
    if odds_col is None or y_col is None or p_model_col is None:
        raise SystemExit(f"Missing required columns in {path}")

    out = pd.DataFrame()
    out["race_id"] = df["race_id"].astype(str)
    out["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    out["odds"] = pd.to_numeric(df[odds_col], errors="coerce")
    out["y_win"] = pd.to_numeric(df[y_col], errors="coerce")
    out["p_model"] = pd.to_numeric(df[p_model_col], errors="coerce")
    if p_hat_col:
        out["p_hat"] = pd.to_numeric(df[p_hat_col], errors="coerce")
    if p_mkt_col:
        out["p_mkt_col"] = pd.to_numeric(df[p_mkt_col], errors="coerce")
    if window_col:
        out["window_id"] = df[window_col].astype(str)
    return out


def _compute_p_mkt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_mkt_raw"] = 1.0 / df["odds"]
    sums = df.groupby("race_id", dropna=False)["p_mkt_raw"].transform("sum")
    df["p_mkt_norm"] = df["p_mkt_raw"] / sums
    return df


def _find_run_dirs(pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
    return sorted(paths, key=lambda p: p.name)


def _ensure_fullfield(run_dir: Path, fullfield_root: Path, split: str) -> Path:
    # expected path
    candidate = fullfield_root / run_dir.name / "fullfield" / f"fullfield_{split}.csv"
    if split == "test":
        alt = fullfield_root / run_dir.name / "fullfield" / "fullfield_preds.csv"
        if alt.exists():
            return alt
    if candidate.exists():
        return candidate

    fullfield_root.mkdir(parents=True, exist_ok=True)
    out_dir = fullfield_root / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "export_fullfield_preds_for_run_dir.py"
    cmd = [sys.executable, str(script_path), "--run_dir", str(run_dir), "--out_dir", str(out_dir), "--split", "all"]
    subprocess.run(cmd, check=True)
    candidate = out_dir / "fullfield" / f"fullfield_{split}.csv"
    if split == "test" and (out_dir / "fullfield" / "fullfield_preds.csv").exists():
        return out_dir / "fullfield" / "fullfield_preds.csv"
    if not candidate.exists():
        raise SystemExit(f"fullfield_{split}.csv not found after export for {run_dir}")
    return candidate


def _find_fullfield(run_dir: Path, fullfield_root: Path, split: str) -> Optional[Path]:
    candidate = fullfield_root / run_dir.name / "fullfield" / f"fullfield_{split}.csv"
    if split == "test":
        alt = fullfield_root / run_dir.name / "fullfield" / "fullfield_preds.csv"
        if alt.exists():
            return alt
    if candidate.exists():
        return candidate
    return None


def _fetch_race_meta(session, race_ids: list[str]) -> pd.DataFrame:
    if not race_ids:
        return pd.DataFrame()
    rows = []
    chunk_size = 1000
    for i in range(0, len(race_ids), chunk_size):
        chunk = race_ids[i : i + chunk_size]
        res = session.execute(
            text(
                """
                SELECT race_id, date, track_code, surface, distance, field_size
                FROM fact_race
                WHERE race_id = ANY(:race_ids)
                """
            ),
            {"race_ids": chunk},
        ).fetchall()
        rows.extend([dict(r._mapping) for r in res])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["race_id"] = df["race_id"].astype(str)
    return df


def _prepare_fullfield(df: pd.DataFrame, market_method: str, model_col: str, session) -> pd.DataFrame:
    df = _compute_p_mkt(df)
    race_meta = _fetch_race_meta(session, df["race_id"].unique().tolist())
    if not race_meta.empty:
        df = df.merge(race_meta, on="race_id", how="left")

    df["odds_band"] = df["odds"].apply(lambda v: _odds_band(float(v)) if np.isfinite(v) else "unknown")
    if "field_size" in df.columns:
        df["field_size_bin"] = df["field_size"].apply(
            lambda v: _field_size_bin(float(v)) if pd.notna(v) else "unknown"
        )
    else:
        df["field_size_bin"] = "unknown"

    if "surface" in df.columns:
        df["surface"] = df["surface"].apply(_surface_label)
    else:
        df["surface"] = "unknown"

    if "distance" in df.columns:
        df["distance_bin"] = df["distance"].apply(
            lambda v: _distance_bin(float(v)) if pd.notna(v) else "unknown"
        )
    else:
        df["distance_bin"] = "unknown"

    if "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
    else:
        df["month"] = "unknown"

    if market_method == "p_mkt_col":
        if "p_mkt_col" in df.columns and df["p_mkt_col"].notna().any():
            df["p_mkt_used"] = df["p_mkt_col"]
        else:
            df["p_mkt_used"] = df["p_mkt_norm"]
    elif market_method == "p_mkt_raw":
        df["p_mkt_used"] = df["p_mkt_raw"]
    else:
        df["p_mkt_used"] = df["p_mkt_norm"]

    df["p_model_used"] = df[model_col]
    df["ev_model"] = df["p_model_used"] * df["odds"] - 1.0
    df["ev_mkt"] = df["p_mkt_used"] * df["odds"] - 1.0
    df["race_key"] = df["race_id"].astype(str)
    df = df[
        df["odds"].notna()
        & (df["odds"] > 0)
        & df["y_win"].isin([0, 1])
        & np.isfinite(df["p_model_used"])
        & np.isfinite(df["p_mkt_used"])
    ].copy()
    return df


def _compute_roi(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_bets": 0, "stake": 0.0, "return": 0.0, "profit": 0.0, "roi": float("nan"), "max_dd": 0.0}
    stake = float(len(df))
    ret = float((df["odds"] * df["y_win"]).sum())
    profit = ret - stake
    roi = profit / stake if stake > 0 else float("nan")

    # max drawdown on cumulative profit by race/date order
    if "date" in df.columns:
        order = df.sort_values(["date", "race_id", "horse_no"])
    else:
        order = df.sort_values(["race_id", "horse_no"])
    pnl = (order["odds"] * order["y_win"] - 1.0).to_numpy()
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0

    return {"n_bets": int(len(df)), "stake": stake, "return": ret, "profit": profit, "roi": roi, "max_dd": max_dd}


def _baseline_bets(run_dir: Path) -> pd.DataFrame:
    rows = []
    for win_dir in run_dir.iterdir():
        if not win_dir.is_dir():
            continue
        bets_path = win_dir / "bets.csv"
        if not bets_path.exists():
            continue
        try:
            b = pd.read_csv(bets_path)
        except Exception:
            continue
        if "race_id" not in b.columns or "horse_no" not in b.columns:
            continue
        b["race_id"] = b["race_id"].astype(str)
        b["horse_no"] = pd.to_numeric(b["horse_no"], errors="coerce").astype("Int64")
        b["window_id"] = win_dir.name
        rows.append(b[["race_id", "horse_no", "window_id"]])
    if not rows:
        return pd.DataFrame(columns=["race_id", "horse_no", "window_id"])
    return pd.concat(rows, ignore_index=True)


def _apply_strategy(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    work = df.copy()
    if strategy.get("include_odds_band"):
        work = work[work["odds_band"].isin(strategy["include_odds_band"])]
    if strategy.get("exclude_odds_band"):
        work = work[~work["odds_band"].isin(strategy["exclude_odds_band"])]
    if strategy.get("field_size_max") is not None:
        if "field_size" in work.columns:
            work = work[work["field_size"] <= strategy["field_size_max"]]
        else:
            work = work.iloc[0:0]
    if strategy.get("ev_thresholds"):
        # band-specific thresholds
        thr_map = strategy["ev_thresholds"]
        work = work[work["odds_band"].map(thr_map).notna()]
        work = work[work["ev_model"] >= work["odds_band"].map(thr_map)]
    if strategy.get("ev_threshold") is not None:
        work = work[work["ev_model"] >= strategy["ev_threshold"]]
    return work


def _fit_band_thresholds(df_fit: pd.DataFrame, grid: list[dict]) -> dict:
    best = None
    best_roi = -math.inf
    for thr_map in grid:
        work = df_fit.copy()
        work = work[work["odds_band"].map(thr_map).notna()]
        work = work[work["ev_model"] >= work["odds_band"].map(thr_map)]
        metrics = _compute_roi(work)
        if metrics["roi"] > best_roi:
            best_roi = metrics["roi"]
            best = thr_map
    return best or grid[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase2 segment strategy evaluation")
    ap.add_argument("--run_dir_glob", required=True)
    ap.add_argument("--fullfield_root", required=True)
    ap.add_argument("--market_prob_method", default="p_mkt_col")
    ap.add_argument("--model_prob_col", default="p_model")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--staging_dir", required=True)
    ap.add_argument("--fit_mode", default="auto", choices=["auto", "train_valid", "walk_forward"])
    args = ap.parse_args()

    run_dirs = _find_run_dirs(args.run_dir_glob)
    if not run_dirs:
        raise SystemExit(f"No run dirs matched: {args.run_dir_glob}")

    out_dir = Path(args.out_dir)
    staging_dir = Path(args.staging_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    fullfield_root = Path(args.fullfield_root)
    session = get_session()

    strategies_base = [
        {"id": "odds_low_only", "include_odds_band": ["<3", "3-5"]},
        {"id": "odds_ex_20", "exclude_odds_band": ["20+"]},
        {"id": "odds_ex_20_5-10", "exclude_odds_band": ["20+", "5-10"]},
        {"id": "field_size_le10", "field_size_max": 10},
    ]

    band_threshold_grid = [
        {"<3": 0.05, "3-5": 0.10, "5-10": 0.20, "10-20": 0.30, "20+": 0.50},
        {"<3": 0.02, "3-5": 0.05, "5-10": 0.15, "10-20": 0.25, "20+": 0.40},
        {"<3": 0.10, "3-5": 0.15, "5-10": 0.25, "10-20": 0.35, "20+": 0.60},
    ]

    results = []
    window_rows = []
    band_contrib_rows = []
    baseline_rows = []
    added_removed_rows = []

    best_candidate = None
    best_roi = -math.inf

    for run_dir in run_dirs:
        run_id = run_dir.name
        ff_test = _ensure_fullfield(run_dir, fullfield_root, "test")
        df_test = _read_fullfield(ff_test)
        df_test = _prepare_fullfield(df_test, args.market_prob_method, args.model_prob_col, session)
        if "window_id" not in df_test.columns:
            df_test["window_id"] = "unknown"

        # baseline bet set from bets.csv
        base_bets = _baseline_bets(run_dir)
        base_keys = set(zip(base_bets["window_id"], base_bets["race_id"], base_bets["horse_no"]))

        # baseline metrics (unit stake)
        base_df = df_test[
            df_test.set_index(["window_id", "race_id", "horse_no"]).index.isin(base_keys)
        ]
        base_metrics = _compute_roi(base_df)
        baseline_rows.append(
            {
                "run_id": run_id,
                "strategy_id": "baseline",
                **base_metrics,
            }
        )
        for win_id, g in base_df.groupby("window_id", dropna=False):
            m = _compute_roi(g)
            window_rows.append(
                {
                    "run_id": run_id,
                    "strategy_id": "baseline",
                    "window_id": win_id,
                    **m,
                }
            )

        # fit band thresholds
        fit_source = "train_valid"
        if args.fit_mode == "walk_forward":
            fit_source = "walk_forward_test"
            df_fit = df_test.copy()
        else:
            ff_train = _find_fullfield(run_dir, fullfield_root, "train")
            ff_valid = _find_fullfield(run_dir, fullfield_root, "valid")
            if ff_train and ff_valid:
                df_train = _prepare_fullfield(
                    _read_fullfield(ff_train), args.market_prob_method, args.model_prob_col, session
                )
                df_valid = _prepare_fullfield(
                    _read_fullfield(ff_valid), args.market_prob_method, args.model_prob_col, session
                )
                df_fit = pd.concat([df_train, df_valid], ignore_index=True)
            else:
                if args.fit_mode == "train_valid":
                    raise SystemExit("train/valid fullfield missing and fit_mode=train_valid")
                fit_source = "walk_forward_test"
                df_fit = df_test.copy()

        best_thr = _fit_band_thresholds(df_fit, band_threshold_grid)
        strategies = list(strategies_base)
        strategies.append({"id": "band_ev_fit", "ev_thresholds": best_thr, "fit_source": fit_source})

        # evaluate strategies
        for strat in strategies:
            cand = _apply_strategy(df_test, strat)
            metrics = _compute_roi(cand)
            results.append(
                {
                    "run_id": run_id,
                    "strategy_id": strat["id"],
                    "market_method": args.market_prob_method,
                    "model_col": args.model_prob_col,
                    "fit_source": strat.get("fit_source", ""),
                    **metrics,
                }
            )

            # window metrics
            for win_id, g in cand.groupby("window_id", dropna=False):
                m = _compute_roi(g)
                window_rows.append(
                    {
                        "run_id": run_id,
                        "strategy_id": strat["id"],
                        "window_id": win_id,
                        **m,
                    }
                )

            # odds band contribution
            if not cand.empty:
                band = cand.groupby("odds_band", dropna=False).apply(_compute_roi).apply(pd.Series).reset_index()
                band["run_id"] = run_id
                band["strategy_id"] = strat["id"]
                band_contrib_rows.append(band)

            # added/removed vs baseline
            cand_keys = set(zip(cand["window_id"], cand["race_id"], cand["horse_no"]))
            added = len(cand_keys - base_keys)
            removed = len(base_keys - cand_keys)
            added_removed_rows.append(
                {
                    "run_id": run_id,
                    "strategy_id": strat["id"],
                    "added": added,
                    "removed": removed,
                }
            )

            if metrics["roi"] > best_roi and metrics["n_bets"] >= 100:
                best_roi = metrics["roi"]
                best_candidate = strat["id"]

    results_df = pd.DataFrame(results)
    window_df = pd.DataFrame(window_rows)
    band_df = pd.concat(band_contrib_rows, ignore_index=True) if band_contrib_rows else pd.DataFrame()
    baseline_df = pd.DataFrame(baseline_rows)
    added_removed_df = pd.DataFrame(added_removed_rows)
    results_all = pd.concat(
        [
            results_df,
            baseline_df.assign(
                market_method=args.market_prob_method,
                model_col=args.model_prob_col,
                fit_source="baseline",
            ),
        ],
        ignore_index=True,
    )

    if not band_df.empty:
        total = band_df.groupby(["run_id", "strategy_id"], dropna=False)["stake"].transform("sum")
        total_profit = band_df.groupby(["run_id", "strategy_id"], dropna=False)["profit"].transform("sum")
        band_df["stake_share"] = band_df["stake"] / total.replace(0, np.nan)
        band_df["profit_contrib"] = band_df["profit"] / total_profit.replace(0, np.nan)

    results_all.to_csv(staging_dir / "segment_strategy_results.csv", index=False, encoding="utf-8")
    window_df.to_csv(staging_dir / "window_metrics.csv", index=False, encoding="utf-8")
    band_df.to_csv(staging_dir / "odds_band_contrib.csv", index=False, encoding="utf-8")
    baseline_df.to_csv(staging_dir / "baseline_metrics.csv", index=False, encoding="utf-8")
    added_removed_df.to_csv(staging_dir / "added_removed.csv", index=False, encoding="utf-8")

    # step14 aggregate (sum profit/stake across windows)
    step14 = (
        window_df.groupby(["strategy_id"], dropna=False)[["profit", "stake"]]
        .sum()
        .reset_index()
    )
    step14["step14_roi"] = step14["profit"] / step14["stake"]
    step14.to_csv(staging_dir / "step14_summary.csv", index=False, encoding="utf-8")

    # window ROI distribution
    window_dist = (
        window_df.groupby("strategy_id", dropna=False)["roi"]
        .agg(min_roi="min", median_roi="median", p90_roi=lambda s: s.quantile(0.9))
        .reset_index()
    )

    # choose best candidate by step14 ROI vs baseline
    baseline_step14 = step14.loc[step14["strategy_id"] == "baseline", "step14_roi"]
    baseline_step14 = float(baseline_step14.values[0]) if len(baseline_step14) else float("nan")
    cand_agg = results_df.groupby("strategy_id", dropna=False).agg(
        n_bets=("n_bets", "sum"),
        stake=("stake", "sum"),
        profit=("profit", "sum"),
    ).reset_index()
    cand_agg["roi"] = cand_agg["profit"] / cand_agg["stake"]
    cand_agg = cand_agg.merge(step14, on="strategy_id", how="left")

    eligible = cand_agg[(cand_agg["n_bets"] >= 100) & (cand_agg["step14_roi"] > baseline_step14)]
    if not eligible.empty:
        best_row = eligible.sort_values("step14_roi", ascending=False).iloc[0]
        best_candidate = str(best_row["strategy_id"])
        best_roi = float(best_row["step14_roi"])
    else:
        best_candidate = None
        best_roi = float("nan")

    # report
    report_lines = [
        "# Phase2 segment strategy evaluation",
        f"- run_dirs: {len(run_dirs)}",
        f"- market_method: {args.market_prob_method}",
        f"- model_prob_col: {args.model_prob_col}",
        "",
        "## Step14 summary (top by ROI)",
        step14.sort_values("step14_roi", ascending=False).head(10).to_string(index=False)
        if not step14.empty
        else "No step14 rows.",
        "",
        "## Results (top by ROI)",
        results_all.sort_values("roi", ascending=False).head(10).to_string(index=False)
        if not results_all.empty
        else "No results rows.",
        "",
        "## Window ROI distribution (min/median/p90)",
        window_dist.to_string(index=False) if not window_dist.empty else "No window dist rows.",
        "",
        "ROI definition: ROI = profit / stake (profit = return - stake).",
    ]
    (staging_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    best_n_bets = "N/A"
    if best_candidate and not cand_agg.empty:
        row = cand_agg[cand_agg["strategy_id"] == best_candidate]
        if not row.empty:
            best_n_bets = int(row.iloc[0]["n_bets"])

    minimal_lines = [
        "# minimal update for chat - Phase2 segment strategy",
        "",
        "## Summary",
        f"- best_candidate: {best_candidate or 'NONE'}",
        f"- best_step14_roi: {best_roi if best_candidate else 'N/A'}",
        f"- best_n_bets: {best_n_bets}",
        "",
        "## Paths",
        f"- staging_dir: {staging_dir}",
        f"- out_dir: {out_dir}",
    ]
    (staging_dir / "minimal_update_for_chat.md").write_text("\n".join(minimal_lines) + "\n", encoding="utf-8")

    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        f"[diag] roi_taskD_segment_strategy_done=true | best_candidate={best_candidate or 'NONE'} | "
        f"step14_roi={best_roi if best_candidate else 'N/A'} | n_bets={best_n_bets}",
        "[plan] decision=segment_strategy_promote | reason=segment_edge"
        if best_candidate
        else "[plan] decision=phase1p5_blend_next | reason=no_candidate",
        f"[paths] out_dir={out_dir} | staging={staging_dir} | zip=<NOT_USED>",
    ]
    (staging_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
