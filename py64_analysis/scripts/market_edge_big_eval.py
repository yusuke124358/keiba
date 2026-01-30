"""
Phase1: Full-field market edge evaluation at scale (edge vs market).
Reads or generates full-field predictions per run_dir, runs integrity checks,
computes p_mkt_norm from odds, evaluates model vs market metrics, and bootstraps CI.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from keiba.db.loader import get_session


RUN_DIR_NAME_RE = "stake_odds_damp_q95_base_2025_w013_022_eval_"

ODDS_CANDIDATES = ["odds", "odds_val", "odds_at_buy", "odds_effective", "odds_final"]
Y_CANDIDATES = ["y_win", "is_win", "winner", "y_true", "y"]
P_MODEL_CANDIDATES = ["p_model", "p_win", "prob", "p_hat"]
P_HAT_CANDIDATES = ["p_hat", "p_cal", "p_blend", "p_used"]
P_MKT_CANDIDATES = ["p_mkt", "p_mkt_race", "p_mkt_raw", "p_market"]
WINDOW_ID_CANDIDATES = ["window_id"]


@dataclass
class RunResult:
    run_id: str
    fullfield_path: Optional[Path]
    generated: bool
    integrity_pass: bool
    dropped_races: int
    dropped_ratio: float


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(score)) + 1
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


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


def _detect_column(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _read_fullfield(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)
    odds_col = _detect_column(cols, ODDS_CANDIDATES)
    y_col = _detect_column(cols, Y_CANDIDATES)
    p_model_col = _detect_column(cols, P_MODEL_CANDIDATES)
    p_hat_col = _detect_column(cols, P_HAT_CANDIDATES)
    p_mkt_col = _detect_column(cols, P_MKT_CANDIDATES)
    window_col = _detect_column(cols, WINDOW_ID_CANDIDATES)

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
        out[p_mkt_col] = pd.to_numeric(df[p_mkt_col], errors="coerce")
    if window_col:
        out["window_id"] = df[window_col].astype(str)
    else:
        out["window_id"] = "unknown"
    return out


def _compute_p_mkt_norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_mkt_raw"] = 1.0 / df["odds"]
    sums = df.groupby("race_id", dropna=False)["p_mkt_raw"].transform("sum")
    df["p_mkt_norm"] = df["p_mkt_raw"] / sums
    return df


def _bootstrap_ci_from_race_sums(
    race_sums: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    if race_sums.empty or n_boot <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    sums_ll_model = race_sums["sum_ll_model"].to_numpy()
    sums_ll_mkt = race_sums["sum_ll_mkt"].to_numpy()
    sums_brier_model = race_sums["sum_brier_model"].to_numpy()
    sums_brier_mkt = race_sums["sum_brier_mkt"].to_numpy()
    counts = race_sums["n_rows"].to_numpy()

    n = len(race_sums)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    idx = rng.integers(0, n, size=(n_boot, n))
    sum_ll_model = sums_ll_model[idx].sum(axis=1)
    sum_ll_mkt = sums_ll_mkt[idx].sum(axis=1)
    sum_brier_model = sums_brier_model[idx].sum(axis=1)
    sum_brier_mkt = sums_brier_mkt[idx].sum(axis=1)
    sum_counts = counts[idx].sum(axis=1)

    delta_ll = (sum_ll_model / sum_counts) - (sum_ll_mkt / sum_counts)
    delta_brier = (sum_brier_model / sum_counts) - (sum_brier_mkt / sum_counts)

    ll_low, ll_high = np.quantile(delta_ll, [0.025, 0.975]).tolist()
    b_low, b_high = np.quantile(delta_brier, [0.025, 0.975]).tolist()
    return (float(ll_low), float(ll_high), float(b_low), float(b_high))


def _metric_row(df: pd.DataFrame, p_col: str, mkt_col: str) -> dict:
    y = df["y_win"].to_numpy()
    p_model = df[p_col].to_numpy()
    p_mkt = df[mkt_col].to_numpy()
    odds = df["odds"].to_numpy()
    return {
        "n_rows": int(len(df)),
        "n_races": int(df["race_key"].nunique()),
        "logloss_model": _logloss(y, p_model),
        "logloss_mkt": _logloss(y, p_mkt),
        "brier_model": _brier(y, p_model),
        "brier_mkt": _brier(y, p_mkt),
        "auc_model": _auc(y, p_model),
        "auc_mkt": _auc(y, p_mkt),
        "mean_ev_model": float(np.mean(p_model * odds - 1.0)),
        "mean_ev_mkt": float(np.mean(p_mkt * odds - 1.0)),
    }


def _race_sums(df: pd.DataFrame, p_col: str, mkt_col: str) -> pd.DataFrame:
    y = df["y_win"].to_numpy()
    p_model = df[p_col].to_numpy()
    p_mkt = df[mkt_col].to_numpy()
    odds = df["odds"].to_numpy()
    ll_model = -(y * np.log(np.clip(p_model, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p_model, 1e-9, 1 - 1e-9)))
    ll_mkt = -(y * np.log(np.clip(p_mkt, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p_mkt, 1e-9, 1 - 1e-9)))
    brier_model = (p_model - y) ** 2
    brier_mkt = (p_mkt - y) ** 2
    tmp = pd.DataFrame(
        {
            "race_key": df["race_key"].values,
            "sum_ll_model": ll_model,
            "sum_ll_mkt": ll_mkt,
            "sum_brier_model": brier_model,
            "sum_brier_mkt": brier_mkt,
            "n_rows": 1,
        }
    )
    return tmp.groupby("race_key", dropna=False).sum().reset_index()


def _run_dir_list(pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
    return sorted(paths, key=lambda p: p.name)


def _find_fullfield(
    run_dir: Path,
    fullfield_root: Optional[Path],
    staging_dir: Path,
    multi_run: bool,
) -> tuple[Optional[Path], bool]:
    candidates: list[Path] = []
    if fullfield_root:
        if fullfield_root.is_file() and fullfield_root.name == "fullfield_preds.csv":
            candidates = [fullfield_root]
        else:
            if (fullfield_root / "fullfield" / "fullfield_preds.csv").exists():
                candidates = [fullfield_root / "fullfield" / "fullfield_preds.csv"]
            else:
                candidates = list(fullfield_root.rglob("fullfield_preds.csv"))

    if candidates:
        matched = [c for c in candidates if run_dir.name in str(c)]
        if len(matched) == 1:
            return matched[0], False
        if not multi_run and len(candidates) == 1:
            return candidates[0], False

    direct = run_dir / "fullfield" / "fullfield_preds.csv"
    if direct.exists():
        return direct, False

    # generate
    out_dir = staging_dir / "fullfield_generated" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "export_fullfield_preds_for_run_dir.py"
    cmd = [sys.executable, str(script_path), "--run_dir", str(run_dir), "--out_dir", str(out_dir)]
    subprocess.run(cmd, check=True)
    gen_path = out_dir / "fullfield" / "fullfield_preds.csv"
    if gen_path.exists():
        return gen_path, True
    return None, True


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


def _integrity_filter(df: pd.DataFrame, field_size_available: bool) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["race_key"] = df["window_id"].astype(str) + "__" + df["race_id"].astype(str)
    df["valid_row"] = (
        df["odds"].notna()
        & (df["odds"] > 0)
        & df["p_mkt_norm"].notna()
        & df["p_model"].notna()
        & df["y_win"].isin([0, 1])
        & np.isfinite(df["odds"])
        & np.isfinite(df["p_mkt_norm"])
        & np.isfinite(df["p_model"])
    )
    # drop invalid rows but keep race if winner label still consistent
    df_valid = df[df["valid_row"]].copy()

    grp_valid = df_valid.groupby("race_key", dropna=False)
    stats = grp_valid.agg(
        n_rows=("race_key", "size"),
        dup_keys=("horse_no", lambda s: s.duplicated().sum()),
        sum_y=("y_win", "sum"),
    ).reset_index()

    invalid_rows = df.groupby("race_key", dropna=False)["valid_row"].apply(lambda s: (~s).sum()).reset_index()
    invalid_rows = invalid_rows.rename(columns={"valid_row": "invalid_rows"})
    stats = stats.merge(invalid_rows, on="race_key", how="left")
    stats["invalid_rows"] = stats["invalid_rows"].fillna(0).astype(int)

    if field_size_available:
        field_map = df.groupby("race_key", dropna=False)["field_size"].first().reset_index()
        stats = stats.merge(field_map, on="race_key", how="left")
        stats["field_size_mismatch"] = stats["field_size"].notna() & (stats["field_size"] != stats["n_rows"])
    else:
        stats["field_size_mismatch"] = False

    stats["sum_y_ok"] = stats["sum_y"] == 1
    stats["race_ok"] = (stats["dup_keys"] == 0) & (stats["sum_y_ok"]) & (stats["n_rows"] > 0)

    bad_races = set(stats.loc[~stats["race_ok"], "race_key"].astype(str).tolist())
    filtered = df_valid[~df_valid["race_key"].isin(bad_races)].copy()

    summary = {
        "total_races": int(stats["race_key"].nunique()),
        "dropped_races": int(len(bad_races)),
        "dropped_ratio": float(len(bad_races) / stats["race_key"].nunique()) if stats["race_key"].nunique() else 0.0,
        "dup_races": int((stats["dup_keys"] > 0).sum()),
        "sumy_bad_races": int((~stats["sum_y_ok"]).sum()),
        "invalid_rows_races": int((stats["invalid_rows"] > 0).sum()),
        "field_size_mismatch_races": int(stats["field_size_mismatch"].sum()),
    }
    return filtered, summary


def _p_mkt_diff_stats(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in [c for c in df.columns if c in P_MKT_CANDIDATES]:
        s = df[col].astype(float)
        base = df["p_mkt_norm"].astype(float)
        mask = np.isfinite(s) & np.isfinite(base)
        if mask.sum() == 0:
            continue
        diff = (s[mask] - base[mask]).abs()
        corr = np.corrcoef(s[mask], base[mask])[0, 1] if mask.sum() > 1 else float("nan")
        rows.append(
            {
                "p_mkt_col": col,
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "corr": float(corr),
                "n_rows": int(mask.sum()),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Market edge fullscale evaluation")
    ap.add_argument("--run_dir_glob", required=True)
    ap.add_argument("--fullfield_root", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--staging_dir", required=True)
    ap.add_argument("--bootstrap_n", type=int, default=500)
    ap.add_argument("--min_races", type=int, default=500)
    ap.add_argument("--track_min_races", type=int, default=200)
    ap.add_argument(
        "--segments",
        default="odds_band,field_size_bin,month,surface,distance_bin,track_code",
    )
    ap.add_argument(
        "--market_prob_method",
        default="p_mkt_norm",
        help="Comma-separated: p_mkt_col,p_mkt_norm,p_mkt_raw,all",
    )
    args = ap.parse_args()

    run_dirs = _run_dir_list(args.run_dir_glob)
    if not run_dirs:
        raise SystemExit(f"No run dirs matched: {args.run_dir_glob}")

    out_dir = Path(args.out_dir)
    staging_dir = Path(args.staging_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    fullfield_root = Path(args.fullfield_root) if args.fullfield_root else None

    session = get_session()

    overall_rows = []
    window_rows = []
    segment_rows = []
    integrity_rows = []
    pmkt_rows = []
    overround_rows = []
    bootstrap_rows = []

    rng = np.random.default_rng(42)
    generated_count = 0
    total_dropped_races = 0

    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    method_arg = (args.market_prob_method or "p_mkt_norm").lower()
    if method_arg == "all":
        market_methods = ["p_mkt_col", "p_mkt_norm", "p_mkt_raw"]
    else:
        market_methods = [m.strip() for m in method_arg.split(",") if m.strip()]
    primary_method = "p_mkt_col" if "p_mkt_col" in market_methods else market_methods[0]

    for run_dir in run_dirs:
        run_id = run_dir.name
        fullfield_path, generated = _find_fullfield(run_dir, fullfield_root, staging_dir, multi_run=len(run_dirs) > 1)
        if generated:
            generated_count += 1
        if fullfield_path is None or not fullfield_path.exists():
            integrity_rows.append(
                {
                    "run_id": run_id,
                    "total_races": 0,
                    "dropped_races": 0,
                    "dropped_ratio": 1.0,
                    "dup_races": 0,
                    "sumy_bad_races": 0,
                    "invalid_rows_races": 0,
                    "field_size_mismatch_races": 0,
                    "integrity_pass": False,
                    "note": "fullfield_missing",
                }
            )
            continue

        df = _read_fullfield(fullfield_path)
        df = _compute_p_mkt_norm(df)
        # choose p_mkt_col if present
        p_mkt_col_name = None
        for cand in P_MKT_CANDIDATES:
            if cand in df.columns:
                p_mkt_col_name = cand
                break
        if p_mkt_col_name:
            df["p_mkt_col"] = df[p_mkt_col_name]
        else:
            df["p_mkt_col"] = np.nan

        race_meta = _fetch_race_meta(session, df["race_id"].unique().tolist())
        field_size_available = False
        if not race_meta.empty:
            df = df.merge(race_meta, on="race_id", how="left")
            field_size_available = "field_size" in df.columns

        df["odds_band"] = df["odds"].apply(lambda v: _odds_band(float(v)) if np.isfinite(v) else "unknown")
        if field_size_available:
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

        if "track_code" not in df.columns:
            df["track_code"] = "unknown"

        if "date" in df.columns:
            df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
        else:
            df["month"] = "unknown"

        df_clean, integ = _integrity_filter(df, field_size_available)
        total_dropped_races += integ["dropped_races"]
        integrity_pass = integ["dropped_ratio"] <= 0.005
        integ_row = {
            "run_id": run_id,
            **integ,
            "integrity_pass": integrity_pass,
            "note": "",
        }
        integrity_rows.append(integ_row)

        df_clean = df_clean.copy()
        df_clean["run_id"] = run_id

        pmkt_stats = _p_mkt_diff_stats(df_clean)
        for row in pmkt_stats:
            pmkt_rows.append({"run_id": run_id, **row})

        # overround stats (sum of 1/odds by race)
        overround = df_clean.groupby("race_id", dropna=False)["p_mkt_raw"].sum()
        if not overround.empty:
            overround_rows.append(
                {
                    "run_id": run_id,
                    "overround_mean": float(overround.mean()),
                    "overround_p50": float(overround.quantile(0.5)),
                    "overround_p90": float(overround.quantile(0.9)),
                    "overround_max": float(overround.max()),
                    "overround_min": float(overround.min()),
                }
            )

        model_variants = ["p_model"]
        if "p_hat" in df_clean.columns:
            model_variants.append("p_hat")

        for market_method in market_methods:
            if market_method == "p_mkt_col":
                df_clean["p_mkt_used"] = df_clean["p_mkt_col"]
            elif market_method == "p_mkt_raw":
                df_clean["p_mkt_used"] = df_clean["p_mkt_raw"]
            else:
                df_clean["p_mkt_used"] = df_clean["p_mkt_norm"]

            # fallback if p_mkt_col missing
            if market_method == "p_mkt_col" and df_clean["p_mkt_used"].isna().all():
                df_clean["p_mkt_used"] = df_clean["p_mkt_norm"]

            df_eval = df_clean[np.isfinite(df_clean["p_mkt_used"])].copy()

            for p_col in model_variants:
                metrics = _metric_row(df_eval, p_col, "p_mkt_used")
                delta_logloss = metrics["logloss_model"] - metrics["logloss_mkt"]
                delta_brier = metrics["brier_model"] - metrics["brier_mkt"]
                delta_auc = metrics["auc_model"] - metrics["auc_mkt"]

                race_sums = _race_sums(df_eval, p_col, "p_mkt_used")
                ll_low, ll_high, b_low, b_high = _bootstrap_ci_from_race_sums(
                    race_sums, args.bootstrap_n, rng
                )

                overall_rows.append(
                    {
                        "run_id": run_id,
                        "market_method": market_method,
                        "model_variant": p_col,
                        "n_races": metrics["n_races"],
                        "n_rows": metrics["n_rows"],
                        "logloss_model": metrics["logloss_model"],
                        "logloss_mkt": metrics["logloss_mkt"],
                        "delta_logloss": delta_logloss,
                        "delta_logloss_ci_low": ll_low,
                        "delta_logloss_ci_high": ll_high,
                        "brier_model": metrics["brier_model"],
                        "brier_mkt": metrics["brier_mkt"],
                        "delta_brier": delta_brier,
                        "delta_brier_ci_low": b_low,
                        "delta_brier_ci_high": b_high,
                        "auc_model": metrics["auc_model"],
                        "auc_mkt": metrics["auc_mkt"],
                        "delta_auc": delta_auc,
                        "mean_ev_model": metrics["mean_ev_model"],
                        "mean_ev_mkt": metrics["mean_ev_mkt"],
                        "dropped_races": integ["dropped_races"],
                        "dropped_ratio": integ["dropped_ratio"],
                        "integrity_pass": integrity_pass,
                    }
                )
                bootstrap_rows.append(
                    {
                        "run_id": run_id,
                        "market_method": market_method,
                        "model_variant": p_col,
                        "group_type": "overall",
                        "segment": "all",
                        "segment_value": "all",
                        "n_races": metrics["n_races"],
                        "delta_logloss_ci_low": ll_low,
                        "delta_logloss_ci_high": ll_high,
                        "delta_brier_ci_low": b_low,
                        "delta_brier_ci_high": b_high,
                        "bootstrap_n": args.bootstrap_n,
                    }
                )

                # window-level metrics
                for window_id, g in df_eval.groupby("window_id", dropna=False):
                    w_metrics = _metric_row(g, p_col, "p_mkt_used")
                    window_rows.append(
                        {
                            "run_id": run_id,
                            "market_method": market_method,
                            "window_id": window_id,
                            "model_variant": p_col,
                            "n_races": w_metrics["n_races"],
                            "n_rows": w_metrics["n_rows"],
                            "logloss_model": w_metrics["logloss_model"],
                            "logloss_mkt": w_metrics["logloss_mkt"],
                            "delta_logloss": w_metrics["logloss_model"] - w_metrics["logloss_mkt"],
                            "brier_model": w_metrics["brier_model"],
                            "brier_mkt": w_metrics["brier_mkt"],
                            "delta_brier": w_metrics["brier_model"] - w_metrics["brier_mkt"],
                            "auc_model": w_metrics["auc_model"],
                            "auc_mkt": w_metrics["auc_mkt"],
                            "delta_auc": w_metrics["auc_model"] - w_metrics["auc_mkt"],
                        }
                    )

                # segment-level metrics
                for seg in segments:
                    if seg not in df_eval.columns:
                        continue
                    seg_groups = df_eval.groupby(seg, dropna=False)
                    seg_window = df_eval.groupby([seg, "window_id"], dropna=False)
                    window_delta_rows = []
                    for (seg_val, win_id), sub in seg_window:
                        m = _metric_row(sub, p_col, "p_mkt_used")
                        window_delta_rows.append(
                            {
                                seg: seg_val,
                                "window_id": win_id,
                                **m,
                            }
                        )
                    window_delta = pd.DataFrame(window_delta_rows)
                    if not window_delta.empty:
                        window_delta["delta_logloss"] = (
                            window_delta["logloss_model"] - window_delta["logloss_mkt"]
                        )
                    for seg_val, g in seg_groups:
                        if seg == "track_code":
                            eligible = g["race_key"].nunique() >= args.track_min_races
                            eligible_500 = g["race_key"].nunique() >= args.min_races
                        else:
                            eligible = g["race_key"].nunique() >= args.min_races
                            eligible_500 = eligible
                        metrics = _metric_row(g, p_col, "p_mkt_used")
                        delta_logloss = metrics["logloss_model"] - metrics["logloss_mkt"]
                        delta_brier = metrics["brier_model"] - metrics["brier_mkt"]
                        delta_auc = metrics["auc_model"] - metrics["auc_mkt"]

                        ll_low = ll_high = b_low = b_high = float("nan")
                        if eligible:
                            race_sums = _race_sums(g, p_col, "p_mkt_used")
                            ll_low, ll_high, b_low, b_high = _bootstrap_ci_from_race_sums(
                                race_sums, args.bootstrap_n, rng
                            )
                            bootstrap_rows.append(
                                {
                                    "run_id": run_id,
                                    "market_method": market_method,
                                    "model_variant": p_col,
                                    "group_type": "segment",
                                    "segment": seg,
                                    "segment_value": seg_val,
                                    "n_races": metrics["n_races"],
                                    "delta_logloss_ci_low": ll_low,
                                    "delta_logloss_ci_high": ll_high,
                                    "delta_brier_ci_low": b_low,
                                    "delta_brier_ci_high": b_high,
                                    "bootstrap_n": args.bootstrap_n,
                                }
                            )

                        win_good = 0
                        win_total = 0
                        if not window_delta.empty:
                            w = window_delta[window_delta[seg] == seg_val]
                            win_total = int(w["window_id"].nunique())
                            win_good = int((w["delta_logloss"] < 0).sum())

                        segment_rows.append(
                            {
                                "run_id": run_id,
                                "market_method": market_method,
                                "model_variant": p_col,
                                "segment": seg,
                                "segment_value": seg_val,
                                "n_races": metrics["n_races"],
                                "n_rows": metrics["n_rows"],
                                "logloss_model": metrics["logloss_model"],
                                "logloss_mkt": metrics["logloss_mkt"],
                                "delta_logloss": delta_logloss,
                                "delta_logloss_ci_low": ll_low,
                                "delta_logloss_ci_high": ll_high,
                                "brier_model": metrics["brier_model"],
                                "brier_mkt": metrics["brier_mkt"],
                                "delta_brier": delta_brier,
                                "delta_brier_ci_low": b_low,
                                "delta_brier_ci_high": b_high,
                                "auc_model": metrics["auc_model"],
                                "auc_mkt": metrics["auc_mkt"],
                                "delta_auc": delta_auc,
                                "windows_total": win_total,
                                "windows_good": win_good,
                                "eligible_500": eligible_500,
                                "eligible_200": g["race_key"].nunique() >= args.track_min_races,
                            }
                        )

    # write outputs
    overall_df = pd.DataFrame(overall_rows)
    window_df = pd.DataFrame(window_rows)
    segment_df = pd.DataFrame(segment_rows)
    integrity_df = pd.DataFrame(integrity_rows)
    pmkt_df = pd.DataFrame(pmkt_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    overround_df = pd.DataFrame(overround_rows)

    overall_df.to_csv(staging_dir / "edge_overall_by_market_method.csv", index=False, encoding="utf-8")
    segment_df.to_csv(staging_dir / "edge_by_segment_by_market_method.csv", index=False, encoding="utf-8")
    window_df.to_csv(staging_dir / "edge_by_window.csv", index=False, encoding="utf-8")
    if not overall_df.empty:
        overall_df[overall_df["market_method"] == primary_method].to_csv(
            staging_dir / "edge_overall.csv", index=False, encoding="utf-8"
        )
    if not segment_df.empty:
        segment_df[segment_df["market_method"] == primary_method].to_csv(
            staging_dir / "edge_by_segment.csv", index=False, encoding="utf-8"
        )
    integrity_df.to_csv(staging_dir / "integrity_summary.csv", index=False, encoding="utf-8")
    pmkt_df.to_csv(staging_dir / "p_mkt_diff_stats.csv", index=False, encoding="utf-8")
    bootstrap_df.to_csv(staging_dir / "bootstrap_ci.csv", index=False, encoding="utf-8")
    overround_df.to_csv(staging_dir / "overround_stats.csv", index=False, encoding="utf-8")

    # decision
    edge_detected_segments = 0
    if not segment_df.empty:
        def _edge_ok(row: pd.Series) -> bool:
            if not row.get("eligible_500", False):
                return False
            if not np.isfinite(row.get("delta_logloss_ci_high", np.nan)):
                return False
            if not np.isfinite(row.get("delta_brier_ci_high", np.nan)):
                return False
            if row["delta_logloss"] >= 0:
                return False
            if row["delta_logloss_ci_high"] >= 0:
                return False
            if row["delta_brier_ci_high"] > 0:
                return False
            if row.get("windows_total", 0) >= 10 and row.get("windows_good", 0) < 7:
                return False
            return True

        seg_primary = segment_df[segment_df["market_method"] == primary_method]
        edge_detected_segments = int(seg_primary.apply(_edge_ok, axis=1).sum())

    edge_detected_any = edge_detected_segments > 0
    decision = "segment_strategy_next" if edge_detected_any else "model_update_next"
    reason = "segment_edge" if edge_detected_any else "no_edge"

    n_runs = len(run_dirs)
    dropped_ratio = (
        float(total_dropped_races) / integrity_df["total_races"].sum()
        if not integrity_df.empty and integrity_df["total_races"].sum() > 0
        else 0.0
    )

    # report
    report_lines = [
        "# Market edge fullscale evaluation (Phase1)",
        "",
        f"- run_dirs: {n_runs}",
        f"- run_dir_glob: {args.run_dir_glob}",
        f"- fullfield_root: {fullfield_root}",
        f"- bootstrap_n: {args.bootstrap_n}",
        f"- market_methods: {','.join(market_methods)}",
        f"- primary_method: {primary_method}",
        "",
        "## Integrity summary",
    ]
    if not integrity_df.empty:
        report_lines.append(integrity_df.to_string(index=False))
    else:
        report_lines.append("No integrity rows.")
        report_lines.extend(
        [
            "",
            "## Overround stats (sum 1/odds by race)",
            overround_df.to_string(index=False) if not overround_df.empty else "No overround stats.",
            "",
            "## Overall edge (first rows)",
            overall_df.head(10).to_string(index=False) if not overall_df.empty else "No overall rows.",
            "",
            "## Segment edge (top 10 by delta_logloss)",
            segment_df.sort_values("delta_logloss").head(10).to_string(index=False)
            if not segment_df.empty
            else "No segment rows.",
            "",
            "## Decision",
            f"- edge_detected_segments={edge_detected_segments}",
            f"- decision={decision} ({reason})",
            "",
            "Note: This Phase1 evaluates full-field edge vs market; ROI is not computed here.",
            "ROI definition: ROI = profit / stake (profit = return - stake).",
        ]
    )
    (staging_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    minimal_lines = [
        "# minimal update for chat - Market edge fullscale (Phase1)",
        "",
        "## Summary",
        f"- run_dirs: {n_runs}",
        f"- fullfield_generated: {generated_count}",
        f"- dropped_races_total: {total_dropped_races}",
        f"- dropped_races_ratio: {dropped_ratio:.6f}",
        f"- edge_detected_any: {edge_detected_any}",
        f"- edge_detected_segments: {edge_detected_segments}",
        f"- decision: {decision} ({reason})",
        "",
        "## Paths",
        f"- staging_dir: {staging_dir}",
        f"- out_dir: {out_dir}",
    ]
    (staging_dir / "minimal_update_for_chat.md").write_text(
        "\n".join(minimal_lines) + "\n", encoding="utf-8"
    )

    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        f"[diag] market_edge_big_eval_done=true | n_runs={n_runs} | fullfield_generated={generated_count} | "
        f"dropped_races={total_dropped_races} | dropped_races_ratio={dropped_ratio:.6f}",
        f"[diag] edge_detected_any={str(edge_detected_any).lower()} | edge_detected_segments={edge_detected_segments} | "
        f"criteria=min_races_{args.min_races}_bootstrap{args.bootstrap_n}",
        f"[plan] decision={decision} | reason={reason}",
        f"[paths] out_dir={out_dir} | staging={staging_dir} | zip=<NOT_USED>",
    ]
    (staging_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
