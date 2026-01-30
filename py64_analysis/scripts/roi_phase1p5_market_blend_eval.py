"""
Phase1.5: market-anchored blend (walk-forward, leak-free).
Fit blend + thresholds on train+valid only per window, evaluate on test (step14 decision).
Outputs are intended for text-only delivery (no zip).
"""
from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


_ensure_import_path()
from keiba.betting.market_blend import logit_blend_prob, odds_band  # noqa: E402

ODDS_PRIORITY = ["odds_val", "odds", "odds_buy", "odds_at_buy", "odds_effective"]
Y_PRIORITY = ["y_win", "is_win", "winner", "y_true", "y"]
P_MODEL_PRIORITY = ["p_model", "p_hat", "p_win", "prob"]
P_HAT_PRIORITY = ["p_hat", "p_cal", "p_used"]
P_MKT_PRIORITY = ["p_mkt_col", "p_mkt", "p_mkt_race", "p_mkt_raw", "p_market"]

WINDOW_RE = re.compile(r"(w\d{3}_\d{8}_\d{8})")


@dataclass
class IntegrityStats:
    total_races: int
    dropped_races: int
    dropped_ratio: float
    dup_rows: int
    invalid_rows: int
    missing_y_races: int


def _detect_column(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _infer_window_id(path: Path) -> Optional[str]:
    match = WINDOW_RE.search(str(path))
    if match:
        return match.group(1)
    return None


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




def _run_dirs(pattern: str) -> list[Path]:
    return sorted([Path(p) for p in glob.glob(pattern) if Path(p).is_dir()], key=lambda p: p.name)


def _window_dirs(run_dir: Path) -> list[Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and WINDOW_RE.match(p.name)], key=lambda p: p.name)


def _check_race_ids(run_dir: Path) -> list[str]:
    missing: list[str] = []
    for window_dir in _window_dirs(run_dir):
        for split in ("train", "valid", "test"):
            path = window_dir / f"race_ids_{split}.txt"
            if not path.exists():
                missing.append(str(path))
    return missing


def _resolve_fullfield_path(run_dir: Path, staging_dir: Path, split: str, allow_export: bool) -> Path:
    # 0) staging_dir/fullfield_generated (reuse if already built)
    staging_fullfield = staging_dir / "fullfield_generated" / run_dir.name / "fullfield"
    if split == "test":
        alias = staging_fullfield / "fullfield_preds.csv"
        if alias.exists():
            return alias
    cand = staging_fullfield / f"fullfield_{split}.csv"
    if cand.exists():
        return cand

    # 1) run_dir/fullfield
    local_fullfield = run_dir / "fullfield"
    if split == "test":
        alias = local_fullfield / "fullfield_preds.csv"
        if alias.exists():
            return alias
    cand = local_fullfield / f"fullfield_{split}.csv"
    if cand.exists():
        return cand

    # 2) output_staging/market_edge_big_eval_* (reuse if exists)
    staging_root = Path(r"C:\Users\yyosh\keiba\output_staging")
    for base in sorted(staging_root.glob("market_edge_big_eval_*"), reverse=True):
        alt = base / "fullfield_generated" / run_dir.name / "fullfield" / f"fullfield_{split}.csv"
        if split == "test":
            alias = base / "fullfield_generated" / run_dir.name / "fullfield" / "fullfield_preds.csv"
            if alias.exists():
                return alias
        if alt.exists():
            return alt

    if not allow_export:
        raise ValueError(f"fullfield_{split}.csv not found (export disabled): {run_dir}")

    # 3) generate into staging_dir/fullfield_generated/<run_id>
    out_dir = staging_dir / "fullfield_generated" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "export_fullfield_preds_for_run_dir.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--run_dir",
        str(run_dir),
        "--out_dir",
        str(out_dir),
        "--splits",
        "train,valid,test",
        "--overwrite",
        "false",
    ]
    subprocess.run(cmd, check=True)
    if split == "test":
        alias = out_dir / "fullfield" / "fullfield_preds.csv"
        if alias.exists():
            return alias
    cand = out_dir / "fullfield" / f"fullfield_{split}.csv"
    if not cand.exists():
        raise SystemExit(f"fullfield_{split}.csv not found after export for {run_dir}")
    return cand


def _read_fullfield(path: Path, market_method: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    cols = list(df.columns)
    odds_col = _detect_column(cols, ODDS_PRIORITY)
    y_col = _detect_column(cols, Y_PRIORITY)
    p_model_col = _detect_column(cols, P_MODEL_PRIORITY)
    p_hat_col = _detect_column(cols, P_HAT_PRIORITY)
    p_mkt_col = _detect_column(cols, P_MKT_PRIORITY)
    if odds_col is None or y_col is None or p_model_col is None:
        raise ValueError(f"Missing required columns in {path}")
    if market_method == "p_mkt_col" and p_mkt_col is None:
        raise ValueError(f"Missing p_mkt_col in {path} (market_method=p_mkt_col)")

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

    if "window_id" in cols:
        out["window_id"] = df["window_id"].astype(str)
    else:
        inferred = _infer_window_id(path)
        if inferred:
            out["window_id"] = inferred
        else:
            out["window_id"] = "unknown"

    meta = {
        "odds_col": odds_col,
        "y_col": y_col,
        "p_model_col": p_model_col,
        "p_hat_col": p_hat_col or "",
        "p_mkt_col": p_mkt_col or "",
    }
    return out, meta


def _compute_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_mkt_raw"] = 1.0 / df["odds"]
    sums = df.groupby(["window_id", "race_id"], dropna=False)["p_mkt_raw"].transform("sum")
    df["p_mkt_norm"] = df["p_mkt_raw"] / sums
    return df


def _select_market_prob(df: pd.DataFrame, method: str) -> pd.Series:
    if method == "p_mkt_col":
        if "p_mkt_col" in df.columns and df["p_mkt_col"].notna().any():
            return df["p_mkt_col"]
        raise SystemExit("market_method=p_mkt_col but column missing or all null")
    if method == "p_mkt_raw":
        return df["p_mkt_raw"]
    return df["p_mkt_norm"]


def _clean_fullfield(df: pd.DataFrame) -> tuple[pd.DataFrame, IntegrityStats]:
    df = df.copy()
    df["race_key"] = df["window_id"].astype(str) + "|" + df["race_id"].astype(str)
    total_races = df["race_key"].nunique()
    dup_rows = int(df.duplicated(["window_id", "race_id", "horse_no"]).sum())
    if dup_rows:
        df = df.drop_duplicates(["window_id", "race_id", "horse_no"])

    invalid_mask = (
        df["odds"].isna()
        | (df["odds"] <= 0)
        | ~np.isfinite(df["p_model"])
        | ~np.isfinite(df["p_mkt_used"])
        | ~df["y_win"].isin([0, 1])
    )
    invalid_rows = int(invalid_mask.sum())
    df = df[~invalid_mask].copy()

    y_sum = df.groupby("race_key", dropna=False)["y_win"].sum()
    valid_races = y_sum[y_sum == 1].index
    missing_y_races = int((y_sum != 1).sum())
    df = df[df["race_key"].isin(valid_races)].copy()

    dropped_races = total_races - len(valid_races)
    dropped_ratio = dropped_races / total_races if total_races else 0.0
    stats = IntegrityStats(
        total_races=total_races,
        dropped_races=dropped_races,
        dropped_ratio=dropped_ratio,
        dup_rows=dup_rows,
        invalid_rows=invalid_rows,
        missing_y_races=missing_y_races,
    )
    return df, stats


def _apply_candidate(df: pd.DataFrame, t_ev: float, odds_cap: Optional[float], exclude_bands: tuple[str, ...]) -> pd.DataFrame:
    out = df
    if odds_cap is not None:
        out = out[out["odds"] <= odds_cap]
    if exclude_bands:
        out = out[~out["odds_band"].isin(exclude_bands)]
    out = out[out["ev_blend"] >= t_ev]
    return out


def _compute_roi(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_bets": 0, "stake": 0.0, "return": 0.0, "profit": 0.0, "roi": float("nan"), "max_dd": 0.0}
    stake = float(len(df))
    ret = float((df["odds"] * df["y_win"]).sum())
    profit = ret - stake
    roi = profit / stake if stake > 0 else float("nan")
    pnl = (df["odds"] * df["y_win"] - 1.0).to_numpy()
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0
    return {"n_bets": int(len(df)), "stake": stake, "return": ret, "profit": profit, "roi": roi, "max_dd": max_dd}


def _race_sums(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    y = df["y_win"].to_numpy()
    a = df[col_a].to_numpy()
    b = df[col_b].to_numpy()
    ll_a = -(y * np.log(np.clip(a, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - a, 1e-9, 1 - 1e-9)))
    ll_b = -(y * np.log(np.clip(b, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - b, 1e-9, 1 - 1e-9)))
    brier_a = (a - y) ** 2
    brier_b = (b - y) ** 2
    tmp = pd.DataFrame(
        {
            "race_key": df["race_key"].values,
            "sum_ll_a": ll_a,
            "sum_ll_b": ll_b,
            "sum_brier_a": brier_a,
            "sum_brier_b": brier_b,
            "n_rows": 1,
        }
    )
    return tmp.groupby("race_key", dropna=False).sum().reset_index()


def _bootstrap_ci_delta(race_sums: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    if race_sums.empty or n_boot <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    n = len(race_sums)
    idx = rng.integers(0, n, size=(n_boot, n))
    ll_a = race_sums["sum_ll_a"].to_numpy()
    ll_b = race_sums["sum_ll_b"].to_numpy()
    brier_a = race_sums["sum_brier_a"].to_numpy()
    brier_b = race_sums["sum_brier_b"].to_numpy()
    counts = race_sums["n_rows"].to_numpy()
    sum_ll_a = ll_a[idx].sum(axis=1)
    sum_ll_b = ll_b[idx].sum(axis=1)
    sum_brier_a = brier_a[idx].sum(axis=1)
    sum_brier_b = brier_b[idx].sum(axis=1)
    sum_counts = counts[idx].sum(axis=1)
    delta_ll = (sum_ll_a / sum_counts) - (sum_ll_b / sum_counts)
    delta_brier = (sum_brier_a / sum_counts) - (sum_brier_b / sum_counts)
    ll_low, ll_high = np.quantile(delta_ll, [0.025, 0.975]).tolist()
    b_low, b_high = np.quantile(delta_brier, [0.025, 0.975]).tolist()
    return float(ll_low), float(ll_high), float(b_low), float(b_high)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase1.5 market-anchored blend (leak-free)")
    ap.add_argument("--run_dir_glob", required=True)
    ap.add_argument("--market_method", default="p_mkt_col")
    ap.add_argument("--bootstrap_n", type=int, default=500)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--staging_dir", required=True)
    ap.add_argument("--no_export", action="store_true")
    args = ap.parse_args()

    run_dirs = _run_dirs(args.run_dir_glob)
    if not run_dirs:
        raise SystemExit(f"No run dirs matched: {args.run_dir_glob}")

    out_dir = Path(args.out_dir)
    staging_dir = Path(args.staging_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    missing_race_ids = []
    for run_dir in run_dirs:
        missing_race_ids.extend(_check_race_ids(run_dir))
    if missing_race_ids:
        minimal_lines = [
            "# minimal update for chat - Phase1.5 market blend",
            "",
            "## Summary",
            "- decision: need_data_fix",
            "- reason: race_ids_{train,valid,test}.txt missing",
            "",
            "## Missing files (first 20)",
        ]
        minimal_lines.extend(missing_race_ids[:20])
        (staging_dir / "minimal_update_for_chat.md").write_text("\n".join(minimal_lines) + "\n", encoding="utf-8")
        stdout_lines = [
            "[audit] agents_md_updated=false | agents_md_zip_included=false",
            "[tests] pytest_passed=false",
            f"[diag] phase1p5_blend_done=false | market_method={args.market_method} | n_runs={len(run_dirs)} | n_windows=0 | dropped_races=0 | dropped_races_ratio=0",
            "[diag] best_candidate=NONE | best_step14_roi=N/A | best_n_bets=N/A | best_params=N/A",
            "[plan] decision=need_data_fix | reason=missing_race_ids",
            f"[paths] out_dir={out_dir} | staging={staging_dir} | zip=<NOT_USED>",
        ]
        (staging_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
        return

    rng = np.random.default_rng(42)
    t_grid = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
    odds_caps = [None, 20, 30]
    exclude_bands_list = [(), ("20+",), ("5-10",), ("3-5",)]
    candidate_grid = []
    for t_ev in t_grid:
        for cap in odds_caps:
            for exclude in exclude_bands_list:
                candidate_grid.append(
                    {
                        "t_ev": t_ev,
                        "odds_cap": cap,
                        "exclude_bands": exclude,
                        "candidate_id": f"t{t_ev:.2f}_cap{cap if cap is not None else 'none'}_ex{'-'.join(exclude) if exclude else 'none'}",
                    }
                )

    metrics_rows = []
    params_rows = []
    cand_train_rows = []
    cand_test_rows = []
    cand_test_window_rows = []
    integrity_rows = []
    window_param_rows = []
    run_windows = 0

    for run_dir in run_dirs:
        run_id = run_dir.name
        try:
            ff_train = _resolve_fullfield_path(run_dir, staging_dir, "train", allow_export=not args.no_export)
            ff_valid = _resolve_fullfield_path(run_dir, staging_dir, "valid", allow_export=not args.no_export)
            ff_test = _resolve_fullfield_path(run_dir, staging_dir, "test", allow_export=not args.no_export)
        except Exception as exc:
            minimal_lines = [
                "# minimal update for chat - Phase1.5 market blend",
                "",
                "## Summary",
                "- decision: need_data_fix",
                "- reason: full-field splits missing",
                f"- detail: {exc}",
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
                f"[diag] phase1p5_blend_done=false | market_method={args.market_method} | n_runs={len(run_dirs)} | n_windows={run_windows} | dropped_races=0 | dropped_races_ratio=0",
                "[diag] best_candidate=NONE | best_step14_roi=N/A | best_n_bets=N/A | best_params=N/A",
                "[plan] decision=need_data_fix | reason=missing_fullfield",
                f"[paths] out_dir={out_dir} | staging={staging_dir} | zip=<NOT_USED>",
            ]
            (staging_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
            return

        try:
            df_train, meta_train = _read_fullfield(ff_train, args.market_method)
            df_valid, meta_valid = _read_fullfield(ff_valid, args.market_method)
            df_test, meta_test = _read_fullfield(ff_test, args.market_method)
        except Exception as exc:
            minimal_lines = [
                "# minimal update for chat - Phase1.5 market blend",
                "",
                "## Summary",
                "- decision: need_data_fix",
                "- reason: required columns missing in full-field",
                f"- detail: {exc}",
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
                f"[diag] phase1p5_blend_done=false | market_method={args.market_method} | n_runs={len(run_dirs)} | n_windows={run_windows} | dropped_races=0 | dropped_races_ratio=0",
                "[diag] best_candidate=NONE | best_step14_roi=N/A | best_n_bets=N/A | best_params=N/A",
                "[plan] decision=need_data_fix | reason=missing_columns",
                f"[paths] out_dir={out_dir} | staging={staging_dir} | zip=<NOT_USED>",
            ]
            (staging_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
            return

        # compute market probs
        df_train = _compute_market_probs(df_train)
        df_valid = _compute_market_probs(df_valid)
        df_test = _compute_market_probs(df_test)

        for df in (df_train, df_valid, df_test):
            df["p_mkt_used"] = _select_market_prob(df, args.market_method)
            df["p_model_used"] = df["p_model"]
            df["odds_band"] = df["odds"].apply(odds_band)

        # integrity (test only for evaluation stats)
        df_test, integrity = _clean_fullfield(df_test)
        integrity_rows.append(
            {
                "run_id": run_id,
                "total_races": integrity.total_races,
                "dropped_races": integrity.dropped_races,
                "dropped_ratio": integrity.dropped_ratio,
                "dup_rows": integrity.dup_rows,
                "invalid_rows": integrity.invalid_rows,
                "missing_y_races": integrity.missing_y_races,
            }
        )

        # Align train/valid with test windows
        for df in (df_train, df_valid):
            df["race_key"] = df["window_id"].astype(str) + "|" + df["race_id"].astype(str)
        df_test = df_test.copy()

        for window_id in sorted(df_test["window_id"].unique()):
            tr = df_train[df_train["window_id"] == window_id].copy()
            va = df_valid[df_valid["window_id"] == window_id].copy()
            te = df_test[df_test["window_id"] == window_id].copy()
            if tr.empty or va.empty or te.empty:
                continue
            run_windows += 1

            # Clean train/valid/test for blending
            def _clean(df):
                df = df[
                    df["y_win"].isin([0, 1])
                    & np.isfinite(df["p_model_used"])
                    & np.isfinite(df["p_mkt_used"])
                    & df["odds"].notna()
                    & (df["odds"] > 0)
                ].copy()
                df["race_key"] = df["window_id"].astype(str) + "|" + df["race_id"].astype(str)
                return df

            tr = _clean(tr)
            va = _clean(va)
            te = _clean(te)
            if tr.empty or va.empty or te.empty:
                continue

            train_valid = pd.concat([tr, va], ignore_index=True)

            # fit logit blend weight on train+valid
            p_model_tv = train_valid["p_model_used"].to_numpy()
            p_mkt_tv = train_valid["p_mkt_used"].to_numpy()
            y_tv = train_valid["y_win"].to_numpy()
            best_w = 0.0
            best_ll = float("inf")
            for w in np.linspace(0, 1, 21):
                p_blend = logit_blend_prob(p_model_tv, p_mkt_tv, w)
                ll = _logloss(y_tv, p_blend)
                if ll < best_ll:
                    best_ll = ll
                    best_w = float(w)

            params_rows.append(
                {
                    "run_id": run_id,
                    "window_id": window_id,
                    "blend_w": best_w,
                    "blend_fit_logloss": best_ll,
                    "market_method": args.market_method,
                }
            )

            # Predict on test
            te = te.copy()
            te["p_blend"] = logit_blend_prob(
                te["p_model_used"].to_numpy(),
                te["p_mkt_used"].to_numpy(),
                best_w,
            )
            te["ev_blend"] = te["p_blend"] * te["odds"] - 1.0

            # Metrics on test
            y_te = te["y_win"].to_numpy()
            p_mkt_te = te["p_mkt_used"].to_numpy()
            p_model_te = te["p_model_used"].to_numpy()
            p_blend_te = te["p_blend"].to_numpy()
            metrics_rows.append(
                {
                    "run_id": run_id,
                    "window_id": window_id,
                    "method": "market",
                    "logloss": _logloss(y_te, p_mkt_te),
                    "brier": _brier(y_te, p_mkt_te),
                    "auc": _auc(y_te, p_mkt_te),
                }
            )
            metrics_rows.append(
                {
                    "run_id": run_id,
                    "window_id": window_id,
                    "method": "model",
                    "logloss": _logloss(y_te, p_model_te),
                    "brier": _brier(y_te, p_model_te),
                    "auc": _auc(y_te, p_model_te),
                }
            )
            metrics_rows.append(
                {
                    "run_id": run_id,
                    "window_id": window_id,
                    "method": "blend",
                    "logloss": _logloss(y_te, p_blend_te),
                    "brier": _brier(y_te, p_blend_te),
                    "auc": _auc(y_te, p_blend_te),
                }
            )

            # Candidate evaluation (train+valid -> test)
            tv = train_valid.copy()
            tv["p_blend"] = logit_blend_prob(
                tv["p_model_used"].to_numpy(),
                tv["p_mkt_used"].to_numpy(),
                best_w,
            )
            tv["ev_blend"] = tv["p_blend"] * tv["odds"] - 1.0
            tv["odds_band"] = tv["odds"].apply(odds_band)

            for cand in candidate_grid:
                cand_id = cand["candidate_id"]
                t_ev = cand["t_ev"]
                cap = cand["odds_cap"]
                exclude = cand["exclude_bands"]
                sel_tv = _apply_candidate(tv, t_ev, cap, exclude)
                m_tv = _compute_roi(sel_tv)
                cand_train_rows.append(
                    {
                        "run_id": run_id,
                        "window_id": window_id,
                        "candidate_id": cand_id,
                        "t_ev": t_ev,
                        "odds_cap": cap,
                        "exclude_bands": ",".join(exclude) if exclude else "",
                        **m_tv,
                    }
                )

                sel_te = _apply_candidate(te, t_ev, cap, exclude)
                m_te = _compute_roi(sel_te)
                cand_test_rows.append(
                    {
                        "run_id": run_id,
                        "window_id": window_id,
                        "candidate_id": cand_id,
                        "t_ev": t_ev,
                        "odds_cap": cap,
                        "exclude_bands": ",".join(exclude) if exclude else "",
                        **m_te,
                    }
                )
                cand_test_window_rows.append(
                    {
                        "run_id": run_id,
                        "window_id": window_id,
                        "candidate_id": cand_id,
                        "t_ev": t_ev,
                        "odds_cap": cap,
                        "exclude_bands": ",".join(exclude) if exclude else "",
                        **m_te,
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    params_df = pd.DataFrame(params_rows)
    cand_train_df = pd.DataFrame(cand_train_rows)
    cand_test_df = pd.DataFrame(cand_test_rows)
    cand_test_window_df = pd.DataFrame(cand_test_window_rows)
    integrity_df = pd.DataFrame(integrity_rows)

    # Aggregate train+valid to select candidate (leak-free)
    cand_train_summary = (
        cand_train_df.groupby(["candidate_id", "t_ev", "odds_cap", "exclude_bands"], dropna=False)[
            ["profit", "stake", "n_bets"]
        ]
        .sum()
        .reset_index()
    )
    cand_train_summary["roi"] = cand_train_summary["profit"] / cand_train_summary["stake"]

    eligible_train = cand_train_summary[cand_train_summary["n_bets"] >= 100].copy()
    if eligible_train.empty:
        best_train = cand_train_summary.sort_values("roi", ascending=False).head(1)
    else:
        best_train = eligible_train.sort_values("roi", ascending=False).head(1)

    if best_train.empty:
        best_candidate_id = "NONE"
        best_params = "N/A"
    else:
        best_candidate_id = best_train.iloc[0]["candidate_id"]
        best_params = f"t_ev={best_train.iloc[0]['t_ev']}, odds_cap={best_train.iloc[0]['odds_cap']}, exclude={best_train.iloc[0]['exclude_bands']}"

    # Evaluate best candidate on test (step14)
    best_test = cand_test_df[cand_test_df["candidate_id"] == best_candidate_id] if best_candidate_id != "NONE" else pd.DataFrame()
    if not best_test.empty:
        best_test_sum = best_test.groupby(["candidate_id"], dropna=False)[["profit", "stake", "n_bets"]].sum().reset_index()
        best_step14_roi = float(best_test_sum["profit"].iloc[0] / best_test_sum["stake"].iloc[0]) if best_test_sum["stake"].iloc[0] > 0 else float("nan")
        best_n_bets = int(best_test_sum["n_bets"].iloc[0])
    else:
        best_step14_roi = float("nan")
        best_n_bets = 0

    decision = "pilot_candidate" if (best_candidate_id != "NONE" and best_step14_roi > 0 and best_n_bets >= 100) else "model_update_next"

    # Bootstrap CI for blend vs market (pooled test)
    blend_boot_rows = []
    for run_id in metrics_df["run_id"].unique():
        # reconstruct pooled test rows by reloading fullfield_test and blend with best_w per window
        # use cand_test_window_df to recover rows count is insufficient; fallback to meta from params_df
        # For CI, reuse test data by re-loading (cheaper than carrying large arrays in memory)
        run_dir = next((p for p in run_dirs if p.name == run_id), None)
        if run_dir is None:
            continue
        df_test_raw, _ = _read_fullfield(
            _resolve_fullfield_path(run_dir, staging_dir, "test", allow_export=False),
            args.market_method,
        )
        df_test_raw = _compute_market_probs(df_test_raw)
        df_test_raw["p_mkt_used"] = _select_market_prob(df_test_raw, args.market_method)
        df_test_raw["p_model_used"] = df_test_raw["p_model"]
        df_test_raw["odds_band"] = df_test_raw["odds"].apply(odds_band)
        df_test_raw, _ = _clean_fullfield(df_test_raw)

        # apply per-window blend params
        df_test_raw["p_blend"] = np.nan
        for _, row in params_df[params_df["run_id"] == run_id].iterrows():
            win = row["window_id"]
            w = float(row["blend_w"])
            mask = df_test_raw["window_id"] == win
            if not mask.any():
                continue
            df_test_raw.loc[mask, "p_blend"] = logit_blend_prob(
                df_test_raw.loc[mask, "p_model_used"].to_numpy(),
                df_test_raw.loc[mask, "p_mkt_used"].to_numpy(),
                w,
            )

        df_test_raw = df_test_raw[df_test_raw["p_blend"].notna()]
        race_sums = _race_sums(df_test_raw, "p_blend", "p_mkt_used")
        ll_low, ll_high, b_low, b_high = _bootstrap_ci_delta(race_sums, args.bootstrap_n, rng)
        blend_boot_rows.append(
            {
                "run_id": run_id,
                "delta_logloss_ci_low": ll_low,
                "delta_logloss_ci_high": ll_high,
                "delta_brier_ci_low": b_low,
                "delta_brier_ci_high": b_high,
                "bootstrap_n": args.bootstrap_n,
            }
        )

    blend_boot_df = pd.DataFrame(blend_boot_rows)

    # Step14 summary (test)
    step14 = (
        cand_test_window_df.groupby(["candidate_id", "t_ev", "odds_cap", "exclude_bands"], dropna=False)[
            ["profit", "stake", "n_bets"]
        ]
        .sum()
        .reset_index()
    )
    step14["step14_roi"] = step14["profit"] / step14["stake"]

    # Overall metrics (mean across windows)
    overall = (
        metrics_df.groupby(["method"], dropna=False)[["logloss", "brier", "auc"]].mean().reset_index()
        if not metrics_df.empty
        else pd.DataFrame()
    )

    # write outputs
    metrics_df.to_csv(staging_dir / "blend_metrics_by_window.csv", index=False, encoding="utf-8")
    overall.to_csv(staging_dir / "blend_metrics_overall.csv", index=False, encoding="utf-8")
    params_df.to_csv(staging_dir / "blend_params.csv", index=False, encoding="utf-8")
    cand_train_df.to_csv(staging_dir / "candidate_trainvalid_metrics.csv", index=False, encoding="utf-8")
    cand_test_df.to_csv(staging_dir / "candidate_test_metrics.csv", index=False, encoding="utf-8")
    step14.to_csv(staging_dir / "step14_summary.csv", index=False, encoding="utf-8")
    cand_test_window_df.to_csv(staging_dir / "candidate_window_metrics.csv", index=False, encoding="utf-8")
    integrity_df.to_csv(staging_dir / "integrity_summary.csv", index=False, encoding="utf-8")
    blend_boot_df.to_csv(staging_dir / "blend_bootstrap_ci.csv", index=False, encoding="utf-8")

    total_dropped = int(integrity_df["dropped_races"].sum()) if not integrity_df.empty else 0
    total_races = int(integrity_df["total_races"].sum()) if not integrity_df.empty else 0
    dropped_ratio = total_dropped / total_races if total_races else 0.0

    minimal_lines = [
        "# minimal update for chat - Phase1.5 market blend",
        "",
        "## Summary",
        "- ops recommendation: NO-BET until step14>0 candidate",
        f"- market_method: {args.market_method}",
        f"- best_candidate: {best_candidate_id}",
        f"- best_step14_roi: {best_step14_roi}",
        f"- best_n_bets: {best_n_bets}",
        f"- decision: {decision}",
        "",
        "## Paths",
        f"- staging_dir: {staging_dir}",
        f"- out_dir: {out_dir}",
    ]
    (staging_dir / "minimal_update_for_chat.md").write_text("\n".join(minimal_lines) + "\n", encoding="utf-8")

    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        f"[diag] phase1p5_blend_done=true | market_method={args.market_method} | n_runs={len(run_dirs)} | n_windows={run_windows} | dropped_races={total_dropped} | dropped_races_ratio={dropped_ratio:.6f}",
        f"[diag] best_candidate={best_candidate_id} | best_step14_roi={best_step14_roi} | best_n_bets={best_n_bets} | best_params={best_params}",
        f"[plan] decision={decision} | reason={'candidate_pass' if decision=='pilot_candidate' else 'no_candidate'}",
        f"[paths] out_dir={out_dir} | staging={staging_dir} | zip=<NOT_USED>",
    ]
    (staging_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Phase1.5 market-anchored blend (leak-free)",
        f"- run_dirs: {len(run_dirs)}",
        f"- market_method: {args.market_method}",
        f"- odds_col priority: {', '.join(ODDS_PRIORITY)}",
        f"- train/valid fit: logit-blend w in [0,1] step 0.05",
        "",
        "## Integrity (test)",
        integrity_df.to_string(index=False) if not integrity_df.empty else "No integrity rows.",
        "",
        "## Blend metrics overall (test)",
        overall.to_string(index=False) if not overall.empty else "No rows.",
        "",
        "## Blend bootstrap CI (blend vs market, pooled test)",
        blend_boot_df.to_string(index=False) if not blend_boot_df.empty else "No rows.",
        "",
        "## Best candidate (selected on train+valid)",
        f"- candidate_id: {best_candidate_id}",
        f"- params: {best_params}",
        f"- step14_roi(test): {best_step14_roi}",
        f"- n_bets(test): {best_n_bets}",
        "",
        "## Step14 summary (top by ROI)",
        step14.sort_values("step14_roi", ascending=False).head(10).to_string(index=False) if not step14.empty else "No rows.",
        "",
        "ROI definition: ROI = profit / stake (profit = return - stake).",
        "Note: reselect is from full-field; unit stake per bet (stake=1).",
    ]
    (staging_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
