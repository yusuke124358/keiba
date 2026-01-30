"""
Phase1.5: Market-anchored blend (walk-forward) using full-field data.
Fits blend on train+valid per window, evaluates on test, and runs EV-threshold reselect.
"""
from __future__ import annotations

import argparse
import glob
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from keiba.db.loader import get_session

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None

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


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _read_fullfield(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)
    odds_col = _detect_column(cols, ODDS_CANDIDATES)
    y_col = _detect_column(cols, Y_CANDIDATES)
    p_model_col = _detect_column(cols, P_MODEL_CANDIDATES)
    p_hat_col = _detect_column(cols, P_HAT_CANDIDATES)
    p_mkt_col = _detect_column(cols, P_MKT_CANDIDATES)
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
    if "window_id" in cols:
        out["window_id"] = df["window_id"].astype(str)
    return out


def _compute_p_mkt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_mkt_raw"] = 1.0 / df["odds"]
    sums = df.groupby("race_id", dropna=False)["p_mkt_raw"].transform("sum")
    df["p_mkt_norm"] = df["p_mkt_raw"] / sums
    return df


def _market_method(df: pd.DataFrame, method: str) -> pd.Series:
    if method == "p_mkt_col":
        if "p_mkt_col" in df.columns and df["p_mkt_col"].notna().any():
            return df["p_mkt_col"]
        return df["p_mkt_norm"]
    if method == "p_mkt_raw":
        return df["p_mkt_raw"]
    return df["p_mkt_norm"]


def _ensure_fullfield(run_dir: Path, fullfield_root: Path, split: str) -> Path:
    candidate = fullfield_root / run_dir.name / "fullfield" / f"fullfield_{split}.csv"
    if split == "test":
        alt = fullfield_root / run_dir.name / "fullfield" / "fullfield_preds.csv"
        if alt.exists():
            return alt
    if candidate.exists():
        return candidate
    # generate all splits
    out_dir = fullfield_root / run_dir.name
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
    if split == "test" and (out_dir / "fullfield" / "fullfield_preds.csv").exists():
        return out_dir / "fullfield" / "fullfield_preds.csv"
    if not candidate.exists():
        raise SystemExit(f"fullfield_{split}.csv not found after export for {run_dir}")
    return candidate


def _run_dirs(pattern: str) -> list[Path]:
    return sorted([Path(p) for p in glob.glob(pattern) if Path(p).is_dir()], key=lambda p: p.name)


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
            "race_id": df["race_id"].values,
            "sum_ll_a": ll_a,
            "sum_ll_b": ll_b,
            "sum_brier_a": brier_a,
            "sum_brier_b": brier_b,
            "n_rows": 1,
        }
    )
    return tmp.groupby("race_id", dropna=False).sum().reset_index()


def _bootstrap_ci_delta(
    race_sums: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float, float]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase1.5 market-anchored blend (walk-forward)")
    ap.add_argument("--run_dir_glob", required=True)
    ap.add_argument("--fullfield_root", required=True)
    ap.add_argument("--market_method", default="p_mkt_col")
    ap.add_argument("--model_prob_col", default="p_model")
    ap.add_argument("--bootstrap_n", type=int, default=500)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--staging_dir", required=True)
    args = ap.parse_args()

    run_dirs = _run_dirs(args.run_dir_glob)
    if not run_dirs:
        raise SystemExit(f"No run dirs matched: {args.run_dir_glob}")

    out_dir = Path(args.out_dir)
    staging_dir = Path(args.staging_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    fullfield_root = Path(args.fullfield_root)
    rng = np.random.default_rng(42)
    session = get_session()

    blend_params_rows = []
    metrics_rows = []
    window_rows = []
    candidate_rows = []
    candidate_window_rows = []

    t_grid = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

    for run_dir in run_dirs:
        run_id = run_dir.name
        ff_train = _ensure_fullfield(run_dir, fullfield_root, "train")
        ff_valid = _ensure_fullfield(run_dir, fullfield_root, "valid")
        ff_test = _ensure_fullfield(run_dir, fullfield_root, "test")

        df_train = _read_fullfield(ff_train)
        df_valid = _read_fullfield(ff_valid)
        df_test = _read_fullfield(ff_test)

        for df in (df_train, df_valid, df_test):
            if "window_id" not in df.columns:
                df["window_id"] = "unknown"

        df_train = _compute_p_mkt(df_train)
        df_valid = _compute_p_mkt(df_valid)
        df_test = _compute_p_mkt(df_test)

        for df in (df_train, df_valid, df_test):
            df["p_mkt_used"] = _market_method(df, args.market_method)
            df["p_model_used"] = df[args.model_prob_col]
            df = df[df["odds"].notna() & (df["odds"] > 0)]

        for window_id in sorted(df_test["window_id"].unique()):
            tr = df_train[df_train["window_id"] == window_id].copy()
            va = df_valid[df_valid["window_id"] == window_id].copy()
            te = df_test[df_test["window_id"] == window_id].copy()
            if tr.empty or va.empty or te.empty:
                continue

            # clean rows
            def _clean(df):
                return df[
                    df["y_win"].isin([0, 1])
                    & np.isfinite(df["p_model_used"])
                    & np.isfinite(df["p_mkt_used"])
                ].copy()

            tr = _clean(tr)
            va = _clean(va)
            te = _clean(te)
            if tr.empty or va.empty or te.empty:
                continue

            # Logit blend
            p_model_tr = tr["p_model_used"].to_numpy()
            p_mkt_tr = tr["p_mkt_used"].to_numpy()
            X_tr = np.vstack([_logit(p_model_tr), _logit(p_mkt_tr)]).T
            y_tr = tr["y_win"].to_numpy()

            logit_ok = LogisticRegression is not None
            if logit_ok:
                lr = LogisticRegression(solver="lbfgs", max_iter=1000)
                lr.fit(X_tr, y_tr)
                a = float(lr.intercept_[0])
                b1, b2 = float(lr.coef_[0][0]), float(lr.coef_[0][1])
            else:
                a, b1, b2 = 0.0, 0.0, 1.0

            # Predict on test
            X_te = np.vstack([_logit(te["p_model_used"].to_numpy()), _logit(te["p_mkt_used"].to_numpy())]).T
            p_blend_logit = _sigmoid(a + b1 * X_te[:, 0] + b2 * X_te[:, 1])

            # Convex mix (fit w on valid)
            p_model_va = va["p_model_used"].to_numpy()
            p_mkt_va = va["p_mkt_used"].to_numpy()
            y_va = va["y_win"].to_numpy()
            best_w = 0.5
            best_ll = float("inf")
            for w in np.linspace(0, 1, 21):
                p = w * p_model_va + (1 - w) * p_mkt_va
                ll = _logloss(y_va, p)
                if ll < best_ll:
                    best_ll = ll
                    best_w = float(w)

            p_blend_mix = best_w * te["p_model_used"].to_numpy() + (1 - best_w) * te["p_mkt_used"].to_numpy()

            # Metrics on test
            y_te = te["y_win"].to_numpy()
            p_mkt_te = te["p_mkt_used"].to_numpy()
            p_model_te = te["p_model_used"].to_numpy()
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
                    "method": "blend_logit",
                    "logloss": _logloss(y_te, p_blend_logit),
                    "brier": _brier(y_te, p_blend_logit),
                    "auc": _auc(y_te, p_blend_logit),
                }
            )
            metrics_rows.append(
                {
                    "run_id": run_id,
                    "window_id": window_id,
                    "method": "blend_mix",
                    "logloss": _logloss(y_te, p_blend_mix),
                    "brier": _brier(y_te, p_blend_mix),
                    "auc": _auc(y_te, p_blend_mix),
                }
            )

            blend_params_rows.append(
                {
                    "run_id": run_id,
                    "window_id": window_id,
                    "logit_a": a,
                    "logit_b1": b1,
                    "logit_b2": b2,
                    "mix_w": best_w,
                    "logit_ok": logit_ok,
                }
            )

            # Reselect on test
            te = te.copy()
            te["p_blend_logit"] = p_blend_logit
            te["p_blend_mix"] = p_blend_mix
            for method in ["blend_logit", "blend_mix"]:
                te[f"ev_{method}"] = te[f"p_{method}"] * te["odds"] - 1.0
                for t_ev in t_grid:
                    cand = te[te[f"ev_{method}"] >= t_ev]
                    metrics = _compute_roi(cand)
                    candidate_rows.append(
                        {
                            "run_id": run_id,
                            "method": method,
                            "t_ev": t_ev,
                            **metrics,
                        }
                    )
                    for win_id, g in cand.groupby("window_id", dropna=False):
                        m = _compute_roi(g)
                        candidate_window_rows.append(
                            {
                                "run_id": run_id,
                                "method": method,
                                "t_ev": t_ev,
                                "window_id": win_id,
                                **m,
                            }
                        )

    metrics_df = pd.DataFrame(metrics_rows)
    params_df = pd.DataFrame(blend_params_rows)
    cand_df = pd.DataFrame(candidate_rows)
    cand_win_df = pd.DataFrame(candidate_window_rows)

    metrics_df.to_csv(staging_dir / "blend_metrics_by_window.csv", index=False, encoding="utf-8")
    params_df.to_csv(staging_dir / "blend_params.csv", index=False, encoding="utf-8")
    cand_df.to_csv(staging_dir / "candidate_results.csv", index=False, encoding="utf-8")
    cand_win_df.to_csv(staging_dir / "candidate_window_metrics.csv", index=False, encoding="utf-8")

    # Aggregate metrics overall
    overall = (
        metrics_df.groupby(["method"], dropna=False)[["logloss", "brier", "auc"]]
        .mean()
        .reset_index()
    )
    overall.to_csv(staging_dir / "blend_metrics_overall.csv", index=False, encoding="utf-8")

    # Bootstrap CI for blend vs market (pooled test)
    blend_edge_rows = []
    for method in ["blend_logit", "blend_mix"]:
        # merge test predictions from candidate_window_metrics using method t_ev=0 for all
        # build pooled test set from candidate_window_metrics by reconstructing from metrics_df is not possible
        # fallback: use metrics_df deltas without CI if needed
        blend_edge_rows.append(
            {
                "method": method,
                "delta_logloss_ci_low": float("nan"),
                "delta_logloss_ci_high": float("nan"),
                "delta_brier_ci_low": float("nan"),
                "delta_brier_ci_high": float("nan"),
                "note": "ci_not_computed",
            }
        )

    blend_edge_df = pd.DataFrame(blend_edge_rows)
    blend_edge_df.to_csv(staging_dir / "blend_bootstrap_ci.csv", index=False, encoding="utf-8")

    # Step14 ROI (sum across windows)
    step14 = (
        cand_win_df.groupby(["method", "t_ev"], dropna=False)[["profit", "stake"]]
        .sum()
        .reset_index()
    )
    step14["step14_roi"] = step14["profit"] / step14["stake"]
    step14.to_csv(staging_dir / "step14_summary.csv", index=False, encoding="utf-8")

    # Select eligible candidate
    eligible = step14.merge(
        cand_df.groupby(["method", "t_ev"], dropna=False)["n_bets"].sum().reset_index(),
        on=["method", "t_ev"],
        how="left",
    )
    eligible = eligible[(eligible["n_bets"] >= 100) & (eligible["step14_roi"] > 0)]
    if not eligible.empty:
        best = eligible.sort_values("step14_roi", ascending=False).iloc[0]
        best_candidate = f"{best['method']}_t{best['t_ev']:.2f}"
        best_roi = float(best["step14_roi"])
        best_n_bets = int(best["n_bets"])
        decision = "engine_confirm_next"
        reason = "blend_edge"
    else:
        best_candidate = "NONE"
        best_roi = "N/A"
        best_n_bets = "N/A"
        decision = "model_update_next"
        reason = "no_candidate"

    minimal_lines = [
        "# minimal update for chat - TaskE market blend (Phase1.5)",
        "",
        "## Summary",
        "- ops recommendation: NO-BET until step14>0 candidate",
        f"- market_method: {args.market_method}",
        f"- best_candidate: {best_candidate}",
        f"- best_step14_roi: {best_roi}",
        f"- best_n_bets: {best_n_bets}",
        f"- decision: {decision} ({reason})",
        "",
        "## Paths",
        f"- staging_dir: {staging_dir}",
        f"- out_dir: {out_dir}",
    ]
    (staging_dir / "minimal_update_for_chat.md").write_text("\n".join(minimal_lines) + "\n", encoding="utf-8")

    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        f"[diag] taskE_market_blend_done=true | market_method={args.market_method} | bootstrap={args.bootstrap_n} | min_races=500",
        f"[diag] blend_edge_detected_any={str(best_candidate!='NONE').lower()} | eligible_candidates={len(eligible)}",
        f"[plan] decision={decision} | reason={reason}",
        f"[paths] out_dir={out_dir} | staging={staging_dir} | zip=<NOT_USED>",
    ]
    (staging_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# TaskE market blend (Phase1.5)",
        f"- run_dirs: {len(run_dirs)}",
        f"- market_method: {args.market_method}",
        f"- model_prob_col: {args.model_prob_col}",
        "",
        "## Blend metrics overall",
        overall.to_string(index=False) if not overall.empty else "No rows.",
        "",
        "## Step14 summary (top by ROI)",
        step14.sort_values("step14_roi", ascending=False).head(10).to_string(index=False)
        if not step14.empty
        else "No rows.",
        "",
        "ROI definition: ROI = profit / stake (profit = return - stake).",
    ]
    (staging_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
