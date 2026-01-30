"""
2つの rolling run（baseline vs variant）を window 単位でペア比較する。

入力:
  - baseline_dir/summary.csv
  - variant_dir/summary.csv

JOIN方針:
  - 原則: (test_start, test_end) でJOIN（窓番号ズレ耐性）
  - オプションで window_idx（w001..）JOINも可能

出力（variant_dir配下に保存）:
  - paired_compare.csv: 1行=1window, ΔROI/ΔMaxDD/Δn_bets 等
  - paired_summary.json: all/design/eval の集計

注意:
  - 窓は独立ではないので、統計検定は参考値（sign testのp値など）
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


WIN_RE = re.compile(r"^w(\d{3})_")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_window_idx(val) -> Optional[int]:
    if pd.isna(val):
        return None
    # int/float
    if isinstance(val, (int, float)) and float(val).is_integer():
        return int(val)
    s = str(val)
    m = WIN_RE.search(s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"w(\d+)", s)
    if m2:
        return int(m2.group(1))
    if s.isdigit():
        return int(s)
    return None


def _binom_two_sided_pvalue(n: int, k: int) -> float:
    """sign test（p=0.5）の両側p値（参考用）"""
    if n <= 0:
        return float("nan")
    k = int(k)
    k = min(k, n - k)
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5**n)
    return min(1.0, 2.0 * p)


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    name_col = _find_col(df, ["name", "window", "window_id", "w"])
    if name_col is None:
        # 最後の手段
        name_col = df.columns[0]

    # required
    roi_col = _find_col(df, ["roi", "ROI"])
    dd_col = _find_col(df, ["max_drawdown", "max_dd", "maxdd", "MaxDD"])
    nb_col = _find_col(df, ["n_bets", "bets", "nBet"])
    ts_col = _find_col(df, ["test_start"])
    te_col = _find_col(df, ["test_end"])
    # optional
    lg_col = _find_col(df, ["log_growth"])
    dd_bank_col = _find_col(df, ["max_drawdown_bankroll", "max_dd_bankroll", "maxdd_bankroll"])
    end_bank_col = _find_col(df, ["ending_bankroll", "final_bankroll"])
    ror03_col = _find_col(df, ["risk_of_ruin_0_3"])
    ror05_col = _find_col(df, ["risk_of_ruin_0_5"])
    ror07_col = _find_col(df, ["risk_of_ruin_0_7"])

    missing = [k for k, v in {"roi": roi_col, "max_drawdown": dd_col, "n_bets": nb_col, "test_start": ts_col, "test_end": te_col}.items() if v is None]
    if missing:
        raise ValueError(f"summary.csv に必要列がありません: {missing}. columns={list(df.columns)}")

    cols = {
        "window_name": name_col,
        "test_start": ts_col,
        "test_end": te_col,
        "roi": roi_col,
        "max_dd": dd_col,
        "n_bets": nb_col,
    }
    if lg_col is not None:
        cols["log_growth"] = lg_col
    if dd_bank_col is not None:
        cols["max_dd_bankroll"] = dd_bank_col
    if end_bank_col is not None:
        cols["ending_bankroll"] = end_bank_col
    if ror03_col is not None:
        cols["risk_of_ruin_0_3"] = ror03_col
    if ror05_col is not None:
        cols["risk_of_ruin_0_5"] = ror05_col
    if ror07_col is not None:
        cols["risk_of_ruin_0_7"] = ror07_col

    out = df[list(cols.values())].copy()
    out.columns = list(cols.keys())

    out["window_idx"] = out["window_name"].apply(_parse_window_idx)
    if out["window_idx"].isna().any():
        # period join では不要だが、design/eval split のために必要
        bad = out[out["window_idx"].isna()][["window_name"]].head(5)
        raise ValueError(f"window_idx を作れませんでした。例: {bad.to_dict(orient='records')}")

    # numeric
    out["roi"] = pd.to_numeric(out["roi"], errors="coerce")
    out["max_dd"] = pd.to_numeric(out["max_dd"], errors="coerce")
    out["n_bets"] = pd.to_numeric(out["n_bets"], errors="coerce").fillna(0).astype(int)
    if "log_growth" in out.columns:
        out["log_growth"] = pd.to_numeric(out["log_growth"], errors="coerce")
    if "max_dd_bankroll" in out.columns:
        out["max_dd_bankroll"] = pd.to_numeric(out["max_dd_bankroll"], errors="coerce")
    if "ending_bankroll" in out.columns:
        out["ending_bankroll"] = pd.to_numeric(out["ending_bankroll"], errors="coerce")
    if "risk_of_ruin_0_3" in out.columns:
        out["risk_of_ruin_0_3"] = pd.to_numeric(out["risk_of_ruin_0_3"], errors="coerce")
    if "risk_of_ruin_0_5" in out.columns:
        out["risk_of_ruin_0_5"] = pd.to_numeric(out["risk_of_ruin_0_5"], errors="coerce")
    if "risk_of_ruin_0_7" in out.columns:
        out["risk_of_ruin_0_7"] = pd.to_numeric(out["risk_of_ruin_0_7"], errors="coerce")

    return out.sort_values(["window_idx"]).reset_index(drop=True)


def summarize_block(df: pd.DataFrame) -> dict:
    n = int(len(df))
    pos = int((df["d_roi"] > 0).sum()) if n else 0
    out = {
        "n_windows": n,
        "improve_rate_roi": (pos / n) if n else None,
        "sign_test_pvalue_ref": float(_binom_two_sided_pvalue(n, pos)) if n else None,
        "median_d_roi": float(df["d_roi"].median()) if n else None,
        "mean_d_roi": float(df["d_roi"].mean()) if n else None,
        "median_d_maxdd": float(df["d_max_dd"].median()) if n else None,
        "mean_d_maxdd": float(df["d_max_dd"].mean()) if n else None,
        "median_d_n_bets": float(df["d_n_bets"].median()) if n else None,
        "mean_d_n_bets": float(df["d_n_bets"].mean()) if n else None,
        "median_roi_base": float(df["roi_base"].median()) if n else None,
        "median_roi_var": float(df["roi_var"].median()) if n else None,
        "median_maxdd_base": float(df["max_dd_base"].median()) if n else None,
        "median_maxdd_var": float(df["max_dd_var"].median()) if n else None,
        "median_n_bets_base": float(df["n_bets_base"].median()) if n else None,
        "median_n_bets_var": float(df["n_bets_var"].median()) if n else None,
    }
    if "d_log_growth" in df.columns:
        out.update({
            "median_d_log_growth": float(df["d_log_growth"].median()) if n else None,
            "mean_d_log_growth": float(df["d_log_growth"].mean()) if n else None,
        })
    if "log_growth_base" in df.columns:
        out.update({
            "median_log_growth_base": float(df["log_growth_base"].median()) if n else None,
            "median_log_growth_var": float(df["log_growth_var"].median()) if n else None,
        })
    if "d_max_dd_bankroll" in df.columns:
        out.update({
            "median_d_maxdd_bankroll": float(df["d_max_dd_bankroll"].median()) if n else None,
            "mean_d_maxdd_bankroll": float(df["d_max_dd_bankroll"].mean()) if n else None,
        })
    if "max_dd_bankroll_base" in df.columns:
        out.update({
            "median_maxdd_bankroll_base": float(df["max_dd_bankroll_base"].median()) if n else None,
            "median_maxdd_bankroll_var": float(df["max_dd_bankroll_var"].median()) if n else None,
        })
    if "d_ending_bankroll" in df.columns:
        out.update({
            "median_d_ending_bankroll": float(df["d_ending_bankroll"].median()) if n else None,
            "mean_d_ending_bankroll": float(df["d_ending_bankroll"].mean()) if n else None,
        })
    if "ending_bankroll_base" in df.columns:
        out.update({
            "median_ending_bankroll_base": float(df["ending_bankroll_base"].median()) if n else None,
            "median_ending_bankroll_var": float(df["ending_bankroll_var"].median()) if n else None,
        })
    if "d_risk_of_ruin_0_3" in df.columns:
        out.update({
            "mean_d_risk_of_ruin_0_3": float(df["d_risk_of_ruin_0_3"].mean()) if n else None,
        })
    if "risk_of_ruin_0_3_base" in df.columns:
        out.update({
            "mean_risk_of_ruin_0_3_base": float(df["risk_of_ruin_0_3_base"].mean()) if n else None,
            "mean_risk_of_ruin_0_3_var": float(df["risk_of_ruin_0_3_var"].mean()) if n else None,
        })
    if "d_risk_of_ruin_0_5" in df.columns:
        out.update({
            "mean_d_risk_of_ruin_0_5": float(df["d_risk_of_ruin_0_5"].mean()) if n else None,
        })
    if "risk_of_ruin_0_5_base" in df.columns:
        out.update({
            "mean_risk_of_ruin_0_5_base": float(df["risk_of_ruin_0_5_base"].mean()) if n else None,
            "mean_risk_of_ruin_0_5_var": float(df["risk_of_ruin_0_5_var"].mean()) if n else None,
        })
    if "d_risk_of_ruin_0_7" in df.columns:
        out.update({
            "mean_d_risk_of_ruin_0_7": float(df["d_risk_of_ruin_0_7"].mean()) if n else None,
        })
    if "risk_of_ruin_0_7_base" in df.columns:
        out.update({
            "mean_risk_of_ruin_0_7_base": float(df["risk_of_ruin_0_7_base"].mean()) if n else None,
            "mean_risk_of_ruin_0_7_var": float(df["risk_of_ruin_0_7_var"].mean()) if n else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired compare rolling runs (baseline vs variant)")
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--variant-dir", type=Path, required=True)
    ap.add_argument("--design-max-idx", type=int, default=12)
    ap.add_argument("--join", choices=["period", "window_idx"], default="period", help="windowの合わせ方")
    args = ap.parse_args()

    base_dir = args.baseline_dir
    var_dir = args.variant_dir
    base_summary_path = base_dir / "summary.csv"
    var_summary_path = var_dir / "summary.csv"
    if not base_summary_path.exists():
        raise SystemExit(f"baseline summary.csv not found: {base_summary_path}")
    if not var_summary_path.exists():
        raise SystemExit(f"variant summary.csv not found: {var_summary_path}")

    base = load_summary(base_summary_path)
    var = load_summary(var_summary_path)

    if args.join == "window_idx":
        merged = base.merge(var, on="window_idx", how="inner", suffixes=("_base", "_var"))
    else:
        # period join（推奨）
        merged = base.merge(var, on=["test_start", "test_end"], how="inner", suffixes=("_base", "_var"))
        # window_idx は baseline側を採用（同一のはず）
        if "window_idx_base" in merged.columns:
            merged["window_idx"] = merged["window_idx_base"]
        elif "window_idx" not in merged.columns:
            merged["window_idx"] = merged["window_idx_var"]

    if merged.empty:
        raise RuntimeError("ペア比較できる窓が0です（JOINキーが合っていない可能性）")

    # 欠損窓チェック（参考）
    base_keys = set(zip(base["test_start"], base["test_end"])) if args.join == "period" else set(base["window_idx"].tolist())
    var_keys = set(zip(var["test_start"], var["test_end"])) if args.join == "period" else set(var["window_idx"].tolist())
    missing_in_var = sorted(base_keys - var_keys)
    missing_in_base = sorted(var_keys - base_keys)

    # delta
    merged["d_roi"] = merged["roi_var"] - merged["roi_base"]
    merged["d_max_dd"] = merged["max_dd_var"] - merged["max_dd_base"]
    merged["d_n_bets"] = merged["n_bets_var"] - merged["n_bets_base"]
    if "log_growth_base" in merged.columns and "log_growth_var" in merged.columns:
        merged["d_log_growth"] = merged["log_growth_var"] - merged["log_growth_base"]
    if "max_dd_bankroll_base" in merged.columns and "max_dd_bankroll_var" in merged.columns:
        merged["d_max_dd_bankroll"] = merged["max_dd_bankroll_var"] - merged["max_dd_bankroll_base"]
    if "ending_bankroll_base" in merged.columns and "ending_bankroll_var" in merged.columns:
        merged["d_ending_bankroll"] = merged["ending_bankroll_var"] - merged["ending_bankroll_base"]
    if "risk_of_ruin_0_3_base" in merged.columns and "risk_of_ruin_0_3_var" in merged.columns:
        merged["d_risk_of_ruin_0_3"] = merged["risk_of_ruin_0_3_var"].astype(int) - merged["risk_of_ruin_0_3_base"].astype(int)
    if "risk_of_ruin_0_5_base" in merged.columns and "risk_of_ruin_0_5_var" in merged.columns:
        merged["d_risk_of_ruin_0_5"] = merged["risk_of_ruin_0_5_var"].astype(int) - merged["risk_of_ruin_0_5_base"].astype(int)
    if "risk_of_ruin_0_7_base" in merged.columns and "risk_of_ruin_0_7_var" in merged.columns:
        merged["d_risk_of_ruin_0_7"] = merged["risk_of_ruin_0_7_var"].astype(int) - merged["risk_of_ruin_0_7_base"].astype(int)

    merged["split"] = merged["window_idx"].apply(lambda i: "design" if int(i) <= int(args.design_max_idx) else "eval")

    out_dir = var_dir
    out_compare = out_dir / "paired_compare.csv"
    out_summary = out_dir / "paired_summary.json"
    merged.sort_values(["window_idx"]).to_csv(out_compare, index=False, encoding="utf-8")

    summary = {
        "meta": {
            "baseline_dir": str(base_dir),
            "variant_dir": str(var_dir),
            "join": args.join,
            "design_max_idx": int(args.design_max_idx),
            "matched_windows": int(len(merged)),
            "missing_in_variant": missing_in_var[:50],  # 念のため上限
            "missing_in_baseline": missing_in_base[:50],
        },
        "all": summarize_block(merged),
        "design": summarize_block(merged[merged["split"] == "design"]),
        "eval": summarize_block(merged[merged["split"] == "eval"]),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote:", out_compare)
    print("Wrote:", out_summary)
    # console summary (eval)
    print("eval:", json.dumps(summary["eval"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


