"""
Post-S1 / Ticket N1:
S1（slippage補正）variant の失敗要因を、bets.csv の slip_r_hat バケットで確定する診断スクリプト。

入力:
  --baseline-dir: baseline rolling group_dir
  --variant-dir : slippage variant rolling group_dir（bets.csv に slip_r_hat がある想定）

出力（variant-dir配下）:
  - slip_rhat_distribution.json
  - slip_rhat_bucket_table.csv
  - slip_rhat_bucket_table_eval.csv

主眼:
  - r_hat の分布（p50/p75/p90、P(>1.00/1.05/1.10)、P(<0.95/<0.90)）
  - r_hat バケット別の ROI / win_rate / odds帯 / ratio_final_to_buy / overlay
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


WIN_DIR_RE = re.compile(r"^w(\d{3})_")


@dataclass(frozen=True)
class WindowFiles:
    idx: int
    name: str
    dir: Path
    bets_csv: Path


def _to_num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _q(s: pd.Series, q: float) -> Optional[float]:
    x = _to_num(s).dropna()
    if x.empty:
        return None
    v = float(x.quantile(q))
    if not np.isfinite(v):
        return None
    return v


def _rate_gt(s: pd.Series, thr: float) -> Optional[float]:
    x = _to_num(s)
    x = x[np.isfinite(x)]
    if x.empty:
        return None
    return float((x > float(thr)).mean())


def _rate_lt(s: pd.Series, thr: float) -> Optional[float]:
    x = _to_num(s)
    x = x[np.isfinite(x)]
    if x.empty:
        return None
    return float((x < float(thr)).mean())


def _parse_windows(group_dir: Path) -> list[WindowFiles]:
    ws: list[WindowFiles] = []
    for p in group_dir.iterdir():
        if not p.is_dir():
            continue
        m = WIN_DIR_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        bc = p / "bets.csv"
        if not bc.exists():
            continue
        ws.append(WindowFiles(idx=idx, name=p.name, dir=p, bets_csv=bc))
    ws.sort(key=lambda w: w.idx)
    return ws


def _load_variant_bets(variant_dir: Path, design_max_idx: int) -> pd.DataFrame:
    windows = _parse_windows(variant_dir)
    rows: list[pd.DataFrame] = []
    for w in windows:
        try:
            df = pd.read_csv(w.bets_csv)
        except Exception:
            continue
        if df.empty:
            continue
        df["window"] = w.name
        df["idx"] = int(w.idx)
        df["split"] = "design" if int(w.idx) <= int(design_max_idx) else "eval"
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    # numeric normalization
    for c in [
        "stake",
        "profit",
        "is_win",
        "odds_at_buy",
        "ratio_final_to_buy",
        "p_hat",
        "p_mkt",
        "slip_r_hat",
    ]:
        if c in out.columns:
            out[c] = _to_num(out[c])

    # derived
    if "p_hat" in out.columns and "p_mkt" in out.columns:
        out["overlay"] = out["p_hat"] - out["p_mkt"]
    else:
        out["overlay"] = np.nan

    return out


def _rhat_bucket(s: pd.Series) -> pd.Series:
    x = _to_num(s)
    # bucket defs: <0.95 / 0.95-1.00 / 1.00-1.05 / >1.05
    b = pd.Series(index=x.index, dtype="object")
    b[(x < 0.95)] = "<0.95"
    b[(x >= 0.95) & (x < 1.00)] = "0.95-1.00"
    b[(x >= 1.00) & (x < 1.05)] = "1.00-1.05"
    b[(x >= 1.05)] = ">1.05"
    b[pd.isna(x)] = "missing"
    return b


def _bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    x = df.copy()
    x["rhat_bucket"] = _rhat_bucket(x.get("slip_r_hat"))

    # pooled ROI per group (profit/stake)
    def _agg(g: pd.DataFrame) -> dict[str, Any]:
        stake = float(_to_num(g.get("stake")).fillna(0).sum())
        profit = float(_to_num(g.get("profit")).fillna(0).sum())
        roi = (profit / stake) if stake > 0 else None
        is_win = _to_num(g.get("is_win")).dropna()
        win_rate = float(is_win.mean()) if not is_win.empty else None
        return {
            "n_bets": int(len(g)),
            "stake": stake,
            "profit": profit,
            "roi": roi,
            "win_rate": win_rate,
            "odds_buy_median": _q(g.get("odds_at_buy"), 0.50),
            "ratio_median": _q(g.get("ratio_final_to_buy"), 0.50),
            "overlay_median": _q(g.get("overlay"), 0.50),
            "rhat_median": _q(g.get("slip_r_hat"), 0.50),
        }

    out_rows: list[dict[str, Any]] = []
    for split in ["all", "design", "eval"]:
        sub = x if split == "all" else x[x["split"] == split]
        if sub.empty:
            continue
        for bucket, g in sub.groupby("rhat_bucket", dropna=False):
            row = {"split": split, "rhat_bucket": str(bucket)}
            row.update(_agg(g))
            out_rows.append(row)

    out = pd.DataFrame(out_rows)
    # ordering
    order = {"<0.95": 1, "0.95-1.00": 2, "1.00-1.05": 3, ">1.05": 4, "missing": 99}
    if not out.empty:
        out["bucket_order"] = out["rhat_bucket"].map(order).fillna(999).astype(int)
        out = out.sort_values(["split", "bucket_order"]).drop(columns=["bucket_order"]).reset_index(drop=True)
    return out


def _distribution_block(s: pd.Series) -> dict[str, Any]:
    return {
        "p50": _q(s, 0.50),
        "p75": _q(s, 0.75),
        "p90": _q(s, 0.90),
        "p_gt_1_00": _rate_gt(s, 1.00),
        "p_gt_1_05": _rate_gt(s, 1.05),
        "p_gt_1_10": _rate_gt(s, 1.10),
        "p_lt_0_95": _rate_lt(s, 0.95),
        "p_lt_0_90": _rate_lt(s, 0.90),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose slippage variant failure (r_hat buckets)")
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--variant-dir", type=Path, required=True)
    p.add_argument("--design-max-idx", type=int, default=12)
    args = p.parse_args()

    base_dir = args.baseline_dir
    var_dir = args.variant_dir

    df = _load_variant_bets(var_dir, design_max_idx=args.design_max_idx)
    if df.empty:
        raise SystemExit(f"No bets.csv found or empty under variant-dir: {var_dir}")

    if "slip_r_hat" not in df.columns:
        raise SystemExit("variant bets.csv に slip_r_hat 列がありません（S1 variant を指定してください）")

    dist = {
        "meta": {
            "baseline_dir": str(base_dir),
            "variant_dir": str(var_dir),
            "design_max_idx": int(args.design_max_idx),
            "n_bets_total": int(len(df)),
            "n_windows": int(df["idx"].nunique()),
        },
        "all": _distribution_block(df["slip_r_hat"]),
        "design": _distribution_block(df.loc[df["split"] == "design", "slip_r_hat"]),
        "eval": _distribution_block(df.loc[df["split"] == "eval", "slip_r_hat"]),
    }

    # r_hat > 1 の pooled ROI（参考）
    def _pooled_roi(mask: pd.Series) -> Optional[float]:
        g = df[mask]
        if g.empty:
            return None
        stake = float(_to_num(g.get("stake")).fillna(0).sum())
        profit = float(_to_num(g.get("profit")).fillna(0).sum())
        return (profit / stake) if stake > 0 else None

    r = df["slip_r_hat"]
    dist["pooled_roi"] = {
        "all_rhat_gt_1": _pooled_roi(r > 1.0),
        "all_rhat_le_1": _pooled_roi(r <= 1.0),
        "design_rhat_gt_1": _pooled_roi((df["split"] == "design") & (r > 1.0)),
        "design_rhat_le_1": _pooled_roi((df["split"] == "design") & (r <= 1.0)),
        "eval_rhat_gt_1": _pooled_roi((df["split"] == "eval") & (r > 1.0)),
        "eval_rhat_le_1": _pooled_roi((df["split"] == "eval") & (r <= 1.0)),
    }

    out_json = var_dir / "slip_rhat_distribution.json"
    out_json.write_text(json.dumps(dist, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    table = _bucket_table(df)
    out_csv = var_dir / "slip_rhat_bucket_table.csv"
    table.to_csv(out_csv, index=False, encoding="utf-8")

    out_eval = var_dir / "slip_rhat_bucket_table_eval.csv"
    if not table.empty:
        table[table["split"] == "eval"].to_csv(out_eval, index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_eval, index=False, encoding="utf-8")

    print("Wrote:", out_json)
    print("Wrote:", out_csv)
    print("Wrote:", out_eval)
    print("eval dist:", json.dumps(dist["eval"], ensure_ascii=False))


if __name__ == "__main__":
    main()


