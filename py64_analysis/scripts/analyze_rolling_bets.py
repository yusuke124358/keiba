"""
rolling holdout の group_dir（w001_... 配下）を集計して、
Step3-A（窓×ドライバー表）と Step3-B（単変量フィルタスイープ）を出力する。

目的:
  - baseline rolling を完走させつつ、完了済みwindowを design として前倒し分析する
  - filter候補（odds上限 / min EV / TSボラ上限）を「1本」に絞る材料を作る

使い方:
  python py64_analysis/scripts/analyze_rolling_bets.py --group-dir data/holdout_runs/<group_dir> --design-max-idx 12
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
    summary_json: Path
    bets_csv: Path


def _project_root() -> Path:
    # keiba/py64_analysis/scripts/ -> keiba/
    return Path(__file__).resolve().parents[2]


def _to_num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _q(s: pd.Series, q: float) -> Optional[float]:
    x = _to_num(s).dropna()
    if x.empty:
        return None
    try:
        v = float(x.quantile(q))
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _rate_lt(s: pd.Series, thr: float) -> Optional[float]:
    x = _to_num(s)
    x = x[np.isfinite(x)]
    if x.empty:
        return None
    return float((x < float(thr)).mean())


def _parse_window_files(group_dir: Path) -> list[WindowFiles]:
    windows: list[WindowFiles] = []
    for p in group_dir.iterdir():
        if not p.is_dir():
            continue
        m = WIN_DIR_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        sj = p / "summary.json"
        bc = p / "bets.csv"
        if not sj.exists() or not bc.exists():
            # 実行中のwindowはここに入らない（=完了済みだけ扱う）
            continue
        windows.append(WindowFiles(idx=idx, name=p.name, dir=p, summary_json=sj, bets_csv=bc))
    windows.sort(key=lambda w: w.idx)
    return windows


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_bets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 数値化しておく（後の集計が安定）
    for c in [
        "stake",
        "odds_at_buy",
        "odds_final",
        "ratio_final_to_buy",
        "odds_effective",
        "p_hat",
        "p_mkt",
        "ev",
        "profit",
        "is_win",
        "log_odds_std_60m",
        "log_odds_slope_60m",
        "snap_age_min",
    ]:
        if c in df.columns:
            df[c] = _to_num(df[c])
    return df


def _bucket_stats(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    label_col: str,
) -> pd.DataFrame:
    """
    1行=1bet の df から、指定ラベル列で集計した表を返す。
    期待列:
      - stake, profit, is_win, p_hat, p_mkt, ratio_final_to_buy
    """
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["n_bets", "stake", "profit", "roi", "win_rate", "overlay_median"])

    d = df.copy()
    d["stake"] = _to_num(d.get("stake")).fillna(0.0)
    d["profit"] = _to_num(d.get("profit")).fillna(0.0)
    d["is_win"] = _to_num(d.get("is_win"))
    d["p_hat"] = _to_num(d.get("p_hat"))
    d["p_mkt"] = _to_num(d.get("p_mkt"))
    d["overlay"] = d["p_hat"] - d["p_mkt"]

    # observed は将来のデフォルト変更に備えて明示（pandas warning抑制）
    g = d.groupby(group_cols, dropna=False, observed=False)
    out = g.agg(
        n_bets=("profit", "size"),
        stake=("stake", "sum"),
        profit=("profit", "sum"),
        win_rate=("is_win", "mean"),
        overlay_median=("overlay", "median"),
    ).reset_index()
    out["roi"] = np.where(out["stake"] > 0, out["profit"] / out["stake"], np.nan)
    # share（同一split内での比率）
    if label_col in out.columns:
        other_keys = [c for c in group_cols if c != label_col]
        if not other_keys:
            denom = float(out["n_bets"].sum())
            out["share"] = out["n_bets"] / denom if denom > 0 else np.nan
        else:
            out["share"] = out.groupby(other_keys)["n_bets"].transform(
                lambda x: x / x.sum() if float(x.sum()) > 0 else np.nan
            )
    return out


def _n1_tables(
    bets: pd.DataFrame,
    *,
    design_max_idx: int,
    out_dir: Path,
) -> None:
    """
    Post-N3 / Ticket N1（m1ベース）向け診断テーブルを出力する。
    - odds帯別
    - overlay帯別（p50/p80/p95）
    - ratio_final_to_buy帯別
    """
    if bets.empty:
        return

    d = bets.copy()
    d["idx"] = pd.to_numeric(d.get("idx"), errors="coerce")
    d["split"] = d["idx"].apply(lambda i: "design" if pd.notna(i) and int(i) <= int(design_max_idx) else "eval")

    splits = ["design", "eval", "all"]

    # odds帯（固定）
    odds_edges = [1.0, 3.0, 5.0, 10.0, 20.0, float("inf")]
    odds_labels = ["1-3", "3-5", "5-10", "10-20", "20+"]

    tables_odds = []
    for sp in splits:
        sub = d if sp == "all" else d[d["split"] == sp]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["odds_band"] = pd.cut(
            pd.to_numeric(sub.get("odds_at_buy"), errors="coerce"),
            bins=odds_edges,
            right=False,
            labels=odds_labels,
        )
        t = _bucket_stats(sub.dropna(subset=["odds_band"]), group_cols=["odds_band"], label_col="odds_band")
        t.insert(0, "split", sp)
        tables_odds.append(t)
    if tables_odds:
        pd.concat(tables_odds, ignore_index=True).to_csv(out_dir / "n1_odds_band_table.csv", index=False, encoding="utf-8")

    # overlay帯（p50/p80/p95）※splitごとに閾値を計算
    tables_overlay = []
    overlay_thresholds: dict[str, dict[str, float]] = {}
    for sp in splits:
        sub = d if sp == "all" else d[d["split"] == sp]
        if sub.empty:
            continue
        ov = pd.to_numeric(sub.get("p_hat"), errors="coerce") - pd.to_numeric(sub.get("p_mkt"), errors="coerce")
        ov = ov.dropna()
        if ov.empty:
            continue
        q50 = float(ov.quantile(0.50))
        q80 = float(ov.quantile(0.80))
        q95 = float(ov.quantile(0.95))
        overlay_thresholds[sp] = {"p50": q50, "p80": q80, "p95": q95}

        sub = sub.copy()
        sub["overlay"] = pd.to_numeric(sub.get("p_hat"), errors="coerce") - pd.to_numeric(sub.get("p_mkt"), errors="coerce")
        sub["overlay_band"] = pd.cut(
            sub["overlay"],
            bins=[-float("inf"), q50, q80, q95, float("inf")],
            right=True,
            labels=["<=p50", "p50-p80", "p80-p95", ">p95"],
        )
        t = _bucket_stats(sub.dropna(subset=["overlay_band"]), group_cols=["overlay_band"], label_col="overlay_band")
        t.insert(0, "split", sp)
        tables_overlay.append(t)
    if tables_overlay:
        pd.concat(tables_overlay, ignore_index=True).to_csv(out_dir / "n1_overlay_band_table.csv", index=False, encoding="utf-8")
        (out_dir / "n1_overlay_thresholds.json").write_text(
            json.dumps(overlay_thresholds, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ratio_final_to_buy帯（固定）
    tables_ratio = []
    for sp in splits:
        sub = d if sp == "all" else d[d["split"] == sp]
        if sub.empty:
            continue
        sub = sub.copy()
        r = pd.to_numeric(sub.get("ratio_final_to_buy"), errors="coerce")
        sub["ratio_bucket"] = pd.cut(
            r,
            bins=[-float("inf"), 0.90, 0.97, 1.03, float("inf")],
            right=False,
            labels=["<0.90", "0.90-0.97", "0.97-1.03", ">1.03"],
        )
        t = _bucket_stats(sub.dropna(subset=["ratio_bucket"]), group_cols=["ratio_bucket"], label_col="ratio_bucket")
        t.insert(0, "split", sp)
        tables_ratio.append(t)
    if tables_ratio:
        pd.concat(tables_ratio, ignore_index=True).to_csv(out_dir / "n1_ratio_bucket_table.csv", index=False, encoding="utf-8")


def _safe_qcut_deciles(x: pd.Series, q: int = 10) -> Optional[pd.Series]:
    """
    EV分位（decile）を作る。ユニーク値不足などで失敗する場合は None。
    戻り値は 1..k のint（k<=q）
    """
    v = pd.to_numeric(x, errors="coerce")
    v = v[np.isfinite(v)]
    if v.empty or len(v) < 20:
        return None
    try:
        # duplicates='drop' で閾値衝突を回避（bin数は減ることがある）
        bins = pd.qcut(v, q=q, labels=False, duplicates="drop")
        if bins is None:
            return None
        return (bins.astype(int) + 1)
    except Exception:
        return None


def _qcut_deciles_with_bins(x: pd.Series, q: int = 10) -> tuple[Optional[pd.Series], Optional[list[float]]]:
    """
    qcutでdecile（1..k）と、実際に使われたbin境界を返す。
    """
    v = pd.to_numeric(x, errors="coerce")
    v = v[np.isfinite(v)]
    if v.empty or len(v) < 20:
        return None, None
    try:
        codes, edges = pd.qcut(v, q=q, labels=False, duplicates="drop", retbins=True)
        if codes is None or edges is None:
            return None, None
        dec = (codes.astype(int) + 1)
        return dec, [float(e) for e in edges]
    except Exception:
        return None, None


def _ev_lift_tables(
    bets: pd.DataFrame,
    *,
    design_max_idx: int,
    out_dir: Path,
    ev_col: str = "ev",
    q: int = 10,
) -> None:
    """
    Ticket N5: EV lift / profit curve 診断

    出力:
      - ev_lift.csv（window×decile）
      - ev_lift_summary.json（design/eval集約 + 単調性チェック）
    """
    if bets.empty or ev_col not in bets.columns:
        return

    d = bets.copy()
    d["idx"] = pd.to_numeric(d.get("idx"), errors="coerce")
    d["split"] = d["idx"].apply(lambda i: "design" if pd.notna(i) and int(i) <= int(design_max_idx) else "eval")

    # 必須列の正規化
    for c in ["stake", "profit", "is_win", "odds_at_buy", ev_col, "p_hat", "p_mkt"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d["overlay"] = d["p_hat"] - d["p_mkt"] if ("p_hat" in d.columns and "p_mkt" in d.columns) else np.nan

    rows: list[dict[str, Any]] = []
    thr_rows: list[dict[str, Any]] = []
    for (w, idx, sp), g in d.groupby(["window", "idx", "split"], dropna=False):
        if g.empty:
            continue

        dec, edges = _qcut_deciles_with_bins(g[ev_col], q=q)
        if dec is None or edges is None:
            continue

        # dec は g の subset index 上なので、元の行に整列して付与
        gg = g.loc[dec.index].copy()
        gg["decile"] = dec.astype(int)
        max_dec = int(gg["decile"].max())
        # 10分位が作れない場合もあるので、ラベルを 1..max_dec として扱う

        thr_rows.append(
            {
                "window": w,
                "idx": int(idx) if pd.notna(idx) else None,
                "split": sp,
                "ev_col": ev_col,
                "q_requested": int(q),
                "decile_max": int(max_dec),
                "ev_bins": json.dumps(edges, ensure_ascii=False),
            }
        )

        for decile, h in gg.groupby("decile", dropna=False):
            stake = float(pd.to_numeric(h.get("stake"), errors="coerce").fillna(0.0).sum())
            profit = float(pd.to_numeric(h.get("profit"), errors="coerce").fillna(0.0).sum())
            roi = (profit / stake) if stake > 0 else None
            evv = pd.to_numeric(h.get(ev_col), errors="coerce")
            odds = pd.to_numeric(h.get("odds_at_buy"), errors="coerce") if "odds_at_buy" in h.columns else None
            overlay = pd.to_numeric(h.get("overlay"), errors="coerce") if "overlay" in h.columns else None
            ratio = pd.to_numeric(h.get("ratio_final_to_buy"), errors="coerce") if "ratio_final_to_buy" in h.columns else None
            snap = pd.to_numeric(h.get("snap_age_min"), errors="coerce") if "snap_age_min" in h.columns else None
            rows.append(
                {
                    "window": w,
                    "idx": int(idx) if pd.notna(idx) else None,
                    "split": sp,
                    "ev_col": ev_col,
                    "decile": int(decile),
                    "decile_max": max_dec,
                    "n_bets": int(len(h)),
                    "stake": stake,
                    "profit": profit,
                    "roi": roi,
                    "hit_rate": float(pd.to_numeric(h.get("is_win"), errors="coerce").mean()) if "is_win" in h.columns else None,
                    "avg_ev": float(pd.to_numeric(h.get(ev_col), errors="coerce").mean()),
                    "median_ev": float(evv.median()) if evv is not None and len(evv.dropna()) else None,
                    "p90_ev": float(evv.quantile(0.90)) if evv is not None and len(evv.dropna()) else None,
                    "avg_odds": float(pd.to_numeric(h.get("odds_at_buy"), errors="coerce").mean()) if "odds_at_buy" in h.columns else None,
                    "median_odds": float(odds.median()) if odds is not None and len(odds.dropna()) else None,
                    "p75_odds": float(odds.quantile(0.75)) if odds is not None and len(odds.dropna()) else None,
                    "avg_overlay": float(overlay.mean()) if overlay is not None and len(overlay.dropna()) else None,
                    "median_overlay": float(overlay.median()) if overlay is not None and len(overlay.dropna()) else None,
                    "p75_overlay": float(overlay.quantile(0.75)) if overlay is not None and len(overlay.dropna()) else None,
                    "ratio_median": float(ratio.median()) if ratio is not None and len(ratio.dropna()) else None,
                    "ratio_lt_0_90_rate": float((ratio < 0.90).mean()) if ratio is not None and len(ratio.dropna()) else None,
                    "snap_age_median": float(snap.median()) if snap is not None and len(snap.dropna()) else None,
                    "snap_age_p90": float(snap.quantile(0.90)) if snap is not None and len(snap.dropna()) else None,
                }
            )

    out_csv = out_dir / "ev_lift.csv"
    out_json = out_dir / "ev_lift_summary.json"
    out_thr = out_dir / "ev_decile_thresholds.csv"
    out_agg = out_dir / "ev_lift_by_split_decile.csv"

    if not rows:
        # 何も作れない場合でも空出力は作らない（上流で判断）
        return

    ev_lift = pd.DataFrame(rows)
    ev_lift.to_csv(out_csv, index=False, encoding="utf-8")
    if thr_rows:
        pd.DataFrame(thr_rows).to_csv(out_thr, index=False, encoding="utf-8")

    # split×decileの集約（windowごとのROIのmedian/mean + pooled ROI）
    agg = (
        ev_lift.groupby(["split", "decile"], dropna=False, observed=False)
        .agg(
            n_windows=("window", "nunique"),
            n_rows=("roi", "size"),
            window_roi_median=("roi", "median"),
            window_roi_mean=("roi", "mean"),
            pooled_stake=("stake", "sum"),
            pooled_profit=("profit", "sum"),
            median_n_bets=("n_bets", "median"),
            avg_ev_mean=("avg_ev", "mean"),
            median_odds_mean=("median_odds", "mean"),
        )
        .reset_index()
    )
    agg["pooled_roi"] = np.where(agg["pooled_stake"] > 0, agg["pooled_profit"] / agg["pooled_stake"], np.nan)
    agg.to_csv(out_agg, index=False, encoding="utf-8")

    def _agg_block(block: pd.DataFrame) -> dict[str, Any]:
        if block.empty:
            return {"n_windows": 0, "deciles": {}}
        # decile別に windowごとのroiを集約（median/mean）
        deciles: dict[str, Any] = {}
        for decile, g in block.groupby("decile", dropna=False):
            rois = pd.to_numeric(g["roi"], errors="coerce").dropna()
            deciles[str(int(decile))] = {
                "n_rows": int(len(g)),
                "median_roi": float(rois.median()) if len(rois) else None,
                "mean_roi": float(rois.mean()) if len(rois) else None,
                "median_n_bets": float(pd.to_numeric(g["n_bets"], errors="coerce").median()),
            }
        return {
            "n_windows": int(block["window"].nunique()),
            "deciles": deciles,
        }

    # 集約（design/eval）
    design = ev_lift[ev_lift["split"] == "design"].copy()
    eval_ = ev_lift[ev_lift["split"] == "eval"].copy()

    # 単調性チェック（evalで D10 が下位より良いか）
    # ※decile_maxが10にならないwindowもあり得るので、「そのwindowの最大decile」をD10相当として扱う
    monotone_flag = None
    try:
        # windowごとに top_decile_roi と bottom_roi（D1..D5の中央値）を作る
        eval_w = []
        for w, g in eval_.groupby("window", dropna=False):
            g = g.copy()
            if g.empty:
                continue
            top_dec = int(pd.to_numeric(g["decile"], errors="coerce").max())
            top_roi = pd.to_numeric(g.loc[g["decile"] == top_dec, "roi"], errors="coerce").dropna()
            if top_roi.empty:
                continue
            bottom = g[g["decile"].isin([1, 2, 3, 4, 5])]
            bottom_rois = pd.to_numeric(bottom["roi"], errors="coerce").dropna()
            if bottom_rois.empty:
                continue
            eval_w.append(
                {
                    "window": w,
                    "top_decile": top_dec,
                    "roi_top": float(top_roi.iloc[0]),
                    "roi_bottom_median": float(bottom_rois.median()),
                    "signal": float(top_roi.iloc[0]) > float(bottom_rois.median()),
                }
            )
        if eval_w:
            dfw = pd.DataFrame(eval_w)
            monotone_flag = {
                "n_windows": int(len(dfw)),
                "signal_rate": float(dfw["signal"].mean()),
                "rule": "roi(top_decile) > median(roi(decile 1..5))",
            }
        else:
            monotone_flag = {"n_windows": 0, "signal_rate": None, "rule": "roi(top_decile) > median(roi(decile 1..5))"}
    except Exception:
        monotone_flag = {"n_windows": None, "signal_rate": None, "rule": "roi(top_decile) > median(roi(decile 1..5))"}

    summary = {
        "meta": {
            "ev_col": ev_col,
            "q": int(q),
            "design_max_idx": int(design_max_idx),
            "paths": {
                "ev_lift_csv": str(out_csv),
                "ev_decile_thresholds_csv": str(out_thr),
                "ev_lift_by_split_decile_csv": str(out_agg),
            },
        },
        "design": _agg_block(design),
        "eval": _agg_block(eval_),
        "monotone_check_eval": monotone_flag,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _window_driver_row(w: WindowFiles) -> dict[str, Any]:
    s = _load_summary(w.summary_json)
    bt = s.get("backtest", {}) or {}
    pq = s.get("pred_quality", {}) or {}

    df = _load_bets(w.bets_csv)
    overlay = None
    if "p_hat" in df.columns and "p_mkt" in df.columns:
        overlay = df["p_hat"] - df["p_mkt"]

    row: dict[str, Any] = {
        "window": w.name,
        "idx": w.idx,
        "test_start": (s.get("test") or {}).get("start"),
        "test_end": (s.get("test") or {}).get("end"),
        # summary.json
        "roi": bt.get("roi"),
        "n_bets": bt.get("n_bets"),
        "max_drawdown": bt.get("max_drawdown"),
        "total_stake": bt.get("total_stake"),
        "total_profit": bt.get("total_profit"),
        "pred_logloss_market": pq.get("logloss_market"),
        "pred_logloss_blend": pq.get("logloss_blend"),
        "pred_logloss_calibrated": pq.get("logloss_calibrated"),
        "pred_brier_market": pq.get("brier_market"),
        "pred_brier_blend": pq.get("brier_blend"),
        "pred_brier_calibrated": pq.get("brier_calibrated"),
        # bets.csv（Step3-A: driver表）
        "odds_median": _q(df.get("odds_at_buy"), 0.50),
        "odds_p75": _q(df.get("odds_at_buy"), 0.75),
        "odds_p90": _q(df.get("odds_at_buy"), 0.90),
        "ratio_median": _q(df.get("ratio_final_to_buy"), 0.50),
        "ratio_p25": _q(df.get("ratio_final_to_buy"), 0.25),
        "ratio_lt_0_90_rate": _rate_lt(df.get("ratio_final_to_buy"), 0.90),
        "ts_std_median": _q(df.get("log_odds_std_60m"), 0.50),
        "ts_std_p90": _q(df.get("log_odds_std_60m"), 0.90),
        "overlay_median": _q(overlay, 0.50) if overlay is not None else None,
        "overlay_p75": _q(overlay, 0.75) if overlay is not None else None,
        "ev_median": _q(df.get("ev"), 0.50),
        "ev_p75": _q(df.get("ev"), 0.75),
    }
    return row


def _calc_window_metrics_from_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: 必ず window 列を持つ（1行=1 bet）
    Returns: 1行=1window の (n_bets, stake, profit, roi)
    """
    if df.empty:
        return pd.DataFrame(columns=["window", "n_bets", "stake", "profit", "roi"])

    g = df.groupby("window", dropna=False)
    out = g.agg(
        n_bets=("profit", "size"),
        stake=("stake", "sum"),
        profit=("profit", "sum"),
    ).reset_index()
    out["roi"] = np.where(out["stake"] > 0, out["profit"] / out["stake"], np.nan)
    return out


def _sweep_filters(
    bets: pd.DataFrame,
    drivers: pd.DataFrame,
    design_max_idx: int,
    odds_thresholds: list[float],
    ev_thresholds: list[float],
    std_quantiles: list[float],
    min_roi_improve: float,
    min_median_n_bets: int,
) -> pd.DataFrame:
    # baseline（design/eval 窓）
    d_design = drivers[drivers["idx"] <= design_max_idx].copy()
    d_eval = drivers[drivers["idx"] > design_max_idx].copy()

    base_median_roi = float(pd.to_numeric(d_design["roi"], errors="coerce").median())
    base_median_n_bets = float(pd.to_numeric(d_design["n_bets"], errors="coerce").median())

    base_eval_median_roi = float(pd.to_numeric(d_eval["roi"], errors="coerce").median()) if len(d_eval) > 0 else None
    base_eval_median_n_bets = float(pd.to_numeric(d_eval["n_bets"], errors="coerce").median()) if len(d_eval) > 0 else None

    def _eval_subset(label: str, d_sub: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
        if d_sub.empty:
            return {
                f"{label}_windows": 0,
                f"{label}_median_roi": None,
                f"{label}_median_n_bets": None,
                f"{label}_no_bet_windows": None,
                f"{label}_median_roi_improve": None,
                f"{label}_pooled_roi": None,
                f"{label}_pooled_stake": None,
                f"{label}_pooled_profit": None,
            }

        m = _calc_window_metrics_from_bets(bets[mask & (bets["window"].isin(d_sub["window"]))])
        merged = d_sub[["window", "roi", "n_bets"]].merge(m, on="window", how="left", suffixes=("_base", ""))
        merged["n_bets"] = merged["n_bets"].fillna(0).astype(int)
        merged["stake"] = pd.to_numeric(merged["stake"], errors="coerce").fillna(0.0)
        merged["profit"] = pd.to_numeric(merged["profit"], errors="coerce").fillna(0.0)

        med_roi = float(pd.to_numeric(merged["roi"], errors="coerce").median())
        med_n = float(pd.to_numeric(merged["n_bets"], errors="coerce").median())
        no_bet_windows = int((merged["n_bets"] == 0).sum())

        # pooled ROI（参考）
        subset_mask = mask & (bets["window"].isin(d_sub["window"]))
        stake = float(pd.to_numeric(bets.loc[subset_mask, "stake"], errors="coerce").sum())
        profit = float(pd.to_numeric(bets.loc[subset_mask, "profit"], errors="coerce").sum())
        pooled_roi = (profit / stake) if stake > 0 else None

        return {
            f"{label}_windows": int(len(d_sub)),
            f"{label}_median_roi": med_roi,
            f"{label}_median_n_bets": med_n,
            f"{label}_no_bet_windows": no_bet_windows,
            f"{label}_pooled_roi": pooled_roi,
            f"{label}_pooled_stake": stake,
            f"{label}_pooled_profit": profit,
        }

    def _as_float(val: Any) -> Optional[float]:
        try:
            v = float(val)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return v

    def eval_candidate(name: str, mask: pd.Series, meta: dict[str, Any]) -> dict[str, Any]:
        design_metrics = _eval_subset("design", d_design, mask)
        eval_metrics = _eval_subset("eval", d_eval, mask)

        improve = None
        design_roi = _as_float(design_metrics["design_median_roi"])
        base_roi = _as_float(base_median_roi)
        if design_roi is not None and base_roi is not None:
            improve = design_roi - base_roi

        design_n_bets = _as_float(design_metrics["design_median_n_bets"])
        design_no_bets = design_metrics["design_no_bet_windows"]

        ok = bool(
            improve is not None
            and improve >= float(min_roi_improve)
            and design_n_bets is not None
            and design_n_bets >= float(min_median_n_bets)
            and design_no_bets is not None
            and int(design_no_bets) == 0
        )

        # eval improvement（参考）
        eval_improve = None
        if base_eval_median_roi is not None and eval_metrics["eval_median_roi"] is not None:
            try:
                eval_improve = float(eval_metrics["eval_median_roi"]) - float(base_eval_median_roi)
            except Exception:
                eval_improve = None

        row = {
            "filter": name,
            "design_max_idx": design_max_idx,
            "baseline_design_median_roi": base_median_roi,
            "baseline_design_median_n_bets": base_median_n_bets,
            "baseline_eval_median_roi": base_eval_median_roi,
            "baseline_eval_median_n_bets": base_eval_median_n_bets,
            "design_median_roi_improve": improve,
            "eval_median_roi_improve": eval_improve,
            "passes_rule": ok,
        }
        row.update(design_metrics)
        row.update(eval_metrics)
        row.update(meta)
        return row

    rows: list[dict[str, Any]] = []

    # odds <= X
    if "odds_at_buy" in bets.columns:
        for x in odds_thresholds:
            mask = bets["odds_at_buy"].notna() & (bets["odds_at_buy"] <= float(x))
            rows.append(eval_candidate("odds_max", mask, {"param": float(x)}))

    # EV >= m
    if "ev" in bets.columns:
        for m in ev_thresholds:
            mask = bets["ev"].notna() & (bets["ev"] >= float(m))
            rows.append(eval_candidate("ev_min", mask, {"param": float(m)}))

    # std <= quantile
    if "log_odds_std_60m" in bets.columns:
        design_bets = bets[bets["idx"] <= design_max_idx]
        x = pd.to_numeric(design_bets["log_odds_std_60m"], errors="coerce").dropna()
        for q in std_quantiles:
            if x.empty:
                continue
            thr = float(x.quantile(float(q)))
            mask = bets["log_odds_std_60m"].notna() & (bets["log_odds_std_60m"] <= thr)
            rows.append(eval_candidate("ts_std_max", mask, {"param": float(q), "threshold": thr}))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 見やすさ: passes_rule → design_median_roi_improve → design_median_n_bets
    out = out.sort_values(
        ["passes_rule", "design_median_roi_improve", "design_median_n_bets", "eval_median_roi_improve"],
        ascending=[False, False, False, False],
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze rolling holdout group_dir (Step3-A/B)")
    p.add_argument("--group-dir", type=Path, required=True)
    p.add_argument("--design-max-idx", type=int, default=12, help="design window の最大idx（w001..w012なら12）")
    p.add_argument("--out-drivers", type=Path, default=None)
    p.add_argument("--out-sweep", type=Path, default=None)
    p.add_argument("--min-roi-improve", type=float, default=0.05)
    p.add_argument("--min-median-n-bets", type=int, default=80)
    p.add_argument("--odds-thresholds", type=str, default="6,8,10,12,15,20,30")
    p.add_argument("--ev-thresholds", type=str, default="0.00,0.02,0.04,0.06,0.08,0.10")
    p.add_argument("--std-quantiles", type=str, default="0.60,0.70,0.80,0.90")
    # Ticket N5
    p.add_argument("--ev-lift", action="store_true", help="EV lift（窓×EV分位）を ev_lift.csv / ev_lift_summary.json に出力する")
    p.add_argument("--ev-col", type=str, default="ev", help="EV列名（default: ev。ev_adj 等を使う場合は指定）")
    args = p.parse_args()

    group_dir = args.group_dir
    if not group_dir.is_absolute():
        group_dir = _project_root() / group_dir
    if not group_dir.exists():
        raise SystemExit(f"group_dir not found: {group_dir}")

    windows = _parse_window_files(group_dir)
    if not windows:
        raise SystemExit("No completed windows (summary.json + bets.csv) found under group_dir")

    drivers_rows = [_window_driver_row(w) for w in windows]
    drivers = pd.DataFrame(drivers_rows)
    drivers["is_design"] = drivers["idx"] <= int(args.design_max_idx)

    out_drivers = args.out_drivers or (group_dir / "drivers.csv")
    if not out_drivers.is_absolute():
        out_drivers = _project_root() / out_drivers
    out_drivers.parent.mkdir(parents=True, exist_ok=True)
    drivers.to_csv(out_drivers, index=False, encoding="utf-8")

    # bets結合（window/idxを付与）
    bets_all = []
    for w in windows:
        df = _load_bets(w.bets_csv)
        df["window"] = w.name
        df["idx"] = int(w.idx)
        bets_all.append(df)
    bets = pd.concat(bets_all, ignore_index=True) if bets_all else pd.DataFrame()

    odds_thresholds = [float(x.strip()) for x in args.odds_thresholds.split(",") if x.strip()]
    ev_thresholds = [float(x.strip()) for x in args.ev_thresholds.split(",") if x.strip()]
    std_quantiles = [float(x.strip()) for x in args.std_quantiles.split(",") if x.strip()]

    sweep = _sweep_filters(
        bets=bets,
        drivers=drivers,
        design_max_idx=int(args.design_max_idx),
        odds_thresholds=odds_thresholds,
        ev_thresholds=ev_thresholds,
        std_quantiles=std_quantiles,
        min_roi_improve=float(args.min_roi_improve),
        min_median_n_bets=int(args.min_median_n_bets),
    )

    out_sweep = args.out_sweep or (group_dir / "filter_sweep.csv")
    if not out_sweep.is_absolute():
        out_sweep = _project_root() / out_sweep
    out_sweep.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out_sweep, index=False, encoding="utf-8")

    # Post-N3: N1（診断）テーブル出力
    _n1_tables(
        bets=bets,
        design_max_idx=int(args.design_max_idx),
        out_dir=group_dir,
    )

    # Ticket N5: EV lift
    if bool(args.ev_lift):
        _ev_lift_tables(
            bets=bets,
            design_max_idx=int(args.design_max_idx),
            out_dir=group_dir,
            ev_col=str(args.ev_col),
            q=10,
        )

    # 推奨を表示
    print("group_dir:", group_dir)
    print("completed_windows:", len(windows))
    print("drivers:", out_drivers)
    print("sweep:", out_sweep)
    print("n1_tables:", group_dir / "n1_odds_band_table.csv")
    if bool(args.ev_lift):
        print("ev_lift:", group_dir / "ev_lift.csv")

    if not sweep.empty:
        best = sweep.iloc[0].to_dict()
        print("best_candidate:", {k: best.get(k) for k in ["filter", "param", "threshold", "design_median_roi", "design_median_n_bets", "design_median_roi_improve", "passes_rule"]})


if __name__ == "__main__":
    main()

