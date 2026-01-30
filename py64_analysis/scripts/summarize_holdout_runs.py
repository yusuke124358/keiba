"""
holdout_runs 配下の summary.json を集計して、一覧CSV + 簡易ランキングを出力する。

使い方:
  python py64_analysis/scripts/summarize_holdout_runs.py
  python py64_analysis/scripts/summarize_holdout_runs.py --root data/holdout_runs --out data/holdout_runs_summary.csv --top 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datetime import datetime


def _project_root() -> Path:
    # keiba/py64_analysis/scripts/ -> keiba/
    return Path(__file__).resolve().parents[2]


def _safe_get(d: dict, path: str, default=None):
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _q(s, q: float):
    try:
        x = _to_num(s).dropna()
        if x.empty:
            return None
        v = float(x.quantile(q))
        return v if pd.notna(v) else None
    except Exception:
        return None


def _rate_lt(s, thr: float):
    try:
        x = _to_num(s)
        x = x[pd.notna(x)]
        if x.empty:
            return None
        return float((x < float(thr)).mean())
    except Exception:
        return None


def _bets_stats(run_dir: Path) -> dict:
    """
    run_dir/bets.csv があれば読み、Step3-A向けの要約統計を返す。
    失敗時は空dict。
    """
    fp = run_dir / "bets.csv"
    if not fp.exists():
        return {}
    try:
        df = pd.read_csv(fp)
    except Exception:
        return {}

    for c in [
        "odds_at_buy",
        "ratio_final_to_buy",
        "log_odds_std_60m",
        "snap_age_min",
        "p_hat",
        "p_mkt",
        "ev",
    ]:
        if c in df.columns:
            df[c] = _to_num(df[c])

    overlay = None
    if "p_hat" in df.columns and "p_mkt" in df.columns:
        overlay = df["p_hat"] - df["p_mkt"]

    out = {
        "odds_median": _q(df.get("odds_at_buy"), 0.50),
        "odds_p75": _q(df.get("odds_at_buy"), 0.75),
        "odds_p90": _q(df.get("odds_at_buy"), 0.90),
        "ratio_median": _q(df.get("ratio_final_to_buy"), 0.50),
        "ratio_p25": _q(df.get("ratio_final_to_buy"), 0.25),
        "ratio_lt_0_90_rate": _rate_lt(df.get("ratio_final_to_buy"), 0.90),
        "ts_std_median": _q(df.get("log_odds_std_60m"), 0.50),
        "ts_std_p90": _q(df.get("log_odds_std_60m"), 0.90),
        "snap_age_median": _q(df.get("snap_age_min"), 0.50),
        "snap_age_p90": _q(df.get("snap_age_min"), 0.90),
        "overlay_median": _q(overlay, 0.50) if overlay is not None else None,
        "overlay_p75": _q(overlay, 0.75) if overlay is not None else None,
        "ev_median": _q(df.get("ev"), 0.50),
        "ev_p75": _q(df.get("ev"), 0.75),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize holdout_runs/*/summary.json to CSV")
    p.add_argument("--root", type=Path, default=Path("data/holdout_runs"), help="holdout_runs root dir")
    p.add_argument("--out", type=Path, default=Path("data/holdout_runs_summary.csv"), help="output CSV path")
    p.add_argument("--top", type=int, default=10, help="print top-N ranking by ROI")
    args = p.parse_args()

    root = args.root
    if not root.is_absolute():
        root = _project_root() / root

    summaries = sorted(root.glob("**/summary.json"))
    rows: list[dict[str, Any]] = []
    for sp in summaries:
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue

        bets_stats = _bets_stats(sp.parent)

        row: dict[str, Any] = {
            "run_dir": str(sp.parent),
            "name": data.get("name"),
            "generated_at": data.get("generated_at"),
            "train_start": _safe_get(data, "train.start"),
            "train_end": _safe_get(data, "train.end"),
            "valid_start": _safe_get(data, "valid.start"),
            "valid_end": _safe_get(data, "valid.end"),
            "test_start": _safe_get(data, "test.start"),
            "test_end": _safe_get(data, "test.end"),
            "buy_t_minus_minutes": data.get("buy_t_minus_minutes"),
            "closing_odds_multiplier": data.get("closing_odds_multiplier"),
            "closing_odds_multiplier_estimated": data.get("closing_odds_multiplier_estimated"),
            "closing_odds_multiplier_quantile": data.get("closing_odds_multiplier_quantile"),
            "n_train_races": _safe_get(data, "train.n_races"),
            "n_test_races": _safe_get(data, "test.n_races"),
            "n_bets": _safe_get(data, "backtest.n_bets"),
            "roi": _safe_get(data, "backtest.roi"),
            "max_drawdown": _safe_get(data, "backtest.max_drawdown"),
            "max_drawdown_bankroll": _safe_get(data, "backtest.max_drawdown_bankroll"),
            "min_bankroll": _safe_get(data, "backtest.min_bankroll"),
            "min_bankroll_frac": _safe_get(data, "backtest.min_bankroll_frac"),
            "total_stake": _safe_get(data, "backtest.total_stake"),
            "total_profit": _safe_get(data, "backtest.total_profit"),
            "ending_bankroll": _safe_get(data, "backtest.ending_bankroll"),
            "log_growth": _safe_get(data, "backtest.log_growth"),
            "risk_of_ruin_0_3": _safe_get(data, "backtest.risk_of_ruin_0_3"),
            "risk_of_ruin_0_5": _safe_get(data, "backtest.risk_of_ruin_0_5"),
            "risk_of_ruin_0_7": _safe_get(data, "backtest.risk_of_ruin_0_7"),
            # 確率品質（market / blend / calibrated を揃えて比較できるように）
            "pred_logloss_market": _safe_get(data, "pred_quality.logloss_market"),
            "pred_logloss_blend": _safe_get(data, "pred_quality.logloss_blend"),
            "pred_logloss_calibrated": _safe_get(data, "pred_quality.logloss_calibrated"),
            "pred_brier_market": _safe_get(data, "pred_quality.brier_market"),
            "pred_brier_blend": _safe_get(data, "pred_quality.brier_blend"),
            "pred_brier_calibrated": _safe_get(data, "pred_quality.brier_calibrated"),
            "train_valid_logloss_calibrated": data.get("train_metrics", {}).get("valid_logloss_calibrated"),
            "train_valid_brier_calibrated": data.get("train_metrics", {}).get("valid_brier_calibrated"),
            "report_path": _safe_get(data, "paths.report"),
        }
        row.update({f"bets_{k}": v for k, v in bets_stats.items()})
        rows.append(row)

    if not rows:
        raise SystemExit(f"No summary.json found under: {root}")

    df = pd.DataFrame(rows)

    # 数値列の整形
    for c in [
        "roi",
        "max_drawdown",
        "max_drawdown_bankroll",
        "min_bankroll",
        "min_bankroll_frac",
        "total_stake",
        "total_profit",
        "ending_bankroll",
        "log_growth",
        "risk_of_ruin_0_3",
        "risk_of_ruin_0_5",
        "risk_of_ruin_0_7",
        "pred_logloss_market",
        "pred_logloss_blend",
        "pred_logloss_calibrated",
        "pred_brier_market",
        "pred_brier_blend",
        "pred_brier_calibrated",
        "train_valid_logloss_calibrated",
        "train_valid_brier_calibrated",
        # bets_*
        "bets_odds_median",
        "bets_odds_p75",
        "bets_odds_p90",
        "bets_ratio_median",
        "bets_ratio_p25",
        "bets_ratio_lt_0_90_rate",
        "bets_ts_std_median",
        "bets_ts_std_p90",
        "bets_snap_age_median",
        "bets_snap_age_p90",
        "bets_overlay_median",
        "bets_overlay_p75",
        "bets_ev_median",
        "bets_ev_p75",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["roi", "n_bets"], ascending=[False, False])

    # 派生指標（比較が楽）
    def _days_between(s: Any, e: Any) -> float | None:
        try:
            if not s or not e:
                return None
            ds = datetime.strptime(str(s), "%Y-%m-%d").date()
            de = datetime.strptime(str(e), "%Y-%m-%d").date()
            return float((de - ds).days + 1)
        except Exception:
            return None

    df["test_days"] = df.apply(lambda r: _days_between(r.get("test_start"), r.get("test_end")), axis=1)
    df["profit_per_day"] = df["total_profit"] / df["test_days"]
    df["stake_per_day"] = df["total_stake"] / df["test_days"]
    df["bets_per_day"] = df["n_bets"] / df["test_days"]
    df["profit_per_bet"] = df["total_profit"] / df["n_bets"]

    out = args.out
    if not out.is_absolute():
        out = _project_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    # 簡易ランキング（コンソール）
    topn = max(1, int(args.top))
    cols = [
        "name",
        "test_start",
        "test_end",
        "roi",
        "n_bets",
        "max_drawdown",
        "pred_brier_market",
        "pred_brier_blend",
        "pred_brier_calibrated",
    ]
    cols = [c for c in cols if c in df.columns]
    print(f"Wrote: {out}")
    print("Top runs by ROI:")
    print(df[cols].head(topn).to_string(index=False))


if __name__ == "__main__":
    main()
