"""Summarize q95 low-odds overlay sweep (replay-from-base)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from keiba.analysis.metrics_utils import roi_footer, sign_mismatch


STAKE_COLS = ["stake", "stake_yen", "stake_amount", "bet_amount", "stake_raw"]
RETURN_COLS = ["return", "return_yen", "payout", "payout_yen"]
PROFIT_COLS = ["profit", "net_profit", "pnl"]
ODDS_COLS = ["odds_at_buy", "odds", "buy_odds", "win_odds"]
P_HAT_COLS = ["p_hat", "p_cal", "p_blend", "p_model", "p_hat_raw"]
PMKT_RACE_COLS = ["p_mkt_race"]
PMKT_COLS = ["p_mkt", "p_mkt_raw", "p_mkt_prob", "p_mkt_probability"]
KEY_COL_CANDIDATES = [
    ["race_id", "horse_no", "asof_time"],
    ["race_id", "horse_id", "asof_time"],
]


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _detect_key_cols(cols: list[str]) -> Optional[list[str]]:
    for cand in KEY_COL_CANDIDATES:
        if all(c in cols for c in cand):
            return cand
    return None


def _detect_stake_col(cols: list[str]) -> Optional[str]:
    return _find_col(cols, STAKE_COLS)


def _detect_return_col(cols: list[str]) -> Optional[str]:
    return _find_col(cols, RETURN_COLS)


def _detect_profit_col(cols: list[str]) -> Optional[str]:
    return _find_col(cols, PROFIT_COLS)


def _detect_odds_col(cols: list[str]) -> Optional[str]:
    return _find_col(cols, ODDS_COLS)


def _detect_p_hat_col(cols: list[str]) -> Optional[str]:
    return _find_col(cols, P_HAT_COLS)


def _detect_p_mkt_cols(cols: list[str]) -> tuple[Optional[str], Optional[str]]:
    p_race = _find_col(cols, PMKT_RACE_COLS)
    if p_race:
        return p_race, "p_mkt_race"
    p_mkt = _find_col(cols, PMKT_COLS)
    if p_mkt:
        return p_mkt, "p_mkt_fallback"
    return None, None


def _parse_window_dates(name: str) -> tuple[str, str]:
    parts = name.split("_")
    if len(parts) < 3:
        return "", ""
    return f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}", f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}"


def _read_expected_windows(run_dir: Path) -> list[str]:
    idx = run_dir / "rolling_index.json"
    if idx.exists():
        try:
            payload = json.loads(idx.read_text(encoding="utf-8"))
            plans = payload.get("plans", []) if isinstance(payload, dict) else []
            names = [p.get("window_name") for p in plans if isinstance(p, dict) and p.get("window_name")]
            return sorted([n for n in names if n])
        except Exception:
            pass
    return sorted([p.name for p in run_dir.glob("w*/")])


def _coverage_status(run_dir: Path, expected_windows: list[str]) -> dict[str, Any]:
    used = []
    missing = []
    empty = []
    for wname in expected_windows:
        bets_path = run_dir / wname / "bets.csv"
        if not bets_path.exists():
            missing.append(wname)
            continue
        used.append(wname)
        try:
            df = pd.read_csv(bets_path)
        except Exception:
            continue
        if df.empty:
            empty.append(wname)
    return {
        "expected": expected_windows,
        "used": used,
        "missing": missing,
        "empty": empty,
    }


def _variant_name(low_odds_max: float, min_overlay: float) -> str:
    low_s = f"{low_odds_max:.1f}".replace(".", "p")
    ov_s = f"{min_overlay:.3f}".replace(".", "p")
    return f"lowodds{low_s}_ov{ov_s}"


def _parse_variant(variant: str) -> tuple[Optional[float], Optional[float]]:
    if not variant.startswith("lowodds"):
        return None, None
    try:
        body = variant.replace("lowodds", "")
        parts = body.split("_ov")
        low = float(parts[0].replace("p", "."))
        ov = float(parts[1].replace("p", "."))
        return low, ov
    except Exception:
        return None, None


def _compute_profit(df: pd.DataFrame, stake_col: str, ret_col: Optional[str], prof_col: Optional[str]) -> pd.Series:
    stake = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
    if ret_col is not None:
        ret = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0)
        return ret - stake
    if prof_col is not None:
        return pd.to_numeric(df[prof_col], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(df))


def _compute_window_summary(bets_path: Path, bankroll: float) -> dict[str, Any]:
    df = pd.read_csv(bets_path)
    if df.empty:
        return {"n_bets": 0, "total_stake": 0.0, "total_profit": 0.0, "roi": 0.0, "max_drawdown": 0.0}
    stake_col = _detect_stake_col(df.columns.tolist())
    odds_col = _detect_odds_col(df.columns.tolist())
    if stake_col is None or odds_col is None:
        return {"n_bets": 0, "total_stake": 0.0, "total_profit": 0.0, "roi": 0.0, "max_drawdown": 0.0}
    ret_col = _detect_return_col(df.columns.tolist())
    prof_col = _detect_profit_col(df.columns.tolist())
    profit = _compute_profit(df, stake_col, ret_col, prof_col)
    stake = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
    total_stake = float(stake.sum())
    total_profit = float(profit.sum())
    roi = (total_profit / total_stake) if total_stake > 0 else 0.0
    n_bets = int(len(df))

    if "asof_time" in df.columns:
        df["bet_date"] = pd.to_datetime(df["asof_time"], errors="coerce").dt.date
        df["__profit__"] = profit
        daily = df.groupby("bet_date", dropna=False).agg(
            stake=(stake_col, "sum"), profit=("__profit__", "sum")
        )
    else:
        daily = None

    if daily is None or daily.empty:
        max_dd = 0.0
    else:
        daily = daily.reset_index()
        daily["cum_profit"] = daily["profit"].cumsum()
        daily["equity"] = bankroll + daily["cum_profit"]
        daily["peak"] = daily["equity"].cummax()
        daily["drawdown_pct"] = (daily["peak"] - daily["equity"]) / daily["peak"]
        max_dd = float(daily["drawdown_pct"].max()) if not daily.empty else 0.0
    return {
        "n_bets": n_bets,
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": roi,
        "max_drawdown": max_dd,
    }


def _read_bankroll(run_dir: Path) -> float:
    for w in sorted(run_dir.glob("w*/")):
        cfg = w / "config_used.yaml"
        if not cfg.exists():
            continue
        try:
            import yaml

            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            return float((data.get("betting") or {}).get("bankroll_yen", 1_000_000.0))
        except Exception:
            continue
    return 1_000_000.0


def _read_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    name_col = _find_col(df.columns.tolist(), ["name", "window", "window_id"])
    roi_col = _find_col(df.columns.tolist(), ["roi", "ROI"])
    dd_col = _find_col(df.columns.tolist(), ["max_drawdown", "max_dd", "maxdd"])
    nb_col = _find_col(df.columns.tolist(), ["n_bets", "bets", "nBet"])
    ts_col = _find_col(df.columns.tolist(), ["test_start"])
    te_col = _find_col(df.columns.tolist(), ["test_end"])
    stake_col = _find_col(df.columns.tolist(), ["total_stake", "stake", "stake_sum", "stake_yen"])
    profit_col = _find_col(df.columns.tolist(), ["total_profit", "profit", "pnl", "net_profit"])
    if not all([name_col, roi_col, dd_col, nb_col, ts_col, te_col]):
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "window": df[name_col],
            "test_start": df[ts_col],
            "test_end": df[te_col],
            "roi": pd.to_numeric(df[roi_col], errors="coerce"),
            "max_drawdown": pd.to_numeric(df[dd_col], errors="coerce"),
            "n_bets": pd.to_numeric(df[nb_col], errors="coerce"),
        }
    )
    if stake_col:
        out["total_stake"] = pd.to_numeric(df[stake_col], errors="coerce")
    if profit_col:
        out["total_profit"] = pd.to_numeric(df[profit_col], errors="coerce")
    return out


def _merge_base_var(base_df: pd.DataFrame, var_df: pd.DataFrame) -> pd.DataFrame:
    merged = base_df.merge(var_df, on=["test_start", "test_end"], suffixes=("_base", "_var"))
    if merged.empty:
        return pd.DataFrame()
    merged["d_roi"] = merged["roi_var"] - merged["roi_base"]
    merged["d_maxdd"] = merged["max_drawdown_var"] - merged["max_drawdown_base"]
    merged["d_n_bets"] = merged["n_bets_var"] - merged["n_bets_base"]
    merged["d_stake"] = merged.get("total_stake_var") - merged.get("total_stake_base")
    merged["d_profit"] = merged.get("total_profit_var") - merged.get("total_profit_base")
    return merged


def _gate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "median_d_roi": float("nan"),
            "improve_rate": float("nan"),
            "median_d_maxdd": float("nan"),
            "median_n_bets_var": float("nan"),
            "zero_bet_windows": 0,
            "gate_pass": False,
        }
    d_roi = df["d_roi"]
    d_maxdd = df["d_maxdd"]
    med_d_roi = float(np.median(d_roi))
    improve_rate = float((d_roi > 0).mean())
    med_d_maxdd = float(np.median(d_maxdd))
    med_n_bets = float(np.median(df["n_bets_var"]))
    zero_bet = int((df["n_bets_var"] == 0).sum())
    gate_pass = bool(
        improve_rate >= 0.6 and med_d_roi > 0 and med_d_maxdd <= 0 and med_n_bets >= 80 and zero_bet == 0
    )
    base_stake = float(df["total_stake_base"].sum()) if "total_stake_base" in df.columns else 0.0
    var_stake = float(df["total_stake_var"].sum()) if "total_stake_var" in df.columns else 0.0
    base_profit = float(df["total_profit_base"].sum()) if "total_profit_base" in df.columns else 0.0
    var_profit = float(df["total_profit_var"].sum()) if "total_profit_var" in df.columns else 0.0
    pooled_roi_base = (base_profit / base_stake) if base_stake else float("nan")
    pooled_roi_var = (var_profit / var_stake) if var_stake else float("nan")
    return {
        "median_d_roi": med_d_roi,
        "improve_rate": improve_rate,
        "median_d_maxdd": med_d_maxdd,
        "median_n_bets_var": med_n_bets,
        "zero_bet_windows": zero_bet,
        "gate_pass": gate_pass,
        "pooled_roi_base": pooled_roi_base,
        "pooled_roi_var": pooled_roi_var,
        "pooled_stake_base": base_stake,
        "pooled_stake_var": var_stake,
    }


def _candidate_scarcity(run_dir: Path) -> dict[str, Any]:
    rows = []
    for w in run_dir.iterdir():
        if not w.is_dir():
            continue
        bets_path = w / "bets.csv"
        if not bets_path.exists():
            continue
        df = pd.read_csv(bets_path)
        if "asof_time" not in df.columns:
            continue
        stake_col = _detect_stake_col(df.columns.tolist())
        if stake_col is None:
            continue
        df["bet_date"] = pd.to_datetime(df["asof_time"], errors="coerce").dt.date
        test_start = pd.to_datetime(w.name.split("_")[1], errors="coerce").date()
        test_end = pd.to_datetime(w.name.split("_")[2], errors="coerce").date()
        df = df[(df["bet_date"] >= test_start) & (df["bet_date"] <= test_end)]
        daily = df.groupby("bet_date")[stake_col].count()
        days_total = int((test_end - test_start).days + 1)
        rows.append(
            {
                "days_total": days_total,
                "days_with_bet": int((daily > 0).sum()),
                "days_bets_ge_1": int((daily >= 1).sum()),
                "days_bets_ge_5": int((daily >= 5).sum()),
                "n_bets": int(len(df)),
            }
        )
    if not rows:
        return {
            "frac_days_any_bet": None,
            "frac_days_bets_ge_1": None,
            "frac_days_bets_ge_5": None,
            "median_bets_per_day": None,
        }
    dfr = pd.DataFrame(rows)
    return {
        "frac_days_any_bet": float((dfr["days_with_bet"] / dfr["days_total"]).median()),
        "frac_days_bets_ge_1": float((dfr["days_bets_ge_1"] / dfr["days_total"]).median()),
        "frac_days_bets_ge_5": float((dfr["days_bets_ge_5"] / dfr["days_total"]).median()),
        "median_bets_per_day": float((dfr["n_bets"] / dfr["days_total"]).median()),
    }


def _hash_keys(keys: list[str]) -> str:
    joined = "|".join(sorted(keys))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _hash_rows(df: pd.DataFrame, key_cols: list[str], stake_col: str, odds_col: str) -> str:
    key = df[key_cols].astype(str).agg(lambda r: "|".join(r.values), axis=1)
    stake = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0).astype(float)
    odds = pd.to_numeric(df[odds_col], errors="coerce").fillna(0.0).astype(float)
    joined = (key + "|" + stake.map(lambda v: f"{v:.1f}") + "|" + odds.map(lambda v: f"{v:.6f}")).tolist()
    return hashlib.sha1("\n".join(sorted(joined)).encode("utf-8")).hexdigest()


def _binding_added_removed(
    base_dir: Path,
    var_dir: Path,
    variant: str,
    year: int,
    split: str,
    expected_windows: list[str],
) -> pd.DataFrame:
    rows = []
    for wname in expected_windows:
        w = base_dir / wname
        base_bets = w / "bets.csv"
        var_bets = var_dir / wname / "bets.csv"
        if not base_bets.exists():
            rows.append(
                {
                    "year": year,
                    "split": split,
                    "variant": variant,
                    "window": wname,
                    "base_n_bets": None,
                    "var_n_bets": None,
                    "filtered_bets": None,
                    "added_bets": None,
                    "common_bets": None,
                    "filtered_stake": None,
                    "added_stake": None,
                    "common_stake": None,
                    "filtered_profit": None,
                    "added_profit": None,
                    "common_profit": None,
                    "identical_passset": None,
                    "pass_set_hash": None,
                    "missing_base": True,
                    "missing_var": bool(var_bets.exists()),
                }
            )
            continue
        bdf = pd.read_csv(base_bets)
        if var_bets.exists():
            vdf = pd.read_csv(var_bets)
            missing_var = False
        else:
            vdf = pd.DataFrame(columns=bdf.columns)
            missing_var = True
        key_cols = _detect_key_cols(bdf.columns.tolist())
        if key_cols is None or any(c not in vdf.columns for c in key_cols):
            continue
        stake_col = _detect_stake_col(bdf.columns.tolist())
        ret_col = _detect_return_col(bdf.columns.tolist())
        prof_col = _detect_profit_col(bdf.columns.tolist())
        if stake_col is None:
            continue
        bdf["key"] = bdf[key_cols].astype(str).agg(lambda r: "|".join(r.values), axis=1)
        b_keys = set(bdf["key"].tolist())
        if vdf.empty:
            v_keys = set()
            v_stake = {}
            v_profit_map = {}
        else:
            vdf["key"] = vdf[key_cols].astype(str).agg(lambda r: "|".join(r.values), axis=1)
            v_keys = set(vdf["key"].tolist())
            v_stake = vdf.set_index("key")[stake_col].to_dict() if stake_col in vdf.columns else {}
            v_profit = _compute_profit(vdf, stake_col, ret_col, prof_col) if stake_col in vdf.columns else pd.Series([])
            vdf["__profit__"] = v_profit
            v_profit_map = vdf.set_index("key")["__profit__"].to_dict()
        filtered = b_keys - v_keys
        added = v_keys - b_keys
        common = b_keys & v_keys
        b_profit = _compute_profit(bdf, stake_col, ret_col, prof_col)
        bdf["__profit__"] = b_profit
        b_stake = bdf.set_index("key")[stake_col].to_dict()
        b_profit_map = bdf.set_index("key")["__profit__"].to_dict()
        rows.append(
            {
                "year": year,
                "split": split,
                "variant": variant,
                "window": w.name,
                "base_n_bets": int(len(b_keys)),
                "var_n_bets": int(len(v_keys)),
                "filtered_bets": int(len(filtered)),
                "added_bets": int(len(added)),
                "common_bets": int(len(common)),
                "filtered_stake": float(sum(b_stake.get(k, 0.0) for k in filtered)),
                "added_stake": float(sum(v_stake.get(k, 0.0) for k in added)),
                "common_stake": float(sum(b_stake.get(k, 0.0) for k in common)),
                "filtered_profit": float(sum(b_profit_map.get(k, 0.0) for k in filtered)),
                "added_profit": float(sum(v_profit_map.get(k, 0.0) for k in added)),
                "common_profit": float(sum(b_profit_map.get(k, 0.0) for k in common)),
                "identical_passset": bool(len(filtered) == 0 and len(added) == 0),
                "pass_set_hash": _hash_keys(list(v_keys)),
                "missing_base": False,
                "missing_var": missing_var,
            }
        )
    return pd.DataFrame(rows)


def _step14_daily_from_run(run_dir: Path, year: int, strategy: str, split: str) -> list[dict[str, Any]]:
    rows = []
    for w in sorted(run_dir.glob("w*/")):
        test_start, test_end = _parse_window_dates(w.name)
        if not test_start or not test_end:
            continue
        seg_start = pd.to_datetime(test_start).date()
        seg_end = seg_start + pd.Timedelta(days=13)
        te = pd.to_datetime(test_end).date()
        if seg_end > te:
            seg_end = te
        bets_path = w / "bets.csv"
        if not bets_path.exists():
            continue
        df = pd.read_csv(bets_path)
        if df.empty or "asof_time" not in df.columns:
            days = pd.date_range(seg_start, seg_end, freq="D")
            for d in days:
                rows.append(
                    {
                        "year": year,
                        "strategy": strategy,
                        "split": split,
                        "date": d.date(),
                        "stake": 0.0,
                        "profit": 0.0,
                        "n_bets": 0,
                    }
                )
            continue
        stake_col = _detect_stake_col(df.columns.tolist())
        if stake_col is None:
            continue
        ret_col = _detect_return_col(df.columns.tolist())
        prof_col = _detect_profit_col(df.columns.tolist())
        df["bet_date"] = pd.to_datetime(df["asof_time"], errors="coerce").dt.date
        df = df[(df["bet_date"] >= seg_start) & (df["bet_date"] <= seg_end)]
        profit = _compute_profit(df, stake_col, ret_col, prof_col)
        df["__profit__"] = profit
        daily = (
            df.groupby("bet_date", dropna=False)
            .agg(stake=(stake_col, "sum"), profit=("__profit__", "sum"), n_bets=(stake_col, "size"))
            .reset_index()
        )
        days = pd.date_range(seg_start, seg_end, freq="D")
        base = pd.DataFrame({"bet_date": [d.date() for d in days]})
        daily = base.merge(daily, on="bet_date", how="left").fillna({"stake": 0.0, "profit": 0.0, "n_bets": 0})
        for _, r in daily.iterrows():
            rows.append(
                {
                    "year": year,
                    "strategy": strategy,
                    "split": split,
                    "date": r["bet_date"],
                    "stake": float(r["stake"]),
                    "profit": float(r["profit"]),
                    "n_bets": int(r["n_bets"]),
                }
            )
    return rows


def _step14_summary(daily_rows: list[dict[str, Any]], bankroll: float) -> pd.DataFrame:
    if not daily_rows:
        return pd.DataFrame()
    df = pd.DataFrame(daily_rows)
    grouped = (
        df.groupby(["year", "strategy", "split", "date"], dropna=False)
        .agg(stake=("stake", "sum"), profit=("profit", "sum"), n_bets=("n_bets", "sum"))
        .reset_index()
    )
    rows = []
    for (year, strategy, split), sub in grouped.groupby(["year", "strategy", "split"], dropna=False):
        sub = sub.sort_values("date").copy()
        sub["cum_profit"] = sub["profit"].cumsum()
        sub["equity"] = bankroll + sub["cum_profit"]
        sub["peak"] = sub["equity"].cummax()
        sub["drawdown_pct"] = (sub["peak"] - sub["equity"]) / sub["peak"]
        total_stake = float(sub["stake"].sum())
        total_profit = float(sub["profit"].sum())
        n_bets = int(sub["n_bets"].sum())
        roi = (total_profit / total_stake) if total_stake > 0 else 0.0
        max_dd = float(sub["drawdown_pct"].max()) if not sub.empty else 0.0
        rows.append(
            {
                "year": int(year),
                "strategy": strategy,
                "split": split,
                "n_bets": n_bets,
                "total_stake": total_stake,
                "total_profit": total_profit,
                "roi": roi,
                "max_drawdown": max_dd,
            }
        )
    return pd.DataFrame(rows)


def _build_virtual_run_dir(
    base_dir: Path,
    low_odds_max: float,
    min_overlay: float,
    out_dir: Path,
    expected_windows: list[str],
) -> tuple[Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bankroll = _read_bankroll(base_dir)
    summary_rows = []
    fallback_used = False
    missing_overlay_rows = 0
    missing_p_hat_rows = 0
    missing_p_mkt_rows = 0
    for wname in expected_windows:
        w = base_dir / wname
        bets_path = w / "bets.csv"
        if not bets_path.exists():
            continue
        df = pd.read_csv(bets_path)
        if df.empty:
            filtered = df.copy()
        else:
            odds_col = _detect_odds_col(df.columns.tolist())
            if odds_col is None:
                continue
            odds = pd.to_numeric(df[odds_col], errors="coerce")
            if low_odds_max <= 0:
                filtered = df.copy()
            else:
                p_hat_col = _detect_p_hat_col(df.columns.tolist())
                p_mkt_col, p_mkt_src = _detect_p_mkt_cols(df.columns.tolist())
                if p_hat_col is None:
                    missing_p_hat_rows += int(len(df))
                    filtered = df[odds >= float(low_odds_max)].copy()
                elif p_mkt_col is None:
                    missing_p_mkt_rows += int(len(df))
                    filtered = df[odds >= float(low_odds_max)].copy()
                else:
                    if p_mkt_src != "p_mkt_race":
                        fallback_used = True
                    p_hat = pd.to_numeric(df[p_hat_col], errors="coerce")
                    p_mkt = pd.to_numeric(df[p_mkt_col], errors="coerce")
                    overlay = p_hat - p_mkt
                    overlay_missing = overlay.isna()
                    missing_overlay_rows += int(overlay_missing.sum())
                    keep_low = overlay >= float(min_overlay)
                    keep = (odds >= float(low_odds_max)) | (keep_low.fillna(False))
                    filtered = df[keep].copy()
        dest_dir = out_dir / w.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(dest_dir / "bets.csv", index=False, encoding="utf-8")
        for copy_name in ["config_used.yaml", "summary.json"]:
            src = w / copy_name
            if src.exists():
                (dest_dir / copy_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        test_start, test_end = _parse_window_dates(w.name)
        metrics = _compute_window_summary(dest_dir / "bets.csv", bankroll=bankroll)
        summary_rows.append(
            {
                "name": w.name,
                "window": w.name,
                "test_start": test_start,
                "test_end": test_end,
                "roi": metrics["roi"],
                "max_drawdown": metrics["max_drawdown"],
                "n_bets": metrics["n_bets"],
                "total_stake": metrics["total_stake"],
                "total_profit": metrics["total_profit"],
            }
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    meta = {
        "fallback_used": fallback_used,
        "missing_overlay_rows": int(missing_overlay_rows),
        "missing_p_hat_rows": int(missing_p_hat_rows),
        "missing_p_mkt_rows": int(missing_p_mkt_rows),
    }
    return out_dir, meta


def _write_debug_samples(out_dir: Path, year: int, variant: str, run_dir: Path) -> None:
    for w in sorted(run_dir.glob("w*/")):
        bets_path = w / "bets.csv"
        if not bets_path.exists():
            continue
        df = pd.read_csv(bets_path)
        if df.empty:
            continue
        dest = out_dir / "debug_samples" / str(year)
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / f"{variant}_{w.name}_head100.csv"
        df.head(100).to_csv(out_path, index=False, encoding="utf-8")
        break


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _load_odds_band_recap(src_dir: Optional[Path]) -> list[str]:
    if src_dir is None:
        return []
    src_dir = Path(src_dir)
    if not src_dir.exists():
        return []
    bins = {"1-2", "2-3", "3-5", "5-10"}
    lines: list[str] = []
    for year in (2024, 2025):
        p = src_dir / f"odds_band_contrib_{year}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["odds_bin"].isin(bins)].copy()
        has_profit_contrib = "profit_contrib" in df.columns
        has_profit_share = "profit_share" in df.columns
        for _, r in df.iterrows():
            profit_txt = ""
            if has_profit_contrib:
                profit_txt = f" profit_contrib={r['profit_contrib']:.3f}"
            elif has_profit_share:
                profit_txt = f" profit_contrib={r['profit_share']:.3f}"
            lines.append(
                f"  - {year} bin={r['odds_bin']} roi={r['roi']:.4f} stake_share={r['stake_share']:.3f}"
                f"{profit_txt} stake={r['stake_sum']:.1f} n_bets={int(r['n_bets'])}"
            )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_2024", required=True)
    ap.add_argument("--base_2025", required=True)
    ap.add_argument("--design_2024")
    ap.add_argument("--design_2025")
    ap.add_argument("--low_odds_max_list", required=True, help="comma-separated list, e.g., 0.0,4.0,5.0")
    ap.add_argument("--min_overlay_list", required=True, help="comma-separated list, e.g., 0.000,0.005,0.010")
    ap.add_argument("--out_ascii", required=True)
    ap.add_argument("--out_jp", required=True)
    ap.add_argument("--odds-band-dir", default=None, help="optional odds_band_analysis directory")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    out_ascii = Path(args.out_ascii)
    out_ascii.mkdir(parents=True, exist_ok=True)
    out_jp = Path(args.out_jp)

    low_odds_max_list = [float(x.strip()) for x in args.low_odds_max_list.split(",") if x.strip()]
    min_overlay_list = [float(x.strip()) for x in args.min_overlay_list.split(",") if x.strip()]
    variants: list[str] = []
    grid: list[tuple[str, float, float]] = []
    for low in low_odds_max_list:
        overlays = [0.0] if low <= 0 else min_overlay_list
        for ov in overlays:
            v = _variant_name(low, ov)
            variants.append(v)
            grid.append((v, low, ov))

    runs: dict[int, dict[str, dict[str, Path]]] = {
        2024: {"eval": {"base": Path(args.base_2024)}},
        2025: {"eval": {"base": Path(args.base_2025)}},
    }
    if args.design_2024:
        runs[2024]["design"] = {"base": Path(args.design_2024)}
    if args.design_2025:
        runs[2025]["design"] = {"base": Path(args.design_2025)}

    coverage_rows = []
    coverage_map: dict[tuple[int, str, str], dict[str, Any]] = {}
    expected_map: dict[tuple[int, str], list[str]] = {}
    overlay_meta: dict[tuple[int, str, str], dict[str, Any]] = {}

    for year, splits in runs.items():
        for split, mapping in splits.items():
            expected_windows = _read_expected_windows(mapping["base"])
            expected_map[(year, split)] = expected_windows
            cov_base = _coverage_status(mapping["base"], expected_windows)
            coverage_map[(year, split, "base")] = cov_base
            coverage_rows.append(
                {
                    "year": year,
                    "split": split,
                    "variant": "base",
                    "expected_windows": len(cov_base["expected"]),
                    "used_windows": len(cov_base["used"]),
                    "missing_windows": len(cov_base["missing"]),
                    "empty_windows": len(cov_base["empty"]),
                }
            )
            for v, low, ov in grid:
                out_dir = out_ascii / "virtual_runs" / split / str(year) / v
                run_dir, meta = _build_virtual_run_dir(
                    mapping["base"],
                    low_odds_max=low,
                    min_overlay=ov,
                    out_dir=out_dir,
                    expected_windows=expected_windows,
                )
                mapping[v] = run_dir
                overlay_meta[(year, split, v)] = meta
                cov_var = _coverage_status(run_dir, expected_windows)
                coverage_map[(year, split, v)] = cov_var
                coverage_rows.append(
                    {
                        "year": year,
                        "split": split,
                        "variant": v,
                        "expected_windows": len(cov_var["expected"]),
                        "used_windows": len(cov_var["used"]),
                        "missing_windows": len(cov_var["missing"]),
                        "empty_windows": len(cov_var["empty"]),
                    }
                )

    eval_rows = []
    scarcity_rows = []
    step14_daily_rows: list[dict[str, Any]] = []
    binding_rows = []

    for year, splits in runs.items():
        for split, mapping in splits.items():
            base_df = _read_summary(mapping["base"])
            if base_df.empty:
                continue
            _ = _read_bankroll(mapping["base"])
            step14_daily_rows.extend(_step14_daily_from_run(mapping["base"], year, "base", split))

            for variant in variants:
                var_df = _read_summary(mapping[variant])
                paired = _merge_base_var(base_df, var_df)
                metrics = _gate_metrics(paired)
                step14_daily_rows.extend(_step14_daily_from_run(mapping[variant], year, variant, split))
                eval_rows.append(
                    {
                        "year": year,
                        "split": split,
                        "variant": variant,
                        **metrics,
                        "step14_roi_base": None,
                        "step14_roi_var": None,
                    }
                )

                scarcity_rows.append(
                    {"year": year, "split": split, "variant": variant, "strategy": "base", **_candidate_scarcity(mapping["base"])}
                )
                scarcity_rows.append(
                    {"year": year, "split": split, "variant": variant, "strategy": "var", **_candidate_scarcity(mapping[variant])}
                )

                if split == "eval":
                    expected_windows = expected_map.get((year, split), [])
                    binding = _binding_added_removed(
                        mapping["base"], mapping[variant], variant, year, split, expected_windows=expected_windows
                    )
                    if not binding.empty:
                        binding_rows.append(binding)

                _write_debug_samples(out_ascii, year, variant, mapping[variant])

            _write_debug_samples(out_ascii, year, "base", mapping["base"])

    bankroll = _read_bankroll(runs[2024]["eval"]["base"])
    step14_summary = _step14_summary(step14_daily_rows, bankroll=bankroll)
    step14_rows = []
    if not step14_summary.empty:
        for _, row in step14_summary.iterrows():
            step14_rows.append(
                {
                    "year": int(row["year"]),
                    "variant": row["strategy"],
                    "split": row["split"],
                    "n_bets": row["n_bets"],
                    "total_stake": row["total_stake"],
                    "total_profit": row["total_profit"],
                    "roi": row["roi"],
                    "max_drawdown": row["max_drawdown"],
                }
            )
    step14_df = pd.DataFrame(step14_rows)

    eval_df = pd.DataFrame(eval_rows)
    if not eval_df.empty and not step14_df.empty:
        for idx, r in eval_df.iterrows():
            y = r["year"]
            split = r["split"]
            base_row = step14_df[
                (step14_df["year"] == y) & (step14_df["variant"] == "base") & (step14_df["split"] == split)
            ]
            var_row = step14_df[
                (step14_df["year"] == y) & (step14_df["variant"] == r["variant"]) & (step14_df["split"] == split)
            ]
            eval_df.loc[idx, "step14_roi_base"] = base_row.iloc[0]["roi"] if not base_row.empty else None
            eval_df.loc[idx, "step14_roi_var"] = var_row.iloc[0]["roi"] if not var_row.empty else None
            eval_df.loc[idx, "step14_profit_base"] = base_row.iloc[0]["total_profit"] if not base_row.empty else None
            eval_df.loc[idx, "step14_profit_var"] = var_row.iloc[0]["total_profit"] if not var_row.empty else None
            eval_df.loc[idx, "step14_stake_base"] = base_row.iloc[0]["total_stake"] if not base_row.empty else None
            eval_df.loc[idx, "step14_stake_var"] = var_row.iloc[0]["total_stake"] if not var_row.empty else None
        eval_df.to_csv(out_ascii / "lowodds_overlay_sweep_summary.csv", index=False, encoding="utf-8")
        eval_df.to_csv(out_ascii / "threshold_sweep_summary.csv", index=False, encoding="utf-8")
        eval_df.to_csv(out_ascii / "eval_gate_summary.csv", index=False, encoding="utf-8")
    elif not eval_df.empty:
        eval_df.to_csv(out_ascii / "lowodds_overlay_sweep_summary.csv", index=False, encoding="utf-8")
        eval_df.to_csv(out_ascii / "threshold_sweep_summary.csv", index=False, encoding="utf-8")
        eval_df.to_csv(out_ascii / "eval_gate_summary.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "lowodds_overlay_sweep_summary.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(out_ascii / "threshold_sweep_summary.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(out_ascii / "eval_gate_summary.csv", index=False, encoding="utf-8")

    scarcity_df = pd.DataFrame(scarcity_rows)
    if not scarcity_df.empty:
        scarcity_df.to_csv(out_ascii / "candidate_scarcity.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "candidate_scarcity.csv", index=False, encoding="utf-8")

    if not step14_df.empty:
        step14_df.to_csv(out_ascii / "walkforward_step14_summary.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "walkforward_step14_summary.csv", index=False, encoding="utf-8")

    added_removed = pd.concat(binding_rows, ignore_index=True) if binding_rows else pd.DataFrame()
    discrete_thresholds = 0
    if not added_removed.empty:
        added_removed["pass_set_discrete"] = False
        for year in added_removed["year"].unique():
            for win in added_removed[added_removed["year"] == year]["window"].unique():
                sub = added_removed[(added_removed["year"] == year) & (added_removed["window"] == win)]
                sub = sub[sub["pass_set_hash"].notna()]
                if sub.empty:
                    continue
                ordered = sub.copy()
                parsed = ordered["variant"].apply(_parse_variant)
                ordered["low_odds_max"] = parsed.apply(lambda v: v[0] if isinstance(v, tuple) else None)
                ordered["min_overlay"] = parsed.apply(lambda v: v[1] if isinstance(v, tuple) else None)
                ordered = ordered.sort_values(["low_odds_max", "min_overlay"])
                hashes = ordered["pass_set_hash"].tolist()
                for i in range(1, len(hashes)):
                    if hashes[i] == hashes[i - 1]:
                        idxs = ordered.iloc[[i - 1, i]].index
                        added_removed.loc[idxs, "pass_set_discrete"] = True
                        discrete_thresholds += 1
        added_removed.to_csv(out_ascii / "binding_diagnosis.csv", index=False, encoding="utf-8")
        added_removed.to_csv(out_ascii / "added_removed_by_window.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "binding_diagnosis.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(out_ascii / "added_removed_by_window.csv", index=False, encoding="utf-8")

    coverage_df = pd.DataFrame(coverage_rows)
    if not coverage_df.empty:
        coverage_df.to_csv(out_ascii / "coverage_summary.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "coverage_summary.csv", index=False, encoding="utf-8")

    noop_variant = None
    for v, low, ov in grid:
        if low <= 0:
            noop_variant = v
            break
    noop_audit: dict[int, dict[str, Any]] = {}
    if noop_variant:
        for year in (2024, 2025):
            expected_windows = expected_map.get((year, "eval"), [])
            rowcount_ok = True
            hash_ok = True
            missing = []
            for wname in expected_windows:
                base_path = runs[year]["eval"]["base"] / wname / "bets.csv"
                var_path = runs[year]["eval"][noop_variant] / wname / "bets.csv"
                if not base_path.exists() or not var_path.exists():
                    missing.append(wname)
                    rowcount_ok = False
                    hash_ok = False
                    continue
                bdf = pd.read_csv(base_path)
                vdf = pd.read_csv(var_path)
                key_cols = _detect_key_cols(bdf.columns.tolist())
                stake_col = _detect_stake_col(bdf.columns.tolist())
                odds_col = _detect_odds_col(bdf.columns.tolist())
                if key_cols is None or stake_col is None or odds_col is None:
                    rowcount_ok = False
                    hash_ok = False
                    continue
                if len(bdf) != len(vdf):
                    rowcount_ok = False
                b_hash = _hash_rows(bdf, key_cols, stake_col, odds_col) if not bdf.empty else ""
                v_hash = _hash_rows(vdf, key_cols, stake_col, odds_col) if not vdf.empty else ""
                if b_hash != v_hash:
                    hash_ok = False
            noop_audit[year] = {
                "variant": noop_variant,
                "rowcount_match": rowcount_ok,
                "hash_match": hash_ok,
                "missing_windows": missing,
            }

    report_lines = ["# coverage status", ""]
    for year in (2024, 2025):
        cov_base = coverage_map.get((year, "eval", "base"))
        if not cov_base:
            continue
        report_lines.append(f"## {year} eval coverage")
        report_lines.append(
            f"- expected={len(cov_base['expected'])} used={len(cov_base['used'])} missing={len(cov_base['missing'])}"
        )
        if cov_base["missing"]:
            report_lines.append(f"- missing_windows={','.join(cov_base['missing'])}")
        if cov_base["empty"]:
            report_lines.append(f"- empty_windows={','.join(cov_base['empty'])}")

        if noop_variant and year in noop_audit:
            audit = noop_audit[year]
            report_lines.append(
                f"- no-op audit ({audit['variant']}): rowcount_match={audit['rowcount_match']} hash_match={audit['hash_match']}"
            )
            if audit["missing_windows"]:
                report_lines.append(f"  noop_missing_windows={','.join(audit['missing_windows'])}")

        for variant in variants:
            cov_var = coverage_map.get((year, "eval", variant))
            if not cov_var:
                continue
            report_lines.append(
                f"- {variant} used={len(cov_var['used'])} missing={len(cov_var['missing'])} empty={len(cov_var['empty'])}"
            )
            if cov_var["missing"]:
                report_lines.append(f"  missing_windows={','.join(cov_var['missing'])}")
            meta = overlay_meta.get((year, "eval", variant))
            if meta:
                report_lines.append(
                    f"  overlay_meta: fallback_used={meta['fallback_used']} "
                    f"missing_p_hat_rows={meta['missing_p_hat_rows']} "
                    f"missing_p_mkt_rows={meta['missing_p_mkt_rows']} "
                    f"missing_overlay_rows={meta['missing_overlay_rows']}"
                )

    if not added_removed.empty:
        report_lines.append("")
        report_lines.append("## binding coverage")
        for year in sorted(added_removed["year"].dropna().unique()):
            sub = added_removed[added_removed["year"] == year]
            used = set(coverage_map.get((int(year), "eval", "base"), {}).get("used", []))
            binding_windows = set(sub["window"].dropna().astype(str).tolist())
            missing_in_binding = sorted(list(used - binding_windows))
            extra_in_binding = sorted(list(binding_windows - used))
            report_lines.append(f"- {int(year)} binding_windows={len(binding_windows)} expected_used={len(used)}")
            if missing_in_binding:
                report_lines.append(f"  missing_in_binding={','.join(missing_in_binding)}")
            if extra_in_binding:
                report_lines.append(f"  extra_in_binding={','.join(extra_in_binding)}")

        added_bug = int((pd.to_numeric(added_removed.get("added_bets"), errors="coerce").fillna(0.0) > 0).sum())
        report_lines.append(f"- added_bets_gt0_windows={added_bug}")
        if added_bug > 0:
            raise SystemExit("added_bets_gt0 detected; tighten-only violation")

    (out_ascii / "report_status.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    cross_rows = []
    if not eval_df.empty:
        for variant in eval_df["variant"].unique():
            sub = eval_df[(eval_df["variant"] == variant) & (eval_df["split"] == "eval")]
            pass_2024 = bool(sub[sub["year"] == 2024]["gate_pass"].iloc[0]) if not sub[sub["year"] == 2024].empty else False
            pass_2025 = bool(sub[sub["year"] == 2025]["gate_pass"].iloc[0]) if not sub[sub["year"] == 2025].empty else False
            decision = "pass_both" if (pass_2024 and pass_2025) else ("pass_single_year" if (pass_2024 or pass_2025) else "fail")
            cross_rows.append({"variant": variant, "pass_2024": pass_2024, "pass_2025": pass_2025, "decision": decision})
        pd.DataFrame(cross_rows).to_csv(out_ascii / "cross_year_summary.csv", index=False, encoding="utf-8")

    chosen_setting = None
    chosen_reason = "no_eval_data"
    chosen_variant = None
    if not eval_df.empty:
        eval_only = eval_df[eval_df["split"] == "eval"].copy()
        if not eval_only.empty:
            stats = []
            for variant in eval_only["variant"].unique():
                sub = eval_only[eval_only["variant"] == variant]
                if sub.empty:
                    continue
                row_2024 = sub[sub["year"] == 2024]
                row_2025 = sub[sub["year"] == 2025]
                if row_2024.empty or row_2025.empty:
                    continue
                profit_2024 = float(row_2024["step14_profit_var"].iloc[0])
                profit_2025 = float(row_2025["step14_profit_var"].iloc[0])
                stake_2024 = float(row_2024["step14_stake_var"].iloc[0])
                stake_2025 = float(row_2025["step14_stake_var"].iloc[0])
                gate_2024 = bool(row_2024["gate_pass"].iloc[0])
                gate_2025 = bool(row_2025["gate_pass"].iloc[0])
                cov_2024 = coverage_map.get((2024, "eval", variant), {}).get("missing", [])
                cov_2025 = coverage_map.get((2025, "eval", variant), {}).get("missing", [])
                coverage_ok = (len(cov_2024) == 0) and (len(cov_2025) == 0)
                stats.append(
                    {
                        "variant": variant,
                        "profit_2024": profit_2024,
                        "profit_2025": profit_2025,
                        "stake_2024": stake_2024,
                        "stake_2025": stake_2025,
                        "gate_2024": gate_2024,
                        "gate_2025": gate_2025,
                        "coverage_ok": coverage_ok,
                        "total_profit": profit_2024 + profit_2025,
                        "total_stake": stake_2024 + stake_2025,
                    }
                )
            stats_df = pd.DataFrame(stats)
            if not stats_df.empty:
                pass_both = stats_df[
                    (stats_df["coverage_ok"])
                    & (stats_df["gate_2024"])
                    & (stats_df["gate_2025"])
                    & (stats_df["profit_2024"] > 0)
                    & (stats_df["profit_2025"] > 0)
                ]
                if pass_both.empty:
                    chosen_reason = "no_pass_both"
                else:
                    candidate = pass_both.sort_values("total_profit", ascending=False).iloc[0]
                    chosen_reason = "pass_both_gate_and_step14_profit_positive"
                    chosen_variant = candidate["variant"]
                    low, ov = _parse_variant(chosen_variant)
                    chosen_setting = (
                        f"low_odds_max={low:.1f}, min_overlay={ov:.3f}" if low is not None and ov is not None else None
                    )

    lines = ["# q95 low-odds overlay sweep (eval-only)", ""]
    lines.append("Pre-flight summary: AGENTS constraints applied; replay-from-base only; no retraining.")
    lines.append("Pooled aggregates overlap across rolling windows; step14 is non-overlap.")
    lines.append("Overlap can flip ROI sign; step14 is the decision signal when signs differ.")
    lines.append("")
    lines.append("- NOTE: replay_from_base=true; stake rescaling is not applied for post-filtered bets.")
    lines.append("- Filter definition: keep bet if odds >= low_odds_max; else require overlay >= min_overlay.")
    lines.append("- overlay = p_hat - p_mkt_race (fallback to p_mkt if p_mkt_race missing).")
    if args.design_2024 or args.design_2025:
        lines.append("- Design split included where provided; eval is used for decision.")
    else:
        lines.append("- Design split not provided; eval-only decision.")
    lines.append("")
    lines.append("## Coverage")
    for year in (2024, 2025):
        cov_base = coverage_map.get((year, "eval", "base"))
        if not cov_base:
            continue
        lines.append(
            f"- {year} eval expected={len(cov_base['expected'])} used={len(cov_base['used'])} missing={len(cov_base['missing'])}"
        )
        if cov_base["missing"]:
            lines.append(f"  missing_windows={','.join(cov_base['missing'])}")
    if not added_removed.empty:
        added_bug = int((pd.to_numeric(added_removed.get("added_bets"), errors="coerce").fillna(0.0) > 0).sum())
        lines.append(f"- added_bets_gt0_windows={added_bug}")
    lines.append("")
    lines.append("## Eval summary (pooled + step14)")
    if not eval_df.empty:
        for _, r in eval_df[eval_df["split"] == "eval"].iterrows():
            base_sm = sign_mismatch(r.get("pooled_roi_base"), r.get("step14_roi_base"))
            var_sm = sign_mismatch(r.get("pooled_roi_var"), r.get("step14_roi_var"))
            lines.append(
                f"- {int(r['year'])} {r['variant']}: gate_pass={bool(r['gate_pass'])} | median_d_roi={r['median_d_roi']:.6f} | "
                f"improve_rate={r['improve_rate']:.3f} | median_d_maxdd={r['median_d_maxdd']:.6f} | "
                f"median_n_bets_var={r['median_n_bets_var']:.1f}"
            )
            lines.append(
                f"  pooled base ROI={r.get('pooled_roi_base')} stake={r.get('pooled_stake_base')} "
                f"| step14 ROI={r.get('step14_roi_base')} | {'SIGN_MISMATCH' if base_sm else 'sign_match'}"
            )
            lines.append(
                f"  pooled var ROI={r.get('pooled_roi_var')} stake={r.get('pooled_stake_var')} "
                f"| step14 ROI={r.get('step14_roi_var')} | {'SIGN_MISMATCH' if var_sm else 'sign_match'}"
            )

    lines.append("")
    lines.append("## Candidate selection")
    if chosen_variant:
        lines.append(f"- chosen_setting={chosen_setting} (variant={chosen_variant}, reason={chosen_reason})")
    else:
        lines.append(f"- chosen_setting=None (reason={chosen_reason})")
        lines.append("- walkforward_step14_daily/segments not emitted (no chosen candidate)")
    if not eval_df.empty:
        eval_only = eval_df[eval_df["split"] == "eval"].copy()
        if not eval_only.empty:
            eval_only["total_step14_profit"] = eval_only["step14_profit_var"]
            eval_only["total_step14_stake"] = eval_only["step14_stake_var"]
            total_by_variant = (
                eval_only.groupby("variant", dropna=False)
                .agg(
                    total_step14_profit=("total_step14_profit", "sum"),
                    total_step14_stake=("total_step14_stake", "sum"),
                )
                .reset_index()
            )
            if "step14_stake_base" in eval_only.columns:
                base_stake = float(eval_only.drop_duplicates(subset=["year"])["step14_stake_base"].sum())
            else:
                base_stake = None
            lines.append("- Top candidates by total_step14_profit (eval):")
            for _, r in total_by_variant.sort_values("total_step14_profit", ascending=False).head(5).iterrows():
                stake_ratio = (r["total_step14_stake"] / base_stake) if base_stake else None
                stake_ratio_str = f"{stake_ratio:.3f}" if stake_ratio is not None else "N/A"
                lines.append(
                    f"  - {r['variant']}: total_step14_profit={r['total_step14_profit']:.2f} "
                    f"total_step14_stake={r['total_step14_stake']:.1f} stake_ratio_vs_base={stake_ratio_str}"
                )

    if not added_removed.empty:
        ident = int(added_removed["identical_passset"].sum())
        lines.append("")
        lines.append("## Added/removed (eval)")
        lines.append("- Replay removes bets only; no added bets are introduced. Window-level removals are in added_removed_by_window.csv.")
        lines.append(f"- identical_passset_windows={ident}")
        lines.append(f"- discrete_thresholds_detected={bool(discrete_thresholds > 0)}")

    lines.append("")
    lines.append("## Odds-band recap (from prior q95 bundle)")
    lines.append("- Source: odds_band_contrib_2024/2025.csv in takeoutq_q95_odds_cap_followup_2024_2025_20260123_214504.")
    odds_band_dir = Path(args.odds_band_dir) if args.odds_band_dir else None
    if odds_band_dir and not odds_band_dir.is_absolute():
        odds_band_dir = project_root / odds_band_dir
    recap = _load_odds_band_recap(odds_band_dir)
    if recap:
        lines.extend(recap)
    else:
        lines.append("- (missing odds_band_contrib CSVs)")

    lines.append("")
    lines.append(roi_footer())
    (out_ascii / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if chosen_variant and not step14_summary.empty:
        chosen_daily = [r for r in step14_daily_rows if r["strategy"] in {"base", chosen_variant} and r["split"] == "eval"]
        pd.DataFrame(chosen_daily).to_csv(out_ascii / "walkforward_step14_daily.csv", index=False, encoding="utf-8")
        if not step14_df.empty:
            seg = step14_df[(step14_df["variant"].isin({"base", chosen_variant})) & (step14_df["split"] == "eval")]
            seg.to_csv(out_ascii / "walkforward_step14_segments.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "walkforward_step14_daily.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(out_ascii / "walkforward_step14_segments.csv", index=False, encoding="utf-8")

    agents_src = Path(__file__).resolve().parents[2] / "AGENTS.md"
    if agents_src.exists():
        shutil.copy2(agents_src, out_ascii / "AGENTS.md")

    try:
        _copy_tree(out_ascii, out_jp)
    except Exception as exc:
        print(f"[warn] failed to copy to JP path: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
