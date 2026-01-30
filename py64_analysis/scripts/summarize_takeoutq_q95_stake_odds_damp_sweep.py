"""Summarize q95 stake odds damp sweep (replay-from-base)."""
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
KEY_COL_CANDIDATES = [
    ["race_id", "horse_no", "asof_time"],
    ["race_id", "horse_id", "asof_time"],
    ["race_id", "horse_no"],
    ["race_id", "horse_id"],
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


def _variant_name(ref_odds: float, power: float, min_mult: float) -> str:
    ref_s = f"{ref_odds:.1f}".replace(".", "p")
    pw_s = f"{power:.1f}".replace(".", "p")
    mm_s = f"{min_mult:.2f}".replace(".", "p")
    return f"refodds{ref_s}_pow{pw_s}_minmult{mm_s}"


def _parse_variant(variant: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not variant.startswith("refodds"):
        return None, None, None
    try:
        body = variant.replace("refodds", "")
        ref_part, rest = body.split("_pow")
        pow_part, mm_part = rest.split("_minmult")
        ref = float(ref_part.replace("p", "."))
        power = float(pow_part.replace("p", "."))
        mm = float(mm_part.replace("p", "."))
        return ref, power, mm
    except Exception:
        return None, None, None


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


def _read_min_yen(run_dir: Path) -> int:
    for w in sorted(run_dir.glob("w*/")):
        cfg = w / "config_used.yaml"
        if not cfg.exists():
            continue
        try:
            import yaml

            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            stake_cfg = (data.get("betting") or {}).get("stake") or {}
            return int(stake_cfg.get("min_yen", 100))
        except Exception:
            continue
    return 100


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


def _odds_band_contrib(
    run_dir: Path,
    *,
    year: int,
    variant: str,
    expected_windows: list[str],
    odds_bins: list[float],
    odds_labels: list[str],
) -> pd.DataFrame:
    rows = []
    for wname in expected_windows:
        bets_path = run_dir / wname / "bets.csv"
        if not bets_path.exists():
            continue
        df = pd.read_csv(bets_path)
        if df.empty:
            continue
        odds_col = _detect_odds_col(df.columns.tolist())
        stake_col = _detect_stake_col(df.columns.tolist())
        if odds_col is None or stake_col is None:
            continue
        ret_col = _detect_return_col(df.columns.tolist())
        prof_col = _detect_profit_col(df.columns.tolist())
        profit = _compute_profit(df, stake_col, ret_col, prof_col)
        df["__profit__"] = profit
        df["__odds__"] = pd.to_numeric(df[odds_col], errors="coerce").fillna(0.0)
        df["__stake__"] = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
        df["odds_bin"] = pd.cut(df["__odds__"], bins=odds_bins, labels=odds_labels, right=True, include_lowest=True)
        grouped = (
            df.groupby("odds_bin", dropna=False)
            .agg(stake_sum=("__stake__", "sum"), profit_sum=("__profit__", "sum"), n_bets=("__stake__", "size"))
            .reset_index()
        )
        for _, r in grouped.iterrows():
            rows.append(
                {
                    "year": year,
                    "variant": variant,
                    "odds_bin": str(r["odds_bin"]),
                    "stake_sum": float(r["stake_sum"]),
                    "profit_sum": float(r["profit_sum"]),
                    "n_bets": int(r["n_bets"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    total_stake = float(out["stake_sum"].sum())
    total_profit = float(out["profit_sum"].sum())
    out["roi"] = out["profit_sum"] / out["stake_sum"].replace({0.0: np.nan})
    out["stake_share"] = out["stake_sum"] / total_stake if total_stake else np.nan
    out["profit_contrib"] = out["profit_sum"] / total_profit if total_profit else np.nan
    return out


def _hash_keys(keys: list[str]) -> str:
    joined = "|".join(sorted(keys))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _hash_rows(df: pd.DataFrame, key_cols: list[str], stake_col: str, odds_col: str) -> str:
    key = df[key_cols].astype(str).agg(lambda r: "|".join(r.values), axis=1)
    stake = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0).astype(float)
    odds = pd.to_numeric(df[odds_col], errors="coerce").fillna(0.0).astype(float)
    joined = (key + "|" + stake.map(lambda v: f"{v:.1f}") + "|" + odds.map(lambda v: f"{v:.6f}")).tolist()
    return hashlib.sha1("\n".join(sorted(joined)).encode("utf-8")).hexdigest()


def _format_key_examples(df: pd.DataFrame, keys: set[str], key_cols: list[str], limit: int = 5) -> str:
    if not keys or df.empty or "key" not in df.columns:
        return ""
    sample_keys = sorted(list(keys))[:limit]
    sub = df[df["key"].isin(sample_keys)].copy()
    if sub.empty:
        return ""
    preferred = [c for c in ["race_id", "horse_no", "horse_id", "ticket_type", "asof_time"] if c in sub.columns]
    cols = preferred if preferred else key_cols
    sub = sub[cols].astype(str).drop_duplicates().head(limit)
    return "; ".join(["|".join(r.values.tolist()) for _, r in sub.iterrows()])


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
                    "base_rowcount": None,
                    "var_rowcount": None,
                    "rowcount_match": None,
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
                    "filtered_examples": None,
                    "added_examples": None,
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
        # Stake is not part of the bet key; keep key set invariant to stake size or zero.
        if stake_col in key_cols:
            key_cols = [c for c in key_cols if c != stake_col]
        base_rows = int(len(bdf))
        var_rows = int(len(vdf))
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
        filtered_examples = _format_key_examples(bdf, filtered, key_cols)
        added_examples = _format_key_examples(vdf, added, key_cols) if not vdf.empty else ""
        rows.append(
            {
                "year": year,
                "split": split,
                "variant": variant,
                "window": w.name,
                "base_n_bets": int(base_rows),
                "var_n_bets": int(var_rows),
                "base_rowcount": int(base_rows),
                "var_rowcount": int(var_rows),
                "rowcount_match": bool(base_rows == var_rows),
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
                "filtered_examples": filtered_examples,
                "added_examples": added_examples,
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
    ref_odds: float,
    power: float,
    min_mult: float,
    out_dir: Path,
    expected_windows: list[str],
) -> tuple[Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bankroll = _read_bankroll(base_dir)
    min_yen = _read_min_yen(base_dir)
    if not np.isfinite(power) or power <= 0:
        power = 1.0
    summary_rows = []
    mult_sum = 0.0
    mult_n = 0
    low_odds_before = 0.0
    low_odds_after = 0.0
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
            stake_col = _detect_stake_col(df.columns.tolist())
            if odds_col is None or stake_col is None:
                continue
            odds = pd.to_numeric(df[odds_col], errors="coerce").fillna(0.0)
            stake_before = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
            safe_min_mult = float(min_mult)
            if not np.isfinite(safe_min_mult):
                safe_min_mult = 0.0
            safe_min_mult = min(1.0, max(0.0, safe_min_mult))
            unit = int(min_yen) if min_yen and min_yen > 0 else 1
            if unit <= 0:
                unit = 1
            if ref_odds <= 0:
                ratio = pd.Series([np.nan] * len(df), index=df.index)
                mult_raw = pd.Series([1.0] * len(df), index=df.index)
                mult_clamped = mult_raw.copy()
                low_mask = pd.Series([False] * len(df), index=df.index)
            else:
                valid = (odds > 0) & np.isfinite(odds)
                ratio = odds / float(ref_odds)
                ratio = ratio.where(valid, np.nan)
                low_mask = ratio < 1.0
                raw = pd.Series(np.power(ratio, power), index=df.index)
                raw = raw.where(np.isfinite(raw) & (raw > 0), 0.0)
                raw = raw.where(low_mask, 1.0)
                mult_raw = raw.where(valid, 1.0)
                mult_clamped = np.maximum(safe_min_mult, mult_raw)
                mult_clamped = np.minimum(1.0, mult_clamped)
                mult_clamped = pd.Series(mult_clamped, index=df.index)
                fallback = max(float(safe_min_mult), 1e-9)
                mult_clamped = mult_clamped.where(~(valid & low_mask & (mult_clamped <= 0)), fallback)
                mult_clamped = mult_clamped.where(valid, 1.0)

            stake_after = stake_before.copy()
            stake_raw = stake_before * mult_clamped
            stake_floor = (stake_raw // unit) * unit if unit > 0 else stake_raw
            if unit > 0:
                stake_floor = stake_floor.where(stake_before <= 0, np.maximum(unit, stake_floor))
            stake_floor = stake_floor.where(stake_floor <= stake_before, stake_before)
            stake_after = stake_after.where(~low_mask, stake_floor)

            mult_used = stake_after / stake_before.replace({0.0: np.nan})
            mult_used = mult_used.fillna(0.0)
            mult_sum += float(mult_used.sum())
            mult_n += int(len(mult_used))

            if ref_odds > 0:
                low_mask = odds < float(ref_odds)
                low_odds_before += float(stake_before[low_mask].sum())
                low_odds_after += float(stake_after[low_mask].sum())

            filtered = df.copy()
            filtered[stake_col] = stake_after
            filtered["stake_before"] = stake_before
            filtered["stake_mult"] = mult_used
            filtered["stake_after"] = stake_after
            filtered["stake_damp_ref_odds"] = float(ref_odds)
            filtered["stake_damp_power"] = float(power)
            filtered["stake_damp_min_mult"] = float(min_mult)
            filtered["stake_damp_odds"] = odds
            filtered["stake_damp_ratio"] = ratio
            filtered["stake_damp_mult_raw"] = mult_raw
            filtered["stake_damp_mult_clamped"] = mult_clamped
            filtered["stake_damp_floor_unit"] = int(unit)

            ret_cols = [c for c in RETURN_COLS if c in filtered.columns]
            if ret_cols:
                for rc in ret_cols:
                    filtered[rc] = pd.to_numeric(filtered[rc], errors="coerce").fillna(0.0) * mult_used
                prof_col = _detect_profit_col(filtered.columns.tolist())
                if prof_col:
                    filtered[prof_col] = filtered[ret_cols[0]] - filtered[stake_col]
            else:
                for pc in [c for c in PROFIT_COLS if c in filtered.columns]:
                    filtered[pc] = pd.to_numeric(filtered[pc], errors="coerce").fillna(0.0) * mult_used
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
        "mean_mult": float(mult_sum / mult_n) if mult_n else 1.0,
        "low_odds_stake_before": float(low_odds_before),
        "low_odds_stake_after": float(low_odds_after),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_2024", required=True)
    ap.add_argument("--base_2025", required=True)
    ap.add_argument("--design_2024")
    ap.add_argument("--design_2025")
    ap.add_argument("--ref_odds_list", required=True, help="comma-separated list, e.g., 0.0,4.0,5.0")
    ap.add_argument("--power_list", required=True, help="comma-separated list, e.g., 1.0,1.5,2.0")
    ap.add_argument("--min_mult", type=float, default=0.0)
    ap.add_argument("--out_ascii", required=True)
    ap.add_argument("--out_jp", required=True)
    args = ap.parse_args()

    out_ascii = Path(args.out_ascii)
    out_ascii.mkdir(parents=True, exist_ok=True)
    out_jp = Path(args.out_jp)

    ref_odds_list = [float(x.strip()) for x in args.ref_odds_list.split(",") if x.strip()]
    power_list = [float(x.strip()) for x in args.power_list.split(",") if x.strip()]
    min_mult = float(args.min_mult)
    variants: list[str] = []
    grid: list[tuple[str, float, float, float]] = []
    for ref in ref_odds_list:
        powers = [1.0] if ref <= 0 else power_list
        for power in powers:
            if not np.isfinite(power) or power <= 0:
                power = 1.0
            v = _variant_name(ref, power, min_mult)
            variants.append(v)
            grid.append((v, ref, power, min_mult))

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
    damp_meta: dict[tuple[int, str, str], dict[str, Any]] = {}

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
            for v, ref, power, mm in grid:
                out_dir = out_ascii / "virtual_runs" / split / str(year) / v
                run_dir, meta = _build_virtual_run_dir(
                    mapping["base"],
                    ref_odds=ref,
                    power=power,
                    min_mult=mm,
                    out_dir=out_dir,
                    expected_windows=expected_windows,
                )
                mapping[v] = run_dir
                damp_meta[(year, split, v)] = meta
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
        eval_df.to_csv(out_ascii / "stake_odds_damp_sweep_summary.csv", index=False, encoding="utf-8")
        eval_df.to_csv(out_ascii / "threshold_sweep_summary.csv", index=False, encoding="utf-8")
        eval_df.to_csv(out_ascii / "eval_gate_summary.csv", index=False, encoding="utf-8")
    elif not eval_df.empty:
        eval_df.to_csv(out_ascii / "stake_odds_damp_sweep_summary.csv", index=False, encoding="utf-8")
        eval_df.to_csv(out_ascii / "threshold_sweep_summary.csv", index=False, encoding="utf-8")
        eval_df.to_csv(out_ascii / "eval_gate_summary.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "stake_odds_damp_sweep_summary.csv", index=False, encoding="utf-8")
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
                ordered["ref_odds"] = parsed.apply(lambda v: v[0] if isinstance(v, tuple) else None)
                ordered["power"] = parsed.apply(lambda v: v[1] if isinstance(v, tuple) else None)
                ordered["min_mult"] = parsed.apply(lambda v: v[2] if isinstance(v, tuple) else None)
                ordered = ordered.sort_values(["ref_odds", "power", "min_mult"])
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

    odds_df = None
    odds_bins = [0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 1000.0]
    odds_labels = ["0-2", "2-3", "3-5", "5-10", "10-15", "15-20", "20-30", "30-50", "50+"]
    odds_rows = []
    for year in (2024, 2025):
        if "eval" not in runs[year]:
            continue
        expected_windows = expected_map.get((year, "eval"), [])
        for variant in ["base"] + variants:
            df = _odds_band_contrib(
                runs[year]["eval"][variant],
                year=year,
                variant=variant,
                expected_windows=expected_windows,
                odds_bins=odds_bins,
                odds_labels=odds_labels,
            )
            if not df.empty:
                odds_rows.append(df)
    if odds_rows:
        odds_df = pd.concat(odds_rows, ignore_index=True)
        odds_df[odds_df["year"] == 2024].to_csv(out_ascii / "odds_band_contrib_2024.csv", index=False, encoding="utf-8")
        odds_df[odds_df["year"] == 2025].to_csv(out_ascii / "odds_band_contrib_2025.csv", index=False, encoding="utf-8")
        joint = (
            odds_df.groupby(["variant", "odds_bin"], dropna=False)
            .agg(stake_sum=("stake_sum", "sum"), profit_sum=("profit_sum", "sum"), n_bets=("n_bets", "sum"))
            .reset_index()
        )
        total_stake = float(joint["stake_sum"].sum())
        total_profit = float(joint["profit_sum"].sum())
        joint["roi"] = joint["profit_sum"] / joint["stake_sum"].replace({0.0: np.nan})
        joint["stake_share"] = joint["stake_sum"] / total_stake if total_stake else np.nan
        joint["profit_contrib"] = joint["profit_sum"] / total_profit if total_profit else np.nan
        joint.to_csv(out_ascii / "odds_band_contrib_joint.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "odds_band_contrib_2024.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(out_ascii / "odds_band_contrib_2025.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(out_ascii / "odds_band_contrib_joint.csv", index=False, encoding="utf-8")

    noop_variant = None
    for v, ref, power, mm in grid:
        if ref <= 0:
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

    report_lines = [
        "# coverage status",
        "",
        "- pooled aggregates overlap across rolling windows; step14 is non-overlap.",
        "- if pooled and step14 ROI signs differ, step14 is the decision signal.",
        "",
    ]
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
            meta = damp_meta.get((year, "eval", variant))
            if meta:
                report_lines.append(
                    f"  damp_meta: mean_mult={meta['mean_mult']:.4f} "
                    f"low_odds_stake_before={meta['low_odds_stake_before']:.1f} "
                    f"low_odds_stake_after={meta['low_odds_stake_after']:.1f}"
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
        removed_bug = int((pd.to_numeric(added_removed.get("filtered_bets"), errors="coerce").fillna(0.0) > 0).sum())
        rowcount_bug = 0
        if "rowcount_match" in added_removed.columns:
            rowcount_bug = int((added_removed["rowcount_match"] == False).sum())
        report_lines.append(f"- added_bets_gt0_windows={added_bug}")
        report_lines.append(f"- removed_bets_gt0_windows={removed_bug}")
        report_lines.append(f"- rowcount_mismatch_windows={rowcount_bug}")
        if added_bug > 0 or removed_bug > 0 or rowcount_bug > 0:
            mismatch = added_removed.copy()
            mask = (
                (pd.to_numeric(mismatch.get("added_bets"), errors="coerce").fillna(0.0) > 0)
                | (pd.to_numeric(mismatch.get("filtered_bets"), errors="coerce").fillna(0.0) > 0)
            )
            if "rowcount_match" in mismatch.columns:
                mask = mask | (mismatch["rowcount_match"] == False)
            sample = mismatch[mask].head(5)
            for _, r in sample.iterrows():
                report_lines.append(
                    "  - mismatch window={window} base_rows={base_rows} var_rows={var_rows} "
                    "filtered={filtered} added={added} filtered_examples={filtered_examples} added_examples={added_examples}".format(
                        window=r.get("window"),
                        base_rows=r.get("base_rowcount"),
                        var_rows=r.get("var_rowcount"),
                        filtered=r.get("filtered_bets"),
                        added=r.get("added_bets"),
                        filtered_examples=r.get("filtered_examples"),
                        added_examples=r.get("added_examples"),
                    )
                )
            raise SystemExit("added/removed bets detected; stake-only invariant violated")

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
                    ref, power, mm = _parse_variant(chosen_variant)
                    chosen_setting = (
                        f"ref_odds={ref:.1f}, power={power:.1f}, min_mult={mm:.2f}"
                        if ref is not None and power is not None and mm is not None
                        else None
                    )

    lines = ["# q95 stake odds damp sweep (eval-only)", ""]
    lines.append("Pre-flight summary: AGENTS constraints applied; replay-from-base only; no retraining.")
    lines.append("Pooled aggregates overlap across rolling windows; step14 is non-overlap.")
    lines.append("Overlap can flip ROI sign; step14 is the decision signal when signs differ.")
    lines.append("")
    lines.append("- NOTE: replay_from_base=true; bet set unchanged; stake rescaling of caps is not applied.")
    lines.append(
        "- Stake damping: mult = min(1, max(min_mult, (odds / ref_odds) ** power)); "
        "stake_after = floor_to_unit(stake_before * mult), then clamped to <= stake_before."
    )
    lines.append("- min_mult applies as a lower bound on mult when enabled.")
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
            best_variant = None
            sorted_total = total_by_variant.sort_values("total_step14_profit", ascending=False)
            if not sorted_total.empty:
                best_variant = str(sorted_total.iloc[0]["variant"])
            for _, r in sorted_total.head(5).iterrows():
                stake_ratio = (r["total_step14_stake"] / base_stake) if base_stake else None
                stake_ratio_str = f"{stake_ratio:.3f}" if stake_ratio is not None else "N/A"
                lines.append(
                    f"  - {r['variant']}: total_step14_profit={r['total_step14_profit']:.2f} "
                    f"total_step14_stake={r['total_step14_stake']:.1f} stake_ratio_vs_base={stake_ratio_str}"
                )

            if odds_df is not None and best_variant is not None:
                target = chosen_variant or best_variant
                lines.append("")
                lines.append("## Low-odds stake/profit share (<=5x)")
                low_bins = {"0-2", "2-3", "3-5"}
                for year in (2024, 2025):
                    for variant in ["base", target]:
                        sub = odds_df[(odds_df["year"] == year) & (odds_df["variant"] == variant)]
                        if sub.empty:
                            continue
                        total_stake = float(sub["stake_sum"].sum())
                        total_profit = float(sub["profit_sum"].sum())
                        low = sub[sub["odds_bin"].isin(low_bins)]
                        low_stake = float(low["stake_sum"].sum())
                        low_profit = float(low["profit_sum"].sum())
                        stake_share = (low_stake / total_stake) if total_stake else 0.0
                        profit_contrib = (low_profit / total_profit) if total_profit else 0.0
                        lines.append(
                            f"- {year} {variant}: low_odds_stake_share={stake_share:.3f} "
                            f"low_odds_profit_contrib={profit_contrib:.3f}"
                        )

    if not added_removed.empty:
        ident = int(added_removed["identical_passset"].sum())
        rowcount_bug = int((added_removed["rowcount_match"] == False).sum()) if "rowcount_match" in added_removed.columns else 0
        lines.append("")
        lines.append("## Added/removed (eval)")
        lines.append("- Stake-damp replay keeps the bet set unchanged; added/removed should remain zero.")
        lines.append(f"- identical_passset_windows={ident}")
        lines.append(f"- rowcount_mismatch_windows={rowcount_bug}")
        lines.append(f"- discrete_thresholds_detected={bool(discrete_thresholds > 0)}")

    lines.append("")
    lines.append("## Odds-band contributions")
    lines.append("- See odds_band_contrib_2024.csv / odds_band_contrib_2025.csv / odds_band_contrib_joint.csv.")

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
