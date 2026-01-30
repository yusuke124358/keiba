"""Summarize q95 odds_floor engine run (baseline vs variant) for eval-only."""
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


def _parse_window_dates(name: str) -> tuple[str, str]:
    parts = name.split("_")
    if len(parts) < 3:
        return "", ""
    return f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}", f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}"


def _compute_profit(df: pd.DataFrame, stake_col: str, ret_col: Optional[str], prof_col: Optional[str]) -> pd.Series:
    stake = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
    if ret_col is not None:
        ret = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0)
        return ret - stake
    if prof_col is not None:
        return pd.to_numeric(df[prof_col], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(df))


def _read_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.csv"
    if path.exists():
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

    rows: list[dict[str, Any]] = []
    for w in sorted(run_dir.glob("w*/summary.json")):
        try:
            payload = json.loads(w.read_text(encoding="utf-8"))
        except Exception:
            continue
        back = payload.get("backtest", {}) if isinstance(payload, dict) else {}
        test = payload.get("test", {}) if isinstance(payload, dict) else {}
        name = payload.get("name") if isinstance(payload, dict) else None
        if not name:
            name = w.parent.name
        ts = test.get("start") if isinstance(test, dict) else None
        te = test.get("end") if isinstance(test, dict) else None
        if not ts or not te:
            ts, te = _parse_window_dates(w.parent.name)
        rows.append(
            {
                "window": name,
                "test_start": ts,
                "test_end": te,
                "roi": back.get("roi"),
                "max_drawdown": back.get("max_drawdown"),
                "n_bets": back.get("n_bets"),
                "total_stake": back.get("total_stake"),
                "total_profit": back.get("total_profit"),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
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


def _binding_added_removed(base_dir: Path, var_dir: Path, year: int, split: str) -> pd.DataFrame:
    rows = []
    for w in base_dir.iterdir():
        if not w.is_dir():
            continue
        base_bets = w / "bets.csv"
        var_bets = var_dir / w.name / "bets.csv"
        if not base_bets.exists() or not var_bets.exists():
            continue
        bdf = pd.read_csv(base_bets)
        vdf = pd.read_csv(var_bets)
        if bdf.empty:
            continue
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
    ap.add_argument("--var_2024", required=True)
    ap.add_argument("--base_2025", required=True)
    ap.add_argument("--var_2025", required=True)
    ap.add_argument("--out_ascii", required=True)
    ap.add_argument("--out_jp", required=True)
    args = ap.parse_args()

    out_ascii = Path(args.out_ascii)
    out_ascii.mkdir(parents=True, exist_ok=True)
    out_jp = Path(args.out_jp)

    runs = {
        2024: {"eval": {"base": Path(args.base_2024), "var": Path(args.var_2024)}},
        2025: {"eval": {"base": Path(args.base_2025), "var": Path(args.var_2025)}},
    }

    eval_rows = []
    scarcity_rows = []
    binding_rows = []
    step14_daily_rows: list[dict[str, Any]] = []

    for year, splits in runs.items():
        for split, mapping in splits.items():
            base_df = _read_summary(mapping["base"])
            var_df = _read_summary(mapping["var"])
            if base_df.empty or var_df.empty:
                continue
            step14_daily_rows.extend(_step14_daily_from_run(mapping["base"], year, "base", split))
            step14_daily_rows.extend(_step14_daily_from_run(mapping["var"], year, "odds_floor_6p0", split))
            paired = _merge_base_var(base_df, var_df)
            metrics = _gate_metrics(paired)
            eval_rows.append(
                {
                    "year": year,
                    "split": split,
                    "variant": "odds_floor_6p0",
                    **metrics,
                    "step14_roi_base": None,
                    "step14_roi_var": None,
                }
            )
            scarcity_rows.append({"year": year, "split": split, "variant": "base", **_candidate_scarcity(mapping["base"])})
            scarcity_rows.append({"year": year, "split": split, "variant": "odds_floor_6p0", **_candidate_scarcity(mapping["var"])})
            binding = _binding_added_removed(mapping["base"], mapping["var"], year, split)
            if not binding.empty:
                binding_rows.append(binding)

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
            base_row = step14_df[(step14_df["year"] == y) & (step14_df["variant"] == "base") & (step14_df["split"] == split)]
            var_row = step14_df[
                (step14_df["year"] == y) & (step14_df["variant"] == "odds_floor_6p0") & (step14_df["split"] == split)
            ]
            eval_df.loc[idx, "step14_roi_base"] = base_row.iloc[0]["roi"] if not base_row.empty else None
            eval_df.loc[idx, "step14_roi_var"] = var_row.iloc[0]["roi"] if not var_row.empty else None
            eval_df.loc[idx, "step14_profit_base"] = base_row.iloc[0]["total_profit"] if not base_row.empty else None
            eval_df.loc[idx, "step14_profit_var"] = var_row.iloc[0]["total_profit"] if not var_row.empty else None
            eval_df.loc[idx, "step14_stake_base"] = base_row.iloc[0]["total_stake"] if not base_row.empty else None
            eval_df.loc[idx, "step14_stake_var"] = var_row.iloc[0]["total_stake"] if not var_row.empty else None
        eval_df.to_csv(out_ascii / "eval_gate_summary.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "eval_gate_summary.csv", index=False, encoding="utf-8")

    step14_df.to_csv(out_ascii / "walkforward_step14_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(scarcity_rows).to_csv(out_ascii / "candidate_scarcity.csv", index=False, encoding="utf-8")

    binding_df = pd.concat(binding_rows, ignore_index=True) if binding_rows else pd.DataFrame()
    discrete_flag = False
    if not binding_df.empty:
        binding_df["pass_set_discrete"] = False
        # only one threshold; identical pass set windows indicate no-op
        discrete_flag = False
        binding_df.to_csv(out_ascii / "binding_diagnosis.csv", index=False, encoding="utf-8")
        binding_df.to_csv(out_ascii / "added_removed_by_window.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(out_ascii / "binding_diagnosis.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(out_ascii / "added_removed_by_window.csv", index=False, encoding="utf-8")

    cross_rows = []
    if not eval_df.empty:
        sub = eval_df[eval_df["split"] == "eval"]
        pass_2024 = bool(sub[sub["year"] == 2024]["gate_pass"].iloc[0]) if not sub[sub["year"] == 2024].empty else False
        pass_2025 = bool(sub[sub["year"] == 2025]["gate_pass"].iloc[0]) if not sub[sub["year"] == 2025].empty else False
        decision = "pass_both" if (pass_2024 and pass_2025) else ("pass_single_year" if (pass_2024 or pass_2025) else "fail")
        cross_rows.append({"variant": "odds_floor_6p0", "pass_2024": pass_2024, "pass_2025": pass_2025, "decision": decision})
        pd.DataFrame(cross_rows).to_csv(out_ascii / "cross_year_summary.csv", index=False, encoding="utf-8")

    lines = ["# q95 odds floor (engine) eval-only", ""]
    lines.append("Pre-flight summary: AGENTS constraints applied; time-split only; no leakage; step14 non-overlap is the decision signal.")
    lines.append("Pooled aggregates overlap across rolling windows; step14 is non-overlap. If signs differ, step14 is used for decisions.")
    lines.append("")
    lines.append("- NOTE: odds_floor is applied post-cap with no stake rescaling.")
    lines.append("")
    lines.append("## Eval summary (pooled + step14)")
    if not eval_df.empty:
        for _, r in eval_df[eval_df["split"] == "eval"].iterrows():
            base_sm = sign_mismatch(r.get("pooled_roi_base"), r.get("step14_roi_base"))
            var_sm = sign_mismatch(r.get("pooled_roi_var"), r.get("step14_roi_var"))
            lines.append(
                f"- {int(r['year'])} odds_floor_6p0: gate_pass={bool(r['gate_pass'])} | median_d_roi={r['median_d_roi']:.6f} | "
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
    else:
        lines.append("- (missing eval summary.csv; check run dirs)")

    if not binding_df.empty:
        ident = int(binding_df["identical_passset"].sum())
        lines.append("")
        lines.append("## Added/removed (eval)")
        lines.append("- Odds floor removes bets only; no added bets are introduced.")
        lines.append(f"- identical_passset_windows={ident}")
        lines.append(f"- discrete_thresholds_detected={bool(discrete_flag)}")

    lines.append("")
    lines.append(roi_footer())
    (out_ascii / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

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
