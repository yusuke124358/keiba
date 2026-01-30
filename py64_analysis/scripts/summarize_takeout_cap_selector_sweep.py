"""
Summarize takeout cap selector sweep across 2024/2025 eval-only rolling runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

WIN_RE = re.compile(r"^w(\d{3})_")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _hash_race_set(race_ids: list[str]) -> str:
    joined = "|".join(sorted(str(r) for r in race_ids))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _window_key_from_summary(summary: dict) -> Optional[str]:
    test = summary.get("test") if isinstance(summary, dict) else None
    if not isinstance(test, dict):
        return None
    start = test.get("start")
    end = test.get("end")
    if not start or not end:
        return None
    return f"{start}__{end}"


def _index_windows_by_period(run_dirs: list[Path]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for run_dir in run_dirs:
        for summary_path in run_dir.glob("w*/summary.json"):
            try:
                summary = _read_json(summary_path)
            except Exception:
                continue
            key = _window_key_from_summary(summary)
            if not key:
                continue
            mapping[key] = summary_path.parent
    return mapping


def _first_valid(series: pd.Series) -> Optional[float]:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[pd.notna(vals)]
    if vals.empty:
        return None
    return float(vals.iloc[0])


def _binding_stats_for_window(
    base_window_dir: Path,
    candidate_caps: list[Any],
    metric: str,
    reject_missing: bool,
) -> list[dict[str, Any]]:
    bets_path = base_window_dir / "bets.csv"
    if not bets_path.exists():
        return []
    df = pd.read_csv(bets_path)
    if df.empty:
        return []

    metric_col = metric if metric in df.columns else None
    if metric_col is None:
        fallback = "takeout_implied" if "takeout_implied" in df.columns else None
        if fallback is None and "overround_sum_inv" in df.columns:
            fallback = "overround_sum_inv"
        metric_col = fallback
    if metric_col is None:
        return []

    df["__metric__"] = pd.to_numeric(df[metric_col], errors="coerce")
    race_group = df.groupby("race_id", dropna=False)
    metric_by_race = race_group["__metric__"].apply(_first_valid)
    race_stats = race_group.agg(
        stake_sum=("stake", "sum"),
        n_bets=("race_id", "size"),
    )
    race_stats["metric_value"] = metric_by_race

    base_races = int(len(race_stats))
    base_bets = int(len(df))
    base_stake = float(pd.to_numeric(df["stake"], errors="coerce").sum())

    rows: list[dict[str, Any]] = []
    for cap in candidate_caps:
        cap_value = None
        if cap is not None:
            try:
                cap_value = float(cap)
            except Exception:
                cap_value = None

        if cap_value is None:
            excluded_mask = pd.Series(False, index=race_stats.index)
        else:
            mvals = race_stats["metric_value"]
            missing = mvals.isna()
            too_high = mvals > float(cap_value)
            excluded_mask = too_high | (missing & bool(reject_missing))

        excluded_races = int(excluded_mask.sum())
        excluded_bets = int(pd.to_numeric(race_stats.loc[excluded_mask, "n_bets"], errors="coerce").sum())
        excluded_stake = float(pd.to_numeric(race_stats.loc[excluded_mask, "stake_sum"], errors="coerce").sum())

        pass_races = [str(r) for r in race_stats.index[~excluded_mask].tolist()]
        pass_hash = _hash_race_set(pass_races)

        rows.append(
            {
                "cap": cap_value,
                "cap_label": "none" if cap_value is None else f"{cap_value:.3f}",
                "metric": metric,
                "reject_if_missing": bool(reject_missing),
                "base_races": base_races,
                "base_bets": base_bets,
                "base_stake": base_stake,
                "excluded_races": excluded_races,
                "excluded_race_rate": (excluded_races / base_races) if base_races else None,
                "excluded_bets": excluded_bets,
                "excluded_bet_rate": (excluded_bets / base_bets) if base_bets else None,
                "excluded_stake": excluded_stake,
                "excluded_stake_rate": (excluded_stake / base_stake) if base_stake else None,
                "pass_set_size": int(len(pass_races)),
                "pass_set_hash": pass_hash,
            }
        )

    group_ids: dict[str, str] = {}
    for row in rows:
        key = row.get("pass_set_hash") or ""
        if key not in group_ids:
            group_ids[key] = f"g{len(group_ids) + 1}"
        row["pass_set_group"] = group_ids[key]

    group_sizes = pd.Series([r.get("pass_set_hash") for r in rows]).value_counts().to_dict()
    for row in rows:
        size = int(group_sizes.get(row.get("pass_set_hash"), 1))
        row["pass_set_group_size"] = size
        row["pass_set_discrete"] = bool(size > 1)

    return rows


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _infer_w_idx_offset(run_dir_name: str) -> int:
    name = run_dir_name.lower()
    if "w013_022" in name or "w013-022" in name:
        return 12
    return 0


def _offset_window_name(val: str, offset: int) -> str:
    m = WIN_RE.match(str(val))
    if not m:
        return str(val)
    idx = int(m.group(1)) + int(offset)
    return f"w{idx:03d}_{str(val)[4:]}"


def _parse_window_idx(val) -> Optional[int]:
    if pd.isna(val):
        return None
    m = re.search(r"w(\d+)", str(val))
    if m:
        return int(m.group(1))
    if str(val).isdigit():
        return int(val)
    return None


def _load_summary(path: Path, offset: int, design_max_idx: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    name_col = _find_col(df, ["name", "window", "window_id", "w"])
    roi_col = _find_col(df, ["roi", "ROI"])
    dd_col = _find_col(df, ["max_drawdown", "max_dd", "maxdd", "MaxDD"])
    nb_col = _find_col(df, ["n_bets", "bets", "nBet"])
    ts_col = _find_col(df, ["test_start"])
    te_col = _find_col(df, ["test_end"])
    stake_col = _find_col(df, ["total_stake", "stake", "stake_sum", "stake_yen"])
    profit_col = _find_col(df, ["total_profit", "profit", "pnl", "net_profit"])

    missing = [k for k, v in {"name": name_col, "roi": roi_col, "max_drawdown": dd_col, "n_bets": nb_col, "test_start": ts_col, "test_end": te_col}.items() if v is None]
    if missing:
        raise ValueError(f"summary.csv missing columns: {missing}. columns={list(df.columns)}")

    out = pd.DataFrame(
        {
            "window_name": df[name_col],
            "test_start": df[ts_col],
            "test_end": df[te_col],
            "roi": pd.to_numeric(df[roi_col], errors="coerce"),
            "max_dd": pd.to_numeric(df[dd_col], errors="coerce"),
            "n_bets": pd.to_numeric(df[nb_col], errors="coerce").fillna(0).astype(int),
        }
    )
    if stake_col is not None:
        out["total_stake"] = pd.to_numeric(df[stake_col], errors="coerce")
    if profit_col is not None:
        out["total_profit"] = pd.to_numeric(df[profit_col], errors="coerce")

    out["window_name"] = out["window_name"].apply(lambda x: _offset_window_name(x, offset))
    out["window_idx"] = out["window_name"].apply(_parse_window_idx)
    out["split"] = out["window_idx"].apply(lambda i: "design" if i is not None and int(i) <= int(design_max_idx) else "eval")
    return out


def _combine_summaries(run_dirs: list[Path], design_max_idx: int) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        offset = _infer_w_idx_offset(run_dir.name)
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            raise SystemExit(f"summary.csv not found: {summary_path}")
        df = _load_summary(summary_path, offset=offset, design_max_idx=design_max_idx)
        df["run_dir"] = str(run_dir)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.sort_values(["window_idx"]).reset_index(drop=True)
    return combined


def _paired_compare(base: pd.DataFrame, var: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(var, on=["test_start", "test_end"], how="inner", suffixes=("_base", "_var"))
    if merged.empty:
        raise RuntimeError("No matched windows for paired compare")
    merged["window_idx"] = merged.get("window_idx_base", merged.get("window_idx_var"))
    merged["split"] = merged.get("split_base", merged.get("split_var"))
    merged["d_roi"] = merged["roi_var"] - merged["roi_base"]
    merged["d_max_dd"] = merged["max_dd_var"] - merged["max_dd_base"]
    merged["d_n_bets"] = merged["n_bets_var"] - merged["n_bets_base"]
    if "total_stake_base" in merged.columns and "total_stake_var" in merged.columns:
        merged["d_total_stake"] = merged["total_stake_var"] - merged["total_stake_base"]
    if "total_profit_base" in merged.columns and "total_profit_var" in merged.columns:
        merged["d_total_profit"] = merged["total_profit_var"] - merged["total_profit_base"]
    return merged.sort_values(["window_idx"]).reset_index(drop=True)


def _eval_gate_metrics(paired: pd.DataFrame) -> dict[str, Any]:
    eval_block = paired[paired["split"] == "eval"].copy()
    if eval_block.empty:
        return {
            "n_windows": 0,
            "improve_rate": None,
            "median_d_roi": None,
            "median_d_maxdd": None,
            "median_d_n_bets": None,
            "median_n_bets_var": None,
            "min_n_bets_var": None,
            "zero_bet_windows": None,
            "pooled_roi_eval": None,
            "pooled_n_bets_eval": None,
            "gate_pass": False,
        }
    improve_rate = float((eval_block["d_roi"] > 0).mean())
    median_d_roi = float(eval_block["d_roi"].median())
    median_d_maxdd = float(eval_block["d_max_dd"].median())
    median_d_n_bets = float(eval_block["d_n_bets"].median())
    median_n_bets_var = float(eval_block["n_bets_var"].median())
    min_n_bets_var = int(eval_block["n_bets_var"].min())
    zero_bet_windows = int((eval_block["n_bets_var"] <= 0).sum())
    stake_sum = pd.to_numeric(eval_block.get("total_stake_var"), errors="coerce").sum()
    profit_sum = pd.to_numeric(eval_block.get("total_profit_var"), errors="coerce").sum()
    pooled_roi = (profit_sum / stake_sum) if stake_sum and pd.notna(stake_sum) else None
    pooled_n_bets = int(pd.to_numeric(eval_block.get("n_bets_var"), errors="coerce").sum())

    gate_pass = (
        improve_rate >= 0.6
        and median_d_roi > 0
        and median_d_maxdd <= 0
        and median_n_bets_var >= 80
        and zero_bet_windows == 0
    )
    return {
        "n_windows": int(len(eval_block)),
        "improve_rate": improve_rate,
        "median_d_roi": median_d_roi,
        "median_d_maxdd": median_d_maxdd,
        "median_d_n_bets": median_d_n_bets,
        "median_n_bets_var": median_n_bets_var,
        "min_n_bets_var": min_n_bets_var,
        "zero_bet_windows": zero_bet_windows,
        "pooled_roi_eval": pooled_roi,
        "pooled_n_bets_eval": pooled_n_bets,
        "gate_pass": gate_pass,
    }


def _year_from_dates(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    val = str(df["test_start"].iloc[0])
    if len(val) >= 4 and val[:4].isdigit():
        return int(val[:4])
    return None


def _aggregate_year_metrics(df: pd.DataFrame) -> dict[str, Any]:
    eval_block = df[df["split"] == "eval"].copy()
    if eval_block.empty:
        return {
            "test_start": None,
            "test_end": None,
            "n_bets": None,
            "total_stake": None,
            "total_profit": None,
            "roi": None,
            "median_max_dd": None,
        }
    test_start = str(eval_block["test_start"].min())
    test_end = str(eval_block["test_end"].max())
    total_stake = pd.to_numeric(eval_block.get("total_stake"), errors="coerce").sum()
    total_profit = pd.to_numeric(eval_block.get("total_profit"), errors="coerce").sum()
    roi = (total_profit / total_stake) if total_stake and pd.notna(total_stake) else None
    n_bets = int(pd.to_numeric(eval_block.get("n_bets"), errors="coerce").sum())
    median_max_dd = float(pd.to_numeric(eval_block.get("max_dd"), errors="coerce").median())
    return {
        "test_start": test_start,
        "test_end": test_end,
        "n_bets": n_bets,
        "total_stake": float(total_stake) if pd.notna(total_stake) else None,
        "total_profit": float(total_profit) if pd.notna(total_profit) else None,
        "roi": float(roi) if roi is not None else None,
        "median_max_dd": median_max_dd,
    }


def _ensure_ev_lift(group_dir: Path, ev_design_max_idx: int) -> Path:
    ev_path = group_dir / "ev_lift_summary.json"
    if ev_path.exists():
        return ev_path
    cmd = [
        sys.executable,
        str(_project_root() / "py64_analysis" / "scripts" / "analyze_rolling_bets.py"),
        "--group-dir",
        str(group_dir),
        "--design-max-idx",
        str(ev_design_max_idx),
        "--ev-lift",
    ]
    subprocess.run(cmd, check=True)
    return ev_path


def _load_ev_lift_metrics(ev_path: Path) -> dict[str, Any]:
    data = _read_json(ev_path)
    eval_block = data.get("eval", {}) if isinstance(data, dict) else {}
    deciles = eval_block.get("deciles", {}) if isinstance(eval_block, dict) else {}
    decile_keys = [int(k) for k in deciles.keys() if str(k).isdigit()]
    decile10 = None
    if 10 in decile_keys:
        decile10 = deciles.get("10", {})
    elif decile_keys:
        decile10 = deciles.get(str(max(decile_keys)), {})
    decile10_median_roi = decile10.get("median_roi") if isinstance(decile10, dict) else None
    monotone = data.get("monotone_check_eval", {}) if isinstance(data, dict) else {}
    signal_rate = monotone.get("signal_rate") if isinstance(monotone, dict) else None
    return {
        "signal_rate": signal_rate,
        "decile10_median_roi": decile10_median_roi,
        "raw": data,
    }


def _date_range(start: str, end: str) -> list[datetime.date]:
    ds = datetime.strptime(start, "%Y-%m-%d").date()
    de = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    cur = ds
    while cur <= de:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _summarize_candidate_scarcity(run_dir: Path) -> Optional[dict[str, Any]]:
    windows = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        if not WIN_RE.match(p.name):
            continue
        if not (p / "summary.json").exists():
            continue
        if not (p / "bets.csv").exists():
            continue
        windows.append(p)
    windows.sort(key=lambda p: p.name)
    if not windows:
        return None

    rows = []
    for w in windows:
        summary = _read_json(w / "summary.json")
        test = summary.get("test") or {}
        test_start = test.get("start")
        test_end = test.get("end")
        if not test_start or not test_end:
            continue
        days = _date_range(test_start, test_end)
        total_days = len(days)
        if total_days <= 0:
            continue
        df = pd.read_csv(w / "bets.csv")
        if "asof_time" in df.columns:
            df["asof_time"] = pd.to_datetime(df["asof_time"], errors="coerce")
            df["asof_date"] = df["asof_time"].dt.date
        else:
            df["asof_date"] = pd.NaT
        days_with_bet = int(df["asof_date"].nunique()) if not df.empty else 0
        n_bets = int(len(df))
        rows.append(
            {
                "test_start": test_start,
                "test_end": test_end,
                "total_days": total_days,
                "days_with_bet": days_with_bet,
                "frac_days_any_bet": days_with_bet / total_days if total_days else None,
                "bets_per_day": n_bets / total_days if total_days else None,
                "n_bets": n_bets,
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    year = int(str(df["test_start"].iloc[0])[:4])
    return {
        "year": year,
        "n_windows": int(len(df)),
        "median_frac_days_any_bet": float(df["frac_days_any_bet"].median()),
        "median_bets_per_day": float(df["bets_per_day"].median()),
        "median_total_days": float(df["total_days"].median()),
        "median_days_with_bet": float(df["days_with_bet"].median()),
        "total_bets": int(df["n_bets"].sum()),
    }


def _selector_decisions(run_dirs: list[Path], design_max_idx: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        offset = _infer_w_idx_offset(run_dir.name)
        for selector_path in sorted(run_dir.glob("w*/artifacts/race_cost_cap_selector.json")):
            window_dir = selector_path.parent.parent
            window_name = _offset_window_name(window_dir.name, offset)
            w_idx = _parse_window_idx(window_name)
            split = "design" if w_idx is not None and int(w_idx) <= int(design_max_idx) else "eval"
            data = _read_json(selector_path)
            base = data.get("baseline", {}) if isinstance(data, dict) else {}
            candidates = data.get("candidates", []) if isinstance(data, dict) else []
            base_valid_bets = base.get("valid_bets")
            candidates_out = []
            eligible_any = False
            for cand in candidates:
                cand_out = dict(cand)
                if cand_out.get("valid_bets_ratio_to_base") is None:
                    try:
                        if base_valid_bets and cand_out.get("valid_bets") is not None:
                            cand_out["valid_bets_ratio_to_base"] = float(cand_out["valid_bets"]) / float(base_valid_bets)
                    except Exception:
                        pass
                if cand_out.get("eligible_reason") is None:
                    cand_out["eligible_reason"] = "eligible" if cand_out.get("eligible") else "not_eligible"
                if cand_out.get("eligible"):
                    eligible_any = True
                candidates_out.append(cand_out)
            rows.append(
                {
                    "window_name": window_name,
                    "window_idx": w_idx,
                    "split": split,
                    "metric": data.get("metric") if isinstance(data, dict) else None,
                    "candidate_set_id": data.get("candidate_set_id") if isinstance(data, dict) else None,
                    "selected_cap": data.get("selected_cap") if isinstance(data, dict) else None,
                    "selected_reason": data.get("selected_reason") if isinstance(data, dict) else None,
                    "selected_from": data.get("selected_from") if isinstance(data, dict) else None,
                    "baseline_valid_roi": base.get("valid_roi"),
                    "baseline_valid_bets": base.get("valid_bets"),
                    "baseline_valid_stake": base.get("valid_stake"),
                    "baseline_valid_profit": base.get("valid_profit"),
                    "candidate_caps": json.dumps(data.get("candidate_caps"), ensure_ascii=False),
                    "candidates_json": json.dumps(candidates_out, ensure_ascii=False),
                    "error": data.get("error") if isinstance(data, dict) else None,
                    "eligible_any": eligible_any,
                }
            )
            table.append(
                {
                    "window_name": window_name,
                    "window_idx": w_idx,
                    "split": split,
                    "baseline": base,
                    "candidate_caps": data.get("candidate_caps") if isinstance(data, dict) else None,
                    "candidates": candidates_out,
                    "thresholds": data.get("thresholds") if isinstance(data, dict) else None,
                }
            )
    return pd.DataFrame(rows), table


def _cap_rate_summary(df: pd.DataFrame, split: str) -> tuple[dict[str, float], float]:
    if df.empty:
        return {}, 0.0
    sub = df[df["split"] == split].copy()
    if sub.empty:
        return {}, 0.0
    sub["selected_cap_label"] = sub["selected_cap"].apply(lambda v: "none" if pd.isna(v) else str(v))
    counts = sub["selected_cap_label"].value_counts(dropna=False)
    total = float(counts.sum()) if not counts.empty else 0.0
    rates = {k: (float(v) / total if total else 0.0) for k, v in counts.items()}
    baseline_rate = rates.get("none", 0.0)
    return rates, baseline_rate


def _decision(pass_2024: bool, pass_2025: bool) -> str:
    if pass_2024 and pass_2025:
        return "pass_both"
    if pass_2024 or pass_2025:
        return "pass_single_year"
    return "fail"


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize takeout cap selector sweep (2024/2025)")
    ap.add_argument("--manifest-2024", type=Path, required=True)
    ap.add_argument("--manifest-2025", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--design-max-idx", type=int, default=12)
    ap.add_argument("--ensure-ev-lift", action="store_true", help="run analyze_rolling_bets if ev_lift_summary.json missing")
    ap.add_argument("--ev-design-max-idx", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "2024": _read_json(args.manifest_2024),
        "2025": _read_json(args.manifest_2025),
    }

    eval_gate_rows = []
    ev_lift_summary = {}
    selector_rate_lines = {}
    binding_rows_all: list[dict[str, Any]] = []
    ood_audit_rows = []
    final_report_lines = []

    final_report_lines.append("# Takeout cap selector sweep (2024/2025)")
    final_report_lines.append("")
    final_report_lines.append("- Split: time-based (train < valid < test), selector uses valid only (no test leakage).")
    final_report_lines.append("")

    for year in ["2024", "2025"]:
        manifest = manifests[year]
        baseline_name = manifest.get("baseline")
        variants = manifest.get("variants", {})
        if baseline_name not in variants:
            raise SystemExit(f"baseline not in manifest {year}: {baseline_name}")
        base_dirs = [Path(p) for p in variants.get(baseline_name, [])]
        selector_dirs = [Path(p) for p in variants.get("cap_selector", [])] or [Path(p) for p in variants.get("selector", [])]
        if not base_dirs or not selector_dirs:
            raise SystemExit(f"missing run dirs in manifest {year}")

        base_df = _combine_summaries(base_dirs, design_max_idx=args.design_max_idx)
        sel_df = _combine_summaries(selector_dirs, design_max_idx=args.design_max_idx)

        paired = _paired_compare(base_df, sel_df)
        gate = _eval_gate_metrics(paired)
        eval_gate_rows.append(
            {
                "year": int(year),
                "variant": "cap_selector",
                **gate,
            }
        )

        base_metrics = _aggregate_year_metrics(base_df)
        sel_metrics = _aggregate_year_metrics(sel_df)

        if args.ensure_ev_lift:
            ev_paths = []
            for run_dir in selector_dirs:
                ev_paths.append(_ensure_ev_lift(run_dir, ev_design_max_idx=int(args.ev_design_max_idx)))
            ev_summary = _load_ev_lift_metrics(ev_paths[-1])
        else:
            ev_path = selector_dirs[-1] / "ev_lift_summary.json"
            ev_summary = _load_ev_lift_metrics(ev_path)
        ev_lift_summary[year] = ev_summary["raw"]

        decisions_df, candidate_table = _selector_decisions(selector_dirs, design_max_idx=args.design_max_idx)
        rates, baseline_rate = _cap_rate_summary(decisions_df, split="eval")
        eval_decisions = decisions_df[decisions_df["split"] == "eval"].copy()
        eligible_windows = int(eval_decisions.get("eligible_any").fillna(False).sum()) if not eval_decisions.empty else 0
        total_windows = int(len(eval_decisions))
        bvb = pd.to_numeric(eval_decisions.get("baseline_valid_bets"), errors="coerce").dropna()
        if bvb.empty:
            baseline_bets_stats = None
        else:
            baseline_bets_stats = {
                "min": float(bvb.min()),
                "median": float(bvb.median()),
                "p90": float(bvb.quantile(0.9)),
                "max": float(bvb.max()),
            }
        selector_rate_lines[year] = {
            "rates": rates,
            "baseline_rate": baseline_rate,
            "eligible_windows": eligible_windows,
            "total_windows": total_windows,
            "baseline_bets_stats": baseline_bets_stats,
        }

        no_op = False
        no_op_reason = None
        if total_windows == 0:
            no_op = True
            no_op_reason = "no_eval_windows"
        elif eligible_windows == 0:
            no_op = True
            no_op_reason = "eligible_windows=0"
        elif len(rates) == 1 and rates.get("none", 0.0) >= 1.0:
            no_op = True
            no_op_reason = "chosen_rate_all_baseline"

        ood_audit_rows.append(
            {
                "year": int(year),
                "fit_min": "N/A (eval-only)",
                "fit_p50": "N/A (eval-only)",
                "fit_p90": "N/A (eval-only)",
                "fit_max": "N/A (eval-only)",
                "test_p90": "N/A (eval-only)",
                "test_max": "N/A (eval-only)",
                "test_max_gt_fit_max": "N/A (eval-only)",
                "note": "N/A (eval-only)",
            }
        )

        final_report_lines.append(f"## {year} eval")
        final_report_lines.append("")
        final_report_lines.append(f"- baseline group_dir: {', '.join(str(p) for p in base_dirs)}")
        final_report_lines.append(f"- selector group_dir: {', '.join(str(p) for p in selector_dirs)}")
        final_report_lines.append("")
        if decisions_df[decisions_df["split"] == "design"].empty:
            final_report_lines.append("- Design metrics: N/A (eval-only)")
            final_report_lines.append("")
        final_report_lines.append("### Test period and headline metrics")
        final_report_lines.append(
            f"- baseline: test={base_metrics['test_start']}..{base_metrics['test_end']} "
            f"roi={base_metrics['roi']} stake={base_metrics['total_stake']} n_bets={base_metrics['n_bets']} "
            f"median_max_dd={base_metrics['median_max_dd']}"
        )
        final_report_lines.append(
            f"- selector: test={sel_metrics['test_start']}..{sel_metrics['test_end']} "
            f"roi={sel_metrics['roi']} stake={sel_metrics['total_stake']} n_bets={sel_metrics['n_bets']} "
            f"median_max_dd={sel_metrics['median_max_dd']}"
        )
        final_report_lines.append("")
        final_report_lines.append("### Paired eval deltas vs baseline")
        final_report_lines.append(
            f"- median_d_roi={gate['median_d_roi']} improve_rate={gate['improve_rate']} "
            f"median_d_maxdd={gate['median_d_maxdd']} median_n_bets_var={gate['median_n_bets_var']} "
            f"n_zero_bet_windows={gate['zero_bet_windows']} gate_pass={gate['gate_pass']}"
        )
        final_report_lines.append("")
        final_report_lines.append("### Selector choice rates (eval)")
        final_report_lines.append(f"- chosen_rate_by_cap={rates} baseline_rate={baseline_rate}")
        final_report_lines.append("")
        final_report_lines.append("### Selector diagnostics (eval)")
        final_report_lines.append(f"- eligible_windows={eligible_windows}/{total_windows}")
        if baseline_bets_stats is None:
            final_report_lines.append("- baseline_valid_bets_min/median/p90/max=N/A (eval-only)")
        else:
            final_report_lines.append(
                "- baseline_valid_bets_min/median/p90/max="
                f"{baseline_bets_stats['min']}/{baseline_bets_stats['median']}/"
                f"{baseline_bets_stats['p90']}/{baseline_bets_stats['max']}"
            )
        final_report_lines.append(f"- no_op={no_op} reason={no_op_reason}")
        final_report_lines.append("")
        final_report_lines.append("### N5 EV lift (eval)")
        final_report_lines.append(
            f"- n5_signal_rate={ev_summary['signal_rate']} decile10_median_roi={ev_summary['decile10_median_roi']}"
        )
        final_report_lines.append("")

        candidate_scarcity_rows = []
        for vname, run_dirs in [("baseline", base_dirs), ("cap_selector", selector_dirs)]:
            for run_dir in run_dirs:
                summary = _summarize_candidate_scarcity(run_dir)
                if not summary:
                    continue
                candidate_scarcity_rows.append(
                    {
                        "year": summary["year"],
                        "variant": vname,
                        "group_dir": str(run_dir),
                        "n_windows": summary["n_windows"],
                        "median_frac_days_any_bet": summary["median_frac_days_any_bet"],
                        "median_bets_per_day": summary["median_bets_per_day"],
                        "median_total_days": summary["median_total_days"],
                        "median_days_with_bet": summary["median_days_with_bet"],
                        "total_bets": summary["total_bets"],
                    }
                )

        if year == "2024":
            scarcity_df_all = candidate_scarcity_rows
        else:
            scarcity_df_all += candidate_scarcity_rows

        final_report_lines.append("### Candidate scarcity (median)")
        if candidate_scarcity_rows:
            final_report_lines.append(_df_to_md(pd.DataFrame(candidate_scarcity_rows)))
        else:
            final_report_lines.append("_no scarcity data_")
        final_report_lines.append("")

        binding_rows_year: list[dict[str, Any]] = []
        base_window_map = _index_windows_by_period(base_dirs)
        for selector_dir in selector_dirs:
            offset = _infer_w_idx_offset(selector_dir.name)
            for selector_path in sorted(selector_dir.glob("w*/artifacts/race_cost_cap_selector.json")):
                window_dir = selector_path.parent.parent
                summary_path = window_dir / "summary.json"
                if not summary_path.exists():
                    continue
                summary = _read_json(summary_path)
                key = _window_key_from_summary(summary)
                if not key:
                    continue
                base_window = base_window_map.get(key)
                if base_window is None:
                    continue
                data = _read_json(selector_path)
                candidate_caps = data.get("candidate_caps") if isinstance(data, dict) else []
                metric = data.get("metric") if isinstance(data, dict) else "takeout_implied"
                cfg_path = window_dir / "config_used.yaml"
                reject_missing = True
                if cfg_path.exists():
                    cfg_used = _read_yaml(cfg_path)
                    reject_missing = bool(
                        (cfg_used.get("betting", {}) or {})
                        .get("race_cost_filter", {})
                        .get("reject_if_missing", True)
                    )
                rows = _binding_stats_for_window(
                    base_window_dir=base_window,
                    candidate_caps=candidate_caps or [None],
                    metric=str(metric or "takeout_implied"),
                    reject_missing=reject_missing,
                )
                window_name = _offset_window_name(window_dir.name, offset)
                w_idx = _parse_window_idx(window_name)
                split = "design" if w_idx is not None and int(w_idx) <= int(args.design_max_idx) else "eval"
                for row in rows:
                    row.update(
                        {
                            "year": int(year),
                            "window_name": window_name,
                            "window_idx": w_idx,
                            "split": split,
                            "base_window_dir": str(base_window),
                            "selector_window_dir": str(window_dir),
                        }
                    )
                binding_rows_year.extend(rows)

        binding_rows_all.extend(binding_rows_year)
        final_report_lines.append("### Binding diagnosis (race_cost_filter)")
        if not binding_rows_year:
            final_report_lines.append("_no binding data_")
            final_report_lines.append("")
        else:
            bind_df = pd.DataFrame(binding_rows_year)
            eval_bind = bind_df[bind_df["split"] == "eval"].copy()
            if not eval_bind.empty:
                bind_summary = (
                    eval_bind.groupby("cap_label", dropna=False)[
                        ["excluded_race_rate", "excluded_bet_rate", "excluded_stake_rate"]
                    ]
                    .median()
                    .reset_index()
                )
                final_report_lines.append(_df_to_md(bind_summary))
            else:
                final_report_lines.append("_no eval binding data_")

            discrete = eval_bind[eval_bind["pass_set_group_size"] > 1].copy() if not eval_bind.empty else pd.DataFrame()
            if not discrete.empty:
                groups = (
                    discrete.groupby(["window_name", "pass_set_group"])["cap_label"]
                    .apply(lambda s: ",".join(sorted(set(str(v) for v in s))))
                    .reset_index()
                )
                final_report_lines.append("")
                final_report_lines.append("- identical pass sets detected (thresholds effectively discrete)")
                final_report_lines.append(_df_to_md(groups))
            else:
                final_report_lines.append("")
                final_report_lines.append("- identical pass sets detected: none")
            final_report_lines.append("")

        if year == "2024":
            selector_decisions_all = decisions_df.copy()
            candidate_table_all = candidate_table
        else:
            selector_decisions_all = pd.concat([selector_decisions_all, decisions_df], ignore_index=True)
            candidate_table_all += candidate_table

    eval_gate_df = pd.DataFrame(eval_gate_rows)
    eval_gate_df.to_csv(out_dir / "eval_gate_summary.csv", index=False, encoding="utf-8")

    selector_decisions_all.to_csv(out_dir / "selector_decisions.csv", index=False, encoding="utf-8")
    (out_dir / "valid_candidate_table.json").write_text(json.dumps(candidate_table_all, ensure_ascii=False, indent=2), encoding="utf-8")

    scarcity_df = pd.DataFrame(scarcity_df_all)
    scarcity_df.to_csv(out_dir / "candidate_scarcity.csv", index=False, encoding="utf-8")

    if binding_rows_all:
        binding_df = pd.DataFrame(binding_rows_all)
    else:
        binding_df = pd.DataFrame(
            columns=[
                "year",
                "window_name",
                "window_idx",
                "split",
                "cap_label",
                "metric",
                "reject_if_missing",
                "base_races",
                "base_bets",
                "base_stake",
                "excluded_races",
                "excluded_race_rate",
                "excluded_bets",
                "excluded_bet_rate",
                "excluded_stake",
                "excluded_stake_rate",
                "pass_set_size",
                "pass_set_hash",
                "pass_set_group",
                "pass_set_group_size",
                "pass_set_discrete",
                "base_window_dir",
                "selector_window_dir",
            ]
        )
    binding_df.to_csv(out_dir / "binding_diagnosis.csv", index=False, encoding="utf-8")

    (out_dir / "ev_lift_summary.json").write_text(json.dumps(ev_lift_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_2024 = eval_gate_df[eval_gate_df["year"] == 2024].copy()
    eval_2025 = eval_gate_df[eval_gate_df["year"] == 2025].copy()
    cross = eval_2024.merge(eval_2025, on="variant", how="outer", suffixes=("_2024", "_2025"))
    cross["pass_2024"] = cross.get("gate_pass_2024").fillna(False).astype(bool)
    cross["pass_2025"] = cross.get("gate_pass_2025").fillna(False).astype(bool)
    cross["decision"] = cross.apply(lambda r: _decision(r["pass_2024"], r["pass_2025"]), axis=1)
    cross_out = cross[["variant", "pass_2024", "pass_2025", "decision"]]
    cross_out.to_csv(out_dir / "cross_year_summary.csv", index=False, encoding="utf-8")

    final_report_lines.append("## Cross-year summary")
    final_report_lines.append(_df_to_md(cross_out))
    final_report_lines.append("")
    final_report_lines.append("## OOD audit (fitted thresholds)")
    final_report_lines.append(_df_to_md(pd.DataFrame(ood_audit_rows)))
    final_report_lines.append("")

    (out_dir / "final_report.md").write_text("\n".join(final_report_lines), encoding="utf-8")

    # Required stdout lines
    for year in ["2024", "2025"]:
        rates = selector_rate_lines[year]["rates"]
        eligible_windows = selector_rate_lines[year]["eligible_windows"]
        total_windows = selector_rate_lines[year]["total_windows"]
        baseline_bets_stats = selector_rate_lines[year]["baseline_bets_stats"]
        if baseline_bets_stats is None:
            bets_stats_str = "N/A (eval-only)"
        else:
            bets_stats_str = (
                f"{baseline_bets_stats['min']}/"
                f"{baseline_bets_stats['median']}/"
                f"{baseline_bets_stats['p90']}/"
                f"{baseline_bets_stats['max']}"
            )
        print(
            f"[selector {year} eval] chosen_rate_by_cap={rates} | "
            f"eligible_windows={eligible_windows}/{total_windows} | "
            f"baseline_valid_bets_min/median/p90/max={bets_stats_str}"
        )

    eval_map = {int(r["year"]): r for r in eval_gate_rows}
    for year in [2024, 2025]:
        row = eval_map.get(year, {})
        print(
            f"[{year} eval] gate_pass={row.get('gate_pass')} | median_d_roi={row.get('median_d_roi')} | "
            f"improve_rate={row.get('improve_rate')} | median_d_maxdd={row.get('median_d_maxdd')} | "
            f"median_n_bets_var={row.get('median_n_bets_var')} | n_zero={row.get('zero_bet_windows')}"
        )

    pass_both = int((cross_out["decision"] == "pass_both").sum()) if not cross_out.empty else 0
    pass_single = int((cross_out["decision"] == "pass_single_year").sum()) if not cross_out.empty else 0
    fail_count = int((cross_out["decision"] == "fail").sum()) if not cross_out.empty else 0
    print(f"[cross-year] pass_both_count={pass_both} | pass_single_year_count={pass_single} | fail_count={fail_count}")
    pass_both_variants = (
        ",".join(sorted(cross_out.loc[cross_out["decision"] == "pass_both", "variant"].unique()))
        if not cross_out.empty and (cross_out["decision"] == "pass_both").any()
        else "none"
    )
    print(f"[cross-year] pass_both_variants={pass_both_variants}")


if __name__ == "__main__":
    main()
