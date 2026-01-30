"""
Summarize candidate-conditional takeout quantile runs with design+eval windows (2024/2025).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from keiba.analysis.metrics_utils import roi_footer, sign_mismatch

WIN_RE = re.compile(r"^w(\d{3})_")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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

    missing = [
        k
        for k, v in {
            "name": name_col,
            "roi": roi_col,
            "max_drawdown": dd_col,
            "n_bets": nb_col,
            "test_start": ts_col,
            "test_end": te_col,
        }.items()
        if v is None
    ]
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
    out["split"] = out["window_idx"].apply(
        lambda i: "design" if i is not None and int(i) <= int(design_max_idx) else "eval"
    )
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


def _split_metrics(paired: pd.DataFrame, split: str) -> dict[str, Any]:
    block = paired[paired["split"] == split].copy()
    if block.empty:
        return {
            "n_windows": 0,
            "improve_rate": None,
            "median_d_roi": None,
            "median_d_maxdd": None,
            "median_d_n_bets": None,
            "median_n_bets_var": None,
            "min_n_bets_var": None,
            "zero_bet_windows": None,
            "pooled_roi": None,
            "pooled_n_bets": None,
            "test_start": None,
            "test_end": None,
        }
    improve_rate = float((block["d_roi"] > 0).mean())
    median_d_roi = float(block["d_roi"].median())
    median_d_maxdd = float(block["d_max_dd"].median())
    median_d_n_bets = float(block["d_n_bets"].median())
    median_n_bets_var = float(block["n_bets_var"].median())
    min_n_bets_var = int(block["n_bets_var"].min())
    zero_bet_windows = int((block["n_bets_var"] <= 0).sum())
    stake_sum = pd.to_numeric(block.get("total_stake_var"), errors="coerce").sum()
    profit_sum = pd.to_numeric(block.get("total_profit_var"), errors="coerce").sum()
    pooled_roi = (profit_sum / stake_sum) if stake_sum and pd.notna(stake_sum) else None
    pooled_n_bets = int(pd.to_numeric(block.get("n_bets_var"), errors="coerce").sum())
    test_start = str(block["test_start"].min())
    test_end = str(block["test_end"].max())
    return {
        "n_windows": int(len(block)),
        "improve_rate": improve_rate,
        "median_d_roi": median_d_roi,
        "median_d_maxdd": median_d_maxdd,
        "median_d_n_bets": median_d_n_bets,
        "median_n_bets_var": median_n_bets_var,
        "min_n_bets_var": min_n_bets_var,
        "zero_bet_windows": zero_bet_windows,
        "pooled_roi": pooled_roi,
        "pooled_n_bets": pooled_n_bets,
        "test_start": test_start,
        "test_end": test_end,
    }


def _gate_pass(metrics: dict[str, Any]) -> bool:
    if not metrics or metrics.get("n_windows", 0) <= 0:
        return False
    return (
        metrics.get("improve_rate") is not None
        and metrics.get("median_d_roi") is not None
        and metrics.get("median_d_maxdd") is not None
        and metrics.get("median_n_bets_var") is not None
        and metrics.get("zero_bet_windows") is not None
        and metrics["improve_rate"] >= 0.6
        and metrics["median_d_roi"] > 0
        and metrics["median_d_maxdd"] <= 0
        and metrics["median_n_bets_var"] >= 80
        and metrics["zero_bet_windows"] == 0
    )


def _aggregate_split_metrics(df: pd.DataFrame, split: str) -> dict[str, Any]:
    block = df[df["split"] == split].copy()
    if block.empty:
        return {
            "test_start": None,
            "test_end": None,
            "n_bets": None,
            "total_stake": None,
            "total_profit": None,
            "roi": None,
            "median_max_dd": None,
        }
    test_start = str(block["test_start"].min())
    test_end = str(block["test_end"].max())
    total_stake = pd.to_numeric(block.get("total_stake"), errors="coerce").sum()
    total_profit = pd.to_numeric(block.get("total_profit"), errors="coerce").sum()
    roi = (total_profit / total_stake) if total_stake and pd.notna(total_stake) else None
    n_bets = int(pd.to_numeric(block.get("n_bets"), errors="coerce").sum())
    median_max_dd = float(pd.to_numeric(block.get("max_dd"), errors="coerce").median())
    return {
        "test_start": test_start,
        "test_end": test_end,
        "n_bets": n_bets,
        "total_stake": float(total_stake) if pd.notna(total_stake) else None,
        "total_profit": float(total_profit) if pd.notna(total_profit) else None,
        "roi": float(roi) if roi is not None else None,
        "median_max_dd": median_max_dd,
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


def _summarize_candidate_scarcity(run_dirs: list[Path], design_max_idx: int) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        offset = _infer_w_idx_offset(run_dir.name)
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
            window_name = _offset_window_name(w.name, offset)
            window_idx = _parse_window_idx(window_name)
            split = "design" if window_idx is not None and int(window_idx) <= int(design_max_idx) else "eval"
            rows.append(
                {
                    "window_name": window_name,
                    "window_idx": window_idx,
                    "split": split,
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
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("split", dropna=False)
        .agg(
            n_windows=("window_name", "count"),
            median_frac_days_any_bet=("frac_days_any_bet", "median"),
            median_bets_per_day=("bets_per_day", "median"),
            median_total_days=("total_days", "median"),
            median_days_with_bet=("days_with_bet", "median"),
            total_bets=("n_bets", "sum"),
        )
        .reset_index()
    )
    return summary


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


def _hash_race_set(race_ids: list[str]) -> str:
    joined = "|".join(sorted(str(r) for r in race_ids))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _binding_stats_for_window(
    base_window_dir: Path,
    cap_value: Optional[float],
    metric: str,
    reject_missing: bool,
) -> dict[str, Any] | None:
    bets_path = base_window_dir / "bets.csv"
    if not bets_path.exists():
        return None
    df = pd.read_csv(bets_path)
    if df.empty:
        return None

    metric_col = metric if metric in df.columns else None
    if metric_col is None:
        if "takeout_implied" in df.columns:
            metric_col = "takeout_implied"
        elif "overround_sum_inv" in df.columns:
            metric_col = "overround_sum_inv"
    if metric_col is None:
        return None

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

    return {
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


def _decision(pass_2024: bool, pass_2025: bool) -> str:
    if pass_2024 and pass_2025:
        return "pass_both"
    if pass_2024 or pass_2025:
        return "pass_single_year"
    return "fail"


def _fmt(val: Any, digits: int = 4) -> str:
    if val is None:
        return "N/A (eval-only)"
    if isinstance(val, float) and (pd.isna(val) or np.isnan(val)):
        return "N/A (eval-only)"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _variant_q_value(cap_meta: dict) -> Optional[float]:
    q = cap_meta.get("train_quantile") if isinstance(cap_meta, dict) else None
    try:
        return float(q) if q is not None else None
    except Exception:
        return None


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def _copy_if_different(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _split_run_dirs(run_dirs: list[Path]) -> tuple[Optional[Path], Optional[Path]]:
    design_dir = None
    eval_dir = None
    for p in run_dirs:
        name = p.name.lower()
        if "w013_022" in name or "w013-022" in name:
            eval_dir = p
        elif "w001_012" in name or "w001-012" in name:
            design_dir = p
    if design_dir is None and run_dirs:
        design_dir = run_dirs[0]
    if eval_dir is None and len(run_dirs) > 1:
        eval_dir = run_dirs[-1]
    return design_dir, eval_dir


def _read_bankroll(run_dir: Path) -> float:
    for window_dir in sorted(run_dir.glob("w*/")):
        cfg_path = window_dir / "config_used.yaml"
        if not cfg_path.exists():
            continue
        try:
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            return float((cfg.get("betting") or {}).get("bankroll_yen", 1_000_000.0))
        except Exception:
            continue
    return 1_000_000.0


def _run_step14(
    year: int,
    strategy: str,
    design_dir: Path,
    eval_dir: Path,
    out_dir: Path,
    initial_bankroll: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(Path("py64_analysis/.venv/Scripts/python.exe")),
        str(Path("py64_analysis/scripts/summarize_walkforward_step14_from_rolling.py")),
        "--year",
        str(year),
        "--strategy",
        strategy,
        "--design-dir",
        str(design_dir),
        "--eval-dir",
        str(eval_dir),
        "--step-days",
        "14",
        "--initial-bankroll",
        str(initial_bankroll),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=False)
    return out_dir / "walkforward_step14_summary.csv"


def _read_step14_summary(path: Path, variant: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(1, "variant", variant)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize candidate-conditional takeout quantile full-year (design+eval) runs (2024/2025)"
    )
    ap.add_argument("--manifest-2024", type=Path, required=True)
    ap.add_argument("--manifest-2025", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--design-max-idx", type=int, default=12)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "2024": _read_json(args.manifest_2024),
        "2025": _read_json(args.manifest_2025),
    }

    eval_gate_rows = []
    scarcity_rows = []
    fitted_rows = []
    binding_rows = []
    step14_rows = []
    step14_lookup: dict[tuple[int, str, str], dict[str, Any]] = {}
    final_report_lines = []

    final_report_lines.append("# Candidate-conditional takeout quantile full-year (2024/2025)")
    final_report_lines.append("")
    final_report_lines.append("- Split: time-based (train < valid < test). Design windows: w001–w012, eval windows: w013+.")
    final_report_lines.append("- Pooled aggregates overlap across rolling windows; step14 is non-overlap.")
    final_report_lines.append("- Overlap can flip ROI sign; step14 is the decision signal when signs differ.")
    final_report_lines.append("")

    for year in ["2024", "2025"]:
        manifest = manifests[year]
        baseline_name = manifest.get("baseline")
        variants = manifest.get("variants", {})
        if baseline_name not in variants:
            raise SystemExit(f"baseline not in manifest {year}: {baseline_name}")

        base_dirs = [Path(p) for p in variants.get(baseline_name, [])]
        if not base_dirs:
            raise SystemExit(f"missing baseline run dirs in manifest {year}")

        final_report_lines.append(f"## {year}")
        final_report_lines.append("")
        final_report_lines.append("Baseline group_dirs:")
        for p in base_dirs:
            final_report_lines.append(f"- {p}")

        base_df = _combine_summaries(base_dirs, design_max_idx=args.design_max_idx)
        base_metrics_design = _aggregate_split_metrics(base_df, "design")
        base_metrics_eval = _aggregate_split_metrics(base_df, "eval")

        base_window_map = _index_windows_by_period(base_dirs)

        base_design_dir, base_eval_dir = _split_run_dirs(base_dirs)
        if base_design_dir is None or base_eval_dir is None:
            raise SystemExit(f"could not resolve design/eval run dirs for baseline {year}")
        base_bankroll = _read_bankroll(base_design_dir)
        base_step_path = _run_step14(
            year=int(year),
            strategy="baseline",
            design_dir=base_design_dir,
            eval_dir=base_eval_dir,
            out_dir=out_dir / f"step14_{year}_baseline",
            initial_bankroll=base_bankroll,
        )
        base_step_df = _read_step14_summary(base_step_path, variant="baseline")
        if not base_step_df.empty:
            step14_rows.append(base_step_df)
            for _, row in base_step_df.iterrows():
                step14_lookup[(int(row["year"]), str(row["strategy"]), str(row["split"]))] = row.to_dict()

        for variant_name, run_list in variants.items():
            if variant_name == baseline_name:
                continue
            var_dirs = [Path(p) for p in run_list]
            if not var_dirs:
                continue

            var_df = _combine_summaries(var_dirs, design_max_idx=args.design_max_idx)
            paired = _paired_compare(base_df, var_df)
            metrics_design = _split_metrics(paired, "design")
            metrics_eval = _split_metrics(paired, "eval")
            gate_pass = _gate_pass(metrics_eval)
            var_metrics_design = _aggregate_split_metrics(var_df, "design")
            var_metrics_eval = _aggregate_split_metrics(var_df, "eval")

            var_design_dir, var_eval_dir = _split_run_dirs(var_dirs)
            if var_design_dir is None or var_eval_dir is None:
                raise SystemExit(f"could not resolve design/eval run dirs for {variant_name} {year}")
            var_bankroll = _read_bankroll(var_design_dir)
            var_step_path = _run_step14(
                year=int(year),
                strategy=str(variant_name),
                design_dir=var_design_dir,
                eval_dir=var_eval_dir,
                out_dir=out_dir / f"step14_{year}_{variant_name}",
                initial_bankroll=var_bankroll,
            )
            var_step_df = _read_step14_summary(var_step_path, variant=str(variant_name))
            if not var_step_df.empty:
                step14_rows.append(var_step_df)
                for _, row in var_step_df.iterrows():
                    step14_lookup[(int(row["year"]), str(row["strategy"]), str(row["split"]))] = row.to_dict()

            q_value = None
            cap_value_median = None
            fit_population_note = None

            for run_dir in var_dirs:
                offset = _infer_w_idx_offset(run_dir.name)
                for cap_path in sorted(run_dir.glob("w*/artifacts/race_cost_cap.json")):
                    window_dir = cap_path.parent.parent
                    window_name = _offset_window_name(window_dir.name, offset)
                    w_idx = _parse_window_idx(window_name)
                    split = "design" if w_idx is not None and int(w_idx) <= int(args.design_max_idx) else "eval"
                    cap_meta = _read_json(cap_path)
                    q_value = _variant_q_value(cap_meta)
                    metric = cap_meta.get("metric")
                    fit_scope = cap_meta.get("fit_scope")
                    fit_population = cap_meta.get("fit_population")
                    fit_population_note = cap_meta.get("fit_population_note")
                    cap_value = cap_meta.get("cap_value")
                    n_fit_races = cap_meta.get("n_train_races_used")
                    fitted_rows.append(
                        {
                            "year": int(year),
                            "variant": variant_name,
                            "window_name": window_name,
                            "window_idx": w_idx,
                            "split": split,
                            "metric": metric,
                            "fit_scope": fit_scope,
                            "fit_population": fit_population,
                            "q": q_value,
                            "cap_value": cap_value,
                            "n_fit_races": n_fit_races,
                            "fit_min": cap_meta.get("fit_metric_min"),
                            "fit_p50": cap_meta.get("fit_metric_p50"),
                            "fit_p90": cap_meta.get("fit_metric_p90"),
                            "fit_max": cap_meta.get("fit_metric_max"),
                            "fit_metric_unique_values": cap_meta.get("fit_metric_unique_values"),
                        }
                    )

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

                    cfg_used = summary.get("race_cost_filter") or {}
                    reject_missing = bool(cfg_used.get("reject_if_missing", True))
                    cap_value = cap_meta.get("cap_value")
                    try:
                        cap_value = float(cap_value) if cap_value is not None else None
                    except Exception:
                        cap_value = None

                    binding = _binding_stats_for_window(
                        base_window_dir=base_window,
                        cap_value=cap_value,
                        metric=str(metric or "takeout_implied"),
                        reject_missing=reject_missing,
                    )
                    if binding is None:
                        continue
                    binding_rows.append(
                        {
                            "year": int(year),
                            "variant": variant_name,
                            "window_name": window_name,
                            "window_idx": w_idx,
                            "split": split,
                            "q": q_value,
                            "cap_value": cap_value,
                            "fit_metric_unique_values": cap_meta.get("fit_metric_unique_values"),
                            **binding,
                        }
                    )

            if fitted_rows:
                cap_vals = [r["cap_value"] for r in fitted_rows if r.get("variant") == variant_name and r.get("year") == int(year)]
                cap_vals = [float(v) for v in cap_vals if v is not None and np.isfinite(v)]
                if cap_vals:
                    cap_value_median = float(np.median(cap_vals))

            eval_gate_rows.append(
                {
                    "year": int(year),
                    "variant": variant_name,
                    "q": q_value,
                    "cap_value_median": cap_value_median,
                    "eval_gate_pass": gate_pass,
                    **_prefix_metrics(metrics_eval, "eval_"),
                    **_prefix_metrics(metrics_design, "design_"),
                }
            )

            scarcity = _summarize_candidate_scarcity(var_dirs, design_max_idx=args.design_max_idx)
            scarcity_eval = None
            scarcity_design = None
            if not scarcity.empty:
                for _, row in scarcity.iterrows():
                    row_dict = row.to_dict()
                    scarcity_rows.append(
                        {
                            "year": int(year),
                            "variant": variant_name,
                            "q": q_value,
                            "cap_value_median": cap_value_median,
                            **row_dict,
                        }
                    )
                eval_rows = scarcity[scarcity["split"] == "eval"]
                if not eval_rows.empty:
                    scarcity_eval = eval_rows.iloc[0].to_dict()
                design_rows = scarcity[scarcity["split"] == "design"]
                if not design_rows.empty:
                    scarcity_design = design_rows.iloc[0].to_dict()

            final_report_lines.append("")
            final_report_lines.append(f"Variant: {variant_name} (q={q_value}, cap_median={cap_value_median})")
            final_report_lines.append("Variant group_dirs:")
            for p in var_dirs:
                final_report_lines.append(f"- {p}")
            final_report_lines.append("Metrics:")
            final_report_lines.append(
                "- eval gate_pass={gate_pass} | median_d_roi={median_d_roi} | improve_rate={improve_rate} | median_d_maxdd={median_d_maxdd} | median_n_bets_var={median_n_bets_var} | n_zero_bet_windows={zero_bet_windows}".format(
                    gate_pass=_fmt(gate_pass, digits=4),
                    median_d_roi=_fmt(metrics_eval.get("median_d_roi")),
                    improve_rate=_fmt(metrics_eval.get("improve_rate")),
                    median_d_maxdd=_fmt(metrics_eval.get("median_d_maxdd")),
                    median_n_bets_var=_fmt(metrics_eval.get("median_n_bets_var"), digits=1),
                    zero_bet_windows=_fmt(metrics_eval.get("zero_bet_windows"), digits=1),
                )
            )
            final_report_lines.append(
                "- eval pooled_roi={roi} | pooled_stake={stake} | pooled_n_bets={n_bets} | median_max_dd={max_dd} | test_period={start}..{end}".format(
                    roi=_fmt(var_metrics_eval.get("roi")),
                    stake=_fmt(var_metrics_eval.get("total_stake"), digits=2),
                    n_bets=_fmt(var_metrics_eval.get("n_bets"), digits=1),
                    max_dd=_fmt(var_metrics_eval.get("median_max_dd")),
                    start=_fmt(var_metrics_eval.get("test_start"), digits=1),
                    end=_fmt(var_metrics_eval.get("test_end"), digits=1),
                )
            )
            base_step_eval = step14_lookup.get((int(year), "baseline", "eval")) or {}
            var_step_eval = step14_lookup.get((int(year), str(variant_name), "eval")) or {}
            base_sm = sign_mismatch(base_metrics_eval.get("roi"), base_step_eval.get("roi"))
            var_sm = sign_mismatch(var_metrics_eval.get("roi"), var_step_eval.get("roi"))
            final_report_lines.append(
                "- eval step14 base roi={roi} | stake={stake} | n_bets={n_bets} | max_dd={max_dd} | {tag}".format(
                    roi=_fmt(base_step_eval.get("roi")),
                    stake=_fmt(base_step_eval.get("total_stake"), digits=2),
                    n_bets=_fmt(base_step_eval.get("n_bets"), digits=1),
                    max_dd=_fmt(base_step_eval.get("max_drawdown")),
                    tag="SIGN_MISMATCH" if base_sm else "sign_match",
                )
            )
            final_report_lines.append(
                "- eval step14 {variant} roi={roi} | stake={stake} | n_bets={n_bets} | max_dd={max_dd} | {tag}".format(
                    variant=variant_name,
                    roi=_fmt(var_step_eval.get("roi")),
                    stake=_fmt(var_step_eval.get("total_stake"), digits=2),
                    n_bets=_fmt(var_step_eval.get("n_bets"), digits=1),
                    max_dd=_fmt(var_step_eval.get("max_drawdown")),
                    tag="SIGN_MISMATCH" if var_sm else "sign_match",
                )
            )
            final_report_lines.append(
                "- design median_d_roi={median_d_roi} | improve_rate={improve_rate} | median_d_maxdd={median_d_maxdd} | median_n_bets_var={median_n_bets_var} | n_zero_bet_windows={zero_bet_windows}".format(
                    median_d_roi=_fmt(metrics_design.get("median_d_roi")),
                    improve_rate=_fmt(metrics_design.get("improve_rate")),
                    median_d_maxdd=_fmt(metrics_design.get("median_d_maxdd")),
                    median_n_bets_var=_fmt(metrics_design.get("median_n_bets_var"), digits=1),
                    zero_bet_windows=_fmt(metrics_design.get("zero_bet_windows"), digits=1),
                )
            )
            final_report_lines.append(
                "- design pooled_roi={roi} | pooled_stake={stake} | pooled_n_bets={n_bets} | median_max_dd={max_dd} | test_period={start}..{end}".format(
                    roi=_fmt(var_metrics_design.get("roi")),
                    stake=_fmt(var_metrics_design.get("total_stake"), digits=2),
                    n_bets=_fmt(var_metrics_design.get("n_bets"), digits=1),
                    max_dd=_fmt(var_metrics_design.get("median_max_dd")),
                    start=_fmt(var_metrics_design.get("test_start"), digits=1),
                    end=_fmt(var_metrics_design.get("test_end"), digits=1),
                )
            )
            base_step_design = step14_lookup.get((int(year), "baseline", "design")) or {}
            var_step_design = step14_lookup.get((int(year), str(variant_name), "design")) or {}
            final_report_lines.append(
                "- design step14 base roi={roi} | stake={stake} | n_bets={n_bets} | max_dd={max_dd}".format(
                    roi=_fmt(base_step_design.get("roi")),
                    stake=_fmt(base_step_design.get("total_stake"), digits=2),
                    n_bets=_fmt(base_step_design.get("n_bets"), digits=1),
                    max_dd=_fmt(base_step_design.get("max_drawdown")),
                )
            )
            final_report_lines.append(
                "- design step14 {variant} roi={roi} | stake={stake} | n_bets={n_bets} | max_dd={max_dd}".format(
                    variant=variant_name,
                    roi=_fmt(var_step_design.get("roi")),
                    stake=_fmt(var_step_design.get("total_stake"), digits=2),
                    n_bets=_fmt(var_step_design.get("n_bets"), digits=1),
                    max_dd=_fmt(var_step_design.get("max_drawdown")),
                )
            )

            final_report_lines.append(
                "- scarcity eval: median_frac_days_any_bet={frac} | median_bets_per_day={bets_per_day}".format(
                    frac=_fmt(scarcity_eval.get("median_frac_days_any_bet") if scarcity_eval else None),
                    bets_per_day=_fmt(scarcity_eval.get("median_bets_per_day") if scarcity_eval else None),
                )
            )
            final_report_lines.append(
                "- scarcity design: median_frac_days_any_bet={frac} | median_bets_per_day={bets_per_day}".format(
                    frac=_fmt(scarcity_design.get("median_frac_days_any_bet") if scarcity_design else None),
                    bets_per_day=_fmt(scarcity_design.get("median_bets_per_day") if scarcity_design else None),
                )
            )
            if fit_population_note:
                final_report_lines.append(f"- fit_population_note: {fit_population_note}")

        final_report_lines.append("")
        final_report_lines.append("Baseline metrics:")
        final_report_lines.append(
            "- eval pooled_roi={roi} | pooled_stake={stake} | pooled_n_bets={n_bets} | median_max_dd={max_dd} | test_period={start}..{end}".format(
                roi=_fmt(base_metrics_eval.get("roi")),
                stake=_fmt(base_metrics_eval.get("total_stake"), digits=2),
                n_bets=_fmt(base_metrics_eval.get("n_bets"), digits=1),
                max_dd=_fmt(base_metrics_eval.get("median_max_dd")),
                start=_fmt(base_metrics_eval.get("test_start"), digits=1),
                end=_fmt(base_metrics_eval.get("test_end"), digits=1),
            )
        )
        base_step_eval = step14_lookup.get((int(year), "baseline", "eval")) or {}
        base_sm = sign_mismatch(base_metrics_eval.get("roi"), base_step_eval.get("roi"))
        final_report_lines.append(
            "- eval step14 base roi={roi} | stake={stake} | n_bets={n_bets} | max_dd={max_dd} | {tag}".format(
                roi=_fmt(base_step_eval.get("roi")),
                stake=_fmt(base_step_eval.get("total_stake"), digits=2),
                n_bets=_fmt(base_step_eval.get("n_bets"), digits=1),
                max_dd=_fmt(base_step_eval.get("max_drawdown")),
                tag="SIGN_MISMATCH" if base_sm else "sign_match",
            )
        )
        final_report_lines.append(
            "- design pooled_roi={roi} | pooled_stake={stake} | pooled_n_bets={n_bets} | median_max_dd={max_dd} | test_period={start}..{end}".format(
                roi=_fmt(base_metrics_design.get("roi")),
                stake=_fmt(base_metrics_design.get("total_stake"), digits=2),
                n_bets=_fmt(base_metrics_design.get("n_bets"), digits=1),
                max_dd=_fmt(base_metrics_design.get("median_max_dd")),
                start=_fmt(base_metrics_design.get("test_start"), digits=1),
                end=_fmt(base_metrics_design.get("test_end"), digits=1),
            )
        )
        base_step_design = step14_lookup.get((int(year), "baseline", "design")) or {}
        final_report_lines.append(
            "- design step14 base roi={roi} | stake={stake} | n_bets={n_bets} | max_dd={max_dd}".format(
                roi=_fmt(base_step_design.get("roi")),
                stake=_fmt(base_step_design.get("total_stake"), digits=2),
                n_bets=_fmt(base_step_design.get("n_bets"), digits=1),
                max_dd=_fmt(base_step_design.get("max_drawdown")),
            )
        )

    eval_gate_df = pd.DataFrame(eval_gate_rows)
    eval_gate_df.to_csv(out_dir / "eval_gate_summary.csv", index=False)

    scarcity_df = pd.DataFrame(scarcity_rows)
    scarcity_df.to_csv(out_dir / "candidate_scarcity.csv", index=False)

    fitted_df = pd.DataFrame(fitted_rows)
    fitted_df.to_csv(out_dir / "fitted_cap_values.csv", index=False)

    binding_df = pd.DataFrame(binding_rows)
    if not binding_df.empty:
        binding_df["pass_set_group_size"] = (
            binding_df.groupby(["year", "window_name", "pass_set_hash"])["variant"]
            .transform("count")
            .astype(int)
        )
        binding_df["pass_set_discrete"] = binding_df["pass_set_group_size"] > 1
    binding_df.to_csv(out_dir / "binding_diagnosis.csv", index=False)

    step14_df = pd.concat(step14_rows, ignore_index=True) if step14_rows else pd.DataFrame()
    if not step14_df.empty:
        step14_df.to_csv(out_dir / "walkforward_step14_summary.csv", index=False)

    eval_2024 = eval_gate_df[eval_gate_df["year"] == 2024].copy()
    eval_2025 = eval_gate_df[eval_gate_df["year"] == 2025].copy()
    cross = eval_2024.merge(eval_2025, on="variant", how="outer", suffixes=("_2024", "_2025"))
    cross["pass_2024"] = cross.get("eval_gate_pass_2024").fillna(False).astype(bool)
    cross["pass_2025"] = cross.get("eval_gate_pass_2025").fillna(False).astype(bool)
    cross["decision"] = cross.apply(lambda r: _decision(r["pass_2024"], r["pass_2025"]), axis=1)
    cross_out = cross[["variant", "pass_2024", "pass_2025", "decision"]]
    cross_out.to_csv(out_dir / "cross_year_summary.csv", index=False)

    final_report_lines.append("")
    final_report_lines.append("## Cross-year summary")
    final_report_lines.append(_df_to_md(cross_out))
    final_report_lines.append("")

    if not step14_df.empty:
        final_report_lines.append("## Step14 non-overlap summary (eval only)")
        eval_step14 = step14_df[step14_df["split"] == "eval"].copy()
        if not eval_step14.empty:
            summary_cols = ["year", "variant", "strategy", "split", "roi", "total_stake", "total_profit", "n_bets", "max_drawdown"]
            final_report_lines.append(_df_to_md(eval_step14[summary_cols]))
            final_report_lines.append("")
            totals = (
                eval_step14.groupby(["variant", "strategy"], dropna=False)[
                    ["total_stake", "total_profit", "n_bets"]
                ]
                .sum()
                .reset_index()
            )
            totals["roi"] = totals.apply(
                lambda r: (r["total_profit"] / r["total_stake"]) if r["total_stake"] else None,
                axis=1,
            )
            totals["year"] = "total"
            totals["split"] = "eval"
            totals["max_drawdown"] = None
            final_report_lines.append("## Step14 totals (2024+2025 eval)")
            final_report_lines.append(_df_to_md(totals[summary_cols]))
            final_report_lines.append("")

    identical_passset_windows = 0
    discrete_thresholds = 0
    if not binding_df.empty:
        passset_groups = binding_df.groupby(["year", "window_name", "pass_set_hash"])["variant"].nunique()
        discrete_thresholds = int((passset_groups > 1).sum())
        identical_passset_windows = int(passset_groups[passset_groups > 1].groupby(level=[0, 1]).ngroups)

        final_report_lines.append("## Binding diagnosis (median excluded rates)")
        bind_summary = (
            binding_df.groupby(["year", "variant", "split"], dropna=False)[
                ["excluded_race_rate", "excluded_bet_rate", "excluded_stake_rate"]
            ]
            .median()
            .reset_index()
        )
        final_report_lines.append(_df_to_md(bind_summary))
        discrete = binding_df[binding_df["pass_set_discrete"]].copy()
        if not discrete.empty:
            final_report_lines.append("")
            final_report_lines.append("Identical pass sets detected (thresholds effectively discrete).")
            discrete_table = discrete[["year", "window_name", "variant", "pass_set_hash"]].drop_duplicates()
            final_report_lines.append(_df_to_md(discrete_table))
        else:
            final_report_lines.append("")
            final_report_lines.append("No identical pass sets detected.")
        final_report_lines.append("")

    (out_dir / "final_report.md").write_text(
        "\n".join(final_report_lines + ["", roi_footer()]) + "\n",
        encoding="utf-8",
    )
    _copy_if_different(args.manifest_2024, out_dir / "manifest_2024.json")
    _copy_if_different(args.manifest_2025, out_dir / "manifest_2025.json")

    # Required stdout lines
    for year in ["2024", "2025"]:
        rows = eval_gate_df[eval_gate_df["year"] == int(year)].copy()
        if rows.empty:
            continue
        rows = rows.sort_values(["q"])
        for _, row in rows.iterrows():
            print(
                f"[{year} eval] q={row.get('q')} gate_pass={row.get('eval_gate_pass')} | "
                f"median_d_roi={row.get('eval_median_d_roi')} | improve_rate={row.get('eval_improve_rate')} | "
                f"median_d_maxdd={row.get('eval_median_d_maxdd')} | median_n_bets_var={row.get('eval_median_n_bets_var')} | "
                f"pooled_roi_eval={row.get('eval_pooled_roi')}"
            )

    pass_both = int((cross_out["decision"] == "pass_both").sum()) if not cross_out.empty else 0
    pass_single = int((cross_out["decision"] == "pass_single_year").sum()) if not cross_out.empty else 0
    fail_count = int((cross_out["decision"] == "fail").sum()) if not cross_out.empty else 0
    pass_both_variants = (
        ",".join(sorted(cross_out.loc[cross_out["decision"] == "pass_both", "variant"].unique()))
        if not cross_out.empty and (cross_out["decision"] == "pass_both").any()
        else "none"
    )
    print(
        f"[cross-year] pass_both_count={pass_both} | pass_single_year_count={pass_single} | fail_count={fail_count} | "
        f"pass_both_variants={pass_both_variants}"
    )
    print(f"[binding] identical_passset_windows={identical_passset_windows} | discrete_thresholds_detected={discrete_thresholds}")


if __name__ == "__main__":
    main()
