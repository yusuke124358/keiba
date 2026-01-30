"""
Summarize candidate-conditional takeout quantile sweep across 2024/2025 eval-only rolling runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

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


def _date_range(start: str, end: str) -> list[datetime.date]:
    ds = datetime.strptime(start, "%Y-%m-%d").date()
    de = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    cur = ds
    while cur <= de:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _summarize_candidate_scarcity(run_dirs: list[Path]) -> Optional[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize candidate-conditional takeout quantile sweep (2024/2025)")
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
    final_report_lines = []

    final_report_lines.append("# Candidate-conditional takeout quantile sweep (2024/2025)")
    final_report_lines.append("")
    final_report_lines.append("- Split: time-based (train < valid < test), eval-only windows (design metrics are N/A).")
    final_report_lines.append("")

    gate_by_year_q: dict[str, dict[float, dict[str, Any]]] = {"2024": {}, "2025": {}}

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
        base_metrics = _aggregate_year_metrics(base_df)

        base_window_map = _index_windows_by_period(base_dirs)

        for variant_name, run_list in variants.items():
            if variant_name == baseline_name:
                continue
            var_dirs = [Path(p) for p in run_list]
            if not var_dirs:
                continue

            var_df = _combine_summaries(var_dirs, design_max_idx=args.design_max_idx)
            paired = _paired_compare(base_df, var_df)
            gate = _eval_gate_metrics(paired)
            var_metrics = _aggregate_year_metrics(var_df)

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
                    **gate,
                }
            )

            scarcity = _summarize_candidate_scarcity(var_dirs)
            if scarcity:
                scarcity_rows.append(
                    {
                        "year": int(year),
                        "variant": variant_name,
                        "q": q_value,
                        "cap_value_median": cap_value_median,
                        **scarcity,
                    }
                )

            if q_value is not None:
                gate_by_year_q[year][q_value] = gate

            final_report_lines.append("")
            final_report_lines.append(f"Variant: {variant_name} (q={q_value}, cap_median={cap_value_median})")
            final_report_lines.append("Variant group_dirs:")
            for p in var_dirs:
                final_report_lines.append(f"- {p}")
            final_report_lines.append("Metrics (eval-only):")
            final_report_lines.append(
                "- gate_pass={gate_pass} | median_d_roi={median_d_roi} | improve_rate={improve_rate} | median_d_maxdd={median_d_maxdd} | median_n_bets_var={median_n_bets_var} | n_zero_bet_windows_var={zero_bet_windows}".format(
                    gate_pass=_fmt(gate.get("gate_pass"), digits=4),
                    median_d_roi=_fmt(gate.get("median_d_roi")),
                    improve_rate=_fmt(gate.get("improve_rate")),
                    median_d_maxdd=_fmt(gate.get("median_d_maxdd")),
                    median_n_bets_var=_fmt(gate.get("median_n_bets_var"), digits=1),
                    zero_bet_windows=_fmt(gate.get("zero_bet_windows"), digits=1),
                )
            )
            final_report_lines.append(
                "- pooled_roi={roi} | pooled_stake={stake} | pooled_n_bets={n_bets} | test_period={start}..{end}".format(
                    roi=_fmt(var_metrics.get("roi")),
                    stake=_fmt(var_metrics.get("total_stake"), digits=2),
                    n_bets=_fmt(var_metrics.get("n_bets"), digits=1),
                    start=_fmt(var_metrics.get("test_start"), digits=1),
                    end=_fmt(var_metrics.get("test_end"), digits=1),
                )
            )

            scarcity_line = "- scarcity: median_frac_days_any_bet={frac} | median_bets_per_day={bets_per_day}".format(
                frac=_fmt(scarcity.get("median_frac_days_any_bet") if scarcity else None),
                bets_per_day=_fmt(scarcity.get("median_bets_per_day") if scarcity else None),
            )
            final_report_lines.append(scarcity_line)
            if fit_population_note:
                final_report_lines.append(f"- fit_population_note: {fit_population_note}")

        final_report_lines.append("")
        final_report_lines.append("Baseline metrics (eval-only):")
        final_report_lines.append(
            "- pooled_roi={roi} | pooled_stake={stake} | pooled_n_bets={n_bets} | test_period={start}..{end}".format(
                roi=_fmt(base_metrics.get("roi")),
                stake=_fmt(base_metrics.get("total_stake"), digits=2),
                n_bets=_fmt(base_metrics.get("n_bets"), digits=1),
                start=_fmt(base_metrics.get("test_start"), digits=1),
                end=_fmt(base_metrics.get("test_end"), digits=1),
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

    eval_2024 = eval_gate_df[eval_gate_df["year"] == 2024].copy()
    eval_2025 = eval_gate_df[eval_gate_df["year"] == 2025].copy()
    cross = eval_2024.merge(eval_2025, on="variant", how="outer", suffixes=("_2024", "_2025"))
    cross["pass_2024"] = cross.get("gate_pass_2024").fillna(False).astype(bool)
    cross["pass_2025"] = cross.get("gate_pass_2025").fillna(False).astype(bool)
    cross["decision"] = cross.apply(lambda r: _decision(r["pass_2024"], r["pass_2025"]), axis=1)
    cross_out = cross[["variant", "pass_2024", "pass_2025", "decision"]]
    cross_out.to_csv(out_dir / "cross_year_summary.csv", index=False)

    final_report_lines.append("")
    final_report_lines.append("## Cross-year summary")
    final_report_lines.append(_df_to_md(cross_out))
    final_report_lines.append("")

    if not binding_df.empty:
        final_report_lines.append("## Binding diagnosis (median excluded rates, eval)")
        bind_eval = binding_df[binding_df["split"] == "eval"].copy()
        if not bind_eval.empty:
            bind_summary = (
                bind_eval.groupby(["year", "variant"], dropna=False)[
                    ["excluded_race_rate", "excluded_bet_rate", "excluded_stake_rate"]
                ]
                .median()
                .reset_index()
            )
            final_report_lines.append(_df_to_md(bind_summary))
            discrete = bind_eval[bind_eval["pass_set_discrete"]].copy()
            if not discrete.empty:
                final_report_lines.append("")
                final_report_lines.append("Identical pass sets detected (thresholds effectively discrete).")
        else:
            final_report_lines.append("_no eval binding data_")
        final_report_lines.append("")

    (out_dir / "final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")
    shutil.copy2(args.manifest_2024, out_dir / "manifest_2024.json")
    shutil.copy2(args.manifest_2025, out_dir / "manifest_2025.json")

    # Required stdout lines
    for year in ["2024", "2025"]:
        rows = eval_gate_df[eval_gate_df["year"] == int(year)].copy()
        if rows.empty:
            continue
        rows = rows.sort_values(["q"])
        for _, row in rows.iterrows():
            print(
                f"[takeoutqCand {year} eval] q={row.get('q')} gate_pass={row.get('gate_pass')} | "
                f"median_d_roi={row.get('median_d_roi')} | improve_rate={row.get('improve_rate')} | "
                f"median_d_maxdd={row.get('median_d_maxdd')} | median_n_bets_var={row.get('median_n_bets_var')} | "
                f"pooled_roi_eval={row.get('pooled_roi_eval')}"
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
