"""
Summarize pace_history sweep runs (baseline vs variant) across 2024/2025.

This combines split run groups (w001_012 + w013_022), computes paired metrics,
gate decision, and step=14 non-overlap summaries.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _parse_window_name(name: str) -> Optional[tuple[int, str, str]]:
    if not isinstance(name, str):
        return None
    m = re.match(r"^w(\d{3})_(\d{8})_(\d{8})$", name)
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def _offset_window_name(name: str, offset: int) -> str:
    parsed = _parse_window_name(name)
    if not parsed:
        return name
    idx, start, end = parsed
    return f"w{idx + offset:03d}_{start}_{end}"


def _load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "name" not in df.columns:
        raise ValueError(f"summary.csv missing name column: {path}")
    return df


def _combine_runs(w001_dir: Path, w013_dir: Path, out_dir: Path, offset: int) -> pd.DataFrame:
    df1 = _load_summary(w001_dir / "summary.csv").copy()
    df2 = _load_summary(w013_dir / "summary.csv").copy()
    df2["name"] = df2["name"].apply(lambda v: _offset_window_name(v, offset))
    combined = pd.concat([df1, df2], ignore_index=True)
    combined["window_idx"] = combined["name"].apply(lambda v: _parse_window_name(v)[0] if _parse_window_name(v) else None)
    combined = combined.sort_values(["window_idx"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    sources = {
        "w001_dir": str(w001_dir),
        "w013_dir": str(w013_dir),
        "offset": offset,
    }
    (out_dir / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    return combined


def _paired_compare(base: pd.DataFrame, var: pd.DataFrame, design_max_idx: int) -> pd.DataFrame:
    merged = base.merge(var, on=["test_start", "test_end"], how="inner", suffixes=("_base", "_var"))
    if merged.empty:
        raise ValueError("No matched windows between baseline and variant summaries.")
    merged["window_idx"] = merged["window_idx_base"].astype(int)
    merged["d_roi"] = merged["roi_var"] - merged["roi_base"]
    merged["d_max_dd"] = merged["max_drawdown_var"] - merged["max_drawdown_base"]
    merged["d_n_bets"] = merged["n_bets_var"] - merged["n_bets_base"]
    merged["split"] = merged["window_idx"].apply(lambda i: "design" if int(i) <= int(design_max_idx) else "eval")
    return merged.sort_values(["window_idx"]).reset_index(drop=True)


def _summarize_block(df: pd.DataFrame) -> dict:
    n = int(len(df))
    pos = int((df["d_roi"] > 0).sum()) if n else 0
    out = {
        "n_windows": n,
        "improve_rate_roi": (pos / n) if n else None,
        "median_d_roi": float(df["d_roi"].median()) if n else None,
        "median_d_maxdd": float(df["d_max_dd"].median()) if n else None,
        "median_d_n_bets": float(df["d_n_bets"].median()) if n else None,
        "median_n_bets_var": float(df["n_bets_var"].median()) if n else None,
        "median_roi_base": float(df["roi_base"].median()) if n else None,
        "median_roi_var": float(df["roi_var"].median()) if n else None,
        "median_maxdd_base": float(df["max_drawdown_base"].median()) if n else None,
        "median_maxdd_var": float(df["max_drawdown_var"].median()) if n else None,
    }
    return out


def _pooled_metrics(df: pd.DataFrame) -> dict:
    total_stake = float(df["total_stake"].sum()) if "total_stake" in df.columns else float("nan")
    total_profit = float(df["total_profit"].sum()) if "total_profit" in df.columns else float("nan")
    roi = (total_profit / total_stake) if total_stake > 0 else float("nan")
    n_bets = int(df["n_bets"].sum()) if "n_bets" in df.columns else 0
    test_start = str(df["test_start"].min()) if "test_start" in df.columns else None
    test_end = str(df["test_end"].max()) if "test_end" in df.columns else None
    return {
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": roi,
        "n_bets": n_bets,
        "test_start": test_start,
        "test_end": test_end,
    }


def _gate_eval(merged_eval: pd.DataFrame) -> dict:
    improve_rate = float((merged_eval["d_roi"] > 0).mean()) if len(merged_eval) else 0.0
    median_d_roi = float(merged_eval["d_roi"].median()) if len(merged_eval) else float("nan")
    median_d_maxdd = float(merged_eval["d_max_dd"].median()) if len(merged_eval) else float("nan")
    median_n_bets_var = float(merged_eval["n_bets_var"].median()) if len(merged_eval) else float("nan")
    zero_bet_windows = int((merged_eval["n_bets_var"] == 0).sum()) if len(merged_eval) else 0
    gate_pass = (
        improve_rate >= 0.6
        and median_d_roi > 0
        and median_d_maxdd <= 0
        and median_n_bets_var >= 80
        and zero_bet_windows == 0
    )
    return {
        "gate_pass": bool(gate_pass),
        "improve_rate_roi": improve_rate,
        "median_d_roi": median_d_roi,
        "median_d_maxdd": median_d_maxdd,
        "median_n_bets_var": median_n_bets_var,
        "zero_bet_windows": zero_bet_windows,
    }


def _run_step14(year: int, strategy: str, design_dir: Path, eval_dir: Path, out_dir: Path) -> Path:
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
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    return out_dir / "walkforward_step14_summary.csv"


def _read_step14_summary(path: Path) -> dict:
    df = pd.read_csv(path)
    out = {}
    for split in ("design", "eval", "all"):
        row = df[df["split"] == split]
        if row.empty:
            continue
        r = row.iloc[0]
        out[split] = {
            "roi": float(r["roi"]),
            "total_stake": float(r["total_stake"]),
            "total_profit": float(r["total_profit"]),
            "n_bets": int(r["n_bets"]),
            "frac_days_any_bet": float(r.get("frac_days_any_bet", np.nan)),
        }
    return out


@dataclass(frozen=True)
class YearConfig:
    year: int
    base_w001: Path
    base_w013: Path
    var_w001: Path
    var_w013: Path


def process_year(cfg: YearConfig, out_dir: Path, design_max_idx: int) -> dict:
    year_dir = out_dir / str(cfg.year)
    combined_root = year_dir / "combined_runs"
    base_combined = combined_root / "base"
    var_combined = combined_root / "pace_history"

    base_df = _combine_runs(cfg.base_w001, cfg.base_w013, base_combined, offset=design_max_idx)
    var_df = _combine_runs(cfg.var_w001, cfg.var_w013, var_combined, offset=design_max_idx)

    paired = _paired_compare(base_df, var_df, design_max_idx)
    paired.to_csv(year_dir / "paired_compare.csv", index=False, encoding="utf-8")

    summary = {
        "all": _summarize_block(paired),
        "design": _summarize_block(paired[paired["split"] == "design"]),
        "eval": _summarize_block(paired[paired["split"] == "eval"]),
    }
    (year_dir / "paired_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_gate = _gate_eval(paired[paired["split"] == "eval"])

    base_eval = base_df[base_df["window_idx"] > design_max_idx]
    var_eval = var_df[var_df["window_idx"] > design_max_idx]
    base_eval_pooled = _pooled_metrics(base_eval)
    var_eval_pooled = _pooled_metrics(var_eval)

    step14_base_path = _run_step14(cfg.year, "base", cfg.base_w001, cfg.base_w013, year_dir / "step14_base")
    step14_var_path = _run_step14(cfg.year, "pace_history", cfg.var_w001, cfg.var_w013, year_dir / "step14_pace_history")
    step14_base = _read_step14_summary(step14_base_path)
    step14_var = _read_step14_summary(step14_var_path)

    eval_gate_row = {
        "variant": "pace_history",
        **eval_gate,
        "median_d_n_bets": summary["eval"]["median_d_n_bets"],
        "pooled_roi_eval": var_eval_pooled["roi"],
        "pooled_stake_eval": var_eval_pooled["total_stake"],
        "pooled_profit_eval": var_eval_pooled["total_profit"],
    }
    pd.DataFrame([eval_gate_row]).to_csv(year_dir / "eval_gate_summary.csv", index=False, encoding="utf-8")

    report_lines = []
    report_lines.append(f"# pace_history {cfg.year} report")
    report_lines.append("")
    report_lines.append("## Pooled (eval) metrics")
    report_lines.append("")
    report_lines.append(f"- base roi={base_eval_pooled['roi']:.6f} stake={base_eval_pooled['total_stake']:.0f} n_bets={base_eval_pooled['n_bets']} test={base_eval_pooled['test_start']}..{base_eval_pooled['test_end']}")
    report_lines.append(f"- pace_history roi={var_eval_pooled['roi']:.6f} stake={var_eval_pooled['total_stake']:.0f} n_bets={var_eval_pooled['n_bets']} test={var_eval_pooled['test_start']}..{var_eval_pooled['test_end']}")
    report_lines.append("")
    report_lines.append("## Paired (eval) deltas")
    report_lines.append("")
    report_lines.append(f"- median_d_roi={eval_gate['median_d_roi']:.6f} improve_rate={eval_gate['improve_rate_roi']:.3f} median_d_maxdd={eval_gate['median_d_maxdd']:.6f} median_n_bets_var={eval_gate['median_n_bets_var']:.1f} zero_bet_windows={eval_gate['zero_bet_windows']}")
    report_lines.append(f"- gate_pass={eval_gate['gate_pass']}")
    report_lines.append("")
    report_lines.append("## Step14 non-overlap (eval)")
    report_lines.append("")
    base_step = step14_base.get("eval", {})
    var_step = step14_var.get("eval", {})
    if base_step and var_step:
        report_lines.append(f"- base roi={base_step['roi']:.6f} stake={base_step['total_stake']:.0f} n_bets={base_step['n_bets']} frac_days_any_bet={base_step.get('frac_days_any_bet', float('nan')):.3f}")
        report_lines.append(f"- pace_history roi={var_step['roi']:.6f} stake={var_step['total_stake']:.0f} n_bets={var_step['n_bets']} frac_days_any_bet={var_step.get('frac_days_any_bet', float('nan')):.3f}")
        if np.sign(base_eval_pooled["roi"]) != np.sign(base_step["roi"]):
            report_lines.append("- NOTE: pooled ROI sign differs from step14 ROI for base; prefer step14 for decisions.")
        if np.sign(var_eval_pooled["roi"]) != np.sign(var_step["roi"]):
            report_lines.append("- NOTE: pooled ROI sign differs from step14 ROI for pace_history; prefer step14 for decisions.")
    else:
        report_lines.append("- Step14 summary missing.")

    (year_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "year": cfg.year,
        "gate_pass": eval_gate["gate_pass"],
        "median_d_roi": eval_gate["median_d_roi"],
        "improve_rate_roi": eval_gate["improve_rate_roi"],
        "median_d_maxdd": eval_gate["median_d_maxdd"],
        "median_n_bets_var": eval_gate["median_n_bets_var"],
        "pooled_roi_eval": var_eval_pooled["roi"],
        "step14_roi_eval": var_step.get("roi") if var_step else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--design-max-idx", type=int, default=12)
    parser.add_argument("--base-2024", nargs=2, required=True)
    parser.add_argument("--var-2024", nargs=2, required=True)
    parser.add_argument("--base-2025", nargs=2, required=True)
    parser.add_argument("--var-2025", nargs=2, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_2024 = YearConfig(
        year=2024,
        base_w001=Path(args.base_2024[0]),
        base_w013=Path(args.base_2024[1]),
        var_w001=Path(args.var_2024[0]),
        var_w013=Path(args.var_2024[1]),
    )
    cfg_2025 = YearConfig(
        year=2025,
        base_w001=Path(args.base_2025[0]),
        base_w013=Path(args.base_2025[1]),
        var_w001=Path(args.var_2025[0]),
        var_w013=Path(args.var_2025[1]),
    )

    res_2024 = process_year(cfg_2024, out_dir, args.design_max_idx)
    res_2025 = process_year(cfg_2025, out_dir, args.design_max_idx)

    cross = pd.DataFrame([res_2024, res_2025])
    cross["variant"] = "pace_history"
    cross["decision"] = cross["gate_pass"].apply(lambda v: "pass" if v else "fail")
    cross.to_csv(out_dir / "cross_year_summary.csv", index=False, encoding="utf-8")

    pass_both = bool(res_2024["gate_pass"] and res_2025["gate_pass"])
    final_lines = []
    final_lines.append("# pace_history cross-year summary")
    final_lines.append("")
    final_lines.append("## 2024 eval")
    final_lines.append(
        f"- gate_pass={res_2024['gate_pass']} median_d_roi={res_2024['median_d_roi']:.6f} improve_rate={res_2024['improve_rate_roi']:.3f} median_d_maxdd={res_2024['median_d_maxdd']:.6f} median_n_bets_var={res_2024['median_n_bets_var']:.1f}"
    )
    final_lines.append("## 2025 eval")
    final_lines.append(
        f"- gate_pass={res_2025['gate_pass']} median_d_roi={res_2025['median_d_roi']:.6f} improve_rate={res_2025['improve_rate_roi']:.3f} median_d_maxdd={res_2025['median_d_maxdd']:.6f} median_n_bets_var={res_2025['median_n_bets_var']:.1f}"
    )
    final_lines.append("")
    final_lines.append("## Decision")
    final_lines.append(f"- pass_both={pass_both}")
    (out_dir / "final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
