from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


WIN_RE = r"^w(\d{3})_(\d{8})_(\d{8})$"


def _read_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def _merge_base_var(base_df: pd.DataFrame, var_df: pd.DataFrame) -> pd.DataFrame:
    if base_df.empty or var_df.empty:
        return pd.DataFrame()
    key_cols = ["test_start", "test_end"]
    merged = base_df.merge(var_df, on=key_cols, suffixes=("_base", "_var"))
    merged["d_roi"] = merged["roi_var"] - merged["roi_base"]
    merged["d_maxdd"] = merged["max_drawdown_var"] - merged["max_drawdown_base"]
    merged["d_n_bets"] = merged["n_bets_var"] - merged["n_bets_base"]
    merged["d_stake"] = merged["total_stake_var"] - merged["total_stake_base"]
    merged["d_profit"] = merged["total_profit_var"] - merged["total_profit_base"]
    merged["split"] = "eval"
    return merged


def _eval_gate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "median_d_roi": float("nan"),
            "improve_rate_roi": float("nan"),
            "median_d_maxdd": float("nan"),
            "median_n_bets_var": float("nan"),
            "zero_bet_windows": 0,
            "gate_pass": False,
        }
    d_roi = df["d_roi"].dropna()
    d_maxdd = df["d_maxdd"].dropna()
    med_d_roi = float(np.median(d_roi)) if len(d_roi) else float("nan")
    improve_rate = float((d_roi > 0).mean()) if len(d_roi) else float("nan")
    med_d_maxdd = float(np.median(d_maxdd)) if len(d_maxdd) else float("nan")
    med_n_bets = float(np.median(df["n_bets_var"])) if "n_bets_var" in df.columns else float("nan")
    zero_bet_windows = int((df["n_bets_var"] == 0).sum()) if "n_bets_var" in df.columns else 0
    gate_pass = bool(
        improve_rate >= 0.6
        and med_d_roi > 0
        and med_d_maxdd <= 0
        and med_n_bets >= 80
        and zero_bet_windows == 0
    )
    return {
        "median_d_roi": med_d_roi,
        "improve_rate_roi": improve_rate,
        "median_d_maxdd": med_d_maxdd,
        "median_n_bets_var": med_n_bets,
        "zero_bet_windows": zero_bet_windows,
        "gate_pass": gate_pass,
    }


def _read_config_bankroll(window_dir: Path) -> float:
    cfg = window_dir / "config_used.yaml"
    if not cfg.exists():
        return 1_000_000.0
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except Exception:
        return 1_000_000.0
    try:
        return float(data.get("betting", {}).get("bankroll_yen", 1_000_000.0))
    except Exception:
        return 1_000_000.0


def _run_step14(run_dir: Path, year: int, label: str, out_dir: Path, bankroll: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(Path("py64_analysis/.venv/Scripts/python.exe")),
        str(Path("py64_analysis/scripts/summarize_walkforward_step14_from_rolling.py")),
        "--year",
        str(year),
        "--strategy",
        label,
        "--design-dir",
        str(run_dir),
        "--eval-dir",
        str(run_dir),
        "--step-days",
        "14",
        "--initial-bankroll",
        str(bankroll),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=False)
    return out_dir / "walkforward_step14_summary.csv"


def _read_step14_eval(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    eval_row = df[df["split"] == "eval"]
    if eval_row.empty:
        return {}
    row = eval_row.iloc[0].to_dict()
    return row


def _binding_diagnosis(base_dir: Path, var_dir: Path, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in paired.iterrows():
        win = r.get("name_base") or r.get("name_var") or ""
        if not win:
            continue
        base_bets = base_dir / str(win) / "bets.csv"
        var_bets = var_dir / str(win) / "bets.csv"
        if not base_bets.exists() or not var_bets.exists():
            continue
        bdf = pd.read_csv(base_bets)
        vdf = pd.read_csv(var_bets)
        key_cols = ["race_id", "horse_no", "asof_time"]
        for c in key_cols:
            if c not in bdf.columns or c not in vdf.columns:
                return pd.DataFrame()
        bdf["key"] = bdf[key_cols].astype(str).agg("|".join, axis=1)
        vdf["key"] = vdf[key_cols].astype(str).agg("|".join, axis=1)
        b_keys = set(bdf["key"].tolist())
        v_keys = set(vdf["key"].tolist())
        filtered = b_keys - v_keys
        added = v_keys - b_keys
        b_stake = bdf.set_index("key")["stake"].to_dict()
        v_stake = vdf.set_index("key")["stake"].to_dict()
        filtered_stake = float(sum(b_stake.get(k, 0.0) for k in filtered))
        added_stake = float(sum(v_stake.get(k, 0.0) for k in added))
        rows.append(
            {
                "window": win,
                "base_n_bets": int(len(b_keys)),
                "var_n_bets": int(len(v_keys)),
                "filtered_bets": int(len(filtered)),
                "added_bets": int(len(added)),
                "filtered_stake": filtered_stake,
                "added_stake": added_stake,
                "identical_passset": bool(len(filtered) == 0 and len(added) == 0),
            }
        )
    return pd.DataFrame(rows)


def _no_op_audit(base_dir: Path, var_dir: Path, window_name: str) -> dict[str, Any]:
    base_bets = base_dir / window_name / "bets.csv"
    var_bets = var_dir / window_name / "bets.csv"
    if not base_bets.exists() or not var_bets.exists():
        return {"status": "missing"}
    bdf = pd.read_csv(base_bets)
    vdf = pd.read_csv(var_bets)
    key_cols = ["race_id", "horse_no", "asof_time", "stake", "odds_at_buy"]
    for c in key_cols:
        if c not in bdf.columns or c not in vdf.columns:
            return {"status": "missing_cols"}
    bdf["key"] = bdf[key_cols].astype(str).agg("|".join, axis=1)
    vdf["key"] = vdf[key_cols].astype(str).agg("|".join, axis=1)
    b_hash = hashlib.md5("".join(sorted(bdf["key"].tolist())).encode("utf-8")).hexdigest()
    v_hash = hashlib.md5("".join(sorted(vdf["key"].tolist())).encode("utf-8")).hexdigest()
    return {
        "status": "ok",
        "base_rows": int(len(bdf)),
        "var_rows": int(len(vdf)),
        "hash_match": bool(b_hash == v_hash),
    }


def _candidate_scarcity(run_dir: Path) -> dict[str, Any]:
    rows = []
    for w in run_dir.iterdir():
        if not w.is_dir():
            continue
        m = re.match(WIN_RE, w.name)
        if not m:
            continue
        test_start = datetime.strptime(m.group(2), "%Y%m%d").date()
        test_end = datetime.strptime(m.group(3), "%Y%m%d").date()
        bets_path = w / "bets.csv"
        if not bets_path.exists():
            continue
        df = pd.read_csv(bets_path)
        if "asof_time" not in df.columns:
            continue
        df["bet_date"] = pd.to_datetime(df["asof_time"], errors="coerce").dt.date
        df = df[(df["bet_date"] >= test_start) & (df["bet_date"] <= test_end)]
        if df.empty:
            rows.append(
                {
                    "window": w.name,
                    "days_total": int((test_end - test_start).days + 1),
                    "days_with_bet": 0,
                    "days_bets_ge_5": 0,
                }
            )
            continue
        daily = df.groupby("bet_date")["stake"].count()
        days_total = int((test_end - test_start).days + 1)
        rows.append(
            {
                "window": w.name,
                "days_total": days_total,
                "days_with_bet": int((daily > 0).sum()),
                "days_bets_ge_5": int((daily >= 5).sum()),
            }
        )
    if not rows:
        return {}
    dfr = pd.DataFrame(rows)
    frac_any = float((dfr["days_with_bet"] / dfr["days_total"]).median())
    frac_ge5 = float((dfr["days_bets_ge_5"] / dfr["days_total"]).median())
    return {"frac_days_any_bet": frac_any, "frac_days_bets_ge_5": frac_ge5}


def _summarize_year(year: int, base_dir: Path, var_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_df = _read_summary(base_dir)
    var_df = _read_summary(var_dir)
    paired = _merge_base_var(base_df, var_df)
    if paired.empty:
        return {"year": year, "status": "missing"}

    paired.to_csv(out_dir / "paired_compare.csv", index=False)
    summary = {
        "year": year,
        "all": {"note": "N/A (eval-only)"},
        "design": {"note": "N/A (eval-only)"},
        "eval": {
            "n_windows": int(len(paired)),
            "roi_base": float((paired["total_profit_base"].sum() / paired["total_stake_base"].sum()) if paired["total_stake_base"].sum() else float("nan")),
            "roi_var": float((paired["total_profit_var"].sum() / paired["total_stake_var"].sum()) if paired["total_stake_var"].sum() else float("nan")),
            "stake_base": float(paired["total_stake_base"].sum()),
            "stake_var": float(paired["total_stake_var"].sum()),
            "n_bets_base": int(paired["n_bets_base"].sum()),
            "n_bets_var": int(paired["n_bets_var"].sum()),
            "median_maxdd_base": float(np.median(paired["max_drawdown_base"])),
            "median_maxdd_var": float(np.median(paired["max_drawdown_var"])),
        },
    }
    (out_dir / "paired_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_gate = _eval_gate_metrics(paired)
    eval_gate_row = {
        "year": year,
        **eval_gate,
        "pooled_roi_eval_base": summary["eval"]["roi_base"],
        "pooled_roi_eval_var": summary["eval"]["roi_var"],
        "test_start": str(paired["test_start"].min()),
        "test_end": str(paired["test_end"].max()),
    }
    pd.DataFrame([eval_gate_row]).to_csv(out_dir / "eval_gate_summary.csv", index=False, encoding="utf-8")

    binding = _binding_diagnosis(base_dir, var_dir, paired)
    binding_path = out_dir / "binding_diagnosis.csv"
    if not binding.empty:
        binding.to_csv(binding_path, index=False, encoding="utf-8")

    sample_window = paired.iloc[0].get("name_base") or paired.iloc[0].get("name_var")
    no_op = _no_op_audit(base_dir, var_dir, str(sample_window))
    (out_dir / "no_op_audit.json").write_text(json.dumps(no_op, ensure_ascii=False, indent=2), encoding="utf-8")

    bankroll = _read_config_bankroll(base_dir / paired.iloc[0]["name_base"])
    step14_base = _read_step14_eval(_run_step14(base_dir, year, "base", out_dir / "step14_base", bankroll))
    step14_var = _read_step14_eval(_run_step14(var_dir, year, "odds_dyn", out_dir / "step14_odds_dyn", bankroll))

    scarcity_base = _candidate_scarcity(base_dir)
    scarcity_var = _candidate_scarcity(var_dir)

    report_lines = []
    report_lines.append(f"# Odds Dynamics eval summary ({year})")
    report_lines.append("")
    report_lines.append(f"- Test period: {eval_gate_row['test_start']} to {eval_gate_row['test_end']}")
    report_lines.append(
        f"- Pooled ROI (base)={summary['eval']['roi_base']:.6f} "
        f"(var)={summary['eval']['roi_var']:.6f}"
    )
    report_lines.append(
        f"- Stake (base)={summary['eval']['stake_base']:.0f} "
        f"(var)={summary['eval']['stake_var']:.0f} | "
        f"n_bets (base)={summary['eval']['n_bets_base']} (var)={summary['eval']['n_bets_var']}"
    )
    report_lines.append(
        f"- Median MaxDD (base)={summary['eval']['median_maxdd_base']:.6f} "
        f"(var)={summary['eval']['median_maxdd_var']:.6f}"
    )
    if step14_base and step14_var:
        report_lines.append(f"- Step14 ROI (base)={step14_base.get('roi')} (var)={step14_var.get('roi')}")
        if (summary['eval']['roi_base'] >= 0) != (step14_base.get('roi', 0) >= 0):
            report_lines.append("- NOTE: pooled ROI sign differs from step14 ROI for base; prefer step14.")
        if (summary['eval']['roi_var'] >= 0) != (step14_var.get('roi', 0) >= 0):
            report_lines.append("- NOTE: pooled ROI sign differs from step14 ROI for variant; prefer step14.")
    report_lines.append("")
    report_lines.append(
        f"- median_d_roi={eval_gate['median_d_roi']:.6f} improve_rate={eval_gate['improve_rate_roi']:.3f} "
        f"median_d_maxdd={eval_gate['median_d_maxdd']:.6f} median_n_bets_var={eval_gate['median_n_bets_var']:.1f} "
        f"zero_bet_windows={eval_gate['zero_bet_windows']}"
    )
    report_lines.append(f"- gate_pass={eval_gate['gate_pass']}")
    report_lines.append("")
    if not binding.empty:
        identical = int(binding["identical_passset"].sum())
        report_lines.append(f"- binding: identical_passset_windows={identical} / {len(binding)}")
        report_lines.append(f"- binding: filtered_bets_total={int(binding['filtered_bets'].sum())}")
        report_lines.append(f"- binding: discrete_thresholds_detected={str(identical > 0).lower()}")
    report_lines.append("")
    if no_op.get("status") == "ok":
        report_lines.append(
            f"- no_op_audit(sample): rows_base={no_op.get('base_rows')} rows_var={no_op.get('var_rows')} "
            f"hash_match={no_op.get('hash_match')}"
        )
        report_lines.append("")
    if scarcity_base:
        report_lines.append(f"- scarcity(base): frac_days_any_bet={scarcity_base.get('frac_days_any_bet'):.3f} frac_days_bets_ge_5={scarcity_base.get('frac_days_bets_ge_5'):.3f}")
    if scarcity_var:
        report_lines.append(f"- scarcity(var): frac_days_any_bet={scarcity_var.get('frac_days_any_bet'):.3f} frac_days_bets_ge_5={scarcity_var.get('frac_days_bets_ge_5'):.3f}")
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "year": year,
        "eval_gate": eval_gate_row,
        "binding": binding,
        "no_op": no_op,
        "step14_base": step14_base,
        "step14_var": step14_var,
        "scarcity_base": scarcity_base,
        "scarcity_var": scarcity_var,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-2024", required=True)
    p.add_argument("--var-2024", required=True)
    p.add_argument("--base-2025", required=True)
    p.add_argument("--var-2025", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res24 = _summarize_year(2024, Path(args.base_2024), Path(args.var_2024), out_dir / "2024")
    res25 = _summarize_year(2025, Path(args.base_2025), Path(args.var_2025), out_dir / "2025")

    cross = {
        "pass_2024": bool(res24.get("eval_gate", {}).get("gate_pass")),
        "pass_2025": bool(res25.get("eval_gate", {}).get("gate_pass")),
    }
    cross["decision"] = "pass_both" if (cross["pass_2024"] and cross["pass_2025"]) else "fail"
    pd.DataFrame(
        [
            {
                "variant": "odds_dyn",
                "pass_2024": cross["pass_2024"],
                "pass_2025": cross["pass_2025"],
                "decision": cross["decision"],
            }
        ]
    ).to_csv(out_dir / "cross_year_summary.csv", index=False, encoding="utf-8")

    final_lines = []
    final_lines.append("# Odds Dynamics eval-only cross-year summary")
    final_lines.append("")
    final_lines.append(f"- 2024 gate_pass={cross['pass_2024']}")
    final_lines.append(f"- 2025 gate_pass={cross['pass_2025']}")
    final_lines.append(f"- decision={cross['decision']}")
    final_lines.append("")
    final_lines.append("Notes:")
    final_lines.append("- Eval-only windows: design metrics are N/A.")
    (out_dir / "final_report.md").write_text("\n".join(final_lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
