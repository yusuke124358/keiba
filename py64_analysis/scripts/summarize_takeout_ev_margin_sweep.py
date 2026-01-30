"""
Summarize takeout EV margin sweep across 2024/2025 eval-only rolling runs.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

WIN_RE = re.compile(r"^w(\d{3})_")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _variant_takeout_ev_margin(run_dirs: list[Path]) -> dict[str, Any]:
    for run_dir in run_dirs:
        for w in sorted(run_dir.glob("w*/summary.json")):
            summary = _read_json(w)
            tem = summary.get("takeout_ev_margin") or {}
            return {
                "enabled": bool(tem.get("enabled", False)),
                "ref_takeout": tem.get("ref_takeout"),
                "slope": tem.get("slope"),
            }
    return {"enabled": None, "ref_takeout": None, "slope": None}


def _decision(pass_2024: bool, pass_2025: bool) -> str:
    if pass_2024 and pass_2025:
        return "pass_both"
    if pass_2024 or pass_2025:
        return "pass_single_year"
    return "fail"


def _fmt(val, digits: int = 4) -> str:
    if val is None:
        return "N/A (eval-only)"
    if isinstance(val, float) and pd.isna(val):
        return "N/A (eval-only)"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize takeout EV margin sweep (2024/2025)")
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
    scarcity_rows = []
    ev_lift_summary: dict[str, dict[str, Any]] = {}
    cross_year_rows = []
    final_report_lines = []

    final_report_lines.append("# Takeout EV margin sweep (2024/2025)")
    final_report_lines.append("")
    final_report_lines.append("- Split: time-based (train < valid < test), eval-only windows (design metrics are N/A).")
    final_report_lines.append("")

    variant_names: list[str] = []
    gate_by_year_variant: dict[str, dict[str, Any]] = {"2024": {}, "2025": {}}

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

        for variant_name, run_list in variants.items():
            if variant_name == baseline_name:
                continue
            if variant_name not in variant_names:
                variant_names.append(variant_name)

            var_dirs = [Path(p) for p in run_list]
            if not var_dirs:
                continue

            base_df = _combine_summaries(base_dirs, design_max_idx=args.design_max_idx)
            var_df = _combine_summaries(var_dirs, design_max_idx=args.design_max_idx)

            paired = _paired_compare(base_df, var_df)
            gate = _eval_gate_metrics(paired)

            base_metrics = _aggregate_year_metrics(base_df)
            var_metrics = _aggregate_year_metrics(var_df)
            tem = _variant_takeout_ev_margin(var_dirs)
            slope = tem.get("slope")

            eval_gate_rows.append(
                {
                    "year": int(year),
                    "variant": variant_name,
                    "slope": slope,
                    **gate,
                }
            )

            gate_by_year_variant[year][variant_name] = gate

            scarcity = _summarize_candidate_scarcity(var_dirs)
            if scarcity:
                scarcity_rows.append(
                    {
                        "year": int(year),
                        "variant": variant_name,
                        "slope": slope,
                        **scarcity,
                    }
                )

            if args.ensure_ev_lift:
                ev_paths = []
                for run_dir in var_dirs:
                    ev_paths.append(_ensure_ev_lift(run_dir, ev_design_max_idx=int(args.ev_design_max_idx)))
                ev_summary = _load_ev_lift_metrics(ev_paths[-1])
            else:
                ev_path = var_dirs[-1] / "ev_lift_summary.json"
                ev_summary = _load_ev_lift_metrics(ev_path) if ev_path.exists() else {"signal_rate": None, "decile10_median_roi": None, "raw": {}}

            ev_lift_summary.setdefault(year, {})[variant_name] = ev_summary.get("raw")

            final_report_lines.append("")
            final_report_lines.append(f"Variant: {variant_name} (slope={slope})")
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

            final_report_lines.append(
                "- N5: signal_rate={signal_rate} | decile10_median_roi={decile10}".format(
                    signal_rate=_fmt(ev_summary.get("signal_rate")),
                    decile10=_fmt(ev_summary.get("decile10_median_roi")),
                )
            )

            print(
                f"[takeout_ev_margin {year} eval] slope={slope} gate_pass={gate.get('gate_pass')} | "
                f"median_d_roi={gate.get('median_d_roi')} | improve_rate={gate.get('improve_rate')} | "
                f"median_d_maxdd={gate.get('median_d_maxdd')} | median_n_bets_var={gate.get('median_n_bets_var')} | "
                f"n_zero={gate.get('zero_bet_windows')}"
            )

    for variant_name in variant_names:
        pass_2024 = bool(gate_by_year_variant.get("2024", {}).get(variant_name, {}).get("gate_pass"))
        pass_2025 = bool(gate_by_year_variant.get("2025", {}).get(variant_name, {}).get("gate_pass"))
        decision = _decision(pass_2024, pass_2025)
        slope = None
        for row in eval_gate_rows:
            if row.get("variant") == variant_name and row.get("slope") is not None:
                slope = row.get("slope")
                break
        cross_year_rows.append(
            {
                "variant": variant_name,
                "slope": slope,
                "pass_2024": pass_2024,
                "pass_2025": pass_2025,
                "decision": decision,
            }
        )

    cross_df = pd.DataFrame(cross_year_rows)
    if not cross_df.empty:
        pass_both = int((cross_df["decision"] == "pass_both").sum())
        pass_single = int((cross_df["decision"] == "pass_single_year").sum())
        fail_count = int((cross_df["decision"] == "fail").sum())
    else:
        pass_both = 0
        pass_single = 0
        fail_count = 0

    print(f"[cross-year] pass_both_count={pass_both} | pass_single_year_count={pass_single} | fail_count={fail_count}")

    pd.DataFrame(eval_gate_rows).to_csv(out_dir / "eval_gate_summary.csv", index=False)
    pd.DataFrame(scarcity_rows).to_csv(out_dir / "candidate_scarcity.csv", index=False)
    cross_df.to_csv(out_dir / "cross_year_summary.csv", index=False)
    (out_dir / "ev_lift_summary.json").write_text(json.dumps(ev_lift_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")

    shutil.copy2(args.manifest_2024, out_dir / "manifest_2024.json")
    shutil.copy2(args.manifest_2025, out_dir / "manifest_2025.json")


if __name__ == "__main__":
    main()
