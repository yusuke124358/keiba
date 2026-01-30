"""Selector summary for odds_dyn_ev_margin (baseline vs s05/s10) using valid-only metrics."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from keiba.analysis.metrics_utils import sign_mismatch
from keiba.analysis.selector_utils import is_eligible, shrink_delta


RUN_TS_RE = re.compile(r"_(\d{8})_(\d{6})$")
WIN_RE = re.compile(r"^w(\d{3})_(\d{8})_(\d{8})$")


@dataclass(frozen=True)
class RunSpec:
    year: int
    variant: str
    run_dir: Path
    timestamp: str


def _parse_timestamp(name: str) -> str:
    m = RUN_TS_RE.search(name)
    if not m:
        return "00000000_000000"
    return f"{m.group(1)}_{m.group(2)}"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _detect_variant_from_cfg(cfg: dict) -> Optional[str]:
    betting = (cfg or {}).get("betting") or {}
    odm = betting.get("odds_dyn_ev_margin") or {}
    enabled = bool(odm.get("enabled", False))
    slope = odm.get("slope", 0.0)
    try:
        slope = float(slope)
    except Exception:
        slope = 0.0
    if not enabled or slope <= 0.0:
        return "base"
    if abs(slope - 0.5) < 1e-6:
        return "s05"
    if abs(slope - 1.0) < 1e-6:
        return "s10"
    return None


def _find_first_window(run_dir: Path) -> Optional[Path]:
    windows = [p for p in run_dir.iterdir() if p.is_dir() and WIN_RE.match(p.name)]
    if not windows:
        return None
    windows.sort(key=lambda p: p.name)
    return windows[0]


def _discover_runs(data_dir: Path, years: list[int]) -> dict[tuple[int, str], RunSpec]:
    found: dict[tuple[int, str], RunSpec] = {}
    for run_dir in data_dir.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if "w013_022" not in name or "eval" not in name:
            continue
        year = None
        for y in years:
            if f"_{y}_" in name:
                year = y
                break
        if year is None:
            continue
        first_win = _find_first_window(run_dir)
        if not first_win:
            continue
        cfg = _read_yaml(first_win / "config_used.yaml")
        variant = _detect_variant_from_cfg(cfg)
        if variant is None:
            continue
        ts = _parse_timestamp(name)
        key = (year, variant)
        if key not in found or ts > found[key].timestamp:
            found[key] = RunSpec(year=year, variant=variant, run_dir=run_dir, timestamp=ts)
    return found


def _load_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "name" in df.columns:
        df = df.rename(columns={"name": "window"})
    return df


def _extract_valid_metrics(summary_json: dict) -> tuple[dict[str, Any], str]:
    valid = summary_json.get("valid_backtest")
    if not isinstance(valid, dict):
        return {}, "missing_valid_backtest"
    required = ["roi", "n_bets", "total_stake", "max_drawdown"]
    if not all(k in valid for k in required):
        return {}, "missing_valid_fields"
    return valid, "valid_backtest"


def _load_window_metrics(run_dir: Path) -> dict[str, dict[str, Any]]:
    metrics = {}
    for w in run_dir.iterdir():
        if not w.is_dir() or not WIN_RE.match(w.name):
            continue
        summary_path = w / "summary.json"
        if not summary_path.exists():
            metrics[w.name] = {"valid_status": "missing_summary_json"}
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        valid, status = _extract_valid_metrics(data)
        metrics[w.name] = {
            "valid_status": status if not valid else "ok",
            "valid_roi": valid.get("roi"),
            "valid_n_bets": valid.get("n_bets"),
            "valid_stake": valid.get("total_stake"),
            "valid_maxdd": valid.get("max_drawdown"),
            "valid_profit": valid.get("total_profit"),
        }
    return metrics


def _candidate_scarcity(run_dir: Path) -> dict[str, float]:
    rows = []
    for w in run_dir.iterdir():
        if not w.is_dir() or not WIN_RE.match(w.name):
            continue
        m = WIN_RE.match(w.name)
        test_start = pd.to_datetime(m.group(2)).date()
        test_end = pd.to_datetime(m.group(3)).date()
        bets_path = w / "bets.csv"
        if not bets_path.exists():
            continue
        df = pd.read_csv(bets_path)
        if "asof_time" not in df.columns:
            continue
        df["bet_date"] = pd.to_datetime(df["asof_time"], errors="coerce").dt.date
        df = df[(df["bet_date"] >= test_start) & (df["bet_date"] <= test_end)]
        daily = df.groupby("bet_date")["stake"].count()
        days_total = int((test_end - test_start).days + 1)
        rows.append(
            {
                "days_total": days_total,
                "days_with_bet": int((daily > 0).sum()),
                "days_bets_ge_5": int((daily >= 5).sum()),
            }
        )
    if not rows:
        return {"frac_days_any_bet": None, "frac_days_bets_ge_5": None}
    dfr = pd.DataFrame(rows)
    return {
        "frac_days_any_bet": float((dfr["days_with_bet"] / dfr["days_total"]).median()),
        "frac_days_bets_ge_5": float((dfr["days_bets_ge_5"] / dfr["days_total"]).median()),
    }


def _copy_selector_bets(out_dir: Path, chosen_rows: list[dict]) -> Path:
    sel_dir = out_dir / "selector_run"
    if sel_dir.exists():
        shutil.rmtree(sel_dir)
    sel_dir.mkdir(parents=True, exist_ok=True)
    for row in chosen_rows:
        src = Path(row["run_dir"]) / row["window"] / "bets.csv"
        if not src.exists():
            continue
        dst_dir = sel_dir / row["window"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "bets.csv")
    return sel_dir


def _summarize_step14(run_dir: Path, year: int, strategy: str, out_dir: Path, initial_bankroll: float) -> Path:
    cmd = [
        str(Path("py64_analysis/.venv/Scripts/python.exe")),
        str(Path("py64_analysis/scripts/summarize_walkforward_step14_from_rolling.py")),
        "--year",
        str(year),
        "--strategy",
        strategy,
        "--design-dir",
        str(run_dir),
        "--eval-dir",
        str(run_dir),
        "--step-days",
        "14",
        "--initial-bankroll",
        str(initial_bankroll),
        "--out-dir",
        str(out_dir),
    ]
    import subprocess

    subprocess.run(cmd, check=False)
    return out_dir / "walkforward_step14_summary.csv"


def _read_step14_eval(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    eval_row = df[df["split"] == "eval"]
    if eval_row.empty:
        return {}
    return eval_row.iloc[0].to_dict()


def _read_bankroll(run_dir: Path) -> float:
    first_win = _find_first_window(run_dir)
    if not first_win:
        return 1_000_000.0
    cfg = _read_yaml(first_win / "config_used.yaml")
    try:
        return float((cfg.get("betting") or {}).get("bankroll_yen", 1_000_000.0))
    except Exception:
        return 1_000_000.0


def _binding_added_removed(base_dir: Path, var_dir: Path) -> pd.DataFrame:
    rows = []
    for w in base_dir.iterdir():
        if not w.is_dir() or not WIN_RE.match(w.name):
            continue
        base_bets = w / "bets.csv"
        var_bets = var_dir / w.name / "bets.csv"
        if not base_bets.exists() or not var_bets.exists():
            continue
        bdf = pd.read_csv(base_bets)
        vdf = pd.read_csv(var_bets)
        key_cols = ["race_id", "horse_no", "asof_time"]
        if any(c not in bdf.columns for c in key_cols) or any(c not in vdf.columns for c in key_cols):
            continue
        bdf["key"] = bdf[key_cols].astype(str).agg("|".join, axis=1)
        vdf["key"] = vdf[key_cols].astype(str).agg("|".join, axis=1)
        b_keys = set(bdf["key"].tolist())
        v_keys = set(vdf["key"].tolist())
        filtered = b_keys - v_keys
        added = v_keys - b_keys
        common = b_keys & v_keys
        b_stake = bdf.set_index("key")["stake"].to_dict() if "stake" in bdf.columns else {}
        v_stake = vdf.set_index("key")["stake"].to_dict() if "stake" in vdf.columns else {}
        rows.append(
            {
                "window": w.name,
                "base_n_bets": int(len(b_keys)),
                "var_n_bets": int(len(v_keys)),
                "filtered_bets": int(len(filtered)),
                "added_bets": int(len(added)),
                "common_bets": int(len(common)),
                "filtered_stake": float(sum(b_stake.get(k, 0.0) for k in filtered)),
                "added_stake": float(sum(v_stake.get(k, 0.0) for k in added)),
                "common_stake": float(sum(b_stake.get(k, 0.0) for k in common)),
                "identical_passset": bool(len(filtered) == 0 and len(added) == 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data/holdout_runs")
    ap.add_argument("--min-valid-bets", type=int, default=30)
    ap.add_argument("--min-valid-bets-ratio", type=float, default=0.50)
    ap.add_argument("--n0", type=float, default=200.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    found = _discover_runs(data_dir, args.years)
    missing = []
    for year in args.years:
        for variant in ("base", "s05", "s10"):
            if (year, variant) not in found:
                missing.append({"year": year, "variant": variant})
    if missing:
        (out_dir / "missing_runs.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    selector_rows = []
    valid_candidate_rows = []
    missing_valid_windows = []
    eval_rows = []
    scarcity_rows = []
    step14_rows = []
    selector_stats_rows = []
    cross_rows = []
    added_removed_rows = []

    for year in args.years:
        runs = {v: found[(year, v)] for v in ("base", "s05", "s10")}
        summary = {v: _load_summary(r.run_dir) for v, r in runs.items()}
        valid_metrics = {v: _load_window_metrics(r.run_dir) for v, r in runs.items()}

        windows = sorted(set(summary["base"]["window"].tolist()))
        baseline_valid_bets = []
        eligible_non_base_windows = 0

        for win in windows:
            base_valid = valid_metrics["base"].get(win, {})
            b_valid_roi = base_valid.get("valid_roi")
            b_valid_bets = base_valid.get("valid_n_bets")
            b_valid_status = base_valid.get("valid_status")
            if b_valid_bets is not None:
                baseline_valid_bets.append(b_valid_bets)

            choices = []
            for v in ("base", "s05", "s10"):
                vm = valid_metrics[v].get(win, {})
                v_roi = vm.get("valid_roi")
                v_bets = vm.get("valid_n_bets")
                status = vm.get("valid_status")
                eligible, reason = is_eligible(
                    v_bets,
                    b_valid_bets,
                    args.min_valid_bets,
                    args.min_valid_bets_ratio,
                )
                if status and status != "ok":
                    eligible = False
                    reason = status
                if b_valid_bets is None:
                    eligible = False
                    reason = "missing_base_valid_bets"
                if v_roi is None or b_valid_roi is None:
                    eligible = False
                    reason = "missing_valid_roi"
                delta = None
                delta_shrunk = None
                if eligible and b_valid_roi is not None and v_roi is not None and v_bets is not None:
                    delta = float(v_roi) - float(b_valid_roi)
                    delta_shrunk = shrink_delta(delta, v_bets, args.n0)
                choices.append(
                    {
                        "variant": v,
                        "eligible": bool(eligible),
                        "eligible_reason": reason,
                        "valid_roi": v_roi,
                        "valid_n_bets": v_bets,
                        "valid_stake": vm.get("valid_stake"),
                        "valid_maxdd": vm.get("valid_maxdd"),
                        "delta_valid_roi": delta,
                        "delta_valid_roi_shrunk": delta_shrunk,
                    }
                )
                valid_candidate_rows.append(
                    {
                        "year": year,
                        "window": win,
                        "variant": v,
                        "valid_status": status,
                        "valid_roi": v_roi,
                        "valid_n_bets": v_bets,
                        "valid_stake": vm.get("valid_stake"),
                        "valid_profit": vm.get("valid_profit"),
                        "valid_maxdd": vm.get("valid_maxdd"),
                    }
                )

            if any(c["eligible"] for c in choices if c["variant"] != "base"):
                eligible_non_base_windows += 1

            chosen = "base"
            eligible_choices = [c for c in choices if c["eligible"] and c["delta_valid_roi_shrunk"] is not None]
            if eligible_choices:
                eligible_choices.sort(key=lambda c: c["delta_valid_roi_shrunk"], reverse=True)
                chosen = eligible_choices[0]["variant"]

            def _get_test_metrics(df: pd.DataFrame, window: str) -> dict[str, Any]:
                if df.empty:
                    return {}
                row = df[df["window"] == window]
                if row.empty:
                    return {}
                return row.iloc[0].to_dict()

            base_test = _get_test_metrics(summary["base"], win)
            chosen_test = _get_test_metrics(summary[chosen], win)

            selector_rows.append(
                {
                    "year": year,
                    "window": win,
                    "chosen": chosen,
                    "base_valid_roi": b_valid_roi,
                    "base_valid_n_bets": b_valid_bets,
                    "base_valid_status": b_valid_status,
                    "s05_valid_roi": choices[1]["valid_roi"],
                    "s05_valid_n_bets": choices[1]["valid_n_bets"],
                    "s05_eligible": choices[1]["eligible"],
                    "s05_eligible_reason": choices[1]["eligible_reason"],
                    "s05_delta_valid_roi_shrunk": choices[1]["delta_valid_roi_shrunk"],
                    "s10_valid_roi": choices[2]["valid_roi"],
                    "s10_valid_n_bets": choices[2]["valid_n_bets"],
                    "s10_eligible": choices[2]["eligible"],
                    "s10_eligible_reason": choices[2]["eligible_reason"],
                    "s10_delta_valid_roi_shrunk": choices[2]["delta_valid_roi_shrunk"],
                }
            )

            if b_valid_roi is None or b_valid_bets is None:
                missing_valid_windows.append({"year": year, "window": win, "reason": b_valid_status})
            for v in ("base", "s05", "s10"):
                vm = valid_metrics[v].get(win, {})
                if vm.get("valid_status") not in (None, "ok"):
                    missing_valid_windows.append(
                        {"year": year, "window": win, "variant": v, "reason": vm.get("valid_status")}
                    )

            eval_rows.append(
                {
                    "year": year,
                    "window": win,
                    "chosen": chosen,
                    "base_roi": base_test.get("roi"),
                    "base_n_bets": base_test.get("n_bets"),
                    "base_stake": base_test.get("total_stake"),
                    "base_profit": base_test.get("total_profit"),
                    "base_maxdd": base_test.get("max_drawdown"),
                    "var_roi": chosen_test.get("roi"),
                    "var_n_bets": chosen_test.get("n_bets"),
                    "var_stake": chosen_test.get("total_stake"),
                    "var_profit": chosen_test.get("total_profit"),
                    "var_maxdd": chosen_test.get("max_drawdown"),
                }
            )

        if baseline_valid_bets:
            baseline_stats = {
                "min": float(np.min(baseline_valid_bets)),
                "median": float(np.median(baseline_valid_bets)),
                "p90": float(np.percentile(baseline_valid_bets, 90)),
                "max": float(np.max(baseline_valid_bets)),
            }
        else:
            baseline_stats = {"min": None, "median": None, "p90": None, "max": None}
        selector_stats_rows.append(
            {
                "year": year,
                "eligible_non_base_windows": int(eligible_non_base_windows),
                "baseline_valid_bets_min": baseline_stats["min"],
                "baseline_valid_bets_median": baseline_stats["median"],
                "baseline_valid_bets_p90": baseline_stats["p90"],
                "baseline_valid_bets_max": baseline_stats["max"],
            }
        )

        # Step14 for baseline and selector
        bankroll = _read_bankroll(runs["base"].run_dir)
        step14_base_path = _summarize_step14(runs["base"].run_dir, year, "base", out_dir / f"step14_base_{year}", bankroll)

        # build selector run dir from chosen windows
        chosen_df = pd.DataFrame([r for r in eval_rows if r["year"] == year])
        chosen_df["run_dir"] = chosen_df["chosen"].map(lambda v: runs[v].run_dir)
        sel_run = _copy_selector_bets(out_dir / f"selector_{year}", chosen_df.to_dict(orient="records"))
        step14_sel_path = _summarize_step14(sel_run, year, "selector", out_dir / f"step14_selector_{year}", bankroll)

        base_step = _read_step14_eval(step14_base_path)
        sel_step = _read_step14_eval(step14_sel_path)
        if base_step:
            step14_rows.append({"year": year, "strategy": "base", **base_step})
        if sel_step:
            step14_rows.append({"year": year, "strategy": "selector", **sel_step})

        # scarcity
        scarcity_rows.append({"year": year, "strategy": "base", **_candidate_scarcity(runs["base"].run_dir)})
        scarcity_rows.append({"year": year, "strategy": "selector", **_candidate_scarcity(sel_run)})

        # added/removed for selector vs base
        diff = _binding_added_removed(runs["base"].run_dir, sel_run)
        if not diff.empty:
            diff.insert(0, "year", year)
            added_removed_rows.append(diff)

    selector_df = pd.DataFrame(selector_rows)
    selector_stats_df = pd.DataFrame(selector_stats_rows)
    if valid_candidate_rows:
        (out_dir / "valid_candidate_table.json").write_text(
            json.dumps(valid_candidate_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if missing_valid_windows:
        (out_dir / "missing_valid_windows.json").write_text(
            json.dumps(missing_valid_windows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if not selector_df.empty:
        selector_df.to_csv(out_dir / "selector_decisions.csv", index=False, encoding="utf-8")

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        return 0

    def _gate_metrics(sub: pd.DataFrame) -> dict[str, Any]:
        d_roi = sub["var_roi"] - sub["base_roi"]
        d_maxdd = sub["var_maxdd"] - sub["base_maxdd"]
        med_d_roi = float(np.median(d_roi))
        improve_rate = float((d_roi > 0).mean())
        med_d_maxdd = float(np.median(d_maxdd))
        med_n_bets = float(np.median(sub["var_n_bets"]))
        zero_bet_windows = int((sub["var_n_bets"] == 0).sum())
        gate_pass = bool(improve_rate >= 0.6 and med_d_roi > 0 and med_d_maxdd <= 0 and med_n_bets >= 80 and zero_bet_windows == 0)
        base_stake = float(sub["base_stake"].sum())
        var_stake = float(sub["var_stake"].sum())
        pooled_roi_base = float(sub["base_profit"].sum() / base_stake) if base_stake else float("nan")
        pooled_roi_var = float(sub["var_profit"].sum() / var_stake) if var_stake else float("nan")
        pooled_maxdd_base = float(np.median(sub["base_maxdd"])) if len(sub["base_maxdd"]) else float("nan")
        pooled_maxdd_var = float(np.median(sub["var_maxdd"])) if len(sub["var_maxdd"]) else float("nan")
        return {
            "median_d_roi": med_d_roi,
            "improve_rate_roi": improve_rate,
            "median_d_maxdd": med_d_maxdd,
            "median_n_bets_var": med_n_bets,
            "zero_bet_windows": zero_bet_windows,
            "gate_pass": gate_pass,
            "pooled_roi_eval_base": pooled_roi_base,
            "pooled_roi_eval_var": pooled_roi_var,
            "pooled_stake_eval_base": base_stake,
            "pooled_stake_eval_var": var_stake,
            "pooled_n_bets_eval_base": float(sub["base_n_bets"].sum()),
            "pooled_n_bets_eval_var": float(sub["var_n_bets"].sum()),
            "pooled_maxdd_eval_base": pooled_maxdd_base,
            "pooled_maxdd_eval_var": pooled_maxdd_var,
        }

    eval_summary_rows = []
    for year in sorted(eval_df["year"].unique()):
        sub = eval_df[eval_df["year"] == year]
        metrics = _gate_metrics(sub)
        metrics.update({"year": year})
        eval_summary_rows.append(metrics)

    eval_summary = pd.DataFrame(eval_summary_rows)
    eval_summary.to_csv(out_dir / "eval_gate_summary.csv", index=False, encoding="utf-8")

    # step14 summary
    step14_df = pd.DataFrame(step14_rows) if step14_rows else pd.DataFrame()
    if not step14_df.empty:
        step14_df.to_csv(out_dir / "step14_eval_summary.csv", index=False, encoding="utf-8")

    # candidate scarcity
    if scarcity_rows:
        pd.DataFrame(scarcity_rows).to_csv(out_dir / "candidate_scarcity.csv", index=False, encoding="utf-8")

    if added_removed_rows:
        pd.concat(added_removed_rows, ignore_index=True).to_csv(
            out_dir / "added_removed_by_window.csv", index=False, encoding="utf-8"
        )

    # cross-year summary
    cross_rows = []
    if not eval_summary.empty:
        for _, r in eval_summary.iterrows():
            cross_rows.append({"year": int(r["year"]), "gate_pass": bool(r["gate_pass"])})
    if cross_rows:
        pass_2024 = next((r["gate_pass"] for r in cross_rows if r["year"] == 2024), False)
        pass_2025 = next((r["gate_pass"] for r in cross_rows if r["year"] == 2025), False)
        decision = "pass_both" if (pass_2024 and pass_2025) else "fail"
        pd.DataFrame(
            [{"pass_2024": pass_2024, "pass_2025": pass_2025, "decision": decision}]
        ).to_csv(out_dir / "cross_year_summary.csv", index=False, encoding="utf-8")

    # final report
    lines = ["# odds_dyn_ev_margin selector (valid-only, offline)", ""]
    lines.append("Pooled aggregates overlap across rolling windows; step14 is non-overlap.")
    lines.append("Overlap can flip ROI sign; step14 is the decision signal when signs differ.")
    lines.append("")

    if missing_valid_windows:
        lines.append("- Missing valid_backtest metrics were detected; affected windows are marked ineligible.")

    if not selector_df.empty:
        for year in sorted(selector_df["year"].unique()):
            sub = selector_df[selector_df["year"] == year]
            total = len(sub)
            chosen_rates = (sub["chosen"].value_counts() / total).to_dict() if total else {}
            eligible_windows = 0
            if not selector_stats_df.empty:
                row = selector_stats_df[selector_stats_df["year"] == year]
                if not row.empty:
                    eligible_windows = int(row.iloc[0]["eligible_non_base_windows"])
            only_base = total > 0 and set(sub["chosen"].unique().tolist()) == {"base"}
            if eligible_windows == 0 or only_base:
                lines.append(f"- {year} selector_noop=true | eligible_windows={eligible_windows}")
            lines.append(f"- {year} chosen_rate_by_option={chosen_rates}")
            if not selector_stats_df.empty:
                row = selector_stats_df[selector_stats_df["year"] == year]
                if not row.empty:
                    lines.append(
                        f"  baseline_valid_bets_min/median/p90/max="
                        f"{row.iloc[0]['baseline_valid_bets_min']}/"
                        f"{row.iloc[0]['baseline_valid_bets_median']}/"
                        f"{row.iloc[0]['baseline_valid_bets_p90']}/"
                        f"{row.iloc[0]['baseline_valid_bets_max']}"
                    )

    if not eval_summary.empty:
        lines.append("")
        for _, r in eval_summary.iterrows():
            year = int(r["year"])
            base_step = {}
            var_step = {}
            if not step14_df.empty:
                base_row = step14_df[(step14_df["year"] == year) & (step14_df["strategy"] == "base")]
                var_row = step14_df[(step14_df["year"] == year) & (step14_df["strategy"] == "selector")]
                if not base_row.empty:
                    base_step = base_row.iloc[0].to_dict()
                if not var_row.empty:
                    var_step = var_row.iloc[0].to_dict()
            base_sm = sign_mismatch(r.get("pooled_roi_eval_base"), base_step.get("roi"))
            var_sm = sign_mismatch(r.get("pooled_roi_eval_var"), var_step.get("roi"))
            base_sm_tag = "SIGN_MISMATCH" if base_sm else "sign_match"
            var_sm_tag = "SIGN_MISMATCH" if var_sm else "sign_match"
            lines.append(
                f"- {year} eval: gate_pass={bool(r['gate_pass'])} | median_d_roi={r['median_d_roi']:.6f} | "
                f"improve_rate={r['improve_rate_roi']:.3f} | median_d_maxdd={r['median_d_maxdd']:.6f} | "
                f"median_n_bets_var={r['median_n_bets_var']:.1f}"
            )
            lines.append(
                f"  pooled base ROI={r.get('pooled_roi_eval_base')} stake={r.get('pooled_stake_eval_base')} "
                f"n_bets={r.get('pooled_n_bets_eval_base')} maxdd_median={r.get('pooled_maxdd_eval_base')} "
                f"| {base_sm_tag}"
            )
            lines.append(
                f"  pooled selector ROI={r.get('pooled_roi_eval_var')} stake={r.get('pooled_stake_eval_var')} "
                f"n_bets={r.get('pooled_n_bets_eval_var')} maxdd_median={r.get('pooled_maxdd_eval_var')} "
                f"| {var_sm_tag}"
            )
            if base_step:
                lines.append(
                    f"  step14 base ROI={base_step.get('roi')} stake={base_step.get('total_stake')} "
                    f"n_bets={base_step.get('n_bets')} maxdd={base_step.get('max_drawdown')}"
                )
            if var_step:
                lines.append(
                    f"  step14 selector ROI={var_step.get('roi')} stake={var_step.get('total_stake')} "
                    f"n_bets={var_step.get('n_bets')} maxdd={var_step.get('max_drawdown')}"
                )

    lines.append("")
    lines.append("ROI definition: ROI = profit / stake, profit = return - stake.")

    report_text = "\n".join(lines) + "\n"
    (out_dir / "final_report.md").write_text(report_text, encoding="utf-8")
    (out_dir / "report_status.md").write_text(
        "Pooled aggregates overlap across rolling windows; step14 is non-overlap.\n"
        "Overlap can flip ROI sign; step14 is the decision signal when signs differ.\n"
        "ROI definition: ROI = profit / stake, profit = return - stake.\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
