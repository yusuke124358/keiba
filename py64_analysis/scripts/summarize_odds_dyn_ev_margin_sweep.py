"""Summarize odds dynamics EV margin sweep across eval-only rolling runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from keiba.analysis.metrics_utils import sign_mismatch

WIN_RE = r"^w(\d{3})_(\d{8})_(\d{8})$"


def _read_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
    return eval_row.iloc[0].to_dict()


def _read_config_bankroll(window_dir: Path) -> float:
    cfg = window_dir / "config_used.yaml"
    if not cfg.exists():
        return 1_000_000.0
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        except Exception:
            return 1_000_000.0
    try:
        return float(data.get("betting", {}).get("bankroll_yen", 1_000_000.0))
    except Exception:
        return 1_000_000.0


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
        base_n = int(len(b_keys))
        rows.append(
            {
                "window": win,
                "base_n_bets": base_n,
                "var_n_bets": int(len(v_keys)),
                "filtered_bets": int(len(filtered)),
                "added_bets": int(len(added)),
                "filtered_ratio": float(len(filtered)) / base_n if base_n else 0.0,
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


def _summarize_variant(year: int, base_dir: Path, var_dir: Path, label: str, out_dir: Path) -> dict[str, Any]:
    base_df = _read_summary(base_dir)
    var_df = _read_summary(var_dir)
    paired = _merge_base_var(base_df, var_df)
    if paired.empty:
        return {"year": year, "variant": label, "status": "missing"}

    eval_gate = _eval_gate_metrics(paired)
    base_stake = float(paired["total_stake_base"].sum()) if "total_stake_base" in paired.columns else 0.0
    var_stake = float(paired["total_stake_var"].sum()) if "total_stake_var" in paired.columns else 0.0
    base_profit = float(paired["total_profit_base"].sum()) if "total_profit_base" in paired.columns else 0.0
    var_profit = float(paired["total_profit_var"].sum()) if "total_profit_var" in paired.columns else 0.0
    pooled_roi_base = float(base_profit / base_stake) if base_stake else float("nan")
    pooled_roi_var = float(var_profit / var_stake) if var_stake else float("nan")
    pooled_maxdd_base = float(np.median(paired["max_drawdown_base"])) if "max_drawdown_base" in paired.columns else float("nan")
    pooled_maxdd_var = float(np.median(paired["max_drawdown_var"])) if "max_drawdown_var" in paired.columns else float("nan")
    eval_gate_row = {
        "year": year,
        "variant": label,
        **eval_gate,
        "pooled_roi_eval_base": pooled_roi_base,
        "pooled_roi_eval_var": pooled_roi_var,
        "pooled_stake_eval_base": base_stake,
        "pooled_stake_eval_var": var_stake,
        "pooled_n_bets_eval_base": float(paired["n_bets_base"].sum()) if "n_bets_base" in paired.columns else float("nan"),
        "pooled_n_bets_eval_var": float(paired["n_bets_var"].sum()) if "n_bets_var" in paired.columns else float("nan"),
        "pooled_maxdd_eval_base": pooled_maxdd_base,
        "pooled_maxdd_eval_var": pooled_maxdd_var,
        "test_start": str(paired["test_start"].min()),
        "test_end": str(paired["test_end"].max()),
    }

    binding = _binding_diagnosis(base_dir, var_dir, paired)

    sample_window = paired.iloc[0].get("name_base") or paired.iloc[0].get("name_var")
    no_op = _no_op_audit(base_dir, var_dir, str(sample_window)) if sample_window else {"status": "missing"}

    bankroll = _read_config_bankroll(base_dir / paired.iloc[0]["name_base"])
    step14_base = _read_step14_eval(_run_step14(base_dir, year, "base", out_dir / f"step14_{label}" / "base", bankroll))
    step14_var = _read_step14_eval(_run_step14(var_dir, year, label, out_dir / f"step14_{label}" / label, bankroll))

    scarcity_base = _candidate_scarcity(base_dir)
    scarcity_var = _candidate_scarcity(var_dir)

    return {
        "year": year,
        "variant": label,
        "eval_gate": eval_gate_row,
        "binding": binding,
        "no_op": no_op,
        "step14_base": step14_base,
        "step14_var": step14_var,
        "scarcity_base": scarcity_base,
        "scarcity_var": scarcity_var,
        "no_op_window": str(sample_window) if sample_window else None,
    }


def _parse_variant(arg: str) -> tuple[str, Path, Path]:
    parts = [p.strip() for p in arg.split("|")]
    if len(parts) != 3:
        raise ValueError("variant must be name|path2024|path2025")
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-2024", required=True)
    p.add_argument("--base-2025", required=True)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--noop-variant", default=None)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    base_2024 = Path(args.base_2024)
    base_2025 = Path(args.base_2025)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    binding_rows = []
    scarcity_rows = []
    step14_rows = []
    no_op_audit = {}
    cross_rows = []

    for raw in args.variant:
        name, var_2024, var_2025 = _parse_variant(raw)
        res24 = _summarize_variant(2024, base_2024, var_2024, name, out_dir / "2024")
        res25 = _summarize_variant(2025, base_2025, var_2025, name, out_dir / "2025")

        if "eval_gate" in res24:
            eval_rows.append(res24["eval_gate"])
        if "eval_gate" in res25:
            eval_rows.append(res25["eval_gate"])

        if isinstance(res24.get("binding"), pd.DataFrame) and not res24["binding"].empty:
            tmp = res24["binding"].copy()
            tmp.insert(0, "variant", name)
            tmp.insert(0, "year", 2024)
            binding_rows.append(tmp)
        if isinstance(res25.get("binding"), pd.DataFrame) and not res25["binding"].empty:
            tmp = res25["binding"].copy()
            tmp.insert(0, "variant", name)
            tmp.insert(0, "year", 2025)
            binding_rows.append(tmp)

        scarcity_base = res24.get("scarcity_base")
        scarcity_var = res24.get("scarcity_var")
        if scarcity_base:
            scarcity_rows.append({"year": 2024, "variant": name, "strategy": "base", **scarcity_base})
        if scarcity_var:
            scarcity_rows.append({"year": 2024, "variant": name, "strategy": "var", **scarcity_var})
        scarcity_base = res25.get("scarcity_base")
        scarcity_var = res25.get("scarcity_var")
        if scarcity_base:
            scarcity_rows.append({"year": 2025, "variant": name, "strategy": "base", **scarcity_base})
        if scarcity_var:
            scarcity_rows.append({"year": 2025, "variant": name, "strategy": "var", **scarcity_var})

        for year, res in [(2024, res24), (2025, res25)]:
            base_step = res.get("step14_base") or {}
            var_step = res.get("step14_var") or {}
            if base_step:
                step14_rows.append({"year": year, "variant": name, "strategy": "base", **base_step})
            if var_step:
                step14_rows.append({"year": year, "variant": name, "strategy": "var", **var_step})

        no_op_audit[name] = {
            "variant": name,
            "sample_window": res24.get("no_op_window"),
            "2024": res24.get("no_op"),
            "2025": res25.get("no_op"),
        }

        pass_2024 = bool(res24.get("eval_gate", {}).get("gate_pass"))
        pass_2025 = bool(res25.get("eval_gate", {}).get("gate_pass"))
        decision = "pass_both" if (pass_2024 and pass_2025) else "fail"
        cross_rows.append({"variant": name, "pass_2024": pass_2024, "pass_2025": pass_2025, "decision": decision})

    if eval_rows:
        pd.DataFrame(eval_rows).to_csv(out_dir / "eval_gate_summary.csv", index=False, encoding="utf-8")
    if binding_rows:
        pd.concat(binding_rows, ignore_index=True).to_csv(out_dir / "binding_diagnosis.csv", index=False, encoding="utf-8")
    if scarcity_rows:
        pd.DataFrame(scarcity_rows).to_csv(out_dir / "candidate_scarcity.csv", index=False, encoding="utf-8")
    if step14_rows:
        pd.DataFrame(step14_rows).to_csv(out_dir / "step14_eval_summary.csv", index=False, encoding="utf-8")

    if no_op_audit:
        (out_dir / "no_op_audit.json").write_text(
            json.dumps(no_op_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if cross_rows:
        pd.DataFrame(cross_rows).to_csv(out_dir / "cross_year_summary.csv", index=False, encoding="utf-8")

    # final report
    lines = ["# odds_dyn_ev_margin sweep", ""]
    lines.append("Pooled aggregates overlap across rolling windows; step14 is non-overlap.")
    lines.append("Overlap can flip ROI sign; step14 is the decision signal when signs differ.")
    lines.append("")

    eval_df = pd.DataFrame(eval_rows) if eval_rows else pd.DataFrame()
    step14_df = pd.DataFrame(step14_rows) if step14_rows else pd.DataFrame()

    if not eval_df.empty:
        for _, r in eval_df.iterrows():
            year = int(r["year"])
            variant = r["variant"]
            base_step = {}
            var_step = {}
            if not step14_df.empty:
                base_row = step14_df[
                    (step14_df["year"] == year) & (step14_df["variant"] == variant) & (step14_df["strategy"] == "base")
                ]
                var_row = step14_df[
                    (step14_df["year"] == year) & (step14_df["variant"] == variant) & (step14_df["strategy"] == "var")
                ]
                if not base_row.empty:
                    base_step = base_row.iloc[0].to_dict()
                if not var_row.empty:
                    var_step = var_row.iloc[0].to_dict()
            base_sm = sign_mismatch(r.get("pooled_roi_eval_base"), base_step.get("roi"))
            var_sm = sign_mismatch(r.get("pooled_roi_eval_var"), var_step.get("roi"))
            base_sm_tag = "SIGN_MISMATCH" if base_sm else "sign_match"
            var_sm_tag = "SIGN_MISMATCH" if var_sm else "sign_match"
            lines.append(
                f"- {year} {variant} eval: gate_pass={bool(r['gate_pass'])} | median_d_roi={r['median_d_roi']:.6f} | "
                f"improve_rate={r['improve_rate_roi']:.3f} | median_d_maxdd={r['median_d_maxdd']:.6f} | "
                f"median_n_bets_var={r['median_n_bets_var']:.1f}"
            )
            lines.append(
                f"  pooled base ROI={r.get('pooled_roi_eval_base')} stake={r.get('pooled_stake_eval_base')} "
                f"n_bets={r.get('pooled_n_bets_eval_base')} maxdd_median={r.get('pooled_maxdd_eval_base')} "
                f"| {base_sm_tag}"
            )
            lines.append(
                f"  pooled var ROI={r.get('pooled_roi_eval_var')} stake={r.get('pooled_stake_eval_var')} "
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
                    f"  step14 var ROI={var_step.get('roi')} stake={var_step.get('total_stake')} "
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
