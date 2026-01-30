"""
Decide between q95 and q97 using step=14 non-overlap aggregates and existing bundles.
"""
from __future__ import annotations

import argparse
import difflib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from keiba.analysis.metrics_utils import roi_footer, sign_mismatch


AGENTS_STEP14_LINE = (
    "- For rolling (60/14/14), pooled ROI includes overlap; include step=14 non-overlap "
    "walk-forward aggregates for annual reproducibility checks."
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    return pd.read_csv(path)


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


def _fmt(val: Any, digits: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def _df_to_md(df: pd.DataFrame, digits: int = 4) -> str:
    def fmt_cell(v: Any) -> str:
        if isinstance(v, float):
            return _fmt(v, digits=digits)
        return _fmt(v, digits=digits)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _cap_stats(df: pd.DataFrame, year: int, split: str) -> dict[str, float | None]:
    sub = df[(df["year"] == year) & (df["split"] == split)].copy()
    vals = pd.to_numeric(sub.get("cap_value"), errors="coerce")
    vals = vals[pd.notna(vals)]
    if vals.empty:
        return {"p10": None, "median": None, "p90": None}
    return {
        "p10": float(np.percentile(vals, 10)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
    }


def _binding_rates(binding_df: pd.DataFrame, year: int, split: str) -> dict[str, float | None]:
    sub = binding_df[(binding_df["year"] == year) & (binding_df["split"] == split)].copy()
    if sub.empty:
        return {"excluded_race_rate": None, "excluded_bet_rate": None, "excluded_stake_rate": None}
    return {
        "excluded_race_rate": float(sub["excluded_race_rate"].median()),
        "excluded_bet_rate": float(sub["excluded_bet_rate"].median()),
        "excluded_stake_rate": float(sub["excluded_stake_rate"].median()),
    }


def _aggregate_eval_daily(daily_df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    df = daily_df[daily_df["split"] == "eval"].copy()
    if year is not None:
        df = df[df["year"] == year].copy()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["strategy", "date"]).copy()
    return df


def _summary_from_daily(df: pd.DataFrame, initial_bankroll: float) -> pd.DataFrame:
    rows = []
    for strategy, sub in df.groupby("strategy", dropna=False):
        sub = sub.sort_values("date").copy()
        total_stake = float(sub["stake"].sum())
        total_profit = float(sub["profit"].sum())
        n_bets = int(sub["n_bets"].sum())
        roi = (total_profit / total_stake) if total_stake > 0 else 0.0
        total_days = int(len(sub))
        days_with_bet = int((sub["stake"] > 0).sum())
        frac_days_any_bet = (days_with_bet / total_days) if total_days > 0 else 0.0
        bets_per_day = (n_bets / total_days) if total_days > 0 else 0.0

        sub["cum_profit"] = sub["profit"].cumsum()
        sub["equity"] = initial_bankroll + sub["cum_profit"]
        sub["peak"] = sub["equity"].cummax()
        sub["drawdown_pct"] = (sub["peak"] - sub["equity"]) / sub["peak"]
        max_dd = float(sub["drawdown_pct"].max()) if not sub.empty else 0.0

        rows.append(
            {
                "strategy": strategy,
                "stake": total_stake,
                "profit": total_profit,
                "roi": roi,
                "maxdd": max_dd,
                "n_bets": n_bets,
                "frac_days_any_bet": frac_days_any_bet,
                "bets_per_day": bets_per_day,
            }
        )
    return pd.DataFrame(rows)


def _decision(
    total_df: pd.DataFrame,
    year_df: dict[int, pd.DataFrame],
    baseline_label: str = "baseline",
) -> tuple[str, str]:
    total = total_df.set_index("strategy")
    if baseline_label not in total.index:
        return "none", "baseline missing in total summary"

    base_roi = float(total.loc[baseline_label, "roi"])
    q95_roi = float(total.loc["q95", "roi"]) if "q95" in total.index else float("nan")
    q97_roi = float(total.loc["q97", "roi"]) if "q97" in total.index else float("nan")

    if q95_roi <= base_roi and q97_roi <= base_roi:
        return "none", "total ROI does not improve over baseline"

    if np.isclose(q95_roi, q97_roi, atol=1e-9):
        q95_all = all(
            year_df[y].set_index("strategy").loc["q95", "roi"]
            > year_df[y].set_index("strategy").loc[baseline_label, "roi"]
            for y in year_df
        )
        q97_all = all(
            year_df[y].set_index("strategy").loc["q97", "roi"]
            > year_df[y].set_index("strategy").loc[baseline_label, "roi"]
            for y in year_df
        )
        if q95_all and not q97_all:
            return "q95", "tie on total ROI; q95 improves both years"
        if q97_all and not q95_all:
            return "q97", "tie on total ROI; q97 improves both years"

        q95_dd = float(total.loc["q95", "maxdd"])
        q97_dd = float(total.loc["q97", "maxdd"])
        if q95_dd < q97_dd:
            return "q95", "tie on ROI; lower max drawdown"
        if q97_dd < q95_dd:
            return "q97", "tie on ROI; lower max drawdown"

        q95_bets = float(total.loc["q95", "n_bets"])
        q97_bets = float(total.loc["q97", "n_bets"])
        if q95_bets >= q97_bets:
            return "q95", "tie on ROI/maxdd; higher n_bets"
        return "q97", "tie on ROI/maxdd; higher n_bets"

    if q95_roi > q97_roi:
        return "q95", "higher total step14 ROI"
    return "q97", "higher total step14 ROI"


def _agents_diff(agents_path: Path) -> str | None:
    if not agents_path.exists():
        return None
    text = agents_path.read_text(encoding="utf-8")
    if AGENTS_STEP14_LINE not in text:
        return None
    before = text.replace(AGENTS_STEP14_LINE + "\n", "")
    diff = difflib.unified_diff(
        before.splitlines(),
        text.splitlines(),
        fromfile="a/AGENTS.md",
        tofile="b/AGENTS.md",
        lineterm="",
    )
    return "\n".join(diff)


def main() -> None:
    ap = argparse.ArgumentParser(description="Decision report for q95 vs q97")
    ap.add_argument("--q95-dir", type=Path, required=True)
    ap.add_argument("--q97-dir", type=Path, required=True)
    ap.add_argument("--step14-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--agents-path", type=Path, required=True)
    ap.add_argument("--initial-bankroll", type=float, default=1_000_000.0)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    q95_eval = _read_csv(args.q95_dir / "eval_gate_summary.csv")
    q97_eval = _read_csv(args.q97_dir / "eval_gate_summary.csv")
    q95_bind = _read_csv(args.q95_dir / "binding_diagnosis.csv")
    q97_bind = _read_csv(args.q97_dir / "binding_diagnosis.csv")
    q95_caps = _read_csv(args.q95_dir / "fitted_cap_values.csv")
    q97_caps = _read_csv(args.q97_dir / "fitted_cap_values.csv")
    step14_daily = _read_csv(args.step14_dir / "walkforward_step14_daily.csv")
    step14_summary = _read_csv(args.step14_dir / "walkforward_step14_summary.csv")

    step14_eval_2024 = _aggregate_eval_daily(step14_daily, year=2024)
    step14_eval_2025 = _aggregate_eval_daily(step14_daily, year=2025)
    step14_eval_total = _aggregate_eval_daily(step14_daily, year=None)

    summary_2024 = _summary_from_daily(step14_eval_2024, args.initial_bankroll)
    summary_2025 = _summary_from_daily(step14_eval_2025, args.initial_bankroll)
    summary_total = _summary_from_daily(step14_eval_total, args.initial_bankroll)

    summary_2024["year"] = 2024
    summary_2025["year"] = 2025
    summary_total["year"] = "total"
    summary_table = pd.concat([summary_2024, summary_2025, summary_total], ignore_index=True)[
        ["year", "strategy", "stake", "profit", "roi", "maxdd"]
    ]

    diff_rows = []
    for label, df in [("2024", summary_2024), ("2025", summary_2025), ("total", summary_total)]:
        if df.empty:
            continue
        s = df.set_index("strategy")
        if "q95" in s.index and "q97" in s.index:
            diff_rows.append(
                {
                    "year": label,
                    "delta_stake_q97_minus_q95": s.loc["q97", "stake"] - s.loc["q95", "stake"],
                    "delta_profit_q97_minus_q95": s.loc["q97", "profit"] - s.loc["q95", "profit"],
                    "delta_roi_q97_minus_q95": s.loc["q97", "roi"] - s.loc["q95", "roi"],
                    "delta_maxdd_q97_minus_q95": s.loc["q97", "maxdd"] - s.loc["q95", "maxdd"],
                }
            )
    diff_table = pd.DataFrame(diff_rows)

    chosen, reason = _decision(
        summary_total,
        {2024: summary_2024, 2025: summary_2025},
        baseline_label="baseline",
    )

    report_lines = []
    report_lines.append("# takeoutq q95 vs q97 decision report")
    report_lines.append("")
    report_lines.append("- Pooled aggregates overlap across rolling windows; step14 is non-overlap.")
    report_lines.append("- Overlap can flip ROI sign; step14 is the decision signal when signs differ.")
    report_lines.append("## Current Summary")
    report_lines.append("- cross-year pass: q95=pass_both, q97=pass_both")
    for year in [2024, 2025]:
        q95_row = q95_eval[(q95_eval["year"] == year) & (q95_eval["variant"] == "q95")].iloc[0]
        q97_row = q97_eval[(q97_eval["year"] == year) & (q97_eval["variant"] == "q97")].iloc[0]
        step_year = summary_2024 if year == 2024 else summary_2025
        step_map = step_year.set_index("strategy") if not step_year.empty else pd.DataFrame()
        q95_step_roi = _safe_float(step_map.loc["q95", "roi"]) if "q95" in step_map.index else float("nan")
        q97_step_roi = _safe_float(step_map.loc["q97", "roi"]) if "q97" in step_map.index else float("nan")
        q95_sm = sign_mismatch(_safe_float(q95_row["eval_pooled_roi"]), q95_step_roi)
        q97_sm = sign_mismatch(_safe_float(q97_row["eval_pooled_roi"]), q97_step_roi)
        report_lines.append(
            f"- {year} eval gate q95: median_d_roi={_fmt(_safe_float(q95_row['eval_median_d_roi']))} | "
            f"improve_rate={_fmt(_safe_float(q95_row['eval_improve_rate']))} | "
            f"median_d_maxdd={_fmt(_safe_float(q95_row['eval_median_d_maxdd']))} | "
            f"median_n_bets={_fmt(_safe_float(q95_row['eval_median_n_bets_var']))}"
        )
        report_lines.append(
            f"- {year} eval gate q97: median_d_roi={_fmt(_safe_float(q97_row['eval_median_d_roi']))} | "
            f"improve_rate={_fmt(_safe_float(q97_row['eval_improve_rate']))} | "
            f"median_d_maxdd={_fmt(_safe_float(q97_row['eval_median_d_maxdd']))} | "
            f"median_n_bets={_fmt(_safe_float(q97_row['eval_median_n_bets_var']))}"
        )

        report_lines.append(
            f"- {year} pooled ROI design/eval: "
            f"q95={_fmt(_safe_float(q95_row['design_pooled_roi']))}/{_fmt(_safe_float(q95_row['eval_pooled_roi']))} | "
            f"q97={_fmt(_safe_float(q97_row['design_pooled_roi']))}/{_fmt(_safe_float(q97_row['eval_pooled_roi']))}"
        )
        report_lines.append(
            f"- {year} step14 ROI eval: q95={_fmt(q95_step_roi)} "
            f"({'SIGN_MISMATCH' if q95_sm else 'sign_match'}) | "
            f"q97={_fmt(q97_step_roi)} ({'SIGN_MISMATCH' if q97_sm else 'sign_match'})"
        )

        cap95 = _cap_stats(q95_caps, year=year, split="eval")
        cap97 = _cap_stats(q97_caps, year=year, split="eval")
        bind95 = _binding_rates(q95_bind, year=year, split="eval")
        bind97 = _binding_rates(q97_bind, year=year, split="eval")
        report_lines.append(
            f"- {year} cap dist (p10/median/p90): q95="
            f"{_fmt(cap95['p10'])}/{_fmt(cap95['median'])}/{_fmt(cap95['p90'])} | "
            f"q97={_fmt(cap97['p10'])}/{_fmt(cap97['median'])}/{_fmt(cap97['p90'])}"
        )
        report_lines.append(
            f"- {year} excluded rate (race/bet/stake): q95="
            f"{_fmt(bind95['excluded_race_rate'])}/{_fmt(bind95['excluded_bet_rate'])}/{_fmt(bind95['excluded_stake_rate'])} | "
            f"q97={_fmt(bind97['excluded_race_rate'])}/{_fmt(bind97['excluded_bet_rate'])}/{_fmt(bind97['excluded_stake_rate'])}"
        )

    report_lines.append("")
    report_lines.append("## Step=14 non-overlap comparison (eval only)")
    report_lines.append(_df_to_md(summary_table))
    report_lines.append("")
    report_lines.append("### q95 vs q97 delta (q97 - q95)")
    report_lines.append(_df_to_md(diff_table))

    report_lines.append("")
    report_lines.append("## Decision Logic")
    report_lines.append("- Primary: choose the highest total(2024+2025) step14 non-overlap ROI")
    report_lines.append("- Tie-break: both-year ROI improvement, then lower maxDD, then higher n_bets")
    report_lines.append("- If total ROI does not improve over baseline, choose none")
    report_lines.append(f"- decision: chosen_variant={chosen} | reason={reason}")
    report_lines.append("")
    report_lines.append(roi_footer())

    agents_diff = _agents_diff(args.agents_path)
    if agents_diff:
        report_lines.append("")
        report_lines.append("## AGENTS.md diff")
        report_lines.append("```diff")
        report_lines.append(agents_diff)
        report_lines.append("```")

    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Copy key CSVs into out_dir
    shutil.copy2(args.q95_dir / "eval_gate_summary.csv", out_dir / "eval_gate_summary_q95.csv")
    shutil.copy2(args.q97_dir / "eval_gate_summary.csv", out_dir / "eval_gate_summary_q97.csv")
    shutil.copy2(args.q95_dir / "binding_diagnosis.csv", out_dir / "binding_diagnosis_q95.csv")
    shutil.copy2(args.q97_dir / "binding_diagnosis.csv", out_dir / "binding_diagnosis_q97.csv")
    shutil.copy2(args.step14_dir / "walkforward_step14_summary.csv", out_dir / "walkforward_step14_summary.csv")

    print(f"[decision] chosen_variant={chosen} | reason={reason}")


if __name__ == "__main__":
    main()
