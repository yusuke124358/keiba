"""
Cross-year summary for segmented blend sweep (2024 vs 2025).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "variant" not in df.columns:
        raise ValueError(f"variant column missing: {path}")
    return df


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _decision(pass_2024: bool, pass_2025: bool) -> str:
    if pass_2024 and pass_2025:
        return "pass_both"
    if pass_2024 or pass_2025:
        return "pass_single_year"
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize cross-year gate results for segblend sweep")
    ap.add_argument("--dir-2024", type=Path, required=True)
    ap.add_argument("--dir-2025", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_2024 = _load_eval(args.dir_2024 / "eval_metrics.csv")
    eval_2025 = _load_eval(args.dir_2025 / "eval_metrics.csv")

    merged = eval_2024.merge(eval_2025, on="variant", how="outer", suffixes=("_2024", "_2025"))
    merged["pass_2024"] = merged.get("gate_pass_2024").fillna(False).astype(bool)
    merged["pass_2025"] = merged.get("gate_pass_2025").fillna(False).astype(bool)
    merged["decision"] = merged.apply(lambda r: _decision(r["pass_2024"], r["pass_2025"]), axis=1)

    merged.to_csv(out_dir / "cross_year_summary.csv", index=False, encoding="utf-8")

    # final report
    lines = []
    lines.append("# Segblend surface cross-year summary")
    lines.append("")
    lines.append("## 2024 eval (gate metrics)")
    lines.append(_df_to_md(eval_2024))
    lines.append("")
    lines.append("## 2025 eval (gate metrics)")
    lines.append(_df_to_md(eval_2025))
    lines.append("")
    lines.append("## Cross-year decision")
    lines.append(_df_to_md(merged[["variant", "pass_2024", "pass_2025", "decision"]]))
    lines.append("")
    lines.append("Gate: improve_rate>=0.6, median_d_roi>0, median_d_maxdd<=0, median_n_bets_var>=80, zero_bet_windows=0")
    (out_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
