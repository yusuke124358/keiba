"""
Summarize odds-window sweep runs (rolling holdout) into combined summaries,
paired comparisons vs baseline, and eval gate metrics.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

WIN_RE = re.compile(r"^w(\d{3})_")


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


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _load_summary(path: Path) -> pd.DataFrame:
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
            "roi": _to_num(df[roi_col]),
            "max_dd": _to_num(df[dd_col]),
            "n_bets": _to_num(df[nb_col]).fillna(0).astype(int),
        }
    )
    if stake_col is not None:
        out["total_stake"] = _to_num(df[stake_col])
    if profit_col is not None:
        out["total_profit"] = _to_num(df[profit_col])
    return out


def _summarize_block(df: pd.DataFrame) -> dict:
    n = int(len(df))
    pos = int((df["d_roi"] > 0).sum()) if n else 0
    out = {
        "n_windows": n,
        "improve_rate_roi": (pos / n) if n else None,
        "median_d_roi": float(df["d_roi"].median()) if n else None,
        "median_d_maxdd": float(df["d_max_dd"].median()) if n else None,
        "median_d_n_bets": float(df["d_n_bets"].median()) if n else None,
        "median_roi_base": float(df["roi_base"].median()) if n else None,
        "median_roi_var": float(df["roi_var"].median()) if n else None,
        "median_maxdd_base": float(df["max_dd_base"].median()) if n else None,
        "median_maxdd_var": float(df["max_dd_var"].median()) if n else None,
        "median_n_bets_base": float(df["n_bets_base"].median()) if n else None,
        "median_n_bets_var": float(df["n_bets_var"].median()) if n else None,
    }
    return out


def _bands_from_bets(
    run_dirs: list[Path],
    design_max_idx: int,
    max_odds: float,
) -> pd.DataFrame:
    stake_cols = ["stake", "stake_yen", "stake_amount", "total_stake", "stake_sum"]
    profit_cols = ["profit", "net_profit", "pnl"]
    odds_cols = ["odds_at_buy", "odds", "buy_odds", "odds_buy"]

    rows = []
    for run_dir in run_dirs:
        offset = _infer_w_idx_offset(run_dir.name)
        for bets_path in sorted(run_dir.glob("w*/bets.csv")):
            window_name = bets_path.parent.name
            w_idx = _parse_window_idx(window_name)
            if w_idx is None:
                continue
            w_idx = int(w_idx) + int(offset)
            split = "design" if w_idx <= int(design_max_idx) else "eval"
            try:
                df = pd.read_csv(bets_path)
            except Exception:
                continue

            stake_col = _find_col(df, stake_cols)
            profit_col = _find_col(df, profit_cols)
            odds_col = _find_col(df, odds_cols)
            if stake_col is None or profit_col is None or odds_col is None:
                continue

            df = df.copy()
            df["stake"] = _to_num(df[stake_col])
            df["profit"] = _to_num(df[profit_col])
            df["odds"] = _to_num(df[odds_col])
            df = df[(df["stake"].notna()) & (df["profit"].notna()) & (df["odds"].notna())]
            df = df[(df["odds"] >= 1.0) & (df["odds"] <= float(max_odds))]

            if df.empty:
                continue

            bins = [1.0, 5.0, 10.0, 20.0, float(max_odds)]
            labels = ["1-5", "5-10", "10-20", "20-30"]
            df["band"] = pd.cut(df["odds"], bins=bins, labels=labels, right=True, include_lowest=True)

            agg = df.groupby("band", observed=True).agg(
                stake=("stake", "sum"),
                profit=("profit", "sum"),
                n_bets=("stake", "count"),
            ).reset_index()
            agg["roi"] = agg.apply(lambda r: (r["profit"] / r["stake"]) if r["stake"] else float("nan"), axis=1)
            agg["split"] = split
            rows.append(agg)

    if not rows:
        return pd.DataFrame(columns=["split", "band", "stake", "profit", "roi", "n_bets"])

    out = pd.concat(rows, ignore_index=True)
    out = out.groupby(["split", "band"], dropna=False).agg(
        stake=("stake", "sum"),
        profit=("profit", "sum"),
        n_bets=("n_bets", "sum"),
    ).reset_index()
    out["roi"] = out.apply(lambda r: (r["profit"] / r["stake"]) if r["stake"] else float("nan"), axis=1)

    pooled = out.groupby(["band"], dropna=False).agg(
        stake=("stake", "sum"),
        profit=("profit", "sum"),
        n_bets=("n_bets", "sum"),
    ).reset_index()
    pooled["roi"] = pooled.apply(lambda r: (r["profit"] / r["stake"]) if r["stake"] else float("nan"), axis=1)
    pooled["split"] = "all"

    return pd.concat([out, pooled], ignore_index=True)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize odds window sweep runs")
    ap.add_argument("--manifest", type=Path, required=True, help="JSON manifest with variant -> run dirs")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--design-max-idx", type=int, default=12)
    ap.add_argument("--max-odds", type=float, default=30.0)
    ap.add_argument("--append-report", action="store_true")
    ap.add_argument("--phase0-base-bands", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    baseline = manifest.get("baseline", "v0_base")
    variants = manifest.get("variants", {})
    if not variants:
        raise SystemExit("manifest missing variants")
    if baseline not in variants:
        raise SystemExit(f"baseline not in variants: {baseline}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_root = out_dir / "combined_runs"
    combined_root.mkdir(parents=True, exist_ok=True)

    combined_summaries: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for vname, run_dirs in variants.items():
        frames = []
        sources = {"variant": vname, "run_dirs": [], "offsets": []}
        for run_dir_str in run_dirs:
            run_dir = Path(run_dir_str)
            sources["run_dirs"].append(str(run_dir))
            if not run_dir.exists():
                raise SystemExit(f"run dir not found: {run_dir}")
            offset = _infer_w_idx_offset(run_dir.name)
            sources["offsets"].append(offset)

            summary_path = run_dir / "summary.csv"
            if not summary_path.exists():
                raise SystemExit(f"summary.csv not found: {summary_path}")
            df = _load_summary(summary_path)
            df = df.copy()
            df["window_name"] = df["window_name"].apply(lambda x: _offset_window_name(x, offset))
            df["window_idx"] = df["window_name"].apply(_parse_window_idx)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["window_idx"]).reset_index(drop=True)

        out_variant_dir = combined_root / vname
        out_variant_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_variant_dir / "summary.csv", index=False, encoding="utf-8")
        (out_variant_dir / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
        combined_summaries[vname] = combined

        for _, row in combined.iterrows():
            w_idx = int(row["window_idx"]) if pd.notna(row.get("window_idx")) else None
            summary_rows.append(
                {
                    "variant": vname,
                    "window_idx": w_idx,
                    "split": "design" if w_idx is not None and w_idx <= int(args.design_max_idx) else "eval",
                    "test_start": row.get("test_start"),
                    "test_end": row.get("test_end"),
                    "roi": row.get("roi"),
                    "max_drawdown": row.get("max_dd"),
                    "n_bets": row.get("n_bets"),
                    "total_stake": row.get("total_stake"),
                    "total_profit": row.get("total_profit"),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")

    base = combined_summaries[baseline].copy()
    paired_rows = []
    paired_summary = {
        "meta": {
            "baseline": baseline,
            "design_max_idx": int(args.design_max_idx),
            "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
        "variants": {},
    }

    for vname, var in combined_summaries.items():
        if vname == baseline:
            continue
        merged = base.merge(var, on=["test_start", "test_end"], how="inner", suffixes=("_base", "_var"))
        if merged.empty:
            raise SystemExit(f"No matched windows for {vname}")

        merged["window_idx"] = merged["window_idx_base"] if "window_idx_base" in merged.columns else merged.get("window_idx")
        merged["d_roi"] = merged["roi_var"] - merged["roi_base"]
        merged["d_max_dd"] = merged["max_dd_var"] - merged["max_dd_base"]
        merged["d_n_bets"] = merged["n_bets_var"] - merged["n_bets_base"]
        merged["split"] = merged["window_idx"].apply(lambda i: "design" if int(i) <= int(args.design_max_idx) else "eval")
        merged["variant"] = vname

        for _, row in merged.iterrows():
            paired_rows.append(
                {
                    "variant": vname,
                    "window_idx": int(row["window_idx"]),
                    "split": row["split"],
                    "test_start": row["test_start"],
                    "test_end": row["test_end"],
                    "roi_base": row["roi_base"],
                    "roi_var": row["roi_var"],
                    "d_roi": row["d_roi"],
                    "max_dd_base": row["max_dd_base"],
                    "max_dd_var": row["max_dd_var"],
                    "d_max_dd": row["d_max_dd"],
                    "n_bets_base": row["n_bets_base"],
                    "n_bets_var": row["n_bets_var"],
                    "d_n_bets": row["d_n_bets"],
                    "total_stake_base": row.get("total_stake_base"),
                    "total_stake_var": row.get("total_stake_var"),
                    "total_profit_base": row.get("total_profit_base"),
                    "total_profit_var": row.get("total_profit_var"),
                }
            )

        paired_summary["variants"][vname] = {
            "all": _summarize_block(merged),
            "design": _summarize_block(merged[merged["split"] == "design"]),
            "eval": _summarize_block(merged[merged["split"] == "eval"]),
        }

    paired_df = pd.DataFrame(paired_rows)
    paired_df.sort_values(["variant", "window_idx"]).to_csv(out_dir / "paired_compare.csv", index=False, encoding="utf-8")
    (out_dir / "paired_summary.json").write_text(json.dumps(paired_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # eval metrics + gate
    eval_rows = []
    for vname in sorted([v for v in combined_summaries.keys() if v != baseline]):
        eval_block = paired_df[(paired_df["variant"] == vname) & (paired_df["split"] == "eval")].copy()
        if eval_block.empty:
            continue

        summary_eval = summary_df[(summary_df["variant"] == vname) & (summary_df["split"] == "eval")].copy()

        stake_sum = _to_num(summary_eval.get("total_stake")).sum() if "total_stake" in summary_eval.columns else float("nan")
        profit_sum = _to_num(summary_eval.get("total_profit")).sum() if "total_profit" in summary_eval.columns else float("nan")
        pooled_roi = (profit_sum / stake_sum) if stake_sum and pd.notna(stake_sum) else float("nan")
        pooled_n_bets = int(_to_num(summary_eval.get("n_bets")).sum()) if "n_bets" in summary_eval.columns else None

        median_n_bets_var = float(eval_block["n_bets_var"].median())
        min_n_bets_var = int(eval_block["n_bets_var"].min())
        zero_bet_windows = int((eval_block["n_bets_var"] <= 0).sum())

        improve_rate = float((eval_block["d_roi"] > 0).mean())
        median_d_roi = float(eval_block["d_roi"].median())
        median_d_maxdd = float(eval_block["d_max_dd"].median())
        median_d_n_bets = float(eval_block["d_n_bets"].median())

        gate_pass = (
            improve_rate >= 0.6
            and median_d_roi > 0
            and median_d_maxdd <= 0
            and median_n_bets_var >= 80
            and zero_bet_windows == 0
        )

        eval_rows.append(
            {
                "variant": vname,
                "median_d_roi": median_d_roi,
                "improve_rate_roi": improve_rate,
                "median_d_maxdd": median_d_maxdd,
                "median_d_n_bets": median_d_n_bets,
                "median_n_bets_var": median_n_bets_var,
                "min_n_bets_var": min_n_bets_var,
                "n_zero_bet_windows_var": zero_bet_windows,
                "pooled_roi_eval": pooled_roi,
                "pooled_n_bets_eval": pooled_n_bets,
                "gate_pass": gate_pass,
            }
        )

    eval_metrics = pd.DataFrame(eval_rows)
    eval_metrics.to_csv(out_dir / "eval_metrics.csv", index=False, encoding="utf-8")

    # report
    report_path = out_dir / "report.md"
    report_lines = []
    if args.append_report and report_path.exists():
        report_lines.append(report_path.read_text(encoding="utf-8"))
        report_lines.append("")
    report_lines.append("## Eval gate summary")
    report_lines.append(_df_to_md(eval_metrics))
    report_lines.append("")
    report_lines.append("Gate criteria: improve_rate>=0.6, median_d_roi>0, median_d_maxdd<=0, median_n_bets_var>=80, zero_bet_windows=0")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    if args.phase0_base_bands:
        base_dirs = [Path(p) for p in variants[baseline]]
        bands = _bands_from_bets(base_dirs, args.design_max_idx, args.max_odds)
        bands.to_csv(out_dir / "phase0_base_odds_bands.csv", index=False, encoding="utf-8")

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
