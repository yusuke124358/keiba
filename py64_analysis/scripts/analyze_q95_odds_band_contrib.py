"""Analyze odds-band contribution for q95 eval bets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


STAKE_COLS = ["stake", "stake_yen", "stake_amount", "bet_amount", "stake_raw"]
RETURN_COLS = ["return", "return_yen", "payout", "payout_yen"]
PROFIT_COLS = ["profit", "net_profit", "pnl"]
ODDS_COLS = ["odds_at_buy", "odds", "buy_odds"]


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _collect_bets(run_dir: Path) -> pd.DataFrame:
    rows = []
    for bets_path in sorted(run_dir.glob("w*/bets.csv")):
        try:
            df = pd.read_csv(bets_path)
        except Exception:
            continue
        if df.empty:
            continue
        stake_col = _find_col(list(df.columns), STAKE_COLS)
        odds_col = _find_col(list(df.columns), ODDS_COLS)
        ret_col = _find_col(list(df.columns), RETURN_COLS)
        prof_col = _find_col(list(df.columns), PROFIT_COLS)
        if stake_col is None or odds_col is None:
            continue
        stake = pd.to_numeric(df[stake_col], errors="coerce")
        odds = pd.to_numeric(df[odds_col], errors="coerce")
        if ret_col is not None:
            ret = pd.to_numeric(df[ret_col], errors="coerce")
            profit = ret - stake
        elif prof_col is not None:
            profit = pd.to_numeric(df[prof_col], errors="coerce")
        else:
            continue
        rows.append(pd.DataFrame({"odds": odds, "stake": stake, "profit": profit}))
    if not rows:
        return pd.DataFrame(columns=["odds", "stake", "profit"])
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["odds", "stake", "profit"])
    return out


def _bin_and_aggregate(df: pd.DataFrame, bins: list[float], labels: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["odds_bin", "stake_sum", "profit_sum", "roi", "n_bets", "stake_share"])
    df = df.copy()
    df["odds_bin"] = pd.cut(df["odds"], bins=bins, labels=labels, right=True, include_lowest=True)
    df = df.dropna(subset=["odds_bin"])
    grouped = (
        df.groupby("odds_bin", dropna=False)
        .agg(stake_sum=("stake", "sum"), profit_sum=("profit", "sum"), n_bets=("stake", "size"))
        .reset_index()
    )
    total_stake = float(grouped["stake_sum"].sum())
    grouped["roi"] = grouped.apply(
        lambda r: (r["profit_sum"] / r["stake_sum"]) if r["stake_sum"] else np.nan,
        axis=1,
    )
    grouped["stake_share"] = grouped["stake_sum"] / total_stake if total_stake else np.nan
    return grouped


def _summarize_md(df: pd.DataFrame, title: str) -> list[str]:
    lines = [f"## {title}", ""]
    if df.empty:
        lines.append("- No bets found.")
        lines.append("")
        return lines
    worst = df.sort_values(["roi", "stake_share"], ascending=[True, False]).head(3)
    lines.append("- Worst ROI bins (low ROI, high stake share):")
    for _, r in worst.iterrows():
        lines.append(
            f"  - bin={r['odds_bin']} roi={r['roi']:.4f} stake_share={r['stake_share']:.3f} "
            f"stake={r['stake_sum']:.1f} n_bets={int(r['n_bets'])}"
        )
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir-2024", required=True)
    ap.add_argument("--run-dir-2025", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, float("inf")]
    labels = ["1-2", "2-3", "3-5", "5-10", "10-15", "15-20", "20-30", "30+"]

    df_2024 = _collect_bets(Path(args.run_dir_2024))
    df_2025 = _collect_bets(Path(args.run_dir_2025))
    df_2024["year"] = 2024
    df_2025["year"] = 2025
    df_all = pd.concat([df_2024, df_2025], ignore_index=True)

    agg_2024 = _bin_and_aggregate(df_2024, bins=bins, labels=labels)
    agg_2025 = _bin_and_aggregate(df_2025, bins=bins, labels=labels)
    agg_all = _bin_and_aggregate(df_all, bins=bins, labels=labels)

    agg_2024.to_csv(out_dir / "odds_band_contrib_2024.csv", index=False, encoding="utf-8")
    agg_2025.to_csv(out_dir / "odds_band_contrib_2025.csv", index=False, encoding="utf-8")
    agg_all.to_csv(out_dir / "odds_band_contrib_joint.csv", index=False, encoding="utf-8")

    lines = ["# q95 odds band contribution", ""]
    lines.extend(_summarize_md(agg_2024, "2024 eval"))
    lines.extend(_summarize_md(agg_2025, "2025 eval"))
    lines.extend(_summarize_md(agg_all, "2024+2025 joint"))
    lines.append("ROI definition: ROI = profit / stake, profit = return - stake.")
    (out_dir / "odds_band_contrib.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        "[analysis] q95_odds_band_contrib_written=true | bins="
        + ",".join(labels)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
