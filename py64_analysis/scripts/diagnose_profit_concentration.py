from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


WINDOW_RE = re.compile(r"^w\d{3}_\d{8}_\d{8}$")


@dataclass
class DatasetSpec:
    name: str
    base_dirs: list[Path]
    cand_dirs: list[Path]


def _infer_window_id(path: Path) -> str:
    for part in path.parts[::-1]:
        if WINDOW_RE.match(part):
            return part
    return "unknown"


def _parse_dataset_arg(value: str) -> DatasetSpec:
    parts = value.split("|")
    if len(parts) != 3:
        raise ValueError("dataset arg must be name|base_dirs|candidate_dirs")
    name, base_raw, cand_raw = parts
    base_dirs = [Path(p.strip()) for p in base_raw.split(";") if p.strip()]
    cand_dirs = [Path(p.strip()) for p in cand_raw.split(";") if p.strip()]
    return DatasetSpec(name=name.strip(), base_dirs=base_dirs, cand_dirs=cand_dirs)


def _load_bets_from_dirs(dirs: list[Path], variant: str, dataset: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for base in dirs:
        if not base.exists():
            continue
        for path in base.rglob("bets.csv"):
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["dataset"] = dataset
            df["variant"] = variant
            df["window_id"] = _infer_window_id(path)
            df["__source_path__"] = str(path)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    merged = pd.concat(rows, ignore_index=True)
    return _dedup_bets(merged)


def _dedup_bets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = []
    for col in ("window_id", "race_id", "horse_no", "asof_time"):
        if col in df.columns:
            keys.append(col)
    if not keys:
        return df
    return df.drop_duplicates(keys, keep="first")


def _extract_day_key(df: pd.DataFrame) -> pd.Series:
    if "race_id" in df.columns:
        race_id = df["race_id"].astype(str)
        mask = race_id.str.len() >= 8
        date_key = pd.Series(index=race_id.index, dtype="object")
        date_key.loc[mask] = race_id.loc[mask].str.slice(0, 8)
        date_key = date_key.where(date_key.str.fullmatch(r"\d{8}"))
        if date_key.notna().any():
            return date_key
    if "asof_time" in df.columns:
        dt = pd.to_datetime(df["asof_time"], errors="coerce")
        date_key = dt.dt.strftime("%Y%m%d")
        if date_key.notna().any():
            return date_key
    cols = ", ".join(df.columns)
    raise SystemExit(f"day_key inference failed; available_cols=[{cols}]")


def _odds_band(val: float) -> str:
    if val < 3:
        return "<3"
    if val < 5:
        return "3-5"
    if val < 10:
        return "5-10"
    if val < 20:
        return "10-20"
    return "20+"


def _prepare_profit(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    profit_col = "profit" if "profit" in work.columns else None
    if profit_col:
        profit = pd.to_numeric(work[profit_col], errors="coerce")
    else:
        payout = pd.to_numeric(work.get("payout"), errors="coerce")
        stake = pd.to_numeric(work.get("stake"), errors="coerce")
        profit = payout - stake
    stake = pd.to_numeric(work.get("stake"), errors="coerce")
    odds_buy = pd.to_numeric(work.get("odds_at_buy"), errors="coerce")
    work["profit_yen"] = profit.fillna(0.0)
    work["stake_yen"] = stake.fillna(0.0)
    work["odds_used"] = odds_buy
    return work


def _top_k_share(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return float("nan")
    total = float(values.sum())
    if total <= 0:
        return float("nan")
    topk = np.sort(values)[-k:]
    return float(topk.sum() / total)


def _segment_label(df: pd.DataFrame) -> pd.Series:
    for col in ("segblend_segment", "segment", "surface"):
        if col in df.columns:
            return df[col].astype(str).fillna("unknown")
    return pd.Series(["unknown"] * len(df))


def _summarize_variant(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_profit": 0.0,
            "total_stake": 0.0,
            "roi": float("nan"),
            "n_bets": 0,
            "top10_profit_share": float("nan"),
            "top50_profit_share": float("nan"),
            "frac_profitable_days": float("nan"),
            "worst5_day_loss_sum": float("nan"),
            "best5_day_gain_sum": float("nan"),
        }

    work = _prepare_profit(df)
    total_profit = float(work["profit_yen"].sum())
    total_stake = float(work["stake_yen"].sum())
    roi = total_profit / total_stake if total_stake > 0 else float("nan")

    day_key = _extract_day_key(work)
    day_key = day_key.str.slice(0, 4) + "-" + day_key.str.slice(4, 6) + "-" + day_key.str.slice(6, 8)
    work["day_key"] = day_key
    day_stats = work.groupby("day_key", dropna=False).agg(
        profit=("profit_yen", "sum"),
        stake=("stake_yen", "sum"),
        n_bets=("profit_yen", "size"),
    ).reset_index()
    day_stats["roi"] = day_stats["profit"] / day_stats["stake"].replace(0, np.nan)
    day_stats["day_dt"] = pd.to_datetime(day_stats["day_key"], errors="coerce")
    day_stats = day_stats.sort_values("day_dt")
    cum = day_stats["profit"].cumsum()
    peak = cum.cummax()
    drawdown = (peak - cum).fillna(0.0)
    max_drawdown = float(drawdown.max()) if len(drawdown) else float("nan")
    dd_p90 = float(drawdown.quantile(0.9)) if len(drawdown) else float("nan")
    dd_p95 = float(drawdown.quantile(0.95)) if len(drawdown) else float("nan")
    frac_profitable_days = float((day_stats["profit"] > 0).mean()) if len(day_stats) else float("nan")
    worst5 = day_stats.sort_values("profit").head(5)["profit"].to_numpy()
    best5 = day_stats.sort_values("profit", ascending=False).head(5)["profit"].to_numpy()

    profits = work["profit_yen"].to_numpy()
    top10_share = _top_k_share(profits, 10)
    top50_share = _top_k_share(profits, 50)

    return {
        "total_profit": total_profit,
        "total_stake": total_stake,
        "roi": roi,
        "n_bets": int(len(work)),
        "top10_profit_share": top10_share,
        "top50_profit_share": top50_share,
        "frac_profitable_days": frac_profitable_days,
        "worst5_day_loss_sum": float(worst5.sum()) if worst5.size else float("nan"),
        "best5_day_gain_sum": float(best5.sum()) if best5.size else float("nan"),
        "max_drawdown": max_drawdown,
        "drawdown_p90": dd_p90,
        "drawdown_p95": dd_p95,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose profit concentration / tail risk.")
    ap.add_argument("--dataset", action="append", required=True, help="name|base_dirs(semi-colon)|candidate_dirs(semi-colon)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [_parse_dataset_arg(v) for v in args.dataset]

    summary_rows = []
    day_rows = []
    odds_rows = []
    seg_rows = []

    for ds in datasets:
        base_df = _load_bets_from_dirs(ds.base_dirs, "base", ds.name)
        cand_df = _load_bets_from_dirs(ds.cand_dirs, "candidate", ds.name)

        for variant, df in [("base", base_df), ("candidate", cand_df)]:
            stats = _summarize_variant(df)
            summary_rows.append({"dataset": ds.name, "variant": variant, **stats})

            if df.empty:
                continue

            work = _prepare_profit(df)
            day_key = _extract_day_key(work)
            day_key = day_key.str.slice(0, 4) + "-" + day_key.str.slice(4, 6) + "-" + day_key.str.slice(6, 8)
            work["day_key"] = day_key
            day_stats = work.groupby("day_key", dropna=False).agg(
                profit=("profit_yen", "sum"),
                stake=("stake_yen", "sum"),
                n_bets=("profit_yen", "size"),
            ).reset_index()
            day_stats["roi"] = day_stats["profit"] / day_stats["stake"].replace(0, np.nan)
            day_stats["dataset"] = ds.name
            day_stats["variant"] = variant
            day_rows.append(day_stats)

            odds = pd.to_numeric(work.get("odds_at_buy"), errors="coerce")
            work["odds_band"] = odds.apply(lambda v: _odds_band(float(v)) if pd.notna(v) else "unknown")
            odds_stats = work.groupby("odds_band", dropna=False).agg(
                profit=("profit_yen", "sum"),
                stake=("stake_yen", "sum"),
                n_bets=("profit_yen", "size"),
            ).reset_index()
            odds_stats["roi"] = odds_stats["profit"] / odds_stats["stake"].replace(0, np.nan)
            odds_stats["dataset"] = ds.name
            odds_stats["variant"] = variant
            odds_rows.append(odds_stats)

            work["segment"] = _segment_label(work)
            seg_stats = work.groupby("segment", dropna=False).agg(
                profit=("profit_yen", "sum"),
                stake=("stake_yen", "sum"),
                n_bets=("profit_yen", "size"),
            ).reset_index()
            seg_stats["roi"] = seg_stats["profit"] / seg_stats["stake"].replace(0, np.nan)
            seg_stats["dataset"] = ds.name
            seg_stats["variant"] = variant
            seg_rows.append(seg_stats)

    summary_df = pd.DataFrame(summary_rows)
    day_df = pd.concat(day_rows, ignore_index=True) if day_rows else pd.DataFrame()
    odds_df = pd.concat(odds_rows, ignore_index=True) if odds_rows else pd.DataFrame()
    seg_df = pd.concat(seg_rows, ignore_index=True) if seg_rows else pd.DataFrame()

    summary_df.to_csv(out_dir / "profit_concentration_summary.csv", index=False, encoding="utf-8")
    day_df.to_csv(out_dir / "profit_concentration_day_stats.csv", index=False, encoding="utf-8")
    odds_df.to_csv(out_dir / "profit_concentration_odds_band.csv", index=False, encoding="utf-8")
    seg_df.to_csv(out_dir / "profit_concentration_segment.csv", index=False, encoding="utf-8")

    report_lines = [
        "# Profit concentration / tail risk diagnostic",
        "",
        "## Summary (per dataset/variant)",
        summary_df.to_string(index=False) if not summary_df.empty else "No data.",
        "",
        "## Notes",
        "- profits use yen profit (payout - stake) when available.",
        "- de-dup by window_id, race_id, horse_no, asof_time across input dirs.",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
