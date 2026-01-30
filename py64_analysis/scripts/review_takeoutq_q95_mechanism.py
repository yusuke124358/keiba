"""
Review where takeoutq q95 helps vs baseline (no reruns). Produces report + CSVs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import re

import numpy as np
import pandas as pd

WIN_RE = re.compile(r"^w\d{3}_(\d{8})_(\d{8})$")

ODDS_BANDS = [(1.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, float("inf"))]

HORSE_COL_CANDIDATES = ["horse_no", "horse_num", "horse_number", "umaban", "horse_id"]
STAKE_COL_CANDIDATES = ["stake", "stake_yen", "stake_amount", "bet_amount", "stake_raw"]
PROFIT_COL_CANDIDATES = ["profit", "net_profit", "pnl"]
RETURN_COL_CANDIDATES = ["return", "return_yen", "payout", "payout_yen"]
ODDS_COL_CANDIDATES = ["odds_at_buy", "odds_effective", "odds_final"]
EV_COL_CANDIDATES = ["ev", "ev_adj"]
TAKEOUT_COL_CANDIDATES = ["takeout_implied", "overround_sum_inv"]
TRACK_COL_CANDIDATES = ["track_code", "track", "track_id"]
SURFACE_COL_CANDIDATES = ["surface", "surface_type"]


@dataclass(frozen=True)
class YearSpec:
    year: int
    baseline_design: Path
    baseline_eval: Path
    q95_design: Path
    q95_eval: Path


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _df_to_md(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_no data_"
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False):
        cells = [_fmt(v, digits=digits) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _read_header(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    return list(pd.read_csv(path, nrows=0).columns)


def _find_col(columns: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def _odds_band(val: float) -> str:
    if not np.isfinite(val):
        return "missing"
    for lo, hi in ODDS_BANDS:
        if lo <= val < hi:
            return f"{lo:.0f}-{hi:.0f}" if hi != float("inf") else f"{lo:.0f}+"
    return "missing"


def _make_bins(values: list[float], n_bins: int) -> Optional[list[float]]:
    if not values:
        return None
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < n_bins:
        return None
    edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1)).tolist()
    edges = [float(e) for e in edges]
    cleaned = [edges[0]]
    for e in edges[1:]:
        if e > cleaned[-1]:
            cleaned.append(e)
    if len(cleaned) < 3:
        return None
    return cleaned


def _assign_bins(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    bins = [-np.inf] + edges[1:-1] + [np.inf]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def _bucket_label(series: pd.Series, bucket_type: str, edges: Optional[list[float]]) -> pd.Series:
    if bucket_type == "odds_band":
        vals = pd.to_numeric(series, errors="coerce")
        return vals.apply(lambda x: _odds_band(x) if pd.notna(x) else "missing")
    if bucket_type == "ev_decile":
        if not edges:
            return pd.Series(["missing"] * len(series))
        labels = [f"D{i}" for i in range(1, len(edges))]
        vals = pd.to_numeric(series, errors="coerce")
        return _assign_bins(vals, edges, labels).astype(str)
    if bucket_type == "takeout_bucket":
        if not edges:
            return pd.Series(["missing"] * len(series))
        labels = [f"Q{i}" for i in range(1, len(edges))]
        vals = pd.to_numeric(series, errors="coerce")
        return _assign_bins(vals, edges, labels).astype(str)
    return series.fillna("missing").astype(str)


def _collect_windows(run_dir: Path) -> dict[str, Path]:
    if not run_dir.exists():
        raise SystemExit(f"missing run dir: {run_dir}")
    windows = {}
    for p in run_dir.iterdir():
        if p.is_dir() and WIN_RE.match(p.name):
            windows[p.name] = p
    if not windows:
        raise SystemExit(f"no windows found in {run_dir}")
    return dict(sorted(windows.items()))


def _aggregate_set(df: pd.DataFrame) -> dict[str, Any]:
    stake = float(df["__stake__"].sum())
    profit = float(df["__profit__"].sum())
    n_bets = int(len(df))
    roi = profit / stake if stake > 0 else 0.0
    return {"stake": stake, "profit": profit, "n_bets": n_bets, "roi": roi}


def _load_bets(path: Path) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
    header = _read_header(path)
    if "race_id" not in header:
        raise SystemExit(f"missing race_id in {path}")
    horse_col = _find_col(header, HORSE_COL_CANDIDATES)
    if horse_col is None:
        raise SystemExit(f"missing horse column in {path}: {HORSE_COL_CANDIDATES}")
    stake_col = _find_col(header, STAKE_COL_CANDIDATES)
    if stake_col is None:
        raise SystemExit(f"missing stake column in {path}: {STAKE_COL_CANDIDATES}")
    profit_col = _find_col(header, PROFIT_COL_CANDIDATES)
    return_col = _find_col(header, RETURN_COL_CANDIDATES)
    odds_col = _find_col(header, ODDS_COL_CANDIDATES)
    ev_col = _find_col(header, EV_COL_CANDIDATES)
    takeout_col = _find_col(header, TAKEOUT_COL_CANDIDATES)
    track_col = _find_col(header, TRACK_COL_CANDIDATES)
    surface_col = _find_col(header, SURFACE_COL_CANDIDATES)

    if profit_col is None and return_col is None:
        raise SystemExit(f"missing profit/return in {path}: {PROFIT_COL_CANDIDATES}/{RETURN_COL_CANDIDATES}")

    usecols = ["race_id", horse_col, stake_col]
    if profit_col:
        usecols.append(profit_col)
    if return_col and return_col not in usecols:
        usecols.append(return_col)
    for col in [odds_col, ev_col, takeout_col, track_col, surface_col]:
        if col and col not in usecols:
            usecols.append(col)

    df = pd.read_csv(path, usecols=usecols)
    df["__race_id__"] = df["race_id"].astype(str)
    df["__horse__"] = df[horse_col].astype(str)
    df["__key__"] = df["__race_id__"] + "|" + df["__horse__"]

    df["__stake__"] = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
    if profit_col:
        df["__profit__"] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0)
    else:
        ret = pd.to_numeric(df[return_col], errors="coerce").fillna(0.0)
        df["__profit__"] = ret - df["__stake__"]

    meta = {
        "horse_col": horse_col,
        "stake_col": stake_col,
        "profit_col": profit_col,
        "return_col": return_col,
        "odds_col": odds_col,
        "ev_col": ev_col,
        "takeout_col": takeout_col,
        "track_col": track_col,
        "surface_col": surface_col,
    }
    return df, meta


def _group_bucket_metrics(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    labels = labels.copy()
    labels.name = "bucket_label"
    grouped = (
        df.groupby(labels, dropna=False)
        .agg(stake=("__stake__", "sum"), profit=("__profit__", "sum"), n_bets=("__stake__", "size"))
        .reset_index()
    )
    grouped["roi"] = grouped["profit"] / grouped["stake"]
    grouped["roi"] = grouped["roi"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return grouped


def _bootstrap_delta_roi(
    daily_df: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    block_days: int,
    n_samples: int,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pivot = daily_df.pivot_table(
        index="date",
        columns="strategy",
        values=["profit", "stake"],
        aggfunc="sum",
        fill_value=0.0,
    )
    if strategy_a not in pivot["profit"].columns or strategy_b not in pivot["profit"].columns:
        return {}

    profits_a = pivot["profit"][strategy_a].to_numpy()
    profits_b = pivot["profit"][strategy_b].to_numpy()
    stakes_a = pivot["stake"][strategy_a].to_numpy()
    stakes_b = pivot["stake"][strategy_b].to_numpy()

    n_days = len(profits_a)
    if n_days == 0:
        return {}

    max_start = max(0, n_days - block_days)
    starts = np.arange(max_start + 1)
    n_blocks = int(np.ceil(n_days / block_days))

    deltas = []
    for _ in range(n_samples):
        idx = []
        for _ in range(n_blocks):
            s = rng.choice(starts)
            idx.extend(range(s, min(s + block_days, n_days)))
            if len(idx) >= n_days:
                break
        idx = np.array(idx[:n_days])
        stake_a = stakes_a[idx].sum()
        stake_b = stakes_b[idx].sum()
        roi_a = profits_a[idx].sum() / stake_a if stake_a > 0 else 0.0
        roi_b = profits_b[idx].sum() / stake_b if stake_b > 0 else 0.0
        deltas.append(roi_b - roi_a)

    deltas = np.asarray(deltas)
    return {
        "n_samples": n_samples,
        "delta_roi_mean": float(np.mean(deltas)),
        "delta_roi_p50": float(np.quantile(deltas, 0.50)),
        "delta_roi_p05": float(np.quantile(deltas, 0.05)),
        "delta_roi_p95": float(np.quantile(deltas, 0.95)),
        "delta_roi_p025": float(np.quantile(deltas, 0.025)),
        "delta_roi_p975": float(np.quantile(deltas, 0.975)),
        "prob_delta_roi_gt_0": float((deltas > 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Review q95 mechanism vs baseline")
    ap.add_argument("--year", action="append", type=int, required=True)
    ap.add_argument("--baseline-design", action="append", required=True)
    ap.add_argument("--baseline-eval", action="append", required=True)
    ap.add_argument("--q95-design", action="append", required=True)
    ap.add_argument("--q95-eval", action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--agents-path", type=Path, required=True)
    ap.add_argument("--step14-daily", type=Path, default=None)
    ap.add_argument("--bootstrap-block-days", type=int, default=7)
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--topk-removed", type=int, default=20)
    args = ap.parse_args()

    if not (
        len(args.year)
        == len(args.baseline_design)
        == len(args.baseline_eval)
        == len(args.q95_design)
        == len(args.q95_eval)
    ):
        raise SystemExit("Mismatched --year/--baseline-*/--q95-* counts.")

    specs = []
    for i in range(len(args.year)):
        specs.append(
            YearSpec(
                year=int(args.year[i]),
                baseline_design=Path(args.baseline_design[i]),
                baseline_eval=Path(args.baseline_eval[i]),
                q95_design=Path(args.q95_design[i]),
                q95_eval=Path(args.q95_eval[i]),
            )
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    bucket_breakdown_rows: list[dict[str, Any]] = []
    bucket_delta_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    total_rows: list[dict[str, Any]] = []
    col_notes: list[dict[str, Any]] = []

    ev_bins: dict[tuple[int, str], Optional[list[float]]] = {}
    takeout_bins: dict[tuple[int, str], Optional[list[float]]] = {}
    takeout_metric: dict[tuple[int, str], Optional[str]] = {}
    ev_available: dict[tuple[int, str], bool] = {}
    takeout_available: dict[tuple[int, str], bool] = {}

    for spec in specs:
        for split, base_dir, q95_dir in [
            ("design", spec.baseline_design, spec.q95_design),
            ("eval", spec.baseline_eval, spec.q95_eval),
        ]:
            base_windows = _collect_windows(base_dir)
            q95_windows = _collect_windows(q95_dir)
            shared = sorted(set(base_windows) & set(q95_windows))
            ev_vals: list[float] = []
            takeout_vals: list[float] = []
            takeout_col_name = None
            ev_col_name = None
            for w in shared:
                for path in [base_windows[w] / "bets.csv", q95_windows[w] / "bets.csv"]:
                    if not path.exists():
                        continue
                    header = _read_header(path)
                    ev_col = _find_col(header, EV_COL_CANDIDATES)
                    takeout_col = _find_col(header, TAKEOUT_COL_CANDIDATES)
                    if ev_col:
                        ev_col_name = ev_col_name or ev_col
                        ev_vals.extend(pd.read_csv(path, usecols=[ev_col])[ev_col].dropna().tolist())
                    if takeout_col:
                        takeout_col_name = takeout_col_name or takeout_col
                        takeout_vals.extend(pd.read_csv(path, usecols=[takeout_col])[takeout_col].dropna().tolist())
            ev_bins[(spec.year, split)] = _make_bins(ev_vals, 10) if ev_col_name else None
            takeout_bins[(spec.year, split)] = _make_bins(takeout_vals, 5) if takeout_col_name else None
            ev_available[(spec.year, split)] = bool(ev_col_name)
            takeout_available[(spec.year, split)] = bool(takeout_col_name)
            takeout_metric[(spec.year, split)] = takeout_col_name

    for spec in specs:
        for split, base_dir, q95_dir in [
            ("design", spec.baseline_design, spec.q95_design),
            ("eval", spec.baseline_eval, spec.q95_eval),
        ]:
            base_windows = _collect_windows(base_dir)
            q95_windows = _collect_windows(q95_dir)
            shared = sorted(set(base_windows) & set(q95_windows))
            missing = sorted(set(base_windows) ^ set(q95_windows))
            for w in missing:
                missing_rows.append(
                    {
                        "year": spec.year,
                        "split": split,
                        "window_name": w,
                        "path": str(base_windows.get(w, q95_windows.get(w))),
                        "reason": "window_missing_in_pair",
                    }
                )

            for w in shared:
                base_path = base_windows[w] / "bets.csv"
                q95_path = q95_windows[w] / "bets.csv"
                if not base_path.exists():
                    missing_rows.append(
                        {
                            "year": spec.year,
                            "split": split,
                            "window_name": w,
                            "path": str(base_path),
                            "reason": "missing_baseline_bets",
                        }
                    )
                    continue
                if not q95_path.exists():
                    missing_rows.append(
                        {
                            "year": spec.year,
                            "split": split,
                            "window_name": w,
                            "path": str(q95_path),
                            "reason": "missing_q95_bets",
                        }
                    )
                    continue

                base_df, base_meta = _load_bets(base_path)
                q95_df, q95_meta = _load_bets(q95_path)

                col_notes.append(
                    {
                        "year": spec.year,
                        "split": split,
                        "window_name": w,
                        "horse_col_base": base_meta["horse_col"],
                        "horse_col_q95": q95_meta["horse_col"],
                        "odds_col": base_meta.get("odds_col") or q95_meta.get("odds_col"),
                        "ev_col": base_meta.get("ev_col") or q95_meta.get("ev_col"),
                        "takeout_col": base_meta.get("takeout_col") or q95_meta.get("takeout_col"),
                    }
                )

                base_total = _aggregate_set(base_df)
                q95_total = _aggregate_set(q95_df)
                total_rows.append(
                    {
                        "year": spec.year,
                        "split": split,
                        "window_name": w,
                        "baseline_stake": base_total["stake"],
                        "baseline_profit": base_total["profit"],
                        "baseline_n_bets": base_total["n_bets"],
                        "q95_stake": q95_total["stake"],
                        "q95_profit": q95_total["profit"],
                        "q95_n_bets": q95_total["n_bets"],
                    }
                )

                base_keys = set(base_df["__key__"])
                q95_keys = set(q95_df["__key__"])
                common_keys = base_keys & q95_keys

                removed = base_df[~base_df["__key__"].isin(common_keys)].copy()
                added = q95_df[~q95_df["__key__"].isin(common_keys)].copy()
                common_base = base_df[base_df["__key__"].isin(common_keys)].copy()
                common_q95 = q95_df[q95_df["__key__"].isin(common_keys)].copy()

                for label, base_set, q95_set in [
                    ("common", common_base, common_q95),
                    ("removed", removed, pd.DataFrame(columns=base_df.columns)),
                    ("added", pd.DataFrame(columns=q95_df.columns), added),
                ]:
                    base_metrics = _aggregate_set(base_set) if not base_set.empty else {"stake": 0.0, "profit": 0.0, "n_bets": 0}
                    q95_metrics = _aggregate_set(q95_set) if not q95_set.empty else {"stake": 0.0, "profit": 0.0, "n_bets": 0}
                    summary_rows.append(
                        {
                            "year": spec.year,
                            "split": split,
                            "window_name": w,
                            "set": label,
                            "baseline_stake": base_metrics["stake"],
                            "baseline_profit": base_metrics["profit"],
                            "baseline_n_bets": base_metrics["n_bets"],
                            "q95_stake": q95_metrics["stake"],
                            "q95_profit": q95_metrics["profit"],
                            "q95_n_bets": q95_metrics["n_bets"],
                        }
                    )

                if not removed.empty:
                    odds_col = base_meta.get("odds_col")
                    ev_col = base_meta.get("ev_col")
                    takeout_col = base_meta.get("takeout_col")
                    track_col = base_meta.get("track_col")
                    surface_col = base_meta.get("surface_col")
                    sample = removed.nsmallest(args.topk_removed, "__profit__").copy()
                    for _, row in sample.iterrows():
                        removed_rows.append(
                            {
                                "year": spec.year,
                                "split": split,
                                "window_name": w,
                                "race_id": row["__race_id__"],
                                "horse_no": row["__horse__"],
                                "stake": row["__stake__"],
                                "profit": row["__profit__"],
                                "odds": row[odds_col] if odds_col else None,
                                "ev": row[ev_col] if ev_col else None,
                                "takeout_metric": takeout_col,
                                "takeout_value": row[takeout_col] if takeout_col else None,
                                "track": row[track_col] if track_col else None,
                                "surface": row[surface_col] if surface_col else None,
                            }
                        )

                bucket_specs: list[tuple[str, Optional[str], Optional[str], Optional[list[float]]]] = []
                if base_meta.get("odds_col") or q95_meta.get("odds_col"):
                    bucket_specs.append(("odds_band", base_meta.get("odds_col"), q95_meta.get("odds_col"), None))
                if ev_available[(spec.year, split)]:
                    bucket_specs.append(
                        ("ev_decile", base_meta.get("ev_col"), q95_meta.get("ev_col"), ev_bins[(spec.year, split)])
                    )
                if takeout_available[(spec.year, split)]:
                    bucket_specs.append(
                        (
                            "takeout_bucket",
                            base_meta.get("takeout_col"),
                            q95_meta.get("takeout_col"),
                            takeout_bins[(spec.year, split)],
                        )
                    )
                if base_meta.get("track_col") or q95_meta.get("track_col"):
                    bucket_specs.append(("track", base_meta.get("track_col"), q95_meta.get("track_col"), None))
                if base_meta.get("surface_col") or q95_meta.get("surface_col"):
                    bucket_specs.append(("surface", base_meta.get("surface_col"), q95_meta.get("surface_col"), None))

                for bucket_type, base_col, q95_col, edges in bucket_specs:
                    base_labels = None
                    q95_labels = None
                    if base_col:
                        base_labels = _bucket_label(base_df[base_col], bucket_type, edges)
                    if q95_col:
                        q95_labels = _bucket_label(q95_df[q95_col], bucket_type, edges)

                    if base_labels is not None:
                        if not removed.empty:
                            group = _group_bucket_metrics(removed, base_labels.loc[removed.index])
                            for _, row in group.iterrows():
                                bucket_breakdown_rows.append(
                                    {
                                        "year": spec.year,
                                        "split": split,
                                        "window_name": w,
                                        "set": "removed",
                                        "bucket_type": bucket_type,
                                        "bucket_label": row["bucket_label"],
                                        "stake": row["stake"],
                                        "profit": row["profit"],
                                        "n_bets": row["n_bets"],
                                        "roi": row["roi"],
                                    }
                                )
                        if not common_base.empty:
                            group = _group_bucket_metrics(common_base, base_labels.loc[common_base.index])
                            for _, row in group.iterrows():
                                bucket_breakdown_rows.append(
                                    {
                                        "year": spec.year,
                                        "split": split,
                                        "window_name": w,
                                        "set": "common",
                                        "bucket_type": bucket_type,
                                        "bucket_label": row["bucket_label"],
                                        "stake": row["stake"],
                                        "profit": row["profit"],
                                        "n_bets": row["n_bets"],
                                        "roi": row["roi"],
                                    }
                                )
                            common_delta = common_base[["__key__", "__profit__"]].rename(
                                columns={"__profit__": "__profit__base"}
                            ).merge(
                                common_q95[["__key__", "__profit__"]].rename(
                                    columns={"__profit__": "__profit__q95"}
                                ),
                                on="__key__",
                                how="inner",
                            )
                            common_delta["delta_profit"] = common_delta["__profit__q95"] - common_delta["__profit__base"]
                            common_delta["bucket_label"] = base_labels.loc[common_base.index].values
                            delta_group = common_delta.groupby("bucket_label")["delta_profit"].sum().reset_index()
                            for _, row in delta_group.iterrows():
                                bucket_delta_rows.append(
                                    {
                                        "year": spec.year,
                                        "split": split,
                                        "window_name": w,
                                        "bucket_type": bucket_type,
                                        "bucket_label": row["bucket_label"],
                                        "common_profit_delta": row["delta_profit"],
                                    }
                                )
                    if q95_labels is not None and not added.empty:
                        group = _group_bucket_metrics(added, q95_labels.loc[added.index])
                        for _, row in group.iterrows():
                            bucket_breakdown_rows.append(
                                {
                                    "year": spec.year,
                                    "split": split,
                                    "window_name": w,
                                    "set": "added",
                                    "bucket_type": bucket_type,
                                    "bucket_label": row["bucket_label"],
                                    "stake": row["stake"],
                                    "profit": row["profit"],
                                    "n_bets": row["n_bets"],
                                    "roi": row["roi"],
                                }
                            )

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(out_dir / "missing_files.csv", index=False)
    if not missing_df.empty:
        raise SystemExit(f"Missing bets.csv/windows detected. See {out_dir / 'missing_files.csv'}")

    summary_df = pd.DataFrame(summary_rows)
    summary_agg = (
        summary_df.groupby(["year", "split", "set"], dropna=False)
        .agg(
            baseline_stake=("baseline_stake", "sum"),
            baseline_profit=("baseline_profit", "sum"),
            baseline_n_bets=("baseline_n_bets", "sum"),
            q95_stake=("q95_stake", "sum"),
            q95_profit=("q95_profit", "sum"),
            q95_n_bets=("q95_n_bets", "sum"),
        )
        .reset_index()
    )
    summary_agg["baseline_roi"] = summary_agg["baseline_profit"] / summary_agg["baseline_stake"]
    summary_agg["q95_roi"] = summary_agg["q95_profit"] / summary_agg["q95_stake"]
    summary_agg["baseline_roi"] = summary_agg["baseline_roi"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    summary_agg["q95_roi"] = summary_agg["q95_roi"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    summary_agg.to_csv(out_dir / "summary_sets.csv", index=False)

    totals_df = pd.DataFrame(total_rows)
    totals_agg = (
        totals_df.groupby(["year", "split"], dropna=False)
        .agg(
            baseline_stake=("baseline_stake", "sum"),
            baseline_profit=("baseline_profit", "sum"),
            baseline_n_bets=("baseline_n_bets", "sum"),
            q95_stake=("q95_stake", "sum"),
            q95_profit=("q95_profit", "sum"),
            q95_n_bets=("q95_n_bets", "sum"),
        )
        .reset_index()
    )
    totals_agg["baseline_roi"] = totals_agg["baseline_profit"] / totals_agg["baseline_stake"]
    totals_agg["q95_roi"] = totals_agg["q95_profit"] / totals_agg["q95_stake"]
    totals_agg["baseline_roi"] = totals_agg["baseline_roi"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    totals_agg["q95_roi"] = totals_agg["q95_roi"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    removed_df = pd.DataFrame(removed_rows)
    if not removed_df.empty:
        removed_df = removed_df.sort_values(["year", "split", "profit"]).reset_index(drop=True)
    removed_df.to_csv(out_dir / "removed_top_losses.csv", index=False)

    bucket_breakdown_df = pd.DataFrame(bucket_breakdown_rows)
    if not bucket_breakdown_df.empty:
        bucket_breakdown_df = (
            bucket_breakdown_df.groupby(
                ["year", "split", "set", "bucket_type", "bucket_label"], dropna=False
            )
            .agg(stake=("stake", "sum"), profit=("profit", "sum"), n_bets=("n_bets", "sum"))
            .reset_index()
        )
        bucket_breakdown_df["roi"] = bucket_breakdown_df["profit"] / bucket_breakdown_df["stake"]
        bucket_breakdown_df["roi"] = bucket_breakdown_df["roi"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    bucket_breakdown_df.to_csv(out_dir / "bucket_breakdown.csv", index=False)

    bucket_delta_df = pd.DataFrame(bucket_delta_rows)
    if not bucket_delta_df.empty:
        bucket_delta_df = (
            bucket_delta_df.groupby(["year", "split", "bucket_type", "bucket_label"], dropna=False)
            .agg(common_profit_delta=("common_profit_delta", "sum"))
            .reset_index()
        )

    removed_profit_df = bucket_breakdown_df[bucket_breakdown_df["set"] == "removed"].copy()
    removed_profit_df = removed_profit_df.rename(columns={"profit": "removed_profit"})[
        ["year", "split", "bucket_type", "bucket_label", "removed_profit"]
    ]
    added_profit_df = bucket_breakdown_df[bucket_breakdown_df["set"] == "added"].copy()
    added_profit_df = added_profit_df.rename(columns={"profit": "added_profit"})[
        ["year", "split", "bucket_type", "bucket_label", "added_profit"]
    ]
    contrib_df = removed_profit_df.merge(
        added_profit_df, on=["year", "split", "bucket_type", "bucket_label"], how="outer"
    )
    contrib_df = contrib_df.merge(
        bucket_delta_df, on=["year", "split", "bucket_type", "bucket_label"], how="outer"
    )
    contrib_df[["removed_profit", "added_profit", "common_profit_delta"]] = contrib_df[
        ["removed_profit", "added_profit", "common_profit_delta"]
    ].fillna(0.0)
    contrib_df["delta_profit_contrib"] = (
        -contrib_df["removed_profit"] + contrib_df["added_profit"] + contrib_df["common_profit_delta"]
    )
    contrib_df.to_csv(out_dir / "bucket_contrib.csv", index=False)

    bootstrap_rows = []
    if args.step14_daily and args.step14_daily.exists():
        daily = pd.read_csv(args.step14_daily)
        for split in sorted(daily["split"].unique()):
            split_df = daily[daily["split"] == split].copy()
            for year in sorted(split_df["year"].unique()):
                sub = split_df[split_df["year"] == year].copy()
                for strategy in ["q95", "q97"]:
                    if strategy not in set(sub["strategy"]):
                        continue
                    result = _bootstrap_delta_roi(
                        sub, "baseline", strategy, args.bootstrap_block_days, args.bootstrap_samples
                    )
                    if result:
                        bootstrap_rows.append(
                            {"split": split, "year": int(year), "variant": strategy, **result}
                        )
            for strategy in ["q95", "q97"]:
                if strategy not in set(split_df["strategy"]):
                    continue
                result = _bootstrap_delta_roi(
                    split_df, "baseline", strategy, args.bootstrap_block_days, args.bootstrap_samples
                )
                if result:
                    bootstrap_rows.append({"split": split, "year": "total", "variant": strategy, **result})

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    report_lines = []
    report_lines.append("# q95 mechanism review")
    report_lines.append("")
    report_lines.append("## Inputs")
    for spec in specs:
        report_lines.append(f"- {spec.year} baseline design: `{spec.baseline_design}`")
        report_lines.append(f"- {spec.year} baseline eval: `{spec.baseline_eval}`")
        report_lines.append(f"- {spec.year} q95 design: `{spec.q95_design}`")
        report_lines.append(f"- {spec.year} q95 eval: `{spec.q95_eval}`")

    report_lines.append("")
    report_lines.append("## Key columns detected")
    col_df = pd.DataFrame(col_notes)
    if not col_df.empty:
        col_df = col_df.groupby(["year", "split"], dropna=False).agg(
            horse_col_base=("horse_col_base", lambda x: ",".join(sorted(set(x)))),
            horse_col_q95=("horse_col_q95", lambda x: ",".join(sorted(set(x)))),
            odds_col=("odds_col", lambda x: ",".join(sorted(set([v for v in x if v])))),
            ev_col=("ev_col", lambda x: ",".join(sorted(set([v for v in x if v])))),
            takeout_col=("takeout_col", lambda x: ",".join(sorted(set([v for v in x if v])))),
        )
        report_lines.append(_df_to_md(col_df.reset_index()))
    else:
        report_lines.append("_no column metadata_")

    report_lines.append("")
    report_lines.append("## Totals (baseline vs q95)")
    totals_show = totals_agg.copy()
    totals_show["baseline_roi"] = totals_show["baseline_roi"].astype(float)
    totals_show["q95_roi"] = totals_show["q95_roi"].astype(float)
    report_lines.append(
        _df_to_md(
            totals_show[
                [
                    "year",
                    "split",
                    "baseline_stake",
                    "baseline_profit",
                    "baseline_roi",
                    "q95_stake",
                    "q95_profit",
                    "q95_roi",
                ]
            ],
            digits=4,
        )
    )

    report_lines.append("")
    report_lines.append("## Set breakdown (common/removed/added)")
    report_lines.append(
        _df_to_md(
            summary_agg[
                [
                    "year",
                    "split",
                    "set",
                    "baseline_stake",
                    "baseline_profit",
                    "baseline_roi",
                    "q95_stake",
                    "q95_profit",
                    "q95_roi",
                ]
            ],
            digits=4,
        )
    )

    report_lines.append("")
    report_lines.append("## Bucket contribution ranking (eval, abs delta profit)")
    if not contrib_df.empty:
        eval_bucket = contrib_df[contrib_df["split"] == "eval"].copy()
        eval_bucket["abs_delta"] = eval_bucket["delta_profit_contrib"].abs()
        top = eval_bucket.sort_values("abs_delta", ascending=False).head(20)
        report_lines.append(
            _df_to_md(top[["year", "bucket_type", "bucket_label", "delta_profit_contrib"]], digits=2)
        )
    else:
        report_lines.append("_no bucket contributions_")

    report_lines.append("")
    report_lines.append("## Fragility check (removed bets)")
    if not removed_df.empty:
        for year in sorted(removed_df["year"].unique()):
            for split in sorted(removed_df["split"].unique()):
                sub = removed_df[(removed_df["year"] == year) & (removed_df["split"] == split)].copy()
                if sub.empty:
                    continue
                delta_row = totals_agg[(totals_agg["year"] == year) & (totals_agg["split"] == split)]
                if delta_row.empty:
                    continue
                delta_profit = float(delta_row["q95_profit"].iloc[0] - delta_row["baseline_profit"].iloc[0])
                if delta_profit == 0:
                    continue
                top5 = sub.nsmallest(5, "profit")["profit"].sum()
                top10 = sub.nsmallest(10, "profit")["profit"].sum()
                frac5 = abs(top5) / abs(delta_profit)
                frac10 = abs(top10) / abs(delta_profit)
                report_lines.append(
                    f"- {year} {split}: top5 removed losses explain {frac5:.2f} of |delta_profit|; "
                    f"top10 explain {frac10:.2f}"
                )
    else:
        report_lines.append("_no removed bets_")

    report_lines.append("")
    report_lines.append("## Bootstrap uncertainty (block=7 days)")
    if not bootstrap_df.empty:
        report_lines.append(_df_to_md(bootstrap_df, digits=4))
    else:
        report_lines.append("_bootstrap skipped or insufficient data_")

    report_lines.append("")
    report_lines.append("## Step=14 non-overlap summary (eval)")
    if args.step14_daily and args.step14_daily.exists():
        daily = pd.read_csv(args.step14_daily)
        daily = daily[daily["split"] == "eval"].copy()
        if not daily.empty:
            table = (
                daily.groupby(["year", "strategy"], dropna=False)
                .agg(stake=("stake", "sum"), profit=("profit", "sum"), n_bets=("n_bets", "sum"))
                .reset_index()
            )
            table["roi"] = table["profit"] / table["stake"]
            table["roi"] = table["roi"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            report_lines.append(_df_to_md(table, digits=4))
        else:
            report_lines.append("_no eval daily rows_")
    else:
        report_lines.append("_walkforward_step14_daily.csv missing; bootstrap skipped_")

    report_lines.append("")
    report_lines.append("## Next minimal experiment proposal")
    proposal = "raise_min_ev"
    proposal_reason = "common bets remain negative and low EV buckets drive losses"
    low_ev_negative = False
    if not bucket_breakdown_df.empty and ev_available:
        eval_common = bucket_breakdown_df[
            (bucket_breakdown_df["split"] == "eval")
            & (bucket_breakdown_df["set"] == "common")
            & (bucket_breakdown_df["bucket_type"] == "ev_decile")
        ]
        low = eval_common[eval_common["bucket_label"].isin(["D1", "D2", "D3"])]
        if not low.empty and low["profit"].sum() < 0:
            low_ev_negative = True
    if not low_ev_negative:
        proposal = "q=0.96"
        proposal_reason = "delta improvement is fragile; try slightly higher quantile to reduce high-takeout exposure"

    report_lines.append(f"- proposed knob: {proposal}")
    if proposal == "raise_min_ev":
        report_lines.append("- config: betting.selection.ev_threshold.min_ev (or equivalent)")
        report_lines.append("- candidates: +0.01, +0.02")
        report_lines.append(f"- expected: fewer low-EV bets; {proposal_reason}")
    else:
        report_lines.append("- config: betting.race_cost_filter.train_quantile")
        report_lines.append("- candidates: 0.96")
        report_lines.append(f"- expected: slightly fewer high-takeout races; {proposal_reason}")

    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
