from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


@dataclass(frozen=True)
class BootstrapSettings:
    """Settings for day-block bootstrap.

    Notes:
    - block_unit is fixed to "day" for now; kept for schema/reporting clarity.
    """

    B: int = 2000
    seed: int = 0
    block_unit: str = "day"


REQUIRED_COLUMNS = ("date", "race_id", "stake", "return", "profit")


def read_per_bet_pnl_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"per_bet_pnl missing required columns: {missing} ({p})")

    # Normalize types; keep date as YYYY-MM-DD strings.
    df = df.copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    df["race_id"] = df["race_id"].astype(str)

    for col in ("stake", "return", "profit"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["stake", "return", "profit"]].isna().any().any():
        bad = df[df[["stake", "return", "profit"]].isna().any(axis=1)].head(5)
        raise ValueError(f"per_bet_pnl has non-numeric stake/return/profit rows: {p}\n{bad}")

    return df


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _point_metrics(df: pd.DataFrame) -> dict[str, Any]:
    total_stake = float(df["stake"].sum())
    total_return = float(df["return"].sum())
    total_profit = float(df["profit"].sum())
    roi = None
    if total_stake > 0:
        roi = total_profit / total_stake
    return {
        "roi": roi,
        "total_stake": total_stake,
        "total_return": total_return,
        "total_profit": total_profit,
        "n_bets": int(len(df)),
        "n_races": int(df["race_id"].nunique()),
        "n_days": int(df["date"].nunique()),
    }


def _daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date", as_index=True)[["stake", "profit"]].sum()
    daily = daily.sort_index()
    return daily


def _percentile_ci95(values: list[float]) -> tuple[float, float]:
    s = pd.Series(values, dtype="float64")
    lo = float(s.quantile(0.025))
    hi = float(s.quantile(0.975))
    return lo, hi


def day_block_bootstrap(
    variant_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    settings: BootstrapSettings,
) -> dict[str, Any]:
    if settings.B < 100:
        raise ValueError("BootstrapSettings.B must be >= 100 for stable CIs.")

    v_daily = _daily_totals(variant_df)
    b_daily = _daily_totals(baseline_df)

    days = sorted(set(v_daily.index) | set(b_daily.index))
    if not days:
        raise ValueError("No days available for bootstrap (empty per_bet_pnl inputs).")

    v_daily = v_daily.reindex(days).fillna(0.0)
    b_daily = b_daily.reindex(days).fillna(0.0)

    rng = np.random.default_rng(settings.seed)
    n_days = len(days)

    v_rois: list[float] = []
    b_rois: list[float] = []
    deltas: list[float] = []

    # When there are zero-stake days, resampling can produce zero total stake.
    # We skip those replicates and keep sampling until we reach B valid reps.
    max_attempts = settings.B * 50
    attempts = 0
    while len(deltas) < settings.B and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n_days, size=n_days)

        v_stake = float(v_daily["stake"].iloc[idx].sum())  # type: ignore[index]
        b_stake = float(b_daily["stake"].iloc[idx].sum())  # type: ignore[index]
        if v_stake <= 0 or b_stake <= 0:
            continue

        v_profit = float(v_daily["profit"].iloc[idx].sum())  # type: ignore[index]
        b_profit = float(b_daily["profit"].iloc[idx].sum())  # type: ignore[index]
        v_roi = v_profit / v_stake
        b_roi = b_profit / b_stake
        v_rois.append(v_roi)
        b_rois.append(b_roi)
        deltas.append(v_roi - b_roi)

    if len(deltas) < settings.B:
        raise RuntimeError(
            f"Unable to generate {settings.B} valid bootstrap replicates "
            f"(got {len(deltas)} after {attempts} attempts)."
        )

    v_ci = _percentile_ci95(v_rois)
    b_ci = _percentile_ci95(b_rois)
    d_ci = _percentile_ci95(deltas)
    p_one_sided = sum(1 for d in deltas if d <= 0.0) / float(len(deltas))

    return {
        "roi_ci95": [v_ci[0], v_ci[1]],
        "baseline_roi_ci95": [b_ci[0], b_ci[1]],
        "delta_roi_ci95": [d_ci[0], d_ci[1]],
        "p_one_sided_delta_le_0": p_one_sided,
        "bootstrap": {
            "B": int(settings.B),
            "seed": int(settings.seed),
            "block_unit": settings.block_unit,
        },
        "resampling_unit": "day-block (YYYY-MM-DD)",
        "union_days": int(n_days),
    }


def _odds_bucket_label(odds: float | None) -> str:
    if odds is None:
        return "unknown"
    if odds < 1.0:
        return "<1"
    if odds < 2.0:
        return "1-2"
    if odds < 5.0:
        return "2-5"
    if odds < 10.0:
        return "5-10"
    if odds < 20.0:
        return "10-20"
    if odds < 30.0:
        return "20-30"
    return "30+"


def _compute_segment_table(
    variant_df: pd.DataFrame, baseline_df: pd.DataFrame, key: str
) -> list[dict[str, Any]]:
    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby(key, as_index=False).agg(
            stake=("stake", "sum"),
            profit=("profit", "sum"),
            bets=("race_id", "size"),
        )
        g["roi"] = g.apply(
            lambda r: (float(r["profit"]) / float(r["stake"])) if float(r["stake"]) > 0 else None,
            axis=1,
        )
        return g[[key, "roi", "stake", "bets"]]

    v = _agg(variant_df)
    b = _agg(baseline_df).rename(
        columns={"roi": "baseline_roi", "stake": "baseline_stake", "bets": "baseline_bets"}
    )
    merged = v.merge(b, how="outer", on=key)
    merged = merged.fillna({"stake": 0.0, "bets": 0, "baseline_stake": 0.0, "baseline_bets": 0})
    merged["bets"] = merged["bets"].astype(int)
    merged["baseline_bets"] = merged["baseline_bets"].astype(int)

    # Stable ordering for odds buckets.
    if key == "odds_bucket":
        order = ["<1", "1-2", "2-5", "5-10", "10-20", "20-30", "30+", "unknown"]
        merged[key] = pd.Categorical(merged[key], categories=order, ordered=True)
        merged = merged.sort_values(key)
        merged[key] = merged[key].astype(str)
    else:
        merged = merged.sort_values(key)

    out: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        out.append(
            {
                key: row[key],
                "roi": _safe_float(row.get("roi")),
                "stake": float(row.get("stake") or 0.0),
                "bets": int(row.get("bets") or 0),
                "baseline_roi": _safe_float(row.get("baseline_roi")),
                "baseline_stake": float(row.get("baseline_stake") or 0.0),
                "baseline_bets": int(row.get("baseline_bets") or 0),
            }
        )
    return out


def compute_breakdowns(
    variant_df: pd.DataFrame, baseline_df: pd.DataFrame
) -> dict[str, list[dict[str, Any]]]:
    v = variant_df.copy()
    b = baseline_df.copy()

    if "odds" in v.columns:
        v["odds_bucket"] = v["odds"].map(lambda x: _odds_bucket_label(_safe_float(x)))
    else:
        v["odds_bucket"] = "unknown"
    if "odds" in b.columns:
        b["odds_bucket"] = b["odds"].map(lambda x: _odds_bucket_label(_safe_float(x)))
    else:
        b["odds_bucket"] = "unknown"

    v["month"] = v["date"].astype(str).str.slice(0, 7)
    b["month"] = b["date"].astype(str).str.slice(0, 7)

    return {
        "odds_bucket": _compute_segment_table(v, b, "odds_bucket"),
        "month": _compute_segment_table(v, b, "month"),
    }


def compute_block_bootstrap_summary(
    variant_path: str | Path,
    baseline_path: str | Path,
    settings: BootstrapSettings,
) -> dict[str, Any]:
    variant_df = read_per_bet_pnl_csv(variant_path)
    baseline_df = read_per_bet_pnl_csv(baseline_path)

    v_metrics = _point_metrics(variant_df)
    b_metrics = _point_metrics(baseline_df)
    delta_roi = None
    if v_metrics["roi"] is not None and b_metrics["roi"] is not None:
        delta_roi = float(v_metrics["roi"]) - float(b_metrics["roi"])

    stats = day_block_bootstrap(variant_df, baseline_df, settings)
    breakdowns = compute_breakdowns(variant_df, baseline_df)

    return {
        "variant": v_metrics,
        "baseline": b_metrics,
        "deltas": {"delta_roi": delta_roi},
        "stats": stats,
        "breakdowns": breakdowns,
    }


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
