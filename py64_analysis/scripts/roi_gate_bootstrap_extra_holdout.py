from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass
class VariantStats:
    roi: float
    n_bets: int
    n_units: int


def _detect_variant(parts: tuple[str, ...]) -> str | None:
    for key in ("base", "candidate"):
        if key in parts:
            return key
    return None


WINDOW_RE = re.compile(r"^w\d{3}_\d{8}_\d{8}$")


def _infer_window_id(path: Path) -> str:
    for part in path.parts[::-1]:
        if WINDOW_RE.match(part):
            return part
    return "unknown"


def _infer_run_id(path: Path) -> str:
    for part in path.parts:
        if part.lower().startswith("run"):
            return part
    return "unknown"


def _load_bets(extra_holdout_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {"base": [], "candidate": []}
    for path in extra_holdout_dir.rglob("bets.csv"):
        parts = tuple(p.lower() for p in path.parts)
        variant = _detect_variant(parts)
        if variant is None:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["__source_path__"] = str(path)
        df["window_id"] = _infer_window_id(path)
        df["run_id"] = _infer_run_id(path)
        out[variant].append(df)
    merged = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in out.items()}
    for key, df in merged.items():
        if df.empty:
            continue
        merged[key] = _dedup_bets(df)
    return merged


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


def _extract_date_key(df: pd.DataFrame) -> pd.Series:
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
    return pd.Series([pd.NA] * len(df))


def _ensure_day_key(df: pd.DataFrame) -> pd.Series:
    day_key = _extract_date_key(df)
    if day_key.isna().all():
        cols = ", ".join(df.columns)
        raise SystemExit(
            "day_key inference failed; required columns missing. "
            f"available_cols=[{cols}]"
        )
    return day_key


def _prepare_bets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    odds_final = pd.to_numeric(work.get("odds_final"), errors="coerce")
    odds_buy = pd.to_numeric(work.get("odds_at_buy"), errors="coerce")
    odds_use = odds_final.where(odds_final.notna(), odds_buy)
    work["odds_use"] = odds_use

    if "is_win" in work.columns:
        is_win = pd.to_numeric(work["is_win"], errors="coerce")
    elif "finish_pos" in work.columns:
        is_win = (pd.to_numeric(work["finish_pos"], errors="coerce") == 1).astype(float)
    else:
        is_win = pd.Series([0.0] * len(work))
    work["is_win_num"] = is_win.fillna(0.0)

    work = work.dropna(subset=["odds_use"])
    work = work[work["odds_use"] > 0]
    return work


def _aggregate_by_unit(df: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, VariantStats]:
    if df.empty:
        return pd.DataFrame(), VariantStats(roi=float("nan"), n_bets=0, n_units=0)

    work = _prepare_bets(df)
    if work.empty:
        return pd.DataFrame(), VariantStats(roi=float("nan"), n_bets=0, n_units=0)

    if unit == "day":
        day_key = _ensure_day_key(work)
        work["unit_key"] = day_key.str.slice(0, 4) + "-" + day_key.str.slice(4, 6) + "-" + day_key.str.slice(6, 8)
    else:
        work["unit_key"] = work["race_id"].astype(str)

    work["profit_unit"] = work["odds_use"] * work["is_win_num"] - 1.0
    grouped = (
        work.groupby("unit_key", dropna=False)
        .agg(profit=("profit_unit", "sum"), stake=("profit_unit", "size"))
        .reset_index()
    )
    total_profit = float(grouped["profit"].sum())
    total_stake = float(grouped["stake"].sum())
    roi = total_profit / total_stake if total_stake > 0 else float("nan")
    stats = VariantStats(roi=roi, n_bets=int(total_stake), n_units=int(len(grouped)))
    return grouped, stats


def _bootstrap_ci(
    *,
    base_group: pd.DataFrame,
    cand_group: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict:
    base_map = {k: (float(p), float(s)) for k, p, s in base_group[["unit_key", "profit", "stake"]].itertuples(index=False)}
    cand_map = {k: (float(p), float(s)) for k, p, s in cand_group[["unit_key", "profit", "stake"]].itertuples(index=False)}

    keys = sorted(set(base_map.keys()) | set(cand_map.keys()))
    n_units = len(keys)
    if n_units == 0:
        return {
            "delta_ci_low": float("nan"),
            "delta_ci_med": float("nan"),
            "delta_ci_high": float("nan"),
            "cand_ci_low": float("nan"),
            "cand_ci_med": float("nan"),
            "cand_ci_high": float("nan"),
            "n_valid_delta": 0,
            "n_valid_cand": 0,
        }

    base_profit = np.array([base_map.get(k, (0.0, 0.0))[0] for k in keys], dtype=float)
    base_stake = np.array([base_map.get(k, (0.0, 0.0))[1] for k in keys], dtype=float)
    cand_profit = np.array([cand_map.get(k, (0.0, 0.0))[0] for k in keys], dtype=float)
    cand_stake = np.array([cand_map.get(k, (0.0, 0.0))[1] for k in keys], dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_units, size=(n_boot, n_units))

    base_profit_s = base_profit[idx].sum(axis=1)
    base_stake_s = base_stake[idx].sum(axis=1)
    cand_profit_s = cand_profit[idx].sum(axis=1)
    cand_stake_s = cand_stake[idx].sum(axis=1)

    base_roi = np.where(base_stake_s > 0, base_profit_s / base_stake_s, np.nan)
    cand_roi = np.where(cand_stake_s > 0, cand_profit_s / cand_stake_s, np.nan)
    delta_roi = cand_roi - base_roi

    cand_valid = cand_roi[np.isfinite(cand_roi)]
    delta_valid = delta_roi[np.isfinite(delta_roi)]

    def _ci(arr: np.ndarray) -> tuple[float, float, float]:
        if arr.size == 0:
            return float("nan"), float("nan"), float("nan")
        low, med, high = np.quantile(arr, [0.025, 0.5, 0.975]).tolist()
        return float(low), float(med), float(high)

    cand_ci = _ci(cand_valid)
    delta_ci = _ci(delta_valid)

    return {
        "delta_ci_low": delta_ci[0],
        "delta_ci_med": delta_ci[1],
        "delta_ci_high": delta_ci[2],
        "cand_ci_low": cand_ci[0],
        "cand_ci_med": cand_ci[1],
        "cand_ci_high": cand_ci[2],
        "n_valid_delta": int(delta_valid.size),
        "n_valid_cand": int(cand_valid.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap CI gate for extra holdout ROI.")
    ap.add_argument("--extra-holdout-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bootstrap-n", type=int, default=500)
    ap.add_argument("--resample-unit", choices=["race", "day"], default="race")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    extra_holdout_dir = Path(args.extra_holdout_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_bets(extra_holdout_dir)
    base_df = data.get("base", pd.DataFrame())
    cand_df = data.get("candidate", pd.DataFrame())

    base_group, base_stats = _aggregate_by_unit(base_df, args.resample_unit)
    cand_group, cand_stats = _aggregate_by_unit(cand_df, args.resample_unit)

    boot = _bootstrap_ci(
        base_group=base_group,
        cand_group=cand_group,
        n_boot=int(args.bootstrap_n),
        seed=int(args.seed),
    )

    delta_roi = cand_stats.roi - base_stats.roi

    row = {
        "resample_unit": args.resample_unit,
        "bootstrap_n": int(args.bootstrap_n),
        "seed": int(args.seed),
        "base_roi": base_stats.roi,
        "candidate_roi": cand_stats.roi,
        "delta_roi": delta_roi,
        "base_n_bets": base_stats.n_bets,
        "candidate_n_bets": cand_stats.n_bets,
        "n_units_base": base_stats.n_units,
        "n_units_candidate": cand_stats.n_units,
        **boot,
    }

    df_out = pd.DataFrame([row])
    csv_path = out_dir / f"bootstrap_ci_{args.resample_unit}.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8")

    report_lines = [
        "# Extra holdout bootstrap CI",
        f"- resample_unit: {args.resample_unit}",
        f"- bootstrap_n: {int(args.bootstrap_n)}",
        f"- base ROI (step14): {base_stats.roi:.6f} | n_bets={base_stats.n_bets}",
        f"- candidate ROI (step14): {cand_stats.roi:.6f} | n_bets={cand_stats.n_bets}",
        f"- delta ROI: {delta_roi:.6f}",
        "",
        "## Candidate ROI CI (95%)",
        f"- low: {boot['cand_ci_low']:.6f}",
        f"- median: {boot['cand_ci_med']:.6f}",
        f"- high: {boot['cand_ci_high']:.6f}",
        "",
        "## Delta ROI CI (candidate - base, 95%)",
        f"- low: {boot['delta_ci_low']:.6f}",
        f"- median: {boot['delta_ci_med']:.6f}",
        f"- high: {boot['delta_ci_high']:.6f}",
    ]
    (out_dir / f"report_{args.resample_unit}.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        f"[ci] delta_roi_ci_low={boot['delta_ci_low']:.6f} | "
        f"delta_roi_ci_med={boot['delta_ci_med']:.6f} | "
        f"delta_roi_ci_high={boot['delta_ci_high']:.6f}"
    )


if __name__ == "__main__":
    main()
