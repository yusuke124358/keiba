from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


WINDOW_RE = re.compile(r"^w\d{3}_\d{8}_\d{8}$")


@dataclass
class AggMetrics:
    n_bets: int
    stake: float
    profit: float
    roi: float
    max_dd: float


def _infer_window_id(path: Path) -> str:
    for part in path.parts[::-1]:
        if WINDOW_RE.match(part):
            return part
    return "unknown"


def _load_bets_from_run(run_dir: Path, variant: str) -> pd.DataFrame:
    base = run_dir / variant
    rows: list[pd.DataFrame] = []
    if not base.exists():
        return pd.DataFrame()
    for path in base.rglob("bets.csv"):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["run_id"] = run_dir.name
        df["window_id"] = _infer_window_id(path)
        df["__source_path__"] = str(path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    merged = pd.concat(rows, ignore_index=True)
    return merged


def _dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        return df, {"dedup_key": [], "rows_before": 0, "rows_after": 0, "rows_dropped": 0}
    keys = []
    for col in ("race_id", "horse_no", "asof_time"):
        if col in df.columns:
            keys.append(col)
    if not keys:
        return df, {"dedup_key": [], "rows_before": len(df), "rows_after": len(df), "rows_dropped": 0}
    before = len(df)
    deduped = df.drop_duplicates(keys, keep="first")
    after = len(deduped)
    return deduped, {
        "dedup_key": keys,
        "rows_before": before,
        "rows_after": after,
        "rows_dropped": before - after,
    }


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


def _compute_metrics(df: pd.DataFrame) -> AggMetrics:
    if df.empty:
        return AggMetrics(n_bets=0, stake=0.0, profit=0.0, roi=float("nan"), max_dd=0.0)
    odds_final = pd.to_numeric(df.get("odds_final"), errors="coerce")
    odds_buy = pd.to_numeric(df.get("odds_at_buy"), errors="coerce")
    odds_use = odds_final.fillna(odds_buy)
    is_win = pd.to_numeric(df.get("is_win"), errors="coerce")
    if is_win.isna().all() and "finish_pos" in df.columns:
        is_win = (pd.to_numeric(df.get("finish_pos"), errors="coerce") == 1).astype(float)
    is_win = is_win.fillna(0.0)
    pnl_unit = odds_use * is_win - 1.0
    profit = float(pnl_unit.sum())
    stake = float(len(df))
    roi = profit / stake if stake > 0 else float("nan")
    cum = np.cumsum(pnl_unit.to_numpy())
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) else 0.0
    return AggMetrics(n_bets=int(stake), stake=stake, profit=profit, roi=roi, max_dd=max_dd)


def _bootstrap_day_ci(base_df: pd.DataFrame, cand_df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    base = base_df.copy()
    cand = cand_df.copy()
    base["day_key"] = _extract_day_key(base)
    cand["day_key"] = _extract_day_key(cand)
    base["day_key"] = base["day_key"].str.slice(0, 4) + "-" + base["day_key"].str.slice(4, 6) + "-" + base["day_key"].str.slice(6, 8)
    cand["day_key"] = cand["day_key"].str.slice(0, 4) + "-" + cand["day_key"].str.slice(4, 6) + "-" + cand["day_key"].str.slice(6, 8)

    def _day_profit(df: pd.DataFrame) -> pd.DataFrame:
        odds_final = pd.to_numeric(df.get("odds_final"), errors="coerce")
        odds_buy = pd.to_numeric(df.get("odds_at_buy"), errors="coerce")
        odds_use = odds_final.fillna(odds_buy)
        is_win = pd.to_numeric(df.get("is_win"), errors="coerce")
        if is_win.isna().all() and "finish_pos" in df.columns:
            is_win = (pd.to_numeric(df.get("finish_pos"), errors="coerce") == 1).astype(float)
        is_win = is_win.fillna(0.0)
        pnl_unit = odds_use * is_win - 1.0
        tmp = pd.DataFrame({"day_key": df["day_key"].values, "profit": pnl_unit.values, "stake": 1})
        return tmp.groupby("day_key", dropna=False).sum().reset_index()

    base_days = _day_profit(base)
    cand_days = _day_profit(cand)

    day_keys = sorted(set(base_days["day_key"]) | set(cand_days["day_key"]))
    if not day_keys:
        return {
            "n_days": 0,
            "delta_ci_low": float("nan"),
            "delta_ci_med": float("nan"),
            "delta_ci_high": float("nan"),
            "base_roi": float("nan"),
            "candidate_roi": float("nan"),
            "delta_roi": float("nan"),
        }

    base_map = {k: (float(p), float(s)) for k, p, s in base_days[["day_key", "profit", "stake"]].itertuples(index=False)}
    cand_map = {k: (float(p), float(s)) for k, p, s in cand_days[["day_key", "profit", "stake"]].itertuples(index=False)}

    base_profit = np.array([base_map.get(k, (0.0, 0.0))[0] for k in day_keys], dtype=float)
    base_stake = np.array([base_map.get(k, (0.0, 0.0))[1] for k in day_keys], dtype=float)
    cand_profit = np.array([cand_map.get(k, (0.0, 0.0))[0] for k in day_keys], dtype=float)
    cand_stake = np.array([cand_map.get(k, (0.0, 0.0))[1] for k in day_keys], dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(day_keys), size=(n_boot, len(day_keys)))
    base_profit_s = base_profit[idx].sum(axis=1)
    base_stake_s = base_stake[idx].sum(axis=1)
    cand_profit_s = cand_profit[idx].sum(axis=1)
    cand_stake_s = cand_stake[idx].sum(axis=1)

    base_roi_s = np.where(base_stake_s > 0, base_profit_s / base_stake_s, np.nan)
    cand_roi_s = np.where(cand_stake_s > 0, cand_profit_s / cand_stake_s, np.nan)
    delta_roi_s = cand_roi_s - base_roi_s

    base_roi = float(base_profit.sum() / base_stake.sum()) if base_stake.sum() > 0 else float("nan")
    cand_roi = float(cand_profit.sum() / cand_stake.sum()) if cand_stake.sum() > 0 else float("nan")
    delta_roi = cand_roi - base_roi

    def _ci(arr: np.ndarray) -> tuple[float, float, float]:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return float("nan"), float("nan"), float("nan")
        low, med, high = np.quantile(valid, [0.025, 0.5, 0.975]).tolist()
        return float(low), float(med), float(high)

    low, med, high = _ci(delta_roi_s)

    return {
        "n_days": int(len(day_keys)),
        "delta_ci_low": low,
        "delta_ci_med": med,
        "delta_ci_high": high,
        "base_roi": base_roi,
        "candidate_roi": cand_roi,
        "delta_roi": delta_roi,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute expanded holdout with fixed definition.")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--market_method", required=True)
    ap.add_argument("--candidate_id", required=True)
    ap.add_argument("--bootstrap_n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.runs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_frames = []
    cand_frames = []
    per_run_rows = []
    manifest = {
        "candidate_id": args.candidate_id,
        "market_method": args.market_method,
        "run_dirs": [str(p) for p in run_dirs],
        "bootstrap_n": int(args.bootstrap_n),
        "seed": int(args.seed),
        "dedup": {},
    }

    for run_dir in run_dirs:
        base_df = _load_bets_from_run(run_dir, "base")
        cand_df = _load_bets_from_run(run_dir, "candidate")
        base_dedup, base_meta = _dedup(base_df)
        cand_dedup, cand_meta = _dedup(cand_df)

        base_metrics = _compute_metrics(base_dedup)
        cand_metrics = _compute_metrics(cand_dedup)

        per_run_rows.append(
            {
                "run_id": run_dir.name,
                "variant": "base",
                "n_bets": base_metrics.n_bets,
                "profit": base_metrics.profit,
                "stake": base_metrics.stake,
                "roi": base_metrics.roi,
                "max_dd": base_metrics.max_dd,
            }
        )
        per_run_rows.append(
            {
                "run_id": run_dir.name,
                "variant": "candidate",
                "n_bets": cand_metrics.n_bets,
                "profit": cand_metrics.profit,
                "stake": cand_metrics.stake,
                "roi": cand_metrics.roi,
                "max_dd": cand_metrics.max_dd,
            }
        )

        manifest["dedup"][run_dir.name] = {
            "base": base_meta,
            "candidate": cand_meta,
        }

        if not base_dedup.empty:
            base_frames.append(base_dedup)
        if not cand_dedup.empty:
            cand_frames.append(cand_dedup)

    base_all = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()
    cand_all = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()

    base_all, base_meta = _dedup(base_all)
    cand_all, cand_meta = _dedup(cand_all)
    manifest["dedup"]["overall"] = {"base": base_meta, "candidate": cand_meta}

    base_metrics = _compute_metrics(base_all)
    cand_metrics = _compute_metrics(cand_all)

    per_run_rows.append(
        {
            "run_id": "overall",
            "variant": "base",
            "n_bets": base_metrics.n_bets,
            "profit": base_metrics.profit,
            "stake": base_metrics.stake,
            "roi": base_metrics.roi,
            "max_dd": base_metrics.max_dd,
        }
    )
    per_run_rows.append(
        {
            "run_id": "overall",
            "variant": "candidate",
            "n_bets": cand_metrics.n_bets,
            "profit": cand_metrics.profit,
            "stake": cand_metrics.stake,
            "roi": cand_metrics.roi,
            "max_dd": cand_metrics.max_dd,
        }
    )

    agg_df = pd.DataFrame(per_run_rows)
    agg_path = out_dir / "expanded_holdout_agg_metrics_fixed.csv"
    agg_df.to_csv(agg_path, index=False, encoding="utf-8")

    day_ci = _bootstrap_day_ci(base_all, cand_all, n_boot=int(args.bootstrap_n), seed=int(args.seed))
    day_ci_row = {
        "bootstrap_n": int(args.bootstrap_n),
        "seed": int(args.seed),
        "n_days": day_ci["n_days"],
        "base_roi": day_ci["base_roi"],
        "candidate_roi": day_ci["candidate_roi"],
        "delta_roi": day_ci["delta_roi"],
        "delta_ci_low": day_ci["delta_ci_low"],
        "delta_ci_med": day_ci["delta_ci_med"],
        "delta_ci_high": day_ci["delta_ci_high"],
    }
    day_ci_path = out_dir / "day_bootstrap_ci_fixed.csv"
    pd.DataFrame([day_ci_row]).to_csv(day_ci_path, index=False, encoding="utf-8")

    manifest["overall"] = {
        "base": base_metrics.__dict__,
        "candidate": cand_metrics.__dict__,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {agg_path}")
    print(f"wrote {day_ci_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
