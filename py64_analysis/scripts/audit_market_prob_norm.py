"""
Audit market probability normalization from a rolling holdout run directory.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import text


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _load_yaml(path: Path) -> Optional[dict]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_race_ids(run_dir: Path) -> list[str]:
    race_ids: set[str] = set()
    for path in run_dir.rglob("race_ids_test.txt"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                rid = line.strip()
                if rid:
                    race_ids.add(rid)
        except Exception:
            continue
    return sorted(race_ids)


def _load_buy_minutes(run_dir: Path) -> int:
    configs = list(run_dir.rglob("config_used.yaml"))
    if not configs:
        return 1
    cfg = _load_yaml(configs[0]) or {}
    backtest_cfg = cfg.get("backtest", {}) if isinstance(cfg, dict) else {}
    val = backtest_cfg.get("buy_t_minus_minutes")
    if val is None:
        val = backtest_cfg.get("buy_minutes")
    try:
        return int(val)
    except Exception:
        return 1


def _chunked(items: Iterable[str], size: int) -> Iterable[list[str]]:
    buf: list[str] = []
    for item in items:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _to_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _sum_or_nan(series: pd.Series) -> float:
    if series.notna().any():
        return float(series.sum(skipna=True))
    return float("nan")


def _summary_quantiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    return {
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def audit_run(run_dir: Path, out_dir: Path, label: str) -> dict:
    _ensure_import_path()
    from keiba.db.loader import get_session
    from keiba.features.build_features import FeatureBuilder

    race_ids = _find_race_ids(run_dir)
    if not race_ids:
        raise SystemExit(f"No race_ids_test.txt found under: {run_dir}")

    buy_minutes = _load_buy_minutes(run_dir)
    feature_version = FeatureBuilder.VERSION

    session = get_session()

    rows: list[dict] = []
    query = text(
        """
        WITH target_races AS (
            SELECT unnest(CAST(:race_ids AS text[])) AS race_id
        ),
        buy_times AS (
            SELECT
                r.race_id,
                r.date,
                r.track_code,
                r.distance,
                ((r.date::timestamp + r.start_time) - make_interval(mins => :buy_minutes)) AS buy_time
            FROM fact_race r
            JOIN target_races t ON r.race_id = t.race_id
            WHERE r.start_time IS NOT NULL
        ),
        latest_features AS (
            SELECT DISTINCT ON (f.race_id, f.horse_id)
                f.race_id,
                f.horse_id,
                f.payload,
                bt.date,
                bt.track_code,
                bt.distance
            FROM features f
            JOIN buy_times bt ON f.race_id = bt.race_id
            WHERE f.feature_version = :feature_version
              AND f.asof_time <= bt.buy_time
            ORDER BY f.race_id, f.horse_id, f.asof_time DESC
        )
        SELECT
            race_id,
            horse_id,
            date,
            track_code,
            distance,
            payload
        FROM latest_features
        """
    )

    for chunk in _chunked(race_ids, 500):
        result = session.execute(
            query,
            {
                "race_ids": chunk,
                "buy_minutes": buy_minutes,
                "feature_version": feature_version,
            },
        ).fetchall()
        for r in result:
            payload = r[5] or {}
            p_raw = payload.get("p_mkt_raw", payload.get("p_mkt"))
            p_race = payload.get("p_mkt_race", payload.get("p_mkt"))
            rows.append(
                {
                    "race_id": r[0],
                    "horse_id": r[1],
                    "date": r[2],
                    "track_code": r[3],
                    "distance": r[4],
                    "p_mkt_raw": _to_float(p_raw),
                    "p_mkt_race": _to_float(p_race),
                    "overround_sum_inv": _to_float(payload.get("overround_sum_inv")),
                    "takeout_implied": _to_float(payload.get("takeout_implied")),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No feature rows found for the requested races.")

    df["abs_diff"] = (df["p_mkt_raw"] - df["p_mkt_race"]).abs()

    race_df = (
        df.groupby("race_id", as_index=False)
        .agg(
            date=("date", "first"),
            track_code=("track_code", "first"),
            distance=("distance", "first"),
            n_horses=("horse_id", "count"),
            sum_p_mkt_raw=("p_mkt_raw", _sum_or_nan),
            sum_p_mkt_race=("p_mkt_race", _sum_or_nan),
            overround_sum_inv=("overround_sum_inv", "first"),
            takeout_implied=("takeout_implied", "first"),
        )
    )
    race_df["year"] = pd.to_datetime(race_df["date"], errors="coerce").dt.year

    valid_race = race_df["sum_p_mkt_race"].replace([np.inf, -np.inf], np.nan).dropna()
    raw_race = race_df["sum_p_mkt_raw"].replace([np.inf, -np.inf], np.nan).dropna()

    missing_rate = 1.0 - (len(valid_race) / len(race_df)) if len(race_df) else 1.0
    p_raw_stats = _summary_quantiles(raw_race.values)
    p_race_stats = _summary_quantiles(valid_race.values)

    pair_df = df[["p_mkt_raw", "p_mkt_race"]].dropna()
    corr = None
    if len(pair_df) >= 2:
        corr = float(pair_df["p_mkt_raw"].corr(pair_df["p_mkt_race"]))

    diff_race = (
        df.groupby("race_id", as_index=False)
        .agg(mean_abs_diff=("abs_diff", "mean"), max_abs_diff=("abs_diff", "max"))
        .merge(
            race_df[["race_id", "date", "track_code", "distance", "sum_p_mkt_raw", "sum_p_mkt_race"]],
            on="race_id",
            how="left",
        )
        .sort_values("mean_abs_diff", ascending=False)
        .head(20)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    race_df.to_csv(out_dir / "race_market_sums.csv", index=False, encoding="utf-8")
    diff_race.to_csv(out_dir / "top_diff_races.csv", index=False, encoding="utf-8")

    takeout_by_year = (
        race_df.groupby("year")["takeout_implied"]
        .agg(["count", "median", "mean", "max"])
        .reset_index()
        .rename(columns={"median": "p50"})
    )
    takeout_by_year.to_csv(out_dir / "takeout_by_year.csv", index=False, encoding="utf-8")

    takeout_by_track = (
        race_df.groupby("track_code")["takeout_implied"]
        .agg(["count", "median", "mean", "max"])
        .reset_index()
        .rename(columns={"median": "p50"})
    )
    takeout_by_track.to_csv(out_dir / "takeout_by_track.csv", index=False, encoding="utf-8")

    takeout_by_distance = (
        race_df.groupby("distance")["takeout_implied"]
        .agg(["count", "median", "mean", "max"])
        .reset_index()
        .rename(columns={"median": "p50"})
    )
    takeout_by_distance.to_csv(out_dir / "takeout_by_distance.csv", index=False, encoding="utf-8")

    summary_lines = [
        f"# Market prob normalization audit ({label})",
        "",
        f"- races: {len(race_df)}",
        f"- horses: {len(df)}",
        f"- feature_version: {feature_version}",
        f"- buy_minutes: {buy_minutes}",
        f"- missing_race_rate: {missing_rate:.3f}",
        "",
        "## Sum(p_mkt_raw) per race",
        f"- p50: {p_raw_stats['p50']}",
        f"- p90: {p_raw_stats['p90']}",
        f"- p99: {p_raw_stats['p99']}",
        f"- max: {p_raw_stats['max']}",
        "",
        "## Sum(p_mkt_race) per race",
        f"- p50: {p_race_stats['p50']}",
        f"- p90: {p_race_stats['p90']}",
        f"- p99: {p_race_stats['p99']}",
        f"- max: {p_race_stats['max']}",
        "",
        "## Correlation (horse-level)",
        f"- corr(p_mkt_raw, p_mkt_race): {corr}",
        "",
        "## Outputs",
        f"- race_market_sums.csv: {out_dir / 'race_market_sums.csv'}",
        f"- top_diff_races.csv: {out_dir / 'top_diff_races.csv'}",
        f"- takeout_by_year.csv: {out_dir / 'takeout_by_year.csv'}",
        f"- takeout_by_track.csv: {out_dir / 'takeout_by_track.csv'}",
        f"- takeout_by_distance.csv: {out_dir / 'takeout_by_distance.csv'}",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "label": label,
        "races": len(race_df),
        "missing_rate": missing_rate,
        "raw_stats": p_raw_stats,
        "race_stats": p_race_stats,
        "corr": corr,
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit market probability normalization for holdout runs.")
    ap.add_argument("--run-dir", type=Path, required=True, help="holdout run dir (group dir)")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory")
    ap.add_argument("--label", type=str, default="run", help="label for the summary")
    args = ap.parse_args()

    result = audit_run(args.run_dir, args.out_dir, args.label)
    race_stats = result["race_stats"]
    print(
        f"[audit {result['label']}] races={result['races']} | "
        f"p_mkt_race_sum_p50={race_stats['p50']} | "
        f"p_mkt_race_sum_p90={race_stats['p90']} | "
        f"missing_race_rate={result['missing_rate']:.3f}"
    )
    print(f"audit_dir={result['out_dir']}")


if __name__ == "__main__":
    main()
