"""
Analyze asof lag vs buy_time and slippage behavior from bets.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


WIN_RE = re.compile(r"^w(\d{3})_(\d{8})_(\d{8})")

STAKE_COLS = ["stake", "stake_yen", "stake_amount", "total_stake", "stake_sum"]
PROFIT_COLS = ["profit", "net_profit", "pnl"]
ASOF_COLS = ["asof_time", "asof_ts", "t_snap", "asof"]
RACE_ID_COLS = ["race_id", "raceid"]
HORSE_NO_COLS = ["horse_no", "horse_num", "horse"]
RATIO_COLS = ["ratio_final_to_buy", "ratio_final_buy"]
ODDS_FINAL_COLS = ["odds_final", "final_odds", "odds_close"]
ODDS_BUY_COLS = ["odds_at_buy", "odds", "buy_odds", "odds_buy"]
BUY_TIME_COLS = ["buy_time", "buy_datetime", "buy_at", "buy_ts", "buy_t", "purchase_time"]

LAG_BINS = [0, 5, 10, 20, 30, 60, 120, 300, float("inf")]


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _infer_w_idx_offset(run_dir_name: str) -> int:
    name = run_dir_name.lower()
    if "w013_022" in name or "w013-022" in name:
        return 12
    return 0


def _parse_window_name(name: str) -> Optional[int]:
    m = WIN_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _load_manifest(manifest_dir: Path, year: str) -> list[Path]:
    path = manifest_dir / f"manifest_{year}.json"
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    variants = manifest.get("variants", {})
    run_dirs = variants.get("b0")
    if not run_dirs:
        raise SystemExit("baseline b0 not found in manifest variants")
    root = Path.cwd()
    resolved = []
    for p in run_dirs:
        p = Path(p)
        if not p.is_absolute():
            p = root / p
        resolved.append(p)
    return resolved


def _get_fact_race_columns(session) -> set[str]:
    from sqlalchemy import text

    try:
        rows = session.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name='fact_race'")
        ).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def _fetch_start_times(session, race_ids: list[str]):
    from sqlalchemy import bindparam, text

    cols = _get_fact_race_columns(session)
    if not cols:
        return None, "fact_race columns unavailable"

    dt_col = None
    date_col = None
    time_col = None

    for c in ["race_datetime", "start_datetime", "race_start_time", "post_time"]:
        if c in cols:
            dt_col = c
            break

    if dt_col is None:
        if "date" in cols and "start_time" in cols:
            date_col = "date"
            time_col = "start_time"
        elif "race_date" in cols and "start_time" in cols:
            date_col = "race_date"
            time_col = "start_time"
        elif "date" in cols and "post_time" in cols:
            date_col = "date"
            time_col = "post_time"
        elif "race_date" in cols and "post_time" in cols:
            date_col = "race_date"
            time_col = "post_time"

    if dt_col is None and (date_col is None or time_col is None):
        return None, "no usable datetime columns in fact_race"

    race_map = {}
    chunk_size = 1000
    if dt_col:
        sql = text(f"SELECT race_id, {dt_col} AS start_dt FROM fact_race WHERE race_id IN :race_ids")
        sql = sql.bindparams(bindparam("race_ids", expanding=True))
        for i in range(0, len(race_ids), chunk_size):
            chunk = race_ids[i : i + chunk_size]
            rows = session.execute(sql, {"race_ids": chunk}).fetchall()
            for race_id, start_dt in rows:
                if start_dt is None:
                    continue
                start_ts = pd.to_datetime(start_dt, errors="coerce")
                if pd.isna(start_ts):
                    continue
                race_map[str(race_id)] = start_ts
    else:
        sql = text(
            f"SELECT race_id, {date_col} AS race_date, {time_col} AS start_time FROM fact_race WHERE race_id IN :race_ids"
        )
        sql = sql.bindparams(bindparam("race_ids", expanding=True))
        for i in range(0, len(race_ids), chunk_size):
            chunk = race_ids[i : i + chunk_size]
            rows = session.execute(sql, {"race_ids": chunk}).fetchall()
            for race_id, race_date, start_time in rows:
                if race_date is None or start_time is None:
                    continue
                start_ts = pd.to_datetime(f"{race_date} {start_time}", errors="coerce")
                if pd.isna(start_ts):
                    continue
                race_map[str(race_id)] = start_ts

    return race_map, None


def _load_eval_bets(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[warn] run_dir not found: {run_dir}", file=sys.stderr)
            continue
        offset = _infer_w_idx_offset(run_dir.name)
        for bets_path in sorted(run_dir.glob("w*/bets.csv")):
            w_idx = _parse_window_name(bets_path.parent.name)
            if w_idx is None:
                continue
            w_idx = int(w_idx) + int(offset)
            if w_idx < 13:
                continue
            try:
                df = pd.read_csv(bets_path)
            except Exception as e:
                print(f"[warn] failed to read {bets_path}: {e}", file=sys.stderr)
                continue

            race_col = _find_col(df, RACE_ID_COLS)
            asof_col = _find_col(df, ASOF_COLS)
            stake_col = _find_col(df, STAKE_COLS)
            profit_col = _find_col(df, PROFIT_COLS)
            if race_col is None or asof_col is None or stake_col is None or profit_col is None:
                print(f"[warn] missing required cols in {bets_path}", file=sys.stderr)
                continue

            ratio_col = _find_col(df, RATIO_COLS)
            odds_final_col = _find_col(df, ODDS_FINAL_COLS)
            odds_buy_col = _find_col(df, ODDS_BUY_COLS)
            buy_time_col = _find_col(df, BUY_TIME_COLS)
            horse_col = _find_col(df, HORSE_NO_COLS)

            tmp = pd.DataFrame(
                {
                    "race_id": df[race_col].astype(str),
                    "horse_no": _to_num(df[horse_col]) if horse_col else np.nan,
                    "asof_time": pd.to_datetime(df[asof_col], errors="coerce"),
                    "stake": _to_num(df[stake_col]),
                    "profit": _to_num(df[profit_col]),
                }
            )

            if ratio_col:
                tmp["ratio_final_to_buy"] = _to_num(df[ratio_col])
            else:
                tmp["ratio_final_to_buy"] = np.nan

            if odds_final_col and odds_buy_col:
                odds_final = _to_num(df[odds_final_col])
                odds_buy = _to_num(df[odds_buy_col])
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = odds_final / odds_buy
                tmp["ratio_final_to_buy"] = tmp["ratio_final_to_buy"].fillna(ratio)

            if buy_time_col:
                tmp["buy_time"] = pd.to_datetime(df[buy_time_col], errors="coerce")

            tmp = tmp.dropna(subset=["race_id", "asof_time", "stake", "profit"])
            if tmp.empty:
                continue
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(
            columns=["race_id", "horse_no", "asof_time", "stake", "profit", "ratio_final_to_buy", "buy_time"]
        )
    return pd.concat(frames, ignore_index=True)


def _bin_stats(df: pd.DataFrame, lag_col: str) -> pd.DataFrame:
    df = df.copy()
    df["lag_bin"] = pd.cut(df[lag_col], bins=LAG_BINS, right=True, include_lowest=True)
    agg = df.groupby("lag_bin", dropna=False).agg(
        stake=("stake", "sum"),
        profit=("profit", "sum"),
        n_bets=("stake", "count"),
    ).reset_index()
    agg["roi"] = agg.apply(lambda r: (r["profit"] / r["stake"]) if r["stake"] else float("nan"), axis=1)
    return agg


def _lag_dist(df: pd.DataFrame, lag_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "p50": float("nan"),
                    "p80": float("nan"),
                    "p90": float("nan"),
                    "p95": float("nan"),
                    "frac_lag_gt30": float("nan"),
                    "frac_lag_gt60": float("nan"),
                    "frac_lag_gt120": float("nan"),
                }
            ]
        )
    vals = df[lag_col].dropna()
    return pd.DataFrame(
        [
            {
                "p50": float(vals.quantile(0.50)),
                "p80": float(vals.quantile(0.80)),
                "p90": float(vals.quantile(0.90)),
                "p95": float(vals.quantile(0.95)),
                "frac_lag_gt30": float((vals > 30).mean()),
                "frac_lag_gt60": float((vals > 60).mean()),
                "frac_lag_gt120": float((vals > 120).mean()),
            }
        ]
    )


def _slippage_by_lag(df: pd.DataFrame, lag_col: str) -> pd.DataFrame:
    df = df.copy()
    df = df[df["ratio_final_to_buy"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["lag_bin", "median_ratio_final_to_buy", "p10", "p90", "n_bets"])
    df["lag_bin"] = pd.cut(df[lag_col], bins=LAG_BINS, right=True, include_lowest=True)
    rows = []
    for lag_bin, g in df.groupby("lag_bin", dropna=False):
        vals = g["ratio_final_to_buy"].dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "lag_bin": lag_bin,
                "median_ratio_final_to_buy": float(vals.median()),
                "p10": float(vals.quantile(0.10)),
                "p90": float(vals.quantile(0.90)),
                "n_bets": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def _year_report(year: int, bins: pd.DataFrame, dist: pd.DataFrame, slip: pd.DataFrame) -> list[str]:
    lines = []
    lines.append(f"### {year} eval")
    if bins.empty:
        lines.append("- no eval bets found")
        return lines
    worst = bins.loc[bins["roi"].idxmin()]
    lines.append(
        f"- worst_roi_bin={worst['lag_bin']} roi={worst['roi']:.6f} stake={worst['stake']:.0f}"
    )
    if not dist.empty:
        row = dist.iloc[0]
        lines.append(
            f"- lag_p50={row['p50']:.2f}s lag_p90={row['p90']:.2f}s frac_gt60={row['frac_lag_gt60']:.3f}"
        )
    if not slip.empty:
        tail = slip.sort_values("lag_bin").tail(1)
        if not tail.empty:
            tail = tail.iloc[0]
            lines.append(
                f"- slippage_tail_bin={tail['lag_bin']} median_ratio_final_to_buy={tail['median_ratio_final_to_buy']:.4f}"
            )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze asof lag and slippage by lag")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--buy-minutes", type=int, default=1)
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_import_path()
    from keiba.db.loader import get_session

    session = get_session()

    reports = []
    for year in [2024, 2025]:
        run_dirs = _load_manifest(manifest_dir, str(year))
        df = _load_eval_bets(run_dirs)
        if df.empty:
            print(f"[warn] no eval bets for {year}", file=sys.stderr)
            bins = pd.DataFrame(columns=["lag_bin", "stake", "profit", "roi", "n_bets"])
            dist = _lag_dist(df, "asof_lag_sec")
            slip = pd.DataFrame(columns=["lag_bin", "median_ratio_final_to_buy", "p10", "p90", "n_bets"])
            bins.to_csv(out_dir / f"asof_lag_bins_{year}_eval.csv", index=False, encoding="utf-8")
            dist.to_csv(out_dir / f"asof_lag_dist_{year}_eval.csv", index=False, encoding="utf-8")
            slip.to_csv(out_dir / f"slippage_by_lag_{year}_eval.csv", index=False, encoding="utf-8")
            reports.extend(_year_report(year, bins, dist, slip))
            continue

        if "buy_time" not in df.columns or df["buy_time"].isna().all():
            race_ids = sorted(df["race_id"].dropna().unique().tolist())
            race_map, err = _fetch_start_times(session, race_ids)
            if race_map is None:
                report_path = out_dir / "asof_lag_report.md"
                report_path.write_text(
                    "asof lag diagnosis unavailable: " + str(err) + "\n", encoding="utf-8"
                )
                for y in [2024, 2025]:
                    empty_bins = pd.DataFrame(columns=["lag_bin", "stake", "profit", "roi", "n_bets"])
                    empty_dist = pd.DataFrame(
                        [
                            {
                                "p50": float("nan"),
                                "p80": float("nan"),
                                "p90": float("nan"),
                                "p95": float("nan"),
                                "frac_lag_gt30": float("nan"),
                                "frac_lag_gt60": float("nan"),
                                "frac_lag_gt120": float("nan"),
                            }
                        ]
                    )
                    empty_slip = pd.DataFrame(
                        columns=["lag_bin", "median_ratio_final_to_buy", "p10", "p90", "n_bets"]
                    )
                    empty_bins.to_csv(out_dir / f"asof_lag_bins_{y}_eval.csv", index=False, encoding="utf-8")
                    empty_dist.to_csv(out_dir / f"asof_lag_dist_{y}_eval.csv", index=False, encoding="utf-8")
                    empty_slip.to_csv(out_dir / f"slippage_by_lag_{y}_eval.csv", index=False, encoding="utf-8")
                print(f"[warn] {err}", file=sys.stderr)
                return
            df["buy_time"] = df["race_id"].map(race_map)

        df = df.dropna(subset=["buy_time", "asof_time"])
        if df.empty:
            print(f"[warn] no valid buy_time/asof_time for {year}", file=sys.stderr)
            continue

        df["asof_lag_sec"] = (df["buy_time"] - df["asof_time"]).dt.total_seconds()
        neg_count = int((df["asof_lag_sec"] < 0).sum())
        if neg_count:
            df = df[df["asof_lag_sec"] >= 0]

        bins = _bin_stats(df, "asof_lag_sec")
        dist = _lag_dist(df, "asof_lag_sec")
        slip = _slippage_by_lag(df, "asof_lag_sec")

        bins.to_csv(out_dir / f"asof_lag_bins_{year}_eval.csv", index=False, encoding="utf-8")
        dist.to_csv(out_dir / f"asof_lag_dist_{year}_eval.csv", index=False, encoding="utf-8")
        slip.to_csv(out_dir / f"slippage_by_lag_{year}_eval.csv", index=False, encoding="utf-8")

        reports.extend(_year_report(year, bins, dist, slip))
        if neg_count:
            reports.append(f"- {year} eval negative_lag_count={neg_count} (excluded)")

    report_path = out_dir / "asof_lag_report.md"
    report_path.write_text("\n".join(reports) + "\n", encoding="utf-8")
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
