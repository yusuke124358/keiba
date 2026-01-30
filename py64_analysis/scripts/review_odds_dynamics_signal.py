import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text

from keiba.db.loader import get_session


ODDS_COLS = ["odds_at_buy", "odds", "buy_odds", "odds_buy"]
PMKT_COLS = ["p_mkt", "market_prob"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        v = float(v)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _parse_year_from_path(path: Path) -> int | None:
    m = re.search(r"(20\d{2})", str(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _list_bets_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("w*/bets.csv"))


def _load_bets(run_dir: Path) -> pd.DataFrame:
    bets_files = _list_bets_files(run_dir)
    if not bets_files:
        return pd.DataFrame()
    parts = []
    for fp in bets_files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df.empty:
            continue
        df["window"] = fp.parent.name
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=0, ignore_index=True)
    out["run_dir"] = str(run_dir)
    return out


def _fetch_snapshot_at(session, race_id: str, t_cutoff: datetime) -> pd.DataFrame:
    query = text(
        """
        SELECT DISTINCT ON (horse_no)
            horse_no, odds, asof_time
        FROM odds_ts_win
        WHERE race_id = :race_id
          AND odds > 0
          AND asof_time <= :t_cutoff
        ORDER BY horse_no, asof_time DESC
        """
    )
    rows = session.execute(query, {"race_id": race_id, "t_cutoff": t_cutoff}).fetchall()
    return pd.DataFrame([dict(r._mapping) for r in rows])


def _fetch_series_between(
    session, race_id: str, horse_nos: list[int], t_start: datetime, t_end: datetime
) -> pd.DataFrame:
    if not horse_nos:
        return pd.DataFrame()
    query = text(
        """
        SELECT horse_no, asof_time, odds
        FROM odds_ts_win
        WHERE race_id = :race_id
          AND horse_no = ANY(:horse_nos)
          AND odds > 0
          AND asof_time BETWEEN :t_start AND :t_end
        ORDER BY horse_no, asof_time
        """
    )
    rows = session.execute(
        query,
        {"race_id": race_id, "horse_nos": horse_nos, "t_start": t_start, "t_end": t_end},
    ).fetchall()
    return pd.DataFrame([dict(r._mapping) for r in rows])


def _compute_odds_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    odds_col = _find_col(df, ODDS_COLS)
    p_mkt_col = _find_col(df, PMKT_COLS)
    if not odds_col:
        df["odds_at_buy"] = np.nan
    else:
        df["odds_at_buy"] = pd.to_numeric(df[odds_col], errors="coerce")
    if not p_mkt_col:
        df["p_mkt_at_buy"] = np.nan
    else:
        df["p_mkt_at_buy"] = pd.to_numeric(df[p_mkt_col], errors="coerce")

    if "odds_chg_10m" in df.columns:
        df["odds_delta_log_10m"] = pd.to_numeric(df["odds_chg_10m"], errors="coerce")
        df.loc[df["odds_delta_log_10m"] > 0, "odds_delta_log_10m"] = np.log(
            df.loc[df["odds_delta_log_10m"] > 0, "odds_delta_log_10m"]
        )
        df.loc[df["odds_delta_log_10m"] <= 0, "odds_delta_log_10m"] = np.nan
    else:
        df["odds_delta_log_10m"] = np.nan

    if "p_mkt_chg_10m" in df.columns:
        df["p_mkt_delta_10m"] = pd.to_numeric(df["p_mkt_chg_10m"], errors="coerce")
    else:
        df["p_mkt_delta_10m"] = np.nan

    return df


def _attach_db_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"race_queries": 0, "series_queries": 0}
    if df.empty:
        return df, stats
    if "asof_time" not in df.columns:
        return df, stats

    df = df.copy()
    df["asof_dt"] = pd.to_datetime(df["asof_time"], errors="coerce")
    df["odds_delta_log_5m"] = np.nan
    df["p_mkt_delta_5m"] = np.nan
    df["odds_vol_10m"] = np.nan
    df["odds_vol_10m_n"] = np.nan

    session = None
    try:
        session = get_session()
    except Exception:
        return df, stats

    try:
        grouped = df.dropna(subset=["asof_dt", "race_id"]).groupby(["race_id", "asof_dt"])
        for (race_id, asof_dt), g in grouped:
            t5 = asof_dt - timedelta(minutes=5)
            t10 = asof_dt - timedelta(minutes=10)

            try:
                snap = _fetch_snapshot_at(session, str(race_id), t5)
                stats["race_queries"] += 1
            except Exception:
                continue
            if snap.empty:
                continue

            snap["odds"] = pd.to_numeric(snap["odds"], errors="coerce")
            snap = snap[snap["odds"] > 0]
            if snap.empty:
                continue

            inv = 1.0 / snap["odds"]
            inv_sum = float(inv.sum())
            snap["p_mkt_t5"] = inv / inv_sum if inv_sum > 0 else np.nan
            odds_map = dict(zip(snap["horse_no"].astype(int), snap["odds"]))
            p_map = dict(zip(snap["horse_no"].astype(int), snap["p_mkt_t5"]))

            idx = g.index
            odds_at_buy = df.loc[idx, "odds_at_buy"].astype(float)
            horse_nos = df.loc[idx, "horse_no"].astype(int)
            odds_5m = horse_nos.map(lambda hn: odds_map.get(int(hn)))
            p_5m = horse_nos.map(lambda hn: p_map.get(int(hn)))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = odds_at_buy / odds_5m
                df.loc[idx, "odds_delta_log_5m"] = np.log(ratio)
            df.loc[idx, "p_mkt_delta_5m"] = df.loc[idx, "p_mkt_at_buy"] - p_5m

            try:
                horse_list = [int(h) for h in horse_nos.unique().tolist() if pd.notna(h)]
                series = _fetch_series_between(session, str(race_id), horse_list, t10, asof_dt)
                stats["series_queries"] += 1
            except Exception:
                continue
            if series.empty:
                continue
            series["odds"] = pd.to_numeric(series["odds"], errors="coerce")
            series = series[series["odds"] > 0]
            if series.empty:
                continue
            series["log_odds"] = np.log(series["odds"].astype(float))
            vol = (
                series.groupby("horse_no")["log_odds"]
                .agg(["std", "count"])
                .reset_index()
            )
            vol = vol[vol["count"] >= 2]
            vol_map = dict(zip(vol["horse_no"].astype(int), vol["std"]))
            vol_n_map = dict(zip(vol["horse_no"].astype(int), vol["count"]))
            df.loc[idx, "odds_vol_10m"] = horse_nos.map(lambda hn: vol_map.get(int(hn)))
            df.loc[idx, "odds_vol_10m_n"] = horse_nos.map(lambda hn: vol_n_map.get(int(hn)))
    finally:
        if session is not None:
            session.close()

    return df, stats


def _bucket_table(
    df: pd.DataFrame, metric: str, buckets: int, year: int
) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[metric], errors="coerce")
    valid = s[np.isfinite(s)]
    if len(valid) < buckets:
        return pd.DataFrame()
    try:
        labels = list(range(1, buckets + 1))
        df = df.copy()
        df["bucket"] = pd.qcut(s, q=buckets, labels=labels, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    grouped = df.dropna(subset=["bucket"]).groupby("bucket")
    out = grouped.agg(
        n_bets=("stake", "count"),
        stake=("stake", "sum"),
        profit=("profit", "sum"),
        winrate=("profit", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
        avg_odds=("odds_at_buy", "mean"),
        avg_metric=(metric, "mean"),
    ).reset_index()
    out["roi"] = out["profit"] / out["stake"]
    out["metric"] = metric
    out["year"] = year
    out["bucket"] = out["bucket"].astype(int)
    return out


def _worst_bucket_direction(table: pd.DataFrame) -> str:
    if table.empty:
        return "unknown"
    table = table.sort_values("roi", ascending=True)
    worst_bucket = int(table.iloc[0]["bucket"])
    if worst_bucket <= 3:
        return "low"
    if worst_bucket >= 8:
        return "high"
    return "mid"


def _write_report(
    out_dir: Path,
    tables: list[pd.DataFrame],
    slippage: dict[int, pd.DataFrame],
    stats: dict[str, dict[str, int]],
    worst_dirs: dict[int, str],
    metric_main: str,
) -> None:
    def _df_to_md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            cols = list(df.columns)
            lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
            for _, row in df.iterrows():
                lines.append("|" + "|".join(str(row[c]) for c in cols) + "|")
            return "\n".join(lines)

    lines = []
    lines.append("# Odds Dynamics Diagnostic (q95 eval)")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Metrics use pre-buy odds snapshots only. Any slippage proxy uses post-buy info for analysis only.")
    lines.append("")
    if stats:
        lines.append("DB query stats:")
        for year, s in stats.items():
            lines.append(f"- {year}: race_queries={s.get('race_queries', 0)} series_queries={s.get('series_queries', 0)}")
        lines.append("")
    for year, direction in worst_dirs.items():
        lines.append(f"- {year} worst_bucket_direction({metric_main})={direction}")
    lines.append("")
    for metric in sorted({t['metric'].iloc[0] for t in tables if not t.empty}):
        lines.append(f"## {metric}")
        for t in tables:
            if t.empty:
                continue
            if t["metric"].iloc[0] != metric:
                continue
            lines.append(f"### {int(t['year'].iloc[0])}")
            lines.append(_df_to_md(t))
            lines.append("")
    if slippage:
        lines.append("## Slippage proxy (ratio_final_to_buy by bucket)")
        for year, t in slippage.items():
            lines.append(f"### {year}")
            if t is None or t.empty:
                lines.append("N/A")
            else:
                lines.append(_df_to_md(t))
            lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", action="append", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--buckets", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tables: list[pd.DataFrame] = []
    slippage_tables: dict[int, pd.DataFrame] = {}
    stats_by_year: dict[int, dict[str, int]] = {}
    worst_dirs: dict[int, str] = {}

    for run in args.run_dir:
        run_dir = Path(run)
        df = _load_bets(run_dir)
        if df.empty:
            continue
        year = _parse_year_from_path(run_dir)
        if year is None:
            continue

        df = _compute_odds_dynamics(df)
        df, stats = _attach_db_metrics(df)
        stats_by_year[year] = stats

        metrics = [
            "odds_delta_log_5m",
            "odds_delta_log_10m",
            "p_mkt_delta_5m",
            "p_mkt_delta_10m",
            "odds_vol_10m",
        ]
        for metric in metrics:
            table = _bucket_table(df, metric, args.buckets, year)
            if not table.empty:
                all_tables.append(table)

        # slippage proxy (analysis only)
        if "ratio_final_to_buy" in df.columns and not df["ratio_final_to_buy"].isna().all():
            sl = df.copy()
            metric = "odds_delta_log_5m" if "odds_delta_log_5m" in sl.columns else "odds_delta_log_10m"
            s = pd.to_numeric(sl[metric], errors="coerce")
            valid = s[np.isfinite(s)]
            if len(valid) >= args.buckets:
                try:
                    sl["bucket"] = pd.qcut(s, q=args.buckets, labels=list(range(1, args.buckets + 1)), duplicates="drop")
                    st = sl.groupby("bucket").agg(
                        n_bets=("ratio_final_to_buy", "count"),
                        avg_ratio=("ratio_final_to_buy", "mean"),
                        p50=("ratio_final_to_buy", "median"),
                    ).reset_index()
                    st["bucket"] = st["bucket"].astype(int)
                    slippage_tables[year] = st
                except Exception:
                    slippage_tables[year] = pd.DataFrame()
            else:
                slippage_tables[year] = pd.DataFrame()

        metric_main = "odds_delta_log_5m" if df["odds_delta_log_5m"].notna().sum() > 0 else "odds_delta_log_10m"
        table_main = next((t for t in all_tables if not t.empty and t["year"].iloc[0] == year and t["metric"].iloc[0] == metric_main), pd.DataFrame())
        worst_dirs[year] = _worst_bucket_direction(table_main)

    if all_tables:
        bucket_tables = pd.concat(all_tables, axis=0, ignore_index=True)
        bucket_tables.to_csv(out_dir / "bucket_tables.csv", index=False, encoding="utf-8")
    else:
        (out_dir / "bucket_tables.csv").write_text("", encoding="utf-8")

    metric_main = "odds_delta_log_5m"
    if metric_main not in {t["metric"].iloc[0] for t in all_tables if not t.empty}:
        metric_main = "odds_delta_log_10m"

    _write_report(out_dir, all_tables, slippage_tables, stats_by_year, worst_dirs, metric_main)

    years = sorted(worst_dirs.keys())
    cross_year_consistent = False
    if len(years) >= 2:
        dirs = [worst_dirs[y] for y in years]
        cross_year_consistent = all(d == dirs[0] for d in dirs) and dirs[0] in ("low", "high")

    print(f"[odds-dyn] computed=true | lookbacks=5m,10m | buckets={args.buckets}")
    print(
        f"[odds-dyn] worst_bucket_direction={worst_dirs} | cross_year_consistent={str(cross_year_consistent).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
