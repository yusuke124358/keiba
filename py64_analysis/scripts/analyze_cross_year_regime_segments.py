
# Analyze cross-year regime segments and replay segment exclusion filters.
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


WIN_RE = re.compile(r"^w(\d{3})_(\d{8})_(\d{8})")

STAKE_COLS = ["stake", "stake_yen", "stake_amount", "total_stake", "stake_sum"]
PROFIT_COLS = ["profit", "net_profit", "pnl"]
RACE_ID_COLS = ["race_id", "raceid"]
HORSE_NO_COLS = ["horse_no", "horse_num", "horse"]
ASOF_COLS = ["asof_time", "asof_ts", "t_snap", "asof"]
ODDS_COLS = ["odds_at_buy", "odds", "buy_odds", "odds_buy"]
P_MKT_COLS = ["p_mkt", "market_prob"]

FIELD_SIZE_COLS = ["field_size", "n_horses", "horse_count", "entry_count", "head_count"]
DATE_COLS = ["date", "race_date"]
TRACK_COLS = ["track_code"]
SURFACE_COLS = [
    "surface",
    "syubetu_cd",
    "syubetu",
    "shubetsu",
    "shubetsu_cd",
    "track_type",
    "course_type",
    "track_surface",
    "surface_type",
]
DIST_COLS = ["distance", "race_distance"]
GRADE_COLS = ["grade", "grade_code", "race_grade", "race_class", "class"]
TURN_COLS = ["course_turn", "turn", "course_dir"]
WEATHER_COLS = ["weather", "weather_code"]
GOING_COLS = ["going", "going_code", "track_condition"]


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


def _parse_window_name(name: str) -> Optional[tuple[int, date, date]]:
    m = WIN_RE.match(name)
    if not m:
        return None
    w_idx = int(m.group(1))
    test_start = datetime.strptime(m.group(2), "%Y%m%d").date()
    test_end = datetime.strptime(m.group(3), "%Y%m%d").date()
    return w_idx, test_start, test_end


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _load_manifest(manifest_dir: Path, variant: str, year: str) -> list[Path]:
    path = manifest_dir / f"manifest_{year}.json"
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    variants = manifest.get("variants", {})
    run_dirs = variants.get(variant)
    if not run_dirs:
        raise SystemExit(f"variant not found in manifest: {variant}")
    root = Path.cwd()
    resolved = []
    for p in run_dirs:
        p = Path(p)
        if not p.is_absolute():
            p = root / p
        resolved.append(p)
    return resolved


def _parse_race_id(race_id: str) -> tuple[Optional[date], Optional[str]]:
    if race_id is None:
        return None, None
    s = str(race_id)
    if len(s) < 10 or not s[:8].isdigit():
        return None, None
    try:
        race_date = datetime.strptime(s[:8], "%Y%m%d").date()
    except Exception:
        race_date = None
    track_code = s[8:10] if len(s) >= 10 else None
    return race_date, track_code


def _surface_label(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "unknown"
    s = str(val).strip().lower()
    if s in {"0", "none", "nan", "null", ""}:
        return "unknown"
    if s in {"1", "01", "turf", "grass", "t"}:
        return "turf"
    if s in {"2", "02", "dirt", "d"}:
        return "dirt"
    if s in {"3", "03", "jump", "j"}:
        return "jump"
    if "\u829d" in s:
        return "turf"
    if "\u30c0" in s or "\u30c0\u30fc\u30c8" in s:
        return "dirt"
    if "\u969c" in s:
        return "jump"
    return "unknown"


def _distance_bucket(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "unknown"
    try:
        d = float(val)
    except Exception:
        return "unknown"
    if d <= 1200:
        return "<=1200"
    if d <= 1600:
        return "1201-1600"
    if d <= 2000:
        return "1601-2000"
    if d <= 2400:
        return "2001-2400"
    if d > 2400:
        return ">2400"
    return "unknown"


def _field_size_bucket(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "unknown"
    try:
        n = int(val)
    except Exception:
        return "unknown"
    if n <= 10:
        return "<=10"
    if n <= 14:
        return "11-14"
    if n <= 18:
        return "15-18"
    return "unknown"


def _get_columns(session, table_name: str) -> set[str]:
    from sqlalchemy import text

    rows = session.execute(
        text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
        ),
        {"t": table_name},
    ).fetchall()
    return {r[0] for r in rows}


def _table_exists(session, table_name: str) -> bool:
    from sqlalchemy import text

    row = session.execute(
        text(
            "SELECT 1 FROM information_schema.tables WHERE table_name = :t"
        ),
        {"t": table_name},
    ).fetchone()
    return row is not None


def _fetch_field_size(session, race_ids: list[str]) -> dict[str, int]:
    from sqlalchemy import bindparam, text

    candidates = ["fact_entry", "fact_entries", "fact_result", "fact_results"]
    table = None
    race_col = None
    for t in candidates:
        if not _table_exists(session, t):
            continue
        cols = _get_columns(session, t)
        if "race_id" in cols:
            table = t
            race_col = "race_id"
            break
        if "raceid" in cols:
            table = t
            race_col = "raceid"
            break
    if table is None:
        return {}

    size_map: dict[str, int] = {}
    chunk = 1000
    sql = text(
        f"SELECT {race_col} AS race_id, COUNT(*) AS field_size "
        f"FROM {table} WHERE {race_col} IN :race_ids GROUP BY {race_col}"
    )
    sql = sql.bindparams(bindparam("race_ids", expanding=True))
    for i in range(0, len(race_ids), chunk):
        part = race_ids[i : i + chunk]
        rows = session.execute(sql, {"race_ids": part}).fetchall()
        for race_id, size in rows:
            if race_id is None or size is None:
                continue
            size_map[str(race_id)] = int(size)
    return size_map


def _fetch_race_attrs(session, race_ids: list[str]) -> pd.DataFrame:
    from sqlalchemy import bindparam, text

    cols = _get_columns(session, "fact_race")

    def pick(candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        return None

    date_col = pick(DATE_COLS)
    track_col = pick(TRACK_COLS)
    surface_candidates = [c for c in SURFACE_COLS if c in cols]
    dist_col = pick(DIST_COLS)
    field_col = pick(FIELD_SIZE_COLS)
    grade_col = pick(GRADE_COLS)
    turn_col = pick(TURN_COLS)
    weather_col = pick(WEATHER_COLS)
    going_col = pick(GOING_COLS)

    select_cols = ["race_id"]
    for col in [date_col, track_col, dist_col, field_col, grade_col, turn_col, weather_col, going_col] + surface_candidates:
        if col and col not in select_cols:
            select_cols.append(col)

    if len(select_cols) == 1:
        df = pd.DataFrame({"race_id": race_ids})
    else:
        sql = text(
            f"SELECT {', '.join(select_cols)} FROM fact_race WHERE race_id IN :race_ids"
        )
        sql = sql.bindparams(bindparam("race_ids", expanding=True))
        rows = []
        chunk = 1000
        for i in range(0, len(race_ids), chunk):
            part = race_ids[i : i + chunk]
            rows.extend(session.execute(sql, {"race_ids": part}).fetchall())
        df = pd.DataFrame(rows, columns=select_cols)

    df["race_id"] = df["race_id"].astype(str)

    if date_col and "race_date" not in df.columns:
        df = df.rename(columns={date_col: "race_date"})
    if track_col and "track_code" not in df.columns:
        df = df.rename(columns={track_col: "track_code"})
    # Surface candidates are handled after load to allow fallback across columns.
    if dist_col and "distance" not in df.columns:
        df = df.rename(columns={dist_col: "distance"})
    if field_col and "field_size" not in df.columns:
        df = df.rename(columns={field_col: "field_size"})
    if grade_col and "grade" not in df.columns:
        df = df.rename(columns={grade_col: "grade"})
    if turn_col and "course_turn" not in df.columns:
        df = df.rename(columns={turn_col: "course_turn"})
    if weather_col and "weather" not in df.columns:
        df = df.rename(columns={weather_col: "weather"})
    if going_col and "going" not in df.columns:
        df = df.rename(columns={going_col: "going"})

    if "race_date" not in df.columns:
        df["race_date"] = None
    if "track_code" not in df.columns:
        df["track_code"] = None
    surface_raw = None
    for col in surface_candidates:
        if col in df.columns:
            if surface_raw is None:
                surface_raw = df[col]
            else:
                surface_raw = surface_raw.combine_first(df[col])
    if surface_raw is None:
        df["surface_raw"] = None
    else:
        df["surface_raw"] = surface_raw
    if "distance" not in df.columns:
        df["distance"] = None
    if "field_size" not in df.columns:
        df["field_size"] = None

    race_dates = []
    track_codes = []
    for rid in df["race_id"].astype(str).tolist():
        r_date, t_code = _parse_race_id(rid)
        race_dates.append(r_date)
        track_codes.append(t_code)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.date
    df["race_date"] = df["race_date"].fillna(pd.Series(race_dates))
    df["track_code"] = df["track_code"].fillna(pd.Series(track_codes))

    if df["field_size"].isna().all():
        size_map = _fetch_field_size(session, race_ids)
        if size_map:
            df["field_size"] = df["race_id"].map(size_map)

    df["surface"] = df["surface_raw"].apply(_surface_label)
    return df


def _load_bets(run_dirs: list[Path], year: int) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[warn] run_dir missing: {run_dir}", file=sys.stderr)
            continue
        offset = _infer_w_idx_offset(run_dir.name)
        for bets_path in sorted(run_dir.glob("w*/bets.csv")):
            win = bets_path.parent.name
            parsed = _parse_window_name(win)
            if not parsed:
                continue
            w_idx, test_start, test_end = parsed
            w_idx = int(w_idx) + int(offset)
            split = "design" if w_idx <= 12 else "eval"

            try:
                df = pd.read_csv(bets_path)
            except Exception as e:
                print(f"[warn] failed to read {bets_path}: {e}", file=sys.stderr)
                continue

            stake_col = _find_col(df, STAKE_COLS)
            profit_col = _find_col(df, PROFIT_COLS)
            race_col = _find_col(df, RACE_ID_COLS)
            horse_col = _find_col(df, HORSE_NO_COLS)
            asof_col = _find_col(df, ASOF_COLS)
            odds_col = _find_col(df, ODDS_COLS)
            p_mkt_col = _find_col(df, P_MKT_COLS)

            if stake_col is None or profit_col is None or race_col is None or horse_col is None:
                print(f"[warn] missing columns in {bets_path}", file=sys.stderr)
                continue

            tmp = pd.DataFrame(
                {
                    "year": int(year),
                    "window_idx": int(w_idx),
                    "split": split,
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "race_id": df[race_col].astype(str),
                    "horse_no": _to_num(df[horse_col]),
                    "stake": _to_num(df[stake_col]),
                    "profit": _to_num(df[profit_col]),
                }
            )

            if asof_col:
                tmp["asof_time"] = pd.to_datetime(df[asof_col], errors="coerce")
            else:
                tmp["asof_time"] = pd.NaT

            if odds_col:
                tmp["odds_at_buy"] = _to_num(df[odds_col])
            else:
                tmp["odds_at_buy"] = np.nan

            if p_mkt_col:
                tmp["p_mkt"] = _to_num(df[p_mkt_col])
            else:
                tmp["p_mkt"] = np.nan

            tmp = tmp.dropna(subset=["stake", "profit", "race_id", "horse_no"])
            if tmp.empty:
                continue
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(
            columns=[
                "year",
                "window_idx",
                "split",
                "test_start",
                "test_end",
                "race_id",
                "horse_no",
                "stake",
                "profit",
                "asof_time",
                "odds_at_buy",
                "p_mkt",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, dropna=False).agg(
        n_bets=("stake", "count"),
        stake_sum=("stake", "sum"),
        profit_sum=("profit", "sum"),
        hit_rate=("profit", lambda s: float((s > 0).mean()) if len(s) else float("nan")),
        avg_odds_at_buy=("odds_at_buy", "mean"),
        avg_p_mkt=("p_mkt", "mean"),
    ).reset_index()
    out["roi"] = out.apply(
        lambda r: (r["profit_sum"] / r["stake_sum"]) if r["stake_sum"] else float("nan"),
        axis=1,
    )
    return out


def _with_segment_key(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["segment_key"] = df[key_cols].astype(str).agg("|".join, axis=1)
    return df


def _drivers_eval(df: pd.DataFrame, segment_type: str, key_cols: list[str]) -> pd.DataFrame:
    eval_df = df[df["split"] == "eval"].copy()
    eval_df = _with_segment_key(eval_df, key_cols)
    pivot = eval_df.pivot_table(
        index="segment_key",
        columns="year",
        values=["roi", "stake_sum"],
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame(
            columns=[
                "segment_type",
                "segment_key",
                "stake_2024",
                "roi_2024",
                "stake_2025",
                "roi_2025",
                "delta_roi",
                "delta_profit_est",
            ]
        )
    pivot.columns = ["_".join([str(a), str(b)]) for a, b in pivot.columns]
    pivot = pivot.reset_index()

    for col in ["stake_sum_2024", "stake_sum_2025", "roi_2024", "roi_2025"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["delta_roi"] = pivot["roi_2025"] - pivot["roi_2024"]
    pivot["stake_total"] = pivot["stake_sum_2024"].fillna(0) + pivot["stake_sum_2025"].fillna(0)
    pivot["delta_profit_est"] = pivot["delta_roi"] * pivot["stake_total"]
    pivot["segment_type"] = segment_type
    pivot = pivot.rename(
        columns={
            "stake_sum_2024": "stake_2024",
            "stake_sum_2025": "stake_2025",
        }
    )
    return pivot[
        [
            "segment_type",
            "segment_key",
            "stake_2024",
            "roi_2024",
            "stake_2025",
            "roi_2025",
            "delta_roi",
            "delta_profit_est",
            "stake_total",
        ]
    ]


def _select_worst_segment(drivers: pd.DataFrame) -> Optional[str]:
    if drivers.empty:
        return None
    df = drivers.copy()
    both_bad = df[(df["roi_2024"] < 0) & (df["roi_2025"] < 0)]
    if not both_bad.empty:
        return both_bad.sort_values("stake_total", ascending=False).iloc[0]["segment_key"]
    df = df.copy()
    df["combined_roi"] = df["delta_profit_est"] / df["stake_total"].replace(0, np.nan)
    df = df.sort_values("combined_roi", ascending=True)
    return df.iloc[0]["segment_key"]


def _get_window_meta(run_dirs: list[Path]) -> dict[int, dict]:
    meta: dict[int, dict] = {}
    for run_dir in run_dirs:
        if not run_dir.exists():
            continue
        offset = _infer_w_idx_offset(run_dir.name)
        for window_dir in run_dir.glob("w*"):
            if not window_dir.is_dir():
                continue
            parsed = _parse_window_name(window_dir.name)
            if not parsed:
                continue
            w_idx, test_start, test_end = parsed
            w_idx = int(w_idx) + int(offset)
            cfg_path = window_dir / "config_used.yaml"
            bankroll = 1_000_000
            if cfg_path.exists():
                try:
                    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                    bankroll = int(cfg.get("betting", {}).get("bankroll_yen", bankroll))
                except Exception:
                    pass
            meta[w_idx] = {
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "bankroll": bankroll,
                "split": "design" if w_idx <= 12 else "eval",
            }
    return meta


def _compute_maxdd(df: pd.DataFrame, initial_bankroll: float) -> float:
    if df.empty:
        return 0.0
    df = df.copy()
    if df["asof_time"].isna().all():
        df = df.sort_values("race_date")
    else:
        df = df.sort_values("asof_time")
    equity = initial_bankroll + df["profit"].cumsum()
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, np.nan)
    return float(dd.max()) if len(dd) else 0.0


def _window_metrics(df: pd.DataFrame, window_meta: dict[int, dict]) -> dict[int, dict]:
    metrics = {}
    for w_idx, g in df.groupby("window_idx"):
        stake_sum = float(g["stake"].sum())
        profit_sum = float(g["profit"].sum())
        roi = (profit_sum / stake_sum) if stake_sum else float("nan")
        n_bets = int(len(g))
        meta = window_meta.get(int(w_idx), {})
        bankroll = float(meta.get("bankroll", 1_000_000))
        maxdd = _compute_maxdd(g, bankroll)
        metrics[int(w_idx)] = {
            "stake_sum": stake_sum,
            "profit_sum": profit_sum,
            "roi": roi,
            "n_bets": n_bets,
            "maxdd": maxdd,
        }
    return metrics


def _paired_compare(
    base_metrics: dict[int, dict],
    var_metrics: dict[int, dict],
    window_meta: dict[int, dict],
    filter_name: str,
) -> pd.DataFrame:
    rows = []
    for w_idx, meta in window_meta.items():
        base = base_metrics.get(w_idx, {"stake_sum": 0.0, "profit_sum": 0.0, "roi": float("nan"), "n_bets": 0, "maxdd": 0.0})
        var = var_metrics.get(w_idx, {"stake_sum": 0.0, "profit_sum": 0.0, "roi": float("nan"), "n_bets": 0, "maxdd": 0.0})
        rows.append(
            {
                "filter": filter_name,
                "window_idx": int(w_idx),
                "split": meta.get("split"),
                "test_start": meta.get("test_start"),
                "test_end": meta.get("test_end"),
                "stake_base": base["stake_sum"],
                "profit_base": base["profit_sum"],
                "roi_base": base["roi"],
                "n_bets_base": base["n_bets"],
                "maxdd_base": base["maxdd"],
                "stake_var": var["stake_sum"],
                "profit_var": var["profit_sum"],
                "roi_var": var["roi"],
                "n_bets_var": var["n_bets"],
                "maxdd_var": var["maxdd"],
                "d_roi": (var["roi"] - base["roi"]) if (not math.isnan(var["roi"]) and not math.isnan(base["roi"])) else float("nan"),
                "d_maxdd": var["maxdd"] - base["maxdd"],
                "d_n_bets": var["n_bets"] - base["n_bets"],
            }
        )
    return pd.DataFrame(rows)


def _summarize_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for filt, g in paired.groupby("filter"):
        for split, s in g.groupby("split"):
            n = int(len(s))
            improve_rate = float((s["d_roi"] > 0).mean()) if n else float("nan")
            median_d_roi = float(s["d_roi"].median()) if n else float("nan")
            median_d_maxdd = float(s["d_maxdd"].median()) if n else float("nan")
            median_d_n_bets = float(s["d_n_bets"].median()) if n else float("nan")
            median_n_bets_var = float(s["n_bets_var"].median()) if n else float("nan")
            min_n_bets_var = int(s["n_bets_var"].min()) if n else 0
            zero_bet_windows = int((s["n_bets_var"] <= 0).sum()) if n else 0
            gate_pass = (
                improve_rate >= 0.6
                and median_d_roi > 0
                and median_d_maxdd <= 0
                and median_n_bets_var >= 80
                and zero_bet_windows == 0
            )
            rows.append(
                {
                    "filter": filt,
                    "split": split,
                    "n_windows": n,
                    "improve_rate_roi": improve_rate,
                    "median_d_roi": median_d_roi,
                    "median_d_maxdd": median_d_maxdd,
                    "median_d_n_bets": median_d_n_bets,
                    "median_n_bets_var": median_n_bets_var,
                    "min_n_bets_var": min_n_bets_var,
                    "n_zero_bet_windows_var": zero_bet_windows,
                    "gate_pass": gate_pass if split == "eval" else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze cross-year segments and replay filters")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_import_path()
    from keiba.db.loader import get_session

    session = get_session()

    run_dirs_2024 = _load_manifest(manifest_dir, args.variant, "2024")
    run_dirs_2025 = _load_manifest(manifest_dir, args.variant, "2025")

    bets_2024 = _load_bets(run_dirs_2024, 2024)
    bets_2025 = _load_bets(run_dirs_2025, 2025)
    bets = pd.concat([bets_2024, bets_2025], ignore_index=True)

    race_ids = sorted(bets["race_id"].dropna().unique().tolist())
    race_attrs = _fetch_race_attrs(session, race_ids)

    bets = bets.merge(race_attrs, on="race_id", how="left")
    bets["race_date"] = pd.to_datetime(bets["race_date"], errors="coerce").dt.date
    bets["month"] = bets["race_date"].apply(lambda d: d.month if isinstance(d, date) else None)
    bets["surface"] = bets.get("surface", "unknown").fillna("unknown")
    bets["distance_bucket"] = bets["distance"].apply(_distance_bucket)
    bets["field_size_bucket"] = bets["field_size"].apply(_field_size_bucket)

    bets["surface_distance"] = bets["surface"].astype(str) + "|" + bets["distance_bucket"].astype(str)
    bets["track_surface_distance"] = bets["track_code"].astype(str) + "|" + bets["surface"].astype(str) + "|" + bets["distance_bucket"].astype(str)

    def _unknown_ratio_eval(df: pd.DataFrame, year: int) -> float:
        sub = df[(df["year"] == year) & (df["split"] == "eval")]
        if sub.empty:
            return float("nan")
        return float((sub["surface"].astype(str) == "unknown").mean())

    unknown_ratio_2024_eval = _unknown_ratio_eval(bets, 2024)
    unknown_ratio_2025_eval = _unknown_ratio_eval(bets, 2025)
    surface_missing = (
        (not math.isnan(unknown_ratio_2024_eval) and unknown_ratio_2024_eval >= 0.99)
        and (not math.isnan(unknown_ratio_2025_eval) and unknown_ratio_2025_eval >= 0.99)
    )

    by_track = _aggregate(bets, ["year", "split", "track_code"])
    by_surface = _aggregate(bets, ["year", "split", "surface"])
    by_distance = _aggregate(bets, ["year", "split", "distance_bucket"])
    by_surface_distance = _aggregate(bets, ["year", "split", "surface_distance"])
    by_field = _aggregate(bets, ["year", "split", "field_size_bucket"])
    by_month = _aggregate(bets, ["year", "split", "month"])
    by_track_surface_distance = _aggregate(bets, ["year", "split", "track_surface_distance"])

    by_track.to_csv(out_dir / "by_track.csv", index=False, encoding="utf-8")
    by_surface.to_csv(out_dir / "by_surface.csv", index=False, encoding="utf-8")
    by_distance.to_csv(out_dir / "by_distance_bucket.csv", index=False, encoding="utf-8")
    by_surface_distance.to_csv(out_dir / "by_surface_distance.csv", index=False, encoding="utf-8")
    by_field.to_csv(out_dir / "by_field_size_bucket.csv", index=False, encoding="utf-8")
    by_month.to_csv(out_dir / "by_month.csv", index=False, encoding="utf-8")
    by_track_surface_distance.to_csv(out_dir / "by_track_surface_distance.csv", index=False, encoding="utf-8")

    drivers = []
    drivers.append(_drivers_eval(by_track, "track_code", ["track_code"]))
    drivers.append(_drivers_eval(by_surface, "surface", ["surface"]))
    drivers.append(_drivers_eval(by_distance, "distance_bucket", ["distance_bucket"]))
    drivers.append(_drivers_eval(by_surface_distance, "surface_distance", ["surface_distance"]))
    drivers.append(_drivers_eval(by_field, "field_size_bucket", ["field_size_bucket"]))
    drivers.append(_drivers_eval(by_month, "month", ["month"]))
    drivers.append(_drivers_eval(by_track_surface_distance, "track_surface_distance", ["track_surface_distance"]))
    drivers_eval = pd.concat(drivers, ignore_index=True)
    drivers_eval = drivers_eval.sort_values(
        ["stake_total", "delta_roi"], ascending=[False, False]
    ).reset_index(drop=True)
    drivers_eval.to_csv(out_dir / "drivers_eval.csv", index=False, encoding="utf-8")

    report_lines = []
    report_lines.append("# Cross-year segment drivers")
    report_lines.append("")
    report_lines.append("## Surface coverage")
    report_lines.append(f"- unknown_ratio_2024_eval={unknown_ratio_2024_eval}")
    report_lines.append(f"- unknown_ratio_2025_eval={unknown_ratio_2025_eval}")
    if surface_missing:
        report_lines.append("- WARNING: surface missing (all unknown in eval)")
    report_lines.append("")
    report_lines.append("## Top eval drivers (stake large + delta ROI)")
    top5 = drivers_eval.head(5)
    for _, row in top5.iterrows():
        report_lines.append(
            f"- {row['segment_type']}:{row['segment_key']} delta_roi={row['delta_roi']:.6f} "
            f"stake_2024={row['stake_2024']:.0f} stake_2025={row['stake_2025']:.0f}"
        )

    report_lines.append("")
    report_lines.append("## Consistent losers (both years ROI<0, stake large)")
    both_bad = drivers_eval[(drivers_eval["roi_2024"] < 0) & (drivers_eval["roi_2025"] < 0)]
    if not both_bad.empty:
        for _, row in both_bad.sort_values("stake_total", ascending=False).head(5).iterrows():
            report_lines.append(
                f"- {row['segment_type']}:{row['segment_key']} roi_2024={row['roi_2024']:.6f} "
                f"roi_2025={row['roi_2025']:.6f} stake_total={row['stake_total']:.0f}"
            )
    else:
        report_lines.append("- none")

    allowed_types = {
        "track_code",
        "month",
        "distance_bucket",
        "field_size_bucket",
        "surface",
        "surface_distance",
        "track_surface_distance",
    }

    consistent_losers = drivers_eval[(drivers_eval["roi_2024"] < 0) & (drivers_eval["roi_2025"] < 0)].copy()
    consistent_losers = consistent_losers.sort_values("stake_total", ascending=False)

    candidates = []
    for _, row in consistent_losers.iterrows():
        if row["segment_type"] not in allowed_types:
            continue
        candidates.append(
            {
                "segment_type": row["segment_type"],
                "segment_key": str(row["segment_key"]),
                "stake_total": row["stake_total"],
                "roi_2024": row["roi_2024"],
                "roi_2025": row["roi_2025"],
            }
        )
        if len(candidates) >= 8:
            break

    manual_targets = [("month", "10"), ("distance_bucket", "2001-2400")]
    for seg_type, seg_key in manual_targets:
        if len(candidates) >= 12:
            break
        if seg_type not in allowed_types:
            continue
        exists = drivers_eval[
            (drivers_eval["segment_type"] == seg_type)
            & (drivers_eval["segment_key"].astype(str).isin([seg_key, f"{seg_key}.0"]))
        ]
        if exists.empty:
            continue
        row = exists.iloc[0]
        candidates.append(
            {
                "segment_type": seg_type,
                "segment_key": str(row["segment_key"]),
                "stake_total": row["stake_total"],
                "roi_2024": row["roi_2024"],
                "roi_2025": row["roi_2025"],
            }
        )

    # Deduplicate and cap
    seen = set()
    filtered = []
    for cand in candidates:
        key = (cand["segment_type"], cand["segment_key"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(cand)
        if len(filtered) >= 12:
            break
    candidates = filtered

    filters = []
    for cand in candidates:
        name = f"EXCL_{cand['segment_type']}_{cand['segment_key']}"
        filters.append(
            {
                "name": name,
                "type": cand["segment_type"],
                "key": cand["segment_key"],
            }
        )

    pd.DataFrame(
        [
            {
                "name": f["name"],
                "segment_type": f["type"],
                "segment_key": f["key"],
                "stake_total": next((c["stake_total"] for c in candidates if c["segment_type"] == f["type"] and c["segment_key"] == f["key"]), None),
                "roi_2024": next((c["roi_2024"] for c in candidates if c["segment_type"] == f["type"] and c["segment_key"] == f["key"]), None),
                "roi_2025": next((c["roi_2025"] for c in candidates if c["segment_type"] == f["type"] and c["segment_key"] == f["key"]), None),
            }
            for f in filters
        ]
    ).to_csv(out_dir / "replay_candidates.csv", index=False, encoding="utf-8")

    window_meta_2024 = _get_window_meta(run_dirs_2024)
    window_meta_2025 = _get_window_meta(run_dirs_2025)

    replay_outputs = {}
    for year, df_year, window_meta in [
        (2024, bets_2024, window_meta_2024),
        (2025, bets_2025, window_meta_2025),
    ]:
        df_year = df_year.merge(race_attrs, on="race_id", how="left")
        df_year["race_date"] = pd.to_datetime(df_year["race_date"], errors="coerce").dt.date
        df_year["surface"] = df_year.get("surface", "unknown").fillna("unknown")
        df_year["distance_bucket"] = df_year["distance"].apply(_distance_bucket)
        df_year["field_size_bucket"] = df_year["field_size"].apply(_field_size_bucket)
        df_year["surface_distance"] = df_year["surface"].astype(str) + "|" + df_year["distance_bucket"].astype(str)
        df_year["track_code"] = df_year.get("track_code")
        df_year["track_surface_distance"] = df_year["track_code"].astype(str) + "|" + df_year["surface"].astype(str) + "|" + df_year["distance_bucket"].astype(str)
        df_year["month"] = df_year["race_date"].apply(lambda d: d.month if isinstance(d, date) else None)

        base_metrics = _window_metrics(df_year, window_meta)

        paired_all = []
        for filt in filters:
            segment_cols = {
                "track_code": "track_code",
                "month": "month",
                "distance_bucket": "distance_bucket",
                "field_size_bucket": "field_size_bucket",
                "surface": "surface",
                "surface_distance": "surface_distance",
                "track_surface_distance": "track_surface_distance",
            }
            col = segment_cols.get(filt["type"])
            if col:
                mask = df_year[col].astype(str) != filt["key"]
            else:
                mask = pd.Series([True] * len(df_year))

            df_var = df_year[mask].copy()
            var_metrics = _window_metrics(df_var, window_meta)
            paired = _paired_compare(base_metrics, var_metrics, window_meta, filt["name"])
            paired_all.append(paired)

        paired_df = pd.concat(paired_all, ignore_index=True) if paired_all else pd.DataFrame()
        summary_df = _summarize_paired(paired_df)

        replay_outputs[year] = {
            "paired": paired_df,
            "summary": summary_df,
        }

    replay_outputs[2024]["paired"].to_csv(out_dir / "replay_paired_compare_2024.csv", index=False, encoding="utf-8")
    replay_outputs[2025]["paired"].to_csv(out_dir / "replay_paired_compare_2025.csv", index=False, encoding="utf-8")
    replay_outputs[2024]["summary"].to_csv(out_dir / "replay_filters_summary_2024.csv", index=False, encoding="utf-8")
    replay_outputs[2025]["summary"].to_csv(out_dir / "replay_filters_summary_2025.csv", index=False, encoding="utf-8")

    s24 = replay_outputs[2024]["summary"]
    s25 = replay_outputs[2025]["summary"]
    eval24 = s24[s24["split"] == "eval"].set_index("filter")
    eval25 = s25[s25["split"] == "eval"].set_index("filter")
    rows = []
    for filt in sorted(set(eval24.index) | set(eval25.index)):
        r24 = eval24.loc[filt] if filt in eval24.index else None
        r25 = eval25.loc[filt] if filt in eval25.index else None
        pass_2024 = bool(r24["gate_pass"]) if r24 is not None else False
        pass_2025 = bool(r25["gate_pass"]) if r25 is not None else False
        if pass_2024 and pass_2025:
            decision = "pass_both"
        elif pass_2024 or pass_2025:
            decision = "pass_single_year"
        else:
            decision = "fail"
        rows.append(
            {
                "filter": filt,
                "median_d_roi_2024": r24["median_d_roi"] if r24 is not None else None,
                "improve_rate_2024": r24["improve_rate_roi"] if r24 is not None else None,
                "median_d_maxdd_2024": r24["median_d_maxdd"] if r24 is not None else None,
                "median_n_bets_var_2024": r24["median_n_bets_var"] if r24 is not None else None,
                "n_zero_bet_windows_2024": r24["n_zero_bet_windows_var"] if r24 is not None else None,
                "gate_pass_2024": pass_2024,
                "median_d_roi_2025": r25["median_d_roi"] if r25 is not None else None,
                "improve_rate_2025": r25["improve_rate_roi"] if r25 is not None else None,
                "median_d_maxdd_2025": r25["median_d_maxdd"] if r25 is not None else None,
                "median_n_bets_var_2025": r25["median_n_bets_var"] if r25 is not None else None,
                "n_zero_bet_windows_2025": r25["n_zero_bet_windows_var"] if r25 is not None else None,
                "gate_pass_2025": pass_2025,
                "decision": decision,
            }
        )

    replay_cross = pd.DataFrame(rows)
    replay_cross.to_csv(out_dir / "replay_cross_year_summary.csv", index=False, encoding="utf-8")

    report_lines.append("")
    report_lines.append("## Top replay candidates by cross-year promise")
    pass_both = replay_cross[replay_cross["decision"] == "pass_both"]
    if not pass_both.empty:
        for _, row in pass_both.iterrows():
            report_lines.append(f"- {row['filter']}: pass_both")
    else:
        pass_single = replay_cross[replay_cross["decision"] == "pass_single_year"].copy()
        pass_single = pass_single[
            (pass_single["median_d_roi_2024"] > 0) & (pass_single["median_d_roi_2025"] > 0)
        ]
        if pass_single.empty:
            report_lines.append("- none")
        else:
            pass_single["score"] = pass_single["median_d_roi_2024"] + pass_single["median_d_roi_2025"]
            for _, row in pass_single.sort_values("score", ascending=False).head(5).iterrows():
                report_lines.append(
                    f"- {row['filter']}: median_d_roi_2024={row['median_d_roi_2024']} median_d_roi_2025={row['median_d_roi_2025']}"
                )

    report_lines.append("")
    report_lines.append("## Replay filters (eval) summary")
    for _, row in replay_cross.iterrows():
        report_lines.append(
            f"- {row['filter']}: decision={row['decision']} "
            f"median_d_roi_2024={row['median_d_roi_2024']} median_d_roi_2025={row['median_d_roi_2025']}"
        )

    (out_dir / "replay_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()

