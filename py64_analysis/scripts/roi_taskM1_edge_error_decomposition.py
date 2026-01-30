"""
Task M1: Edge error decomposition vs market (full-field, split-aware).
Produces segment metrics (odds_band/field_size/surface/distance/track/month),
overall metrics, and day-bootstrap CI for delta logloss/brier.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from keiba.db.loader import get_session


ODDS_CANDIDATES = ["odds", "odds_val", "odds_at_buy", "odds_effective", "odds_final"]
Y_CANDIDATES = ["y_win", "is_win", "winner", "y_true", "y"]
FINISH_CANDIDATES = ["finish_pos", "finish_position"]
P_MODEL_CANDIDATES = ["p_model", "p_hat", "p_win", "prob"]
DATE_CANDIDATES = ["asof_time", "race_date", "date", "event_date", "buy_time", "bet_time"]


@dataclass(frozen=True)
class FullfieldFile:
    path: Path
    split: str


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(score)) + 1
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


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


def _field_size_bin(val: float) -> str:
    if val <= 10:
        return "<=10"
    if val <= 14:
        return "11-14"
    if val <= 18:
        return "15-18"
    return "19+"


def _distance_bin(val: float) -> str:
    if val < 1400:
        return "<1400"
    if val < 1800:
        return "1400-1799"
    if val < 2200:
        return "1800-2199"
    return "2200+"


def _surface_label(val) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "unknown"
    s = str(val).strip().lower()
    if s in ("1", "turf", "grass", "shiba"):
        return "turf"
    if s in ("2", "dirt", "dar"):
        return "dirt"
    if s in ("3", "jump"):
        return "jump"
    return "unknown"


def _detect_column(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _resolve_run_dirs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend([Path(p) for p in glob.glob(pat) if Path(p).is_dir()])
    return sorted(out, key=lambda p: p.name)


def _infer_split_from_name(name: str) -> Optional[str]:
    low = name.lower()
    if "train" in low:
        return "train"
    if "valid" in low or "val" in low:
        return "valid"
    if "test" in low or "preds" in low:
        return "test"
    return None


def _collect_fullfield_files(
    run_dir: Path, fullfield_root: Optional[Path]
) -> list[FullfieldFile]:
    run_id = run_dir.name
    roots: list[Path] = []
    if fullfield_root:
        roots.append(fullfield_root)
    roots.append(run_dir)

    collected: list[FullfieldFile] = []

    # Prefer combined fullfield_{split}.csv if present
    for root in roots:
        if root.is_file():
            split = _infer_split_from_name(root.name)
            if split:
                return [FullfieldFile(path=root, split=split)]
            continue
        for split in ["train", "valid", "test"]:
            candidates = list(root.rglob(f"fullfield_{split}.csv"))
            if not candidates:
                continue
            # prefer path containing run_id
            matched = [c for c in candidates if run_id in str(c)]
            use = matched[0] if matched else candidates[0]
            collected.append(FullfieldFile(path=use, split=split))
        test_alias = list(root.rglob("fullfield_preds.csv"))
        if test_alias and not any(f.split == "test" for f in collected):
            matched = [c for c in test_alias if run_id in str(c)]
            use = matched[0] if matched else test_alias[0]
            collected.append(FullfieldFile(path=use, split="test"))

    if collected:
        # If combined found, skip per-window files
        return collected

    # Fallback: per-window files
    for root in roots:
        if not root.exists() or root.is_file():
            continue
        for p in root.rglob("*_fullfield_*.csv"):
            if run_id not in str(p):
                continue
            split = _infer_split_from_name(p.name)
            if split:
                collected.append(FullfieldFile(path=p, split=split))
    return collected


def _generate_fullfield(run_dir: Path, staging_dir: Path) -> Optional[list[FullfieldFile]]:
    out_dir = staging_dir / "fullfield_generated" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "export_fullfield_preds_for_run_dir.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--run_dir",
        str(run_dir),
        "--out_dir",
        str(out_dir),
        "--splits",
        "train,valid,test",
    ]
    try:
        print(f"[info] generate fullfield: {run_dir}")
        import subprocess

        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[warn] fullfield generation failed for {run_dir}: {exc}")
        return None
    files = _collect_fullfield_files(out_dir, None)
    return files if files else None


def _fetch_race_meta(session, race_ids: list[str]) -> pd.DataFrame:
    if not race_ids:
        return pd.DataFrame()
    rows = []
    chunk_size = 1000
    for i in range(0, len(race_ids), chunk_size):
        chunk = race_ids[i : i + chunk_size]
        res = session.execute(
            text(
                """
                SELECT race_id, date, track_code, surface, distance, field_size
                FROM fact_race
                WHERE race_id = ANY(:race_ids)
                """
            ),
            {"race_ids": chunk},
        ).fetchall()
        rows.extend([dict(r._mapping) for r in res])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["race_id"] = df["race_id"].astype(str)
    return df


def _load_fullfield_file(path: Path, market_method: str) -> tuple[pd.DataFrame, dict]:
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    odds_col = _detect_column(cols, ODDS_CANDIDATES)
    y_col = _detect_column(cols, Y_CANDIDATES)
    finish_col = _detect_column(cols, FINISH_CANDIDATES)
    p_model_col = _detect_column(cols, P_MODEL_CANDIDATES)

    if market_method == "p_mkt_col":
        if "p_mkt_col" in cols:
            p_mkt_col = "p_mkt_col"
            mkt_alias = "p_mkt_col"
        elif "p_mkt" in cols:
            p_mkt_col = "p_mkt"
            mkt_alias = "p_mkt"
        else:
            raise SystemExit(
                f"p_mkt_col missing in {path}; available p_mkt candidates: "
                f"{[c for c in cols if c.startswith('p_mkt')]}"
            )
    else:
        if market_method in cols:
            p_mkt_col = market_method
            mkt_alias = market_method
        else:
            raise SystemExit(f"market_method {market_method} not found in {path}")

    if odds_col is None or p_model_col is None or p_mkt_col is None:
        raise SystemExit(f"Missing required columns in {path}")

    usecols = {odds_col, p_model_col, p_mkt_col, "race_id", "horse_no"}
    if y_col:
        usecols.add(y_col)
    if finish_col:
        usecols.add(finish_col)
    for col in DATE_CANDIDATES:
        if col in cols:
            usecols.add(col)
    if "split" in cols:
        usecols.add("split")
    if "window_id" in cols:
        usecols.add("window_id")

    df = pd.read_csv(path, usecols=sorted(usecols))
    out = pd.DataFrame()
    out["race_id"] = df["race_id"].astype(str)
    out["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    out["odds"] = pd.to_numeric(df[odds_col], errors="coerce")
    out["p_model"] = pd.to_numeric(df[p_model_col], errors="coerce")
    out["p_mkt"] = pd.to_numeric(df[p_mkt_col], errors="coerce")

    if y_col and y_col in df.columns:
        out["y_win"] = pd.to_numeric(df[y_col], errors="coerce")
    elif finish_col and finish_col in df.columns:
        out["y_win"] = (pd.to_numeric(df[finish_col], errors="coerce") == 1).astype(int)
    else:
        raise SystemExit(f"y_win not found in {path}")

    for col in DATE_CANDIDATES:
        if col in df.columns:
            out[col] = df[col]
    if "split" in df.columns:
        out["split"] = df["split"].astype(str)
    if "window_id" in df.columns:
        out["window_id"] = df["window_id"].astype(str)

    meta = {
        "odds_col": odds_col,
        "p_model_col": p_model_col,
        "p_mkt_col": p_mkt_col,
        "p_mkt_alias": mkt_alias,
        "y_col": y_col or finish_col or "missing",
    }
    return out, meta


def _infer_day_key(df: pd.DataFrame) -> pd.Series:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                return dt.dt.date
    if "race_id" in df.columns:
        race_id = df["race_id"].astype(str).str.slice(0, 8)
        dt = pd.to_datetime(race_id, errors="coerce", format="%Y%m%d")
        if dt.notna().any():
            return dt.dt.date
    cols = ", ".join(df.columns)
    raise SystemExit(f"day_key inference failed; available_cols=[{cols}]")


def _dedup(df: pd.DataFrame, dedup_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        return df, {"dedup_key": dedup_cols, "rows_before": 0, "rows_after": 0, "rows_dropped": 0}
    valid_cols = [c for c in dedup_cols if c in df.columns]
    if not valid_cols:
        return df, {
            "dedup_key": [],
            "rows_before": len(df),
            "rows_after": len(df),
            "rows_dropped": 0,
        }
    before = len(df)
    deduped = df.drop_duplicates(valid_cols, keep="first")
    after = len(deduped)
    return deduped, {
        "dedup_key": valid_cols,
        "rows_before": before,
        "rows_after": after,
        "rows_dropped": before - after,
    }


def _integrity_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["valid_row"] = (
        df["odds"].notna()
        & (df["odds"] > 0)
        & df["p_model"].notna()
        & df["p_mkt"].notna()
        & df["y_win"].isin([0, 1])
        & np.isfinite(df["odds"])
        & np.isfinite(df["p_model"])
        & np.isfinite(df["p_mkt"])
    )
    df_valid = df[df["valid_row"]].copy()

    stats = (
        df_valid.groupby("race_id", dropna=False)
        .agg(
            n_rows=("race_id", "size"),
            dup_horse=("horse_no", lambda s: s.duplicated().sum()),
            sum_y=("y_win", "sum"),
        )
        .reset_index()
    )
    stats["sum_y_ok"] = stats["sum_y"] == 1
    stats["race_ok"] = (stats["dup_horse"] == 0) & (stats["sum_y_ok"]) & (stats["n_rows"] > 0)
    bad_races = set(stats.loc[~stats["race_ok"], "race_id"].astype(str).tolist())
    filtered = df_valid[~df_valid["race_id"].isin(bad_races)].copy()

    summary = {
        "total_races": int(stats["race_id"].nunique()),
        "dropped_races": int(len(bad_races)),
        "dropped_ratio": float(len(bad_races) / stats["race_id"].nunique()) if stats["race_id"].nunique() else 0.0,
        "dup_races": int((stats["dup_horse"] > 0).sum()),
        "sumy_bad_races": int((~stats["sum_y_ok"]).sum()),
    }
    return filtered, summary


def _metric_row(df: pd.DataFrame) -> dict:
    y = df["y_win"].to_numpy()
    p_model = df["p_model"].to_numpy()
    p_mkt = df["p_mkt"].to_numpy()
    return {
        "n_rows": int(len(df)),
        "n_races": int(df["race_id"].nunique()),
        "logloss_model": _logloss(y, p_model),
        "logloss_mkt": _logloss(y, p_mkt),
        "brier_model": _brier(y, p_model),
        "brier_mkt": _brier(y, p_mkt),
        "auc_model": _auc(y, p_model),
        "auc_mkt": _auc(y, p_mkt),
        "avg_p_model": float(np.mean(p_model)),
        "win_rate": float(np.mean(y)),
    }


def _bootstrap_ci_day(
    df: pd.DataFrame, n_boot: int, seed: int
) -> dict:
    if df.empty or n_boot <= 0:
        return {
            "n_days": 0,
            "delta_logloss_ci_low": float("nan"),
            "delta_logloss_ci_med": float("nan"),
            "delta_logloss_ci_high": float("nan"),
            "delta_brier_ci_low": float("nan"),
            "delta_brier_ci_med": float("nan"),
            "delta_brier_ci_high": float("nan"),
        }

    df = df.copy()
    df["day_key"] = _infer_day_key(df).astype(str)

    y = df["y_win"].to_numpy()
    p_model = df["p_model"].to_numpy()
    p_mkt = df["p_mkt"].to_numpy()

    ll_model = -(y * np.log(np.clip(p_model, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p_model, 1e-9, 1 - 1e-9)))
    ll_mkt = -(y * np.log(np.clip(p_mkt, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p_mkt, 1e-9, 1 - 1e-9)))
    brier_model = (p_model - y) ** 2
    brier_mkt = (p_mkt - y) ** 2

    sums = (
        pd.DataFrame(
            {
                "day_key": df["day_key"].values,
                "sum_ll_model": ll_model,
                "sum_ll_mkt": ll_mkt,
                "sum_brier_model": brier_model,
                "sum_brier_mkt": brier_mkt,
                "n_rows": 1,
            }
        )
        .groupby("day_key", dropna=False)
        .sum()
        .reset_index()
    )
    if sums.empty or len(sums) < 2:
        return {
            "n_days": int(len(sums)),
            "delta_logloss_ci_low": float("nan"),
            "delta_logloss_ci_med": float("nan"),
            "delta_logloss_ci_high": float("nan"),
            "delta_brier_ci_low": float("nan"),
            "delta_brier_ci_med": float("nan"),
            "delta_brier_ci_high": float("nan"),
        }

    sums_ll_model = sums["sum_ll_model"].to_numpy()
    sums_ll_mkt = sums["sum_ll_mkt"].to_numpy()
    sums_brier_model = sums["sum_brier_model"].to_numpy()
    sums_brier_mkt = sums["sum_brier_mkt"].to_numpy()
    counts = sums["n_rows"].to_numpy()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    ll_model_s = sums_ll_model[idx].sum(axis=1)
    ll_mkt_s = sums_ll_mkt[idx].sum(axis=1)
    brier_model_s = sums_brier_model[idx].sum(axis=1)
    brier_mkt_s = sums_brier_mkt[idx].sum(axis=1)
    counts_s = counts[idx].sum(axis=1)

    delta_ll = (ll_model_s / counts_s) - (ll_mkt_s / counts_s)
    delta_brier = (brier_model_s / counts_s) - (brier_mkt_s / counts_s)

    ll_low, ll_med, ll_high = np.quantile(delta_ll, [0.025, 0.5, 0.975]).tolist()
    b_low, b_med, b_high = np.quantile(delta_brier, [0.025, 0.5, 0.975]).tolist()
    return {
        "n_days": int(len(sums)),
        "delta_logloss_ci_low": float(ll_low),
        "delta_logloss_ci_med": float(ll_med),
        "delta_logloss_ci_high": float(ll_high),
        "delta_brier_ci_low": float(b_low),
        "delta_brier_ci_med": float(b_med),
        "delta_brier_ci_high": float(b_high),
    }


def _segment_metrics(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    rows = []
    for seg_val, sub in df.groupby(segment_col, dropna=False):
        metrics = _metric_row(sub)
        delta_logloss = metrics["logloss_model"] - metrics["logloss_mkt"]
        delta_brier = metrics["brier_model"] - metrics["brier_mkt"]
        delta_auc = metrics["auc_model"] - metrics["auc_mkt"]
        calib_gap = metrics["avg_p_model"] - metrics["win_rate"]
        rows.append(
            {
                "segment": segment_col,
                "segment_value": seg_val,
                "n_races": metrics["n_races"],
                "n_rows": metrics["n_rows"],
                "delta_logloss": delta_logloss,
                "delta_brier": delta_brier,
                "delta_auc": delta_auc,
                "calib_gap": calib_gap,
                "abs_calib_gap": abs(calib_gap),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="TaskM1 edge error decomposition (full-field)")
    ap.add_argument("--run_dir_glob", nargs="+", required=True)
    ap.add_argument("--fullfield_root", default=None)
    ap.add_argument("--market_method", default="p_mkt_col")
    ap.add_argument("--dedup_key", default="race_id,horse_no,asof_time")
    ap.add_argument("--min_races", type=int, default=500)
    ap.add_argument("--bootstrap_n", type=int, default=500)
    ap.add_argument("--resample_unit", default="day", choices=["day"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_jp", required=True)
    ap.add_argument("--out_ascii", required=True)
    ap.add_argument("--generate_if_missing", action="store_true")
    args = ap.parse_args()

    run_dirs = _resolve_run_dirs(args.run_dir_glob)
    if not run_dirs:
        raise SystemExit(f"No run dirs matched: {args.run_dir_glob}")

    out_ascii = Path(args.out_ascii)
    out_jp = Path(args.out_jp)
    out_ascii.mkdir(parents=True, exist_ok=True)
    out_jp.mkdir(parents=True, exist_ok=True)
    tables_dir = out_ascii / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    fullfield_root = Path(args.fullfield_root) if args.fullfield_root else None
    dedup_cols = [c.strip() for c in args.dedup_key.split(",") if c.strip()]

    session = None
    try:
        session = get_session()
    except Exception:
        session = None

    manifest = {
        "run_dirs": [str(p) for p in run_dirs],
        "market_method": args.market_method,
        "dedup_key": dedup_cols,
        "bootstrap_n": int(args.bootstrap_n),
        "resample_unit": args.resample_unit,
        "missing_fullfield": [],
        "dedup": {},
        "integrity": {},
        "meta": {},
    }

    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}
    split_meta: dict[str, dict] = {}

    for run_dir in run_dirs:
        files = _collect_fullfield_files(run_dir, fullfield_root)
        if args.generate_if_missing:
            found_splits = {f.split for f in files}
            need_splits = {"train", "valid", "test"}
            if not files or not need_splits.issubset(found_splits):
                generated = _generate_fullfield(run_dir, out_ascii)
                if generated:
                    # keep existing, fill missing splits from generated
                    by_split = {f.split: f for f in files}
                    for g in generated:
                        if g.split not in by_split:
                            by_split[g.split] = g
                    files = list(by_split.values())
        if not files:
            manifest["missing_fullfield"].append(str(run_dir))
            continue

        for f in files:
            df, meta = _load_fullfield_file(f.path, args.market_method)
            if "split" in df.columns:
                # honor split column if present
                for split_name, sub in df.groupby("split", dropna=False):
                    split = str(split_name)
                    if split not in split_frames:
                        continue
                    sub = sub.copy()
                    sub["run_id"] = run_dir.name
                    split_frames[split].append(sub)
            else:
                split = f.split
                if split not in split_frames:
                    continue
                df = df.copy()
                df["run_id"] = run_dir.name
                split_frames[split].append(df)
            if f.split not in split_meta:
                split_meta[f.split] = meta

    overall_rows = []
    bootstrap_rows = []
    segment_rows = []
    segment_bootstrap_rows = []
    dedup_rows = []

    for split, frames in split_frames.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df, dedup_meta = _dedup(df, dedup_cols)
        manifest["dedup"][split] = dedup_meta

        df_clean, integ = _integrity_filter(df)
        manifest["integrity"][split] = integ

        if session is not None:
            race_meta = _fetch_race_meta(session, df_clean["race_id"].unique().tolist())
        else:
            race_meta = pd.DataFrame()

        if not race_meta.empty:
            df_clean = df_clean.merge(race_meta, on="race_id", how="left")
        else:
            df_clean["track_code"] = "unknown"
            df_clean["surface"] = "unknown"
            df_clean["distance"] = np.nan
            df_clean["field_size"] = np.nan

        # derive date/month
        df_clean["day_key"] = _infer_day_key(df_clean)
        df_clean["month"] = pd.to_datetime(df_clean["day_key"], errors="coerce").dt.month

        # derive field_size from data if missing
        if df_clean["field_size"].isna().all():
            field_sizes = df_clean.groupby("race_id", dropna=False)["horse_no"].nunique().reset_index()
            field_sizes = field_sizes.rename(columns={"horse_no": "field_size"})
            df_clean = df_clean.merge(field_sizes, on="race_id", how="left")

        df_clean["odds_band"] = df_clean["odds"].apply(lambda v: _odds_band(float(v)) if np.isfinite(v) else "unknown")
        df_clean["field_size_bin"] = df_clean["field_size"].apply(
            lambda v: _field_size_bin(float(v)) if pd.notna(v) else "unknown"
        )
        df_clean["surface"] = df_clean["surface"].apply(_surface_label)
        df_clean["distance_bin"] = df_clean["distance"].apply(
            lambda v: _distance_bin(float(v)) if pd.notna(v) else "unknown"
        )
        if "track_code" not in df_clean.columns:
            df_clean["track_code"] = "unknown"

        metrics = _metric_row(df_clean)
        delta_logloss = metrics["logloss_model"] - metrics["logloss_mkt"]
        delta_brier = metrics["brier_model"] - metrics["brier_mkt"]
        delta_auc = metrics["auc_model"] - metrics["auc_mkt"]
        calib_gap = metrics["avg_p_model"] - metrics["win_rate"]

        overall_rows.append(
            {
                "split": split,
                "n_races": metrics["n_races"],
                "n_rows": metrics["n_rows"],
                "logloss_model": metrics["logloss_model"],
                "logloss_mkt": metrics["logloss_mkt"],
                "delta_logloss": delta_logloss,
                "brier_model": metrics["brier_model"],
                "brier_mkt": metrics["brier_mkt"],
                "delta_brier": delta_brier,
                "auc_model": metrics["auc_model"],
                "auc_mkt": metrics["auc_mkt"],
                "delta_auc": delta_auc,
                "calib_gap": calib_gap,
                "abs_calib_gap": abs(calib_gap),
            }
        )

        ci = _bootstrap_ci_day(df_clean, args.bootstrap_n, args.seed)
        bootstrap_rows.append(
            {
                "split": split,
                "bootstrap_n": int(args.bootstrap_n),
                **ci,
            }
        )

        for seg in ["odds_band", "field_size_bin", "surface", "distance_bin", "track_code", "month"]:
            if seg not in df_clean.columns:
                continue
            seg_df = _segment_metrics(df_clean, seg)
            seg_df["split"] = split
            segment_rows.append(seg_df)

            # bootstrap CI per segment (min_races)
            for seg_val, sub in df_clean.groupby(seg, dropna=False):
                if sub["race_id"].nunique() < args.min_races:
                    continue
                ci_seg = _bootstrap_ci_day(sub, args.bootstrap_n, args.seed)
                segment_bootstrap_rows.append(
                    {
                        "split": split,
                        "segment": seg,
                        "segment_value": seg_val,
                        "n_races": int(sub["race_id"].nunique()),
                        "bootstrap_n": int(args.bootstrap_n),
                        **ci_seg,
                    }
                )

    overall_df = pd.DataFrame(overall_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    segment_df = pd.concat(segment_rows, ignore_index=True) if segment_rows else pd.DataFrame()
    segment_boot_df = pd.DataFrame(segment_bootstrap_rows)

    overall_df.to_csv(out_ascii / "edge_overall_by_split.csv", index=False, encoding="utf-8")
    bootstrap_df.to_csv(out_ascii / "bootstrap_ci_by_split.csv", index=False, encoding="utf-8")
    if not segment_df.empty:
        segment_df.to_csv(out_ascii / "edge_by_segment.csv", index=False, encoding="utf-8")
    if not segment_boot_df.empty:
        segment_boot_df.to_csv(out_ascii / "bootstrap_ci_by_segment.csv", index=False, encoding="utf-8")

    manifest["meta"] = split_meta
    (out_ascii / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # identify good/bad segments on test split
    bad_rows_df = pd.DataFrame()
    good_rows_df = pd.DataFrame()
    if not segment_df.empty:
        seg_test = segment_df[segment_df["split"] == "test"].copy()
        seg_test = seg_test[seg_test["n_races"] >= args.min_races]
        if not seg_test.empty:
            bad_rows_df = seg_test.sort_values("delta_logloss", ascending=False).head(10)
            good_rows_df = seg_test.sort_values("delta_logloss", ascending=True).head(10)
            bad_rows_df.to_csv(out_ascii / "segments_worst_delta_logloss.csv", index=False, encoding="utf-8")
            good_rows_df.to_csv(out_ascii / "segments_best_delta_logloss.csv", index=False, encoding="utf-8")

    # report
    report_lines = [
        "# TaskM1 edge error decomposition (full-field)",
        "",
        f"- run_dirs: {len(run_dirs)}",
        f"- run_dir_glob: {args.run_dir_glob}",
        f"- market_method: {args.market_method}",
        f"- dedup_key: {dedup_cols}",
        f"- bootstrap_n: {args.bootstrap_n}",
        f"- resample_unit: {args.resample_unit}",
        f"- missing_fullfield: {len(manifest['missing_fullfield'])}",
        "",
        "## Overall metrics by split",
        overall_df.to_string(index=False) if not overall_df.empty else "No overall rows.",
        "",
        "## Day-bootstrap CI by split",
        bootstrap_df.to_string(index=False) if not bootstrap_df.empty else "No bootstrap rows.",
    ]

    if not bad_rows_df.empty:
        report_lines.extend(
            [
                "",
                "## Worst segments (test, delta_logloss high)",
                bad_rows_df.to_string(index=False),
            ]
        )
    if not good_rows_df.empty:
        report_lines.extend(
            [
                "",
                "## Best segments (test, delta_logloss low)",
                good_rows_df.to_string(index=False),
            ]
        )
    report_lines.append("")
    report_lines.append("ROI definition: ROI = profit / stake (profit = return - stake).")
    (out_ascii / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # minimal update + stdout (placeholders for tests)
    today = datetime.now().strftime("%Y-%m-%d")
    if not overall_df.empty and "test" in overall_df["split"].values:
        test_row = overall_df[overall_df["split"] == "test"].iloc[0].to_dict()
        test_ci = bootstrap_df[bootstrap_df["split"] == "test"].iloc[0].to_dict() if not bootstrap_df.empty else {}
    else:
        test_row = {}
        test_ci = {}

    minimal_lines = [
        f"# minimal update for chat - TaskM1 edge decomposition ({today})",
        "",
        "## Summary",
        f"- run_dirs: {len(run_dirs)} | splits: {', '.join([s for s in ['train','valid','test'] if s in overall_df.get('split', [])])}",
    ]
    if test_row:
        minimal_lines.extend(
            [
                f"- test delta_logloss={test_row.get('delta_logloss'):.6f} | delta_auc={test_row.get('delta_auc'):.6f}",
                f"- test calib_gap={test_row.get('calib_gap'):.6f} | abs={test_row.get('abs_calib_gap'):.6f}",
            ]
        )
    if test_ci:
        minimal_lines.append(
            f"- test day CI delta_logloss [{test_ci.get('delta_logloss_ci_low'):.6f}, {test_ci.get('delta_logloss_ci_med'):.6f}, {test_ci.get('delta_logloss_ci_high'):.6f}]"
        )
    minimal_lines.extend(
        [
            "",
            "## Paths",
            f"- out_ascii: {out_ascii}",
            f"- out_jp: {out_jp}",
        ]
    )
    (out_ascii / "minimal_update_for_chat.md").write_text("\n".join(minimal_lines) + "\n", encoding="utf-8")

    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        f"[diag] taskM1_edge_decomp_done=true | n_runs={len(run_dirs)} | missing_fullfield={len(manifest['missing_fullfield'])}",
        f"[plan] decision=model_update_next | reason=await_m1_summary",
        f"[paths] out_dir={out_jp} | staging={out_ascii} | zip=<NOT_USED>",
    ]
    (out_ascii / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
