"""
Task M2: Market-anchored logit blend evaluation.
Fits blend weight w on train+valid (logloss), evaluates on test,
and reports delta_logloss vs market with race/day bootstrap CI.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


_ensure_import_path()
from keiba.utils.run_paths import make_analysis_out_dir, require_existing_dir  # noqa: E402


ODDS_CANDIDATES = ["odds", "odds_val", "odds_at_buy", "odds_effective", "odds_final"]
Y_CANDIDATES = ["y_win", "is_win", "winner", "y_true", "y"]
FINISH_CANDIDATES = ["finish_pos", "finish_position"]
P_MODEL_CANDIDATES = ["p_model", "p_hat", "p_win", "prob"]
DATE_CANDIDATES = ["asof_time", "race_date", "date", "event_date", "buy_time", "bet_time"]


@dataclass(frozen=True)
class FullfieldFile:
    path: Path
    split: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


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


def _collect_fullfield_files(run_dir: Path, fullfield_root: Optional[Path]) -> list[FullfieldFile]:
    run_id = run_dir.name
    roots: list[Path] = []
    if fullfield_root:
        roots.append(fullfield_root)
    roots.append(run_dir)

    collected: list[FullfieldFile] = []
    for root in roots:
        if root.is_file():
            split = _infer_split_from_name(root.name)
            if split:
                return [FullfieldFile(path=root, split=split)]
            continue
        if not root.exists():
            continue
        for split in ["train", "valid", "test"]:
            candidates = list(root.rglob(f"fullfield_{split}.csv"))
            if not candidates:
                continue
            matched = [c for c in candidates if run_id in str(c)]
            use = matched[0] if matched else candidates[0]
            collected.append(FullfieldFile(path=use, split=split))
        alias = list(root.rglob("fullfield_preds.csv"))
        if alias and not any(f.split == "test" for f in collected):
            matched = [c for c in alias if run_id in str(c)]
            use = matched[0] if matched else alias[0]
            collected.append(FullfieldFile(path=use, split="test"))
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
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[warn] fullfield generation failed for {run_dir}: {exc}")
        return None
    files = _collect_fullfield_files(out_dir, None)
    return files if files else None


def _load_fullfield_file(path: Path, market_method: str) -> tuple[pd.DataFrame, dict]:
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    odds_col = _detect_column(cols, ODDS_CANDIDATES)
    y_col = _detect_column(cols, Y_CANDIDATES)
    finish_col = _detect_column(cols, FINISH_CANDIDATES)
    p_model_col = _detect_column(cols, P_MODEL_CANDIDATES)

    if market_method != "p_mkt_col":
        raise SystemExit(f"market_method unsupported: {market_method}")

    if "p_mkt_col" in cols:
        p_mkt_col = "p_mkt_col"
    elif "p_mkt" in cols:
        p_mkt_col = "p_mkt"
    else:
        raise SystemExit(f"p_mkt_col missing in {path}")

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
    if "asof_time" in cols:
        usecols.add("asof_time")

    df = pd.read_csv(path, usecols=sorted(usecols))
    out = pd.DataFrame()
    out["race_id"] = df["race_id"].astype(str)
    out["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    out["odds"] = pd.to_numeric(df[odds_col], errors="coerce")
    out["p_model"] = pd.to_numeric(df[p_model_col], errors="coerce")
    out["p_mkt"] = pd.to_numeric(df[p_mkt_col], errors="coerce")
    if "p_hat" in df.columns:
        out["p_hat"] = pd.to_numeric(df["p_hat"], errors="coerce")

    if y_col and y_col in df.columns:
        out["y_win"] = pd.to_numeric(df[y_col], errors="coerce")
    elif finish_col and finish_col in df.columns:
        out["y_win"] = (pd.to_numeric(df[finish_col], errors="coerce") == 1).astype(int)
    else:
        raise SystemExit(f"y_win not found in {path}")

    if "window_id" in df.columns:
        out["window_id"] = df["window_id"].astype(str)
    if "asof_time" in df.columns:
        out["asof_time"] = df["asof_time"]
    for col in DATE_CANDIDATES:
        if col in df.columns:
            out[col] = df[col]
    if "split" in df.columns:
        out["split"] = df["split"].astype(str)

    meta = {
        "odds_col": odds_col,
        "p_model_col": p_model_col,
        "p_mkt_col": p_mkt_col,
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


def _integrity_filter(df: pd.DataFrame, model_col: str) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["valid_row"] = (
        df["odds"].notna()
        & (df["odds"] > 0)
        & df["p_mkt"].notna()
        & df[model_col].notna()
        & df["y_win"].isin([0, 1])
        & np.isfinite(df["odds"])
        & np.isfinite(df["p_mkt"])
        & np.isfinite(df[model_col])
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
    invalid_rows = df.groupby("race_id", dropna=False)["valid_row"].apply(lambda s: (~s).sum()).reset_index()
    invalid_rows = invalid_rows.rename(columns={"valid_row": "invalid_rows"})
    stats = stats.merge(invalid_rows, on="race_id", how="left")
    stats["invalid_rows"] = stats["invalid_rows"].fillna(0).astype(int)

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
        "invalid_rows_races": int((stats["invalid_rows"] > 0).sum()),
    }
    return filtered, summary


def _fit_w_grid(y: np.ndarray, logit_model: np.ndarray, logit_mkt: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.0, 1.0, 1001)
    delta = logit_model - logit_mkt
    base = logit_mkt
    best_w = 0.0
    best_ll = math.inf
    for w in grid:
        logit_blend = base + w * delta
        p = _sigmoid(logit_blend)
        ll = _logloss(y, p)
        if ll < best_ll:
            best_ll = ll
            best_w = float(w)
    return best_w, float(best_ll)


def _metrics_for_split(df: pd.DataFrame, w: float, model_col: str) -> dict:
    y = df["y_win"].to_numpy()
    p_model = df[model_col].to_numpy()
    p_mkt = df["p_mkt"].to_numpy()
    logit_model = _logit(p_model)
    logit_mkt = _logit(p_mkt)
    logit_blend = logit_mkt + w * (logit_model - logit_mkt)
    p_blend = _sigmoid(logit_blend)
    return {
        "n_rows": int(len(df)),
        "n_races": int(df["race_id"].nunique()),
        "logloss_blend": _logloss(y, p_blend),
        "logloss_mkt": _logloss(y, p_mkt),
        "brier_blend": _brier(y, p_blend),
        "brier_mkt": _brier(y, p_mkt),
        "auc_blend": _auc(y, p_blend),
        "auc_mkt": _auc(y, p_mkt),
        "avg_p_blend": float(np.mean(p_blend)),
        "win_rate": float(np.mean(y)),
    }


def _bootstrap_ci(df: pd.DataFrame, w: float, model_col: str, unit: str, n_boot: int, seed: int) -> dict:
    if df.empty or n_boot <= 0:
        return {
            "n_groups": 0,
            "delta_logloss_ci_low": float("nan"),
            "delta_logloss_ci_med": float("nan"),
            "delta_logloss_ci_high": float("nan"),
            "delta_brier_ci_low": float("nan"),
            "delta_brier_ci_med": float("nan"),
            "delta_brier_ci_high": float("nan"),
        }
    df = df.copy()
    if unit == "race":
        key = "race_id"
    elif unit == "day":
        df["day_key"] = _infer_day_key(df)
        key = "day_key"
    else:
        raise SystemExit(f"Unknown resample unit: {unit}")

    y = df["y_win"].to_numpy()
    p_model = df[model_col].to_numpy()
    p_mkt = df["p_mkt"].to_numpy()
    logit_model = _logit(p_model)
    logit_mkt = _logit(p_mkt)
    logit_blend = logit_mkt + w * (logit_model - logit_mkt)
    p_blend = _sigmoid(logit_blend)

    ll_blend = -(y * np.log(np.clip(p_blend, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p_blend, 1e-9, 1 - 1e-9)))
    ll_mkt = -(y * np.log(np.clip(p_mkt, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p_mkt, 1e-9, 1 - 1e-9)))
    brier_blend = (p_blend - y) ** 2
    brier_mkt = (p_mkt - y) ** 2

    sums = (
        pd.DataFrame(
            {
                key: df[key].values,
                "sum_ll_blend": ll_blend,
                "sum_ll_mkt": ll_mkt,
                "sum_brier_blend": brier_blend,
                "sum_brier_mkt": brier_mkt,
                "n_rows": 1,
            }
        )
        .groupby(key, dropna=False)
        .sum()
        .reset_index()
    )

    if len(sums) < 2:
        return {
            "n_groups": int(len(sums)),
            "delta_logloss_ci_low": float("nan"),
            "delta_logloss_ci_med": float("nan"),
            "delta_logloss_ci_high": float("nan"),
            "delta_brier_ci_low": float("nan"),
            "delta_brier_ci_med": float("nan"),
            "delta_brier_ci_high": float("nan"),
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    ll_blend_s = sums["sum_ll_blend"].to_numpy()[idx].sum(axis=1)
    ll_mkt_s = sums["sum_ll_mkt"].to_numpy()[idx].sum(axis=1)
    brier_blend_s = sums["sum_brier_blend"].to_numpy()[idx].sum(axis=1)
    brier_mkt_s = sums["sum_brier_mkt"].to_numpy()[idx].sum(axis=1)
    counts_s = sums["n_rows"].to_numpy()[idx].sum(axis=1)

    delta_ll = (ll_blend_s / counts_s) - (ll_mkt_s / counts_s)
    delta_brier = (brier_blend_s / counts_s) - (brier_mkt_s / counts_s)

    ll_low, ll_med, ll_high = np.quantile(delta_ll, [0.025, 0.5, 0.975]).tolist()
    b_low, b_med, b_high = np.quantile(delta_brier, [0.025, 0.5, 0.975]).tolist()
    return {
        "n_groups": int(len(sums)),
        "delta_logloss_ci_low": float(ll_low),
        "delta_logloss_ci_med": float(ll_med),
        "delta_logloss_ci_high": float(ll_high),
        "delta_brier_ci_low": float(b_low),
        "delta_brier_ci_med": float(b_med),
        "delta_brier_ci_high": float(b_high),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TaskM2 market-anchored logit blend (full-field)")
    ap.add_argument("--run_dir", action="append", default=None, help="run_dir (repeatable)")
    ap.add_argument("--run_dir_glob", action="append", default=None, help="glob for run_dir (repeatable)")
    ap.add_argument("--market_method", default="p_mkt_col")
    ap.add_argument("--model_prob_col", default="p_model")
    ap.add_argument("--fullfield_root", default=None)
    ap.add_argument("--bootstrap_n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None, help="output directory (default: run_dir/analysis/...)")
    ap.add_argument("--out_ascii", default=None, help="alias for --out_dir (deprecated)")
    ap.add_argument("--out_jp", default=None, help="optional secondary output directory")
    args = ap.parse_args()

    run_dirs: list[Path] = []
    if args.run_dir:
        run_dirs.extend([Path(p) for p in args.run_dir])
    if args.run_dir_glob:
        run_dirs.extend(_resolve_run_dirs(args.run_dir_glob))
    if not run_dirs:
        raise SystemExit("No run_dir specified (use --run_dir or --run_dir_glob)")
    run_dirs = [require_existing_dir(p, "run_dir") for p in run_dirs]

    if args.out_dir and args.out_ascii:
        raise SystemExit("Use --out_dir or --out_ascii (alias), not both.")
    out_dir = (
        Path(args.out_dir or args.out_ascii)
        if (args.out_dir or args.out_ascii)
        else make_analysis_out_dir(run_dirs[0], "roi_taskM2_market_anchor_logit_blend")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ascii = out_dir
    out_jp = Path(args.out_jp) if args.out_jp else None
    if out_jp:
        out_jp.mkdir(parents=True, exist_ok=True)

    fullfield_root = Path(args.fullfield_root) if args.fullfield_root else None
    dedup_cols = ["window_id", "race_id", "horse_no", "asof_time"]

    manifest = {
        "run_dirs": [str(p) for p in run_dirs],
        "market_method": args.market_method,
        "dedup_key": dedup_cols,
        "bootstrap_n": int(args.bootstrap_n),
        "missing_fullfield": [],
        "dedup": {},
        "integrity": {},
        "meta": {},
    }

    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}
    split_meta: dict[str, dict] = {}

    for run_dir in run_dirs:
        files = _collect_fullfield_files(run_dir, fullfield_root)
        found_splits = {f.split for f in files}
        if not {"train", "valid", "test"}.issubset(found_splits):
            generated = _generate_fullfield(run_dir, out_ascii)
            if generated:
                # merge missing splits from generated
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

    if manifest["missing_fullfield"]:
        print(f"[warn] missing fullfield for {len(manifest['missing_fullfield'])} runs")

    # prepare per split dataset (dedup + integrity on primary model col)
    model_cols = [c.strip() for c in args.model_prob_col.split(",") if c.strip()]
    # optional p_hat
    if "p_hat" not in model_cols:
        if any(not frames[0].empty and "p_hat" in frames[0].columns for frames in split_frames.values() if frames):
            model_cols.append("p_hat")

    split_data: dict[str, pd.DataFrame] = {}
    primary_model = model_cols[0] if model_cols else "p_model"

    for split, frames in split_frames.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df, dedup_meta = _dedup(df, dedup_cols)
        manifest["dedup"][split] = dedup_meta
        df_clean, integ = _integrity_filter(df, primary_model)
        manifest["integrity"][split] = integ
        split_data[split] = df_clean

    manifest["meta"] = split_meta
    (out_ascii / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not all(s in split_data for s in ["train", "valid", "test"]):
        raise SystemExit("train/valid/test fullfield data not available after generation.")

    # fit w
    def _fit_on_split(df: pd.DataFrame, model_col: str) -> tuple[float, float]:
        sub = df[df[model_col].notna()].copy()
        y = sub["y_win"].to_numpy()
        logit_model = _logit(sub[model_col].to_numpy())
        logit_mkt = _logit(sub["p_mkt"].to_numpy())
        return _fit_w_grid(y, logit_model, logit_mkt)

    blend_params = {}
    edge_rows = []
    bootstrap_rows = []

    for model_col in model_cols:
        train_df = split_data["train"][split_data["train"][model_col].notna()]
        valid_df = split_data["valid"][split_data["valid"][model_col].notna()]
        test_df = split_data["test"][split_data["test"][model_col].notna()]

        tv_df = pd.concat([train_df, valid_df], ignore_index=True)
        w_tv, ll_tv = _fit_on_split(tv_df, model_col)
        w_tr, ll_tr = _fit_on_split(train_df, model_col)
        w_va, ll_va = _fit_on_split(valid_df, model_col)

        blend_params[model_col] = {
            "w_fit_train_valid": w_tv,
            "logloss_train_valid": ll_tv,
            "w_fit_train_only": w_tr,
            "logloss_train_only": ll_tr,
            "w_fit_valid_only": w_va,
            "logloss_valid_only": ll_va,
            "grid_step": 0.001,
        }

        for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            metrics = _metrics_for_split(df, w_tv, model_col)
            edge_rows.append(
                {
                    "model_col": model_col,
                    "split": split_name,
                    "w_used": w_tv,
                    **metrics,
                    "delta_logloss": metrics["logloss_blend"] - metrics["logloss_mkt"],
                    "delta_brier": metrics["brier_blend"] - metrics["brier_mkt"],
                    "delta_auc": metrics["auc_blend"] - metrics["auc_mkt"],
                    "calib_gap": metrics["avg_p_blend"] - metrics["win_rate"],
                    "abs_calib_gap": abs(metrics["avg_p_blend"] - metrics["win_rate"]),
                }
            )

        # bootstraps on test
        for unit in ["race", "day"]:
            ci = _bootstrap_ci(test_df, w_tv, model_col, unit, args.bootstrap_n, args.seed)
            bootstrap_rows.append(
                {
                    "model_col": model_col,
                    "split": "test",
                    "unit": unit,
                    "bootstrap_n": int(args.bootstrap_n),
                    **ci,
                }
            )

    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(out_ascii / "edge_metrics.csv", index=False, encoding="utf-8")

    ci_df = pd.DataFrame(bootstrap_rows)
    ci_df.to_csv(out_ascii / "bootstrap_ci.csv", index=False, encoding="utf-8")

    (out_ascii / "blend_params.json").write_text(json.dumps(blend_params, indent=2), encoding="utf-8")

    # decision for primary model col based on day CI
    def _ci_row(unit: str) -> Optional[dict]:
        sub = ci_df[(ci_df["model_col"] == primary_model) & (ci_df["unit"] == unit)]
        if sub.empty:
            return None
        return sub.iloc[0].to_dict()

    primary_test = edge_df[(edge_df["model_col"] == primary_model) & (edge_df["split"] == "test")]
    if primary_test.empty:
        raise SystemExit("primary model test metrics missing")
    primary_test_row = primary_test.iloc[0].to_dict()
    day_ci = _ci_row("day")
    race_ci = _ci_row("race")

    decision = "model_update_next"
    reason = "test_edge_negative"
    if day_ci and np.isfinite(day_ci.get("delta_logloss_ci_high", np.nan)):
        if day_ci["delta_logloss_ci_high"] < 0:
            decision = "edge_positive_then_taskM3"
            reason = "ci_high_below_zero"
        elif day_ci["delta_logloss_ci_med"] < 0:
            decision = "edge_positive_then_taskM3"
            reason = "ci_cross_0_promising"
        else:
            decision = "model_update_next"
            reason = "ci_all_positive_or_median_pos"

    report_lines = [
        "# TaskM2 market-anchored logit blend",
        "",
        f"- run_dirs: {len(run_dirs)}",
        f"- run_dir_glob: {args.run_dir_glob}",
        f"- market_method: {args.market_method}",
        f"- model_cols: {', '.join(model_cols)}",
        f"- bootstrap_n: {args.bootstrap_n}",
        "",
        "## Dedup summary",
    ]
    for split, meta in manifest["dedup"].items():
        report_lines.append(f"- {split}: {meta}")
    report_lines.append("")
    report_lines.append("## Integrity summary (primary model)")
    for split, meta in manifest["integrity"].items():
        report_lines.append(f"- {split}: {meta}")
    report_lines.extend(
        [
            "",
            "## Blend params",
            json.dumps(blend_params, indent=2),
            "",
            "## Edge metrics (by split)",
            edge_df.to_string(index=False) if not edge_df.empty else "No edge rows.",
            "",
            "## Bootstrap CI (test)",
            ci_df.to_string(index=False) if not ci_df.empty else "No CI rows.",
            "",
            "## Decision",
            f"- decision={decision} ({reason})",
            "",
            "ROI definition: ROI = profit / stake (profit = return - stake).",
        ]
    )
    (out_ascii / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    today = datetime.now().strftime("%Y-%m-%d")
    minimal_lines = [
        f"# minimal update for chat - TaskM2 market anchor blend ({today})",
        "",
        "## Summary",
        f"- market_method=p_mkt_col | model_col={primary_model}",
        f"- w_fit_train_valid={blend_params[primary_model]['w_fit_train_valid']:.3f} "
        f"(train_only={blend_params[primary_model]['w_fit_train_only']:.3f}, valid_only={blend_params[primary_model]['w_fit_valid_only']:.3f})",
        f"- test delta_logloss={primary_test_row['delta_logloss']:.6f} | "
        f"delta_auc={primary_test_row['delta_auc']:.6f}",
    ]
    if day_ci:
        minimal_lines.append(
            f"- day CI delta_logloss [{day_ci['delta_logloss_ci_low']:.6f}, "
            f"{day_ci['delta_logloss_ci_med']:.6f}, {day_ci['delta_logloss_ci_high']:.6f}]"
        )
    if race_ci:
        minimal_lines.append(
            f"- race CI delta_logloss [{race_ci['delta_logloss_ci_low']:.6f}, "
            f"{race_ci['delta_logloss_ci_med']:.6f}, {race_ci['delta_logloss_ci_high']:.6f}]"
        )
    minimal_lines.extend(
        [
            f"- decision={decision} ({reason})",
            "",
            "## Paths",
            f"- out_dir: {out_dir}",
            f"- out_jp: {out_jp if out_jp else 'N/A'}",
        ]
    )
    (out_ascii / "minimal_update_for_chat.md").write_text("\n".join(minimal_lines) + "\n", encoding="utf-8")

    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        f"[diag] taskM2_market_anchor_blend_done=true | market_method=p_mkt_col | w={blend_params[primary_model]['w_fit_train_valid']:.3f}",
    ]
    if day_ci:
        stdout_lines.append(
            f"[diag] test_delta_logloss={primary_test_row['delta_logloss']:.6f} | "
            f"test_ci_low={day_ci['delta_logloss_ci_low']:.6f} | "
            f"test_ci_high={day_ci['delta_logloss_ci_high']:.6f} | resample=race/day"
        )
    else:
        stdout_lines.append(
            f"[diag] test_delta_logloss={primary_test_row['delta_logloss']:.6f} | "
            "test_ci_low=nan | test_ci_high=nan | resample=race/day"
        )
    stdout_lines.extend(
        [
            f"[plan] decision={decision} | reason={reason}",
            f"[paths] out_dir={out_jp if out_jp else out_dir} | staging={out_dir} | zip=<NOT_USED>",
        ]
    )
    (out_ascii / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
