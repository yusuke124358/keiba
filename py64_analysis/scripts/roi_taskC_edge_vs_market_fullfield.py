"""
Task C: evaluate model edge vs market on full-field (pre-selection) data if available.
If full-field data is missing, emits a diagnostic report and stops short of re-running.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WIN_RE = re.compile(r"^w\d{3}_(\d{8})_(\d{8})$")

P_MODEL_CANDIDATES = ["p_model", "p_hat", "p_hat_capped", "prob", "p_win"]
P_MKT_CANDIDATES = ["p_mkt", "p_mkt_raw", "p_mkt_race"]
ODDS_CANDIDATES = ["odds_at_buy", "odds_effective", "odds_final", "odds"]
Y_CANDIDATES = ["is_win", "y_win"]
FINISH_CANDIDATES = ["finish_pos", "finish_position"]

FULLFIELD_NAME_RE = re.compile(r"(candidate|candidates|pred|preds|score|scores|horse|horses|full|all)", re.I)


@dataclass(frozen=True)
class FullFile:
    path: Path
    window_id: Optional[str]


def _resolve_run_dir(path: Path) -> Path:
    if path.exists():
        return path
    base = PROJECT_ROOT / "data" / "holdout_runs"
    candidate = base / path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"run_dir not found: {path}")


def _infer_window_id(path: Path) -> Optional[str]:
    for parent in path.parents:
        if WIN_RE.match(parent.name):
            return parent.name
    return None


def _find_fullfield_files(root: Path) -> list[FullFile]:
    files: list[FullFile] = []
    if root.is_file():
        return [FullFile(path=root, window_id=_infer_window_id(root))]
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower() == "bets.csv":
            continue
        if not FULLFIELD_NAME_RE.search(p.name):
            continue
        if p.suffix.lower() not in {".csv", ".parquet"}:
            continue
        files.append(FullFile(path=p, window_id=_infer_window_id(p)))
    return files


def _detect_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


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


def _auc_from_scores(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(score)) + 1
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _load_fullfile(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _prep_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cols = list(df.columns)
    p_model_col = _detect_column(cols, P_MODEL_CANDIDATES)
    p_mkt_col = _detect_column(cols, P_MKT_CANDIDATES)
    odds_col = _detect_column(cols, ODDS_CANDIDATES)
    y_col = _detect_column(cols, Y_CANDIDATES)
    finish_col = _detect_column(cols, FINISH_CANDIDATES)

    meta = {
        "p_model_col": p_model_col or "missing",
        "p_mkt_col": p_mkt_col or "missing",
        "odds_col": odds_col or "missing",
        "y_col": y_col or (finish_col or "missing"),
    }

    if odds_col is None:
        raise SystemExit("odds column not found in fullfield data")
    if p_model_col is None:
        raise SystemExit("p_model column not found in fullfield data")

    df = df.copy()
    df["p_model"] = pd.to_numeric(df[p_model_col], errors="coerce")
    df["odds_val"] = pd.to_numeric(df[odds_col], errors="coerce")
    df["p_mkt"] = (
        pd.to_numeric(df[p_mkt_col], errors="coerce") if p_mkt_col else (1.0 / df["odds_val"])
    )
    if y_col:
        df["y_win"] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)
    elif finish_col:
        df["y_win"] = (pd.to_numeric(df[finish_col], errors="coerce") == 1).astype(int)
    else:
        raise SystemExit("y_win not found (is_win/finish_pos missing)")

    df = df[(df["odds_val"] > 0) & df["p_model"].notna() & df["p_mkt"].notna()]
    df["odds_band"] = df["odds_val"].apply(_odds_band)
    return df, meta


def _aggregate_edge(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if not group_cols:
        sub = df
        y = sub["y_win"].to_numpy()
        p_model = sub["p_model"].to_numpy()
        p_mkt = sub["p_mkt"].to_numpy()
        odds = sub["odds_val"].to_numpy()
        rows.append(
            {
                "n_samples": len(sub),
                "logloss_model": _logloss(y, p_model),
                "logloss_mkt": _logloss(y, p_mkt),
                "brier_model": _brier(y, p_model),
                "brier_mkt": _brier(y, p_mkt),
                "auc_model": _auc_from_scores(y, p_model),
                "auc_mkt": _auc_from_scores(y, p_mkt),
                "mean_ev_model": float(np.mean(p_model * odds - 1.0)),
                "mean_ev_mkt": float(np.mean(p_mkt * odds - 1.0)),
            }
        )
        return pd.DataFrame(rows)

    for key, sub in df.groupby(group_cols, dropna=False):
        y = sub["y_win"].to_numpy()
        p_model = sub["p_model"].to_numpy()
        p_mkt = sub["p_mkt"].to_numpy()
        odds = sub["odds_val"].to_numpy()
        rows.append(
            {
                **({} if isinstance(key, tuple) else {}),
                "n_samples": len(sub),
                "logloss_model": _logloss(y, p_model),
                "logloss_mkt": _logloss(y, p_mkt),
                "brier_model": _brier(y, p_model),
                "brier_mkt": _brier(y, p_mkt),
                "auc_model": _auc_from_scores(y, p_model),
                "auc_mkt": _auc_from_scores(y, p_mkt),
                "mean_ev_model": float(np.mean(p_model * odds - 1.0)),
                "mean_ev_mkt": float(np.mean(p_mkt * odds - 1.0)),
            }
        )
    out = pd.DataFrame(rows)
    if group_cols:
        if isinstance(key, tuple):
            out.insert(0, "group", list(df.groupby(group_cols, dropna=False).groups.keys()))
        else:
            out.insert(0, group_cols[0], list(df.groupby(group_cols, dropna=False).groups.keys()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="TaskC edge vs market (full-field) diagnostic")
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--fullfield_source", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    fullfield_root = args.fullfield_source or run_dir
    fullfiles = _find_fullfield_files(fullfield_root)
    if not fullfiles:
        report = [
            "# TaskC edge vs market (full-field) diagnostic",
            "",
            f"- run_dir: {run_dir}",
            f"- fullfield_source: {fullfield_root} (missing)",
            "- decision: model_update_next (no full-field data)",
            "",
            "ROI definition: ROI = profit / stake (profit = return - stake).",
        ]
        (out_dir / "report_edge.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        minimal = [
            "# minimal update for chat - TaskC edge vs market (2026-01-28)",
            "",
            "## Summary",
            "- fullfield data not found under run_dir; cannot verify market edge.",
            "",
            "## Decision",
            "- model_update_next (need full-field preds to assess edge).",
            "",
            "## Paths",
            f"- out_dir: {out_dir}",
        ]
        (out_dir / "minimal_update_for_chat.md").write_text("\n".join(minimal) + "\n", encoding="utf-8")
        stdout_lines = [
            "[audit] agents_md_updated=false | agents_md_zip_included=false",
            "[tests] pytest_passed=false",
            f"[diag] edge_vs_market_fullfield_done=false | fullfield_source={fullfield_root}",
            "[plan] decision=model_update_next | reason=fullfield_missing",
            f"[paths] out_dir={out_dir} | staging={out_dir} | zip=<NOT_USED>",
        ]
        (out_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
        return

    dfs = []
    meta_info = {}
    for f in fullfiles:
        df = _load_fullfile(f.path)
        df, meta = _prep_df(df)
        df["window_id"] = f.window_id or "unknown"
        dfs.append(df)
        if not meta_info:
            meta_info = meta

    all_df = pd.concat(dfs, ignore_index=True)
    by_window = _aggregate_edge(all_df, ["window_id"])
    by_window.to_csv(tables_dir / "edge_by_window.csv", index=False, encoding="utf-8")

    by_band = _aggregate_edge(all_df, ["odds_band"])
    by_band.to_csv(tables_dir / "edge_by_odds_band.csv", index=False, encoding="utf-8")

    # aggregate decision
    agg = _aggregate_edge(all_df, [])
    agg_row = agg.iloc[0].to_dict() if not agg.empty else {}
    better_ll = agg_row.get("logloss_model", math.inf) < agg_row.get("logloss_mkt", math.inf)
    better_auc = agg_row.get("auc_model", -math.inf) > agg_row.get("auc_mkt", -math.inf)
    edge_ok = bool(better_ll and better_auc)

    report = [
        "# TaskC edge vs market (full-field) diagnostic",
        "",
        f"- run_dir: {run_dir}",
        f"- fullfield_source: {fullfield_root}",
        f"- files_used: {len(fullfiles)}",
        f"- detected columns: {meta_info}",
        "",
        "## Aggregate edge",
        f"- logloss_model={agg_row.get('logloss_model')} vs logloss_mkt={agg_row.get('logloss_mkt')}",
        f"- brier_model={agg_row.get('brier_model')} vs brier_mkt={agg_row.get('brier_mkt')}",
        f"- auc_model={agg_row.get('auc_model')} vs auc_mkt={agg_row.get('auc_mkt')}",
        f"- mean_ev_model={agg_row.get('mean_ev_model')} vs mean_ev_mkt={agg_row.get('mean_ev_mkt')}",
        "",
        "## Conclusion",
        f"- market_edge_detected={edge_ok}",
        "",
        "ROI definition: ROI = profit / stake (profit = return - stake).",
    ]
    (out_dir / "report_edge.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    minimal = [
        "# minimal update for chat - TaskC edge vs market (2026-01-28)",
        "",
        "## Summary",
        f"- market_edge_detected={edge_ok}",
        "",
        "## Paths",
        f"- out_dir: {out_dir}",
    ]
    (out_dir / "minimal_update_for_chat.md").write_text("\n".join(minimal) + "\n", encoding="utf-8")

    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        f"[diag] edge_vs_market_fullfield_done=true | fullfield_source={fullfield_root}",
        f"[plan] decision={'proceed_reselect_eval' if edge_ok else 'model_update_next'} | "
        f"reason={'edge_detected' if edge_ok else 'no_edge'}",
        f"[paths] out_dir={out_dir} | staging={out_dir} | zip=<NOT_USED>",
    ]
    (out_dir / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
