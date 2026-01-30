"""
Export full-field (pre-selection) predictions for test races in a holdout run.
Uses BacktestEngine._predict_race to score all horses at buy-time.
"""
from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from keiba.backtest.engine import BacktestEngine
from keiba.config import get_config, reset_config
from keiba.db.loader import get_session
from keiba.modeling.train import WinProbabilityModel

WIN_RE = re.compile(r"^w\d{3}_(\d{8})_(\d{8})$")


def _resolve_run_dir(path: Path) -> Path:
    if path.exists():
        return path
    base = Path(r"C:\Users\yyosh\keiba\data\holdout_runs")
    candidate = base / path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"run_dir not found: {path}")


def _window_dirs(run_dir: Path) -> list[Path]:
    windows = [p for p in run_dir.iterdir() if p.is_dir() and WIN_RE.match(p.name)]
    return sorted(windows, key=lambda p: p.name)


def _read_race_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        val = line.strip()
        if val:
            lines.append(val)
    return lines


def _find_model_path(window_dir: Path) -> Path:
    candidates = [
        window_dir / "artifacts" / "model.pkl",
        window_dir / "model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(f"model.pkl not found under {window_dir}")


def _chunked(seq: list[str], size: int = 500) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _fetch_finish_map(session, race_ids: list[str]) -> dict[tuple[str, int], Optional[int]]:
    mapping: dict[tuple[str, int], Optional[int]] = {}
    if not race_ids:
        return mapping
    for chunk in _chunked(race_ids):
        rows = session.execute(
            text(
                """
                SELECT race_id, horse_no, finish_pos
                FROM fact_result
                WHERE race_id = ANY(:race_ids)
                """
            ),
            {"race_ids": chunk},
        ).fetchall()
        for r in rows:
            race_id = str(r[0])
            horse_no = int(r[1])
            finish_pos = r[2]
            mapping[(race_id, horse_no)] = int(finish_pos) if finish_pos is not None else None
    return mapping


def _set_config(config_path: Optional[Path]) -> None:
    if config_path and config_path.exists():
        os.environ["KEIBA_CONFIG_PATH"] = str(config_path)
    else:
        os.environ.pop("KEIBA_CONFIG_PATH", None)
    reset_config()
    _ = get_config()


def _to_float(val) -> float:
    try:
        out = float(val)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float("nan")


def _build_rows(
    engine: BacktestEngine,
    model: WinProbabilityModel,
    race_ids: list[str],
    finish_map: dict[tuple[str, int], Optional[int]],
    window_id: str,
) -> list[dict]:
    rows: list[dict] = []
    for race_id in race_ids:
        buy_time = engine._get_buy_time(race_id)
        if buy_time is None:
            continue
        try:
            preds = engine._predict_race(race_id, buy_time, model)
        except Exception as exc:
            print(f"[warn] predict failed race_id={race_id}: {exc}")
            continue
        for pred in preds:
            horse_no = pred.get("horse_no")
            if horse_no is None:
                continue
            horse_no_int = int(horse_no)
            finish_pos = finish_map.get((race_id, horse_no_int))
            y_win = 1 if finish_pos == 1 else 0 if finish_pos is not None else None
            odds_val = pred.get("odds")
            p_mkt = pred.get("p_mkt")
            if p_mkt is None:
                p_mkt = pred.get("p_mkt_race") if pred.get("p_mkt_race") is not None else pred.get("p_mkt_raw")
            p_model = pred.get("p_model")
            if p_model is None:
                p_model = pred.get("p_hat")
            odds_f = _to_float(odds_val)
            p_model_f = _to_float(p_model)
            ev_raw = p_model_f * odds_f - 1.0 if np.isfinite(odds_f) and np.isfinite(p_model_f) else float("nan")
            rows.append(
                {
                    "window_id": window_id,
                    "race_id": race_id,
                    "horse_no": horse_no_int,
                    "horse_id": pred.get("horse_id"),
                    "asof_time": buy_time,
                    "odds": odds_f,
                    "p_mkt": _to_float(p_mkt),
                    "p_mkt_raw": _to_float(pred.get("p_mkt_raw")),
                    "p_mkt_race": _to_float(pred.get("p_mkt_race")),
                    "p_model": p_model_f,
                    "p_hat": _to_float(pred.get("p_hat")),
                    "ev_raw": ev_raw,
                    "finish_pos": finish_pos,
                    "y_win": y_win,
                    "has_ts_odds": pred.get("has_ts_odds"),
                    "snap_age_min": pred.get("snap_age_min"),
                    "segblend_segment": pred.get("segblend_segment"),
                    "segblend_w_used": pred.get("segblend_w_used"),
                    "segblend_w_global": pred.get("segblend_w_global"),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Export full-field predictions for holdout run")
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to export: test/train/valid/all (default: test)",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated splits, e.g. train,valid,test (overrides --split)",
    )
    ap.add_argument(
        "--overwrite",
        type=str,
        default="false",
        help="Overwrite existing outputs (true/false)",
    )
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fullfield_dir = out_dir / "fullfield"
    fullfield_dir.mkdir(parents=True, exist_ok=True)

    windows = _window_dirs(run_dir)
    if not windows:
        raise SystemExit(f"No windows found under run_dir: {run_dir}")

    session = get_session()
    all_rows: list[dict] = []

    overwrite = str(args.overwrite).strip().lower() in ("1", "true", "yes", "y")
    if args.splits:
        split_arg = args.splits.lower()
    else:
        split_arg = (args.split or "test").lower()
    if split_arg == "all":
        splits = ["train", "valid", "test"]
    else:
        splits = [s.strip() for s in split_arg.split(",") if s.strip()]

    for window_dir in windows:
        window_id = window_dir.name
        config_path = window_dir / "config_used.yaml"
        _set_config(config_path if config_path.exists() else None)

        model_path = _find_model_path(window_dir)
        model = WinProbabilityModel.load(model_path)
        engine = BacktestEngine(session)

        for split in splits:
            per_window_path = fullfield_dir / f"{window_id}_fullfield_{split}.csv"
            if per_window_path.exists() and not overwrite:
                # reuse existing file to avoid recompute
                try:
                    existing = pd.read_csv(per_window_path)
                    if "split" not in existing.columns:
                        existing["split"] = split
                    all_rows.extend(existing.to_dict("records"))
                    print(f"[info] {window_id} ({split}): reuse {per_window_path}")
                    continue
                except Exception:
                    pass

            race_ids = _read_race_ids(window_dir / f"race_ids_{split}.txt")
            if not race_ids:
                print(f"[warn] race_ids_{split}.txt missing or empty: {window_dir}")
                continue

            finish_map = _fetch_finish_map(session, race_ids)
            rows = _build_rows(engine, model, race_ids, finish_map, window_id)
            if not rows:
                print(f"[warn] no predictions generated for {window_id} ({split})")
                continue
            for row in rows:
                row["split"] = split

            df = pd.DataFrame(rows)
            df["split"] = split
            df = df.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
            df.to_csv(per_window_path, index=False, encoding="utf-8")
            all_rows.extend(rows)
            print(f"[info] {window_id} ({split}): rows={len(df)} -> {per_window_path}")

    if not all_rows:
        raise SystemExit("No full-field rows produced; check DB/features availability.")

    df_all = pd.DataFrame(all_rows)
    if df_all.empty:
        # allow when reuse-only and outputs already exist
        print("[warn] no new rows produced; ensure existing outputs are present")
        return

    df_all = df_all.sort_values(["window_id", "race_id", "horse_no"]).reset_index(drop=True)
    for split in splits:
        df_split = df_all[df_all.get("split") == split].copy()
        if df_split.empty:
            continue
        combined_path = fullfield_dir / f"fullfield_{split}.csv"
        if combined_path.exists() and not overwrite:
            print(f"[info] combined exists (skip): {combined_path}")
        else:
            df_split.to_csv(combined_path, index=False, encoding="utf-8")
        print(f"[info] combined rows={len(df_split)} -> {combined_path}")
        if split == "test":
            alias_path = fullfield_dir / "fullfield_preds.csv"
            if alias_path.exists() and not overwrite:
                print(f"[info] alias exists (skip): {alias_path}")
            else:
                df_split.to_csv(alias_path, index=False, encoding="utf-8")
                print(f"[info] alias test -> {alias_path}")


if __name__ == "__main__":
    main()
