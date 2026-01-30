"""
Analyze race-level probability mass (sum of p_used) for eval windows.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


WIN_RE = re.compile(r"^w(\d{3})_(\d{8})_(\d{8})$")


def _infer_w_idx_offset(run_dir_name: str) -> int:
    name = run_dir_name.lower()
    if "w013_022" in name or "w013-022" in name:
        return 12
    return 0


def _parse_window_name(name: str) -> Optional[tuple[int, str, str]]:
    m = WIN_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def _load_manifest(manifest_dir: Path, year: str, variant: str) -> list[Path]:
    manifest_path = manifest_dir / f"manifest_{year}.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = data.get("variants", {})
    if variant not in variants:
        raise SystemExit(f"variant not found in {manifest_path}: {variant}")
    return [Path(p) for p in variants[variant]]


def _ensure_import_path() -> None:
    import sys
    from pathlib import Path as P

    project_root = P(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X2 = X.reindex(columns=feature_names, fill_value=0.0)
    return X2.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def analyze_year(manifest_dir: Path, variant: str, year: str, out_dir: Path) -> None:
    _ensure_import_path()
    from keiba.db.loader import get_session
    from keiba.modeling.train import prepare_training_data, WinProbabilityModel

    run_dirs = _load_manifest(manifest_dir, year, variant)
    session = get_session()

    sum_rows = []
    roi_rows = []
    for run_dir in run_dirs:
        offset = _infer_w_idx_offset(run_dir.name)
        for window_dir in sorted(run_dir.glob("w*/")):
            parsed = _parse_window_name(window_dir.name)
            if not parsed:
                continue
            w_idx, test_start, test_end = parsed
            w_idx = int(w_idx) + int(offset)
            if w_idx < 13:
                continue

            model_path = window_dir / "artifacts" / "model.pkl"
            if not model_path.exists():
                continue
            model = WinProbabilityModel.load(model_path)

            X, y, p_mkt, race_ids = prepare_training_data(
                session, f"{test_start[:4]}-{test_start[4:6]}-{test_start[6:8]}", f"{test_end[:4]}-{test_end[4:6]}-{test_end[6:8]}"
            )
            if X is None or len(X) == 0:
                continue

            X2 = _align_features(X, model.feature_names)
            p_used = model.predict(X2, p_mkt, calibrate=True)

            df = pd.DataFrame(
                {
                    "race_id": race_ids.values,
                    "p_used": np.asarray(p_used, dtype=float),
                }
            )
            sum_p = df.groupby("race_id", dropna=False)["p_used"].sum()

            sum_rows.extend(sum_p.values.tolist())

            bets_path = window_dir / "bets.csv"
            if bets_path.exists():
                try:
                    bets = pd.read_csv(bets_path)
                except Exception:
                    bets = None
                if bets is not None and not bets.empty:
                    bets["stake"] = pd.to_numeric(bets.get("stake"), errors="coerce")
                    bets["profit"] = pd.to_numeric(bets.get("profit"), errors="coerce")
                    bets = bets.dropna(subset=["race_id", "stake", "profit"])
                    bets["sum_p"] = bets["race_id"].map(sum_p)
                    bins = [-np.inf, 0.9, 1.1, np.inf]
                    labels = ["<0.9", "0.9-1.1", ">1.1"]
                    bets["sum_p_bin"] = pd.cut(bets["sum_p"], bins=bins, labels=labels)
                    agg = bets.groupby("sum_p_bin", dropna=False).agg(
                        stake=("stake", "sum"),
                        profit=("profit", "sum"),
                        n_bets=("stake", "count"),
                    ).reset_index()
                    agg["roi"] = agg.apply(lambda r: (r["profit"] / r["stake"]) if r["stake"] else float("nan"), axis=1)
                    agg["window"] = window_dir.name
                    roi_rows.append(agg)

    if sum_rows:
        arr = np.asarray(sum_rows, dtype=float)
        dist = {
            "year": year,
            "split": "eval",
            "n_races": int(len(arr)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
            "frac_gt_1_2": float(np.mean(arr > 1.2)),
            "frac_lt_0_8": float(np.mean(arr < 0.8)),
        }
        pd.DataFrame([dist]).to_csv(out_dir / f"prob_mass_dist_{year}_eval.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame([{"year": year, "split": "eval", "n_races": 0}]).to_csv(
            out_dir / f"prob_mass_dist_{year}_eval.csv", index=False, encoding="utf-8"
        )

    if roi_rows:
        roi_df = pd.concat(roi_rows, ignore_index=True)
        roi_df = roi_df.groupby("sum_p_bin", dropna=False).agg(
            stake=("stake", "sum"),
            profit=("profit", "sum"),
            n_bets=("n_bets", "sum"),
        ).reset_index()
        roi_df["roi"] = roi_df.apply(lambda r: (r["profit"] / r["stake"]) if r["stake"] else float("nan"), axis=1)
        roi_df.to_csv(out_dir / f"prob_mass_roi_bins_{year}_eval.csv", index=False, encoding="utf-8")


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze race-level probability mass for eval windows")
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    analyze_year(args.manifest_dir, args.variant, "2024", out_dir)
    analyze_year(args.manifest_dir, args.variant, "2025", out_dir)

    # simple report
    lines = []
    lines.append("# Race-level probability mass (eval)")
    for year in ["2024", "2025"]:
        dist_path = out_dir / f"prob_mass_dist_{year}_eval.csv"
        if not dist_path.exists():
            continue
        dist = pd.read_csv(dist_path)
        lines.append(f"\n## {year} eval")
        lines.append(_df_to_md(dist))
        roi_path = out_dir / f"prob_mass_roi_bins_{year}_eval.csv"
        if roi_path.exists():
            roi = pd.read_csv(roi_path)
            lines.append("")
            lines.append(_df_to_md(roi))
    (out_dir / "prob_mass_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
