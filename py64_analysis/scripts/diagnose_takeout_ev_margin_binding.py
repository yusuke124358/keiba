"""
Diagnose takeout_ev_margin binding (eval-only samples).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DEFAULT_BUNDLE = Path(r"C:\Users\yyosh\keiba\output for 5.2pro\単勝\takeout_ev_margin_sweep_2024_2025_20260118_014611")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_run_dir(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _find_first_summary(run_dir: Path) -> Optional[dict[str, Any]]:
    for w in sorted(run_dir.glob("w*/summary.json")):
        try:
            return _read_json(w)
        except Exception:
            continue
    return None


def _variant_meta(run_dir: Path) -> dict[str, Any]:
    summary = _find_first_summary(run_dir) or {}
    tem = summary.get("takeout_ev_margin") or {}
    return {
        "enabled": bool(tem.get("enabled", False)),
        "ref_takeout": tem.get("ref_takeout"),
        "slope": tem.get("slope"),
    }


def _sample_bets(run_dir: Path, max_windows: int) -> pd.DataFrame:
    frames = []
    windows = sorted(run_dir.glob("w*/bets.csv"))[:max_windows]
    for path in windows:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["_window"] = path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _calc_metrics(df: pd.DataFrame, ref_takeout_default: float) -> dict[str, Any]:
    n_bets = int(len(df))
    if n_bets == 0:
        return {
            "n_bets": 0,
            "takeout_nonnull_rate": None,
            "takeout_gt_ref_rate": None,
            "min_ev_eff_gt_base_rate": None,
            "failed_count": None,
            "failed_rate": None,
            "diff_p1": None,
            "diff_p5": None,
            "diff_p50": None,
            "diff_p95": None,
        }

    takeout = pd.to_numeric(df.get("takeout_implied"), errors="coerce")
    takeout_nonnull = takeout.dropna()
    takeout_nonnull_rate = float(len(takeout_nonnull)) / float(n_bets) if n_bets else None

    takeout_ref = pd.to_numeric(df.get("takeout_ref"), errors="coerce")
    ref_val = takeout_ref.dropna().iloc[0] if takeout_ref is not None and not takeout_ref.dropna().empty else ref_takeout_default

    takeout_gt_ref = (takeout_nonnull > ref_val).mean() if not takeout_nonnull.empty else None

    base_min_ev = pd.to_numeric(df.get("base_min_ev"), errors="coerce")
    min_ev_eff = pd.to_numeric(df.get("min_ev_eff"), errors="coerce")
    mask_min = base_min_ev.notna() & min_ev_eff.notna()
    min_ev_eff_gt_base = (min_ev_eff[mask_min] > base_min_ev[mask_min]).mean() if mask_min.any() else None

    passed_col = df.get("passed_takeout_ev_margin")
    failed_count = None
    failed_rate = None
    if passed_col is not None:
        passed = pd.to_numeric(passed_col, errors="coerce")
        failed_mask = passed == 0
        failed_count = int(failed_mask.sum())
        failed_rate = float(failed_count) / float(n_bets) if n_bets else None

    diff = pd.to_numeric(df.get("ev"), errors="coerce") - min_ev_eff
    diff = diff.dropna()
    if diff.empty:
        diff_p1 = diff_p5 = diff_p50 = diff_p95 = None
    else:
        diff_p1 = float(np.quantile(diff, 0.01))
        diff_p5 = float(np.quantile(diff, 0.05))
        diff_p50 = float(np.quantile(diff, 0.50))
        diff_p95 = float(np.quantile(diff, 0.95))

    return {
        "n_bets": n_bets,
        "takeout_nonnull_rate": takeout_nonnull_rate,
        "takeout_gt_ref_rate": float(takeout_gt_ref) if takeout_gt_ref is not None else None,
        "min_ev_eff_gt_base_rate": float(min_ev_eff_gt_base) if min_ev_eff_gt_base is not None else None,
        "failed_count": failed_count,
        "failed_rate": failed_rate,
        "diff_p1": diff_p1,
        "diff_p5": diff_p5,
        "diff_p50": diff_p50,
        "diff_p95": diff_p95,
    }


def _fmt(val: Any, digits: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose takeout_ev_margin binding")
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sample-windows", type=int, default=3)
    ap.add_argument("--target-slopes", type=float, nargs="*", default=[0.0, 2.0])
    args = ap.parse_args()

    bundle_dir = args.bundle_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in ["2024", "2025"]:
        manifest_path = bundle_dir / f"manifest_{year}.json"
        if not manifest_path.exists():
            raise SystemExit(f"manifest not found: {manifest_path}")
        manifest = _read_json(manifest_path)
        variants = manifest.get("variants", {})

        for variant, run_dirs in variants.items():
            if not run_dirs:
                continue
            run_dir = _resolve_run_dir(run_dirs[0])
            meta = _variant_meta(run_dir)
            slope = _safe_float(meta.get("slope"))
            enabled = bool(meta.get("enabled", False))

            if slope is None:
                continue
            if slope not in args.target_slopes:
                continue
            if not enabled and slope == 0.0:
                # Prefer enabled variants for slope=0.0; skip baseline if disabled.
                continue

            df = _sample_bets(run_dir, max_windows=int(args.sample_windows))
            metrics = _calc_metrics(df, ref_takeout_default=0.215)
            rows.append(
                {
                    "year": int(year),
                    "variant": variant,
                    "slope": slope,
                    "enabled": enabled,
                    "sample_windows": int(args.sample_windows),
                    **metrics,
                }
            )

    if not rows:
        raise SystemExit("No rows collected for diagnosis")

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["year", "slope", "variant"]).reset_index(drop=True)

    csv_path = out_dir / "takeout_ev_margin_binding_diagnosis.csv"
    md_path = out_dir / "takeout_ev_margin_binding_diagnosis.md"

    out_df.to_csv(csv_path, index=False)

    lines = ["# Takeout EV margin binding diagnosis", ""]
    for year in sorted(out_df["year"].unique()):
        lines.append(f"## {year}")
        lines.append("")
        sub = out_df[out_df["year"] == year]
        for _, row in sub.iterrows():
            lines.append(f"- slope={row['slope']} (variant={row['variant']})")
            lines.append(
                "  - n_bets={n_bets} | takeout_nonnull_rate={tnr} | takeout_gt_ref_rate={tgr} | min_ev_eff_gt_base_rate={mbr}".format(
                    n_bets=row.get("n_bets"),
                    tnr=_fmt(row.get("takeout_nonnull_rate")),
                    tgr=_fmt(row.get("takeout_gt_ref_rate")),
                    mbr=_fmt(row.get("min_ev_eff_gt_base_rate")),
                )
            )
            lines.append(
                "  - failed_rate={fr} ({fc}) | ev_minus_min_ev_eff_p1/p5/p50/p95={p1}/{p5}/{p50}/{p95}".format(
                    fr=_fmt(row.get("failed_rate")),
                    fc=row.get("failed_count"),
                    p1=_fmt(row.get("diff_p1")),
                    p5=_fmt(row.get("diff_p5")),
                    p50=_fmt(row.get("diff_p50")),
                    p95=_fmt(row.get("diff_p95")),
                )
            )
        lines.append("")

    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
