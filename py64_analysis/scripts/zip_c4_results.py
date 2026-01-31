"""
Zip rolling holdout results for a C4 variant.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        import sys

        sys.path.insert(0, str(src))


_ensure_import_path()
from keiba.utils.run_paths import make_analysis_out_dir, require_existing_dir  # noqa: E402


def _rotate_existing_zips(out_dir: Path) -> None:
    existing = list(out_dir.glob("*.zip"))
    if not existing:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    old_dir = out_dir / "old" / ts
    old_dir.mkdir(parents=True, exist_ok=True)
    for zip_file in existing:
        shutil.move(str(zip_file), str(old_dir / zip_file.name))
    print(f"Moved {len(existing)} existing ZIP files to {old_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Zip C4 rolling holdout results.")
    ap.add_argument("--run-dir", required=True, type=Path, help="C4 run_dir")
    ap.add_argument("--baseline-dir", required=True, type=Path, help="baseline run_dir")
    ap.add_argument("--out-dir", default=None, type=Path, help="output directory (default: run_dir/analysis/...)")
    ap.add_argument("--zip-name", default=None, help="ZIP filename (default: C4_rolling_results_<ts>.zip)")
    args = ap.parse_args()

    run_dir = require_existing_dir(args.run_dir, "run_dir")
    baseline_dir = require_existing_dir(args.baseline_dir, "baseline_dir")
    out_dir = (
        args.out_dir
        if args.out_dir
        else make_analysis_out_dir(run_dir, "zip_c4_results")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _rotate_existing_zips(out_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = args.zip_name or f"C4_rolling_results_{ts}.zip"
    zip_path = out_dir / zip_name

    files_to_include = [
        ("paired_summary.json", run_dir / "paired_summary.json"),
        ("paired_compare.csv", run_dir / "paired_compare.csv"),
        ("ev_lift_summary.json", run_dir / "ev_lift_summary.json"),
        ("ev_lift.csv", run_dir / "ev_lift.csv"),
        ("summary.csv", run_dir / "summary.csv"),
        ("drivers.csv", run_dir / "drivers.csv"),
        ("baseline_ev_lift_summary.json", baseline_dir / "ev_lift_summary.json"),
        ("baseline_summary.csv", baseline_dir / "summary.csv"),
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, filepath in files_to_include:
            if filepath.exists():
                zf.write(filepath, arcname)
                print(f"Added: {arcname}")
            else:
                print(f"Warning: {filepath} not found, skipping")

    print(f"\nZIP created: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
