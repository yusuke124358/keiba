from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]  # keiba/
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_import_path()
    from keiba.eval.extract_metrics import write_metrics_json

    p = argparse.ArgumentParser(description="Extract metrics.json from holdout or rolling run")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path, help="holdout run dir (summary.json in dir)")
    g.add_argument("--rolling-group-dir", type=Path, help="rolling group dir (summary.csv in dir)")
    args = p.parse_args()

    if args.run_dir:
        out = write_metrics_json(args.run_dir, run_kind="holdout")
    else:
        out = write_metrics_json(args.rolling_group_dir, run_kind="rolling_holdout")
    print("metrics:", out)


if __name__ == "__main__":
    main()
