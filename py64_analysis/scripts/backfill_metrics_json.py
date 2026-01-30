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

    p = argparse.ArgumentParser(description="Backfill metrics.json for holdout_runs")
    p.add_argument("--root", type=Path, default=Path("data/holdout_runs"), help="holdout_runs root")
    args = p.parse_args()

    root = args.root
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    if not root.exists():
        raise SystemExit(f"holdout_runs not found: {root}")

    errors = []
    created = 0

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        metrics_path = child / "metrics.json"
        if metrics_path.exists():
            continue

        if (child / "summary.csv").exists():
            try:
                write_metrics_json(child, run_kind="rolling_holdout")
                created += 1
            except Exception as e:
                errors.append((str(child), f"rolling:{e}"))
            continue

        if (child / "summary.json").exists():
            try:
                write_metrics_json(child, run_kind="holdout")
                created += 1
            except Exception as e:
                errors.append((str(child), f"holdout:{e}"))

    print(f"created: {created}")
    if errors:
        print("errors:")
        for path, err in errors:
            print(f"- {path}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
