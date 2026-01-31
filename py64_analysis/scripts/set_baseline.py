from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]  # keiba/
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_import_path()
    from keiba.eval.extract_metrics import extract_metrics_from_holdout_run, extract_metrics_from_rolling_run

    p = argparse.ArgumentParser(description="Register baseline metrics for a scenario")
    p.add_argument("--scenario", required=True)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--note", default="")
    args = p.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        # infer type by summary
        if (run_dir / "summary.json").exists():
            metrics = extract_metrics_from_holdout_run(run_dir)
        elif (run_dir / "summary.csv").exists():
            metrics = extract_metrics_from_rolling_run(run_dir)
        else:
            raise SystemExit("run_dir has no summary.json or summary.csv")
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "baselines" / args.scenario
    base_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(metrics_path, base_dir / "metrics.json")
    baseline_meta = {
        "scenario": args.scenario,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": metrics.get("git_commit"),
        "config_hash_sha256": metrics.get("config_hash_sha256"),
        "data_cutoff": metrics.get("data_cutoff"),
        "prob_variant_used": metrics.get("prob_variant_used"),
        "run_dir": metrics.get("run_dir"),
        "note": args.note,
    }
    (base_dir / "baseline.json").write_text(
        json.dumps(baseline_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("baseline:", base_dir)


if __name__ == "__main__":
    main()
