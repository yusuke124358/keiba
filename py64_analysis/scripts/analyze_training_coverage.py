"""
Analyze training coverage for rolling holdout windows (cold start).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


WIN_RE = re.compile(r"^w(\d{3})_(\d{8})_(\d{8})")


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _infer_w_idx_offset(run_dir_name: str) -> int:
    name = run_dir_name.lower()
    if "w013_022" in name or "w013-022" in name:
        return 12
    return 0


def _parse_window_name(name: str) -> Optional[tuple[int, date, date]]:
    m = WIN_RE.match(name)
    if not m:
        return None
    w_idx = int(m.group(1))
    test_start = datetime.strptime(m.group(2), "%Y%m%d").date()
    test_end = datetime.strptime(m.group(3), "%Y%m%d").date()
    return w_idx, test_start, test_end


def _load_yaml(path: Path) -> Optional[dict]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_train_params(cfg: Optional[dict]) -> tuple[int, int, int, bool]:
    estimated = False
    valid_window_days = None
    gap_days = None
    train_lookback_days = None

    if cfg:
        valid_window_days = cfg.get("valid_window_days")
        gap_days = cfg.get("gap_days")
        train_lookback_days = cfg.get("train_lookback_days")

    if valid_window_days is None:
        valid_window_days = 14
        estimated = True
    if gap_days is None:
        gap_days = 0
        estimated = True
    if train_lookback_days is None:
        train_lookback_days = 365
        estimated = True

    return int(valid_window_days), int(gap_days), int(train_lookback_days), estimated


def _get_db_min_date() -> Optional[date]:
    _ensure_import_path()
    from keiba.db.loader import get_session
    from sqlalchemy import text

    session = get_session()
    for col in ["date", "race_date"]:
        try:
            row = session.execute(text(f"SELECT MIN(r.{col}) FROM fact_race r")).fetchone()
            if row and row[0]:
                value = row[0]
                if isinstance(value, datetime):
                    return value.date()
                if isinstance(value, date):
                    return value
        except Exception:
            continue
    return None


def _load_manifest(manifest_dir: Path, year: str) -> list[Path]:
    path = manifest_dir / f"manifest_{year}.json"
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    variants = manifest.get("variants", {})
    run_dirs = variants.get("b0")
    if not run_dirs:
        raise SystemExit("baseline b0 not found in manifest variants")
    root = Path.cwd()
    resolved = []
    for p in run_dirs:
        p = Path(p)
        if not p.is_absolute():
            p = root / p
        resolved.append(p)
    return resolved


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze training coverage (cold start)")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_min_date = _get_db_min_date()
    if db_min_date is None:
        raise SystemExit("failed to fetch db min date")

    rows = []
    estimated_count = {"2024": 0, "2025": 0}

    for year in ["2024", "2025"]:
        run_dirs = _load_manifest(manifest_dir, year)
        for run_dir in run_dirs:
            if not run_dir.exists():
                print(f"[warn] run_dir not found: {run_dir}", file=sys.stderr)
                continue
            offset = _infer_w_idx_offset(run_dir.name)
            for window_dir in sorted(run_dir.glob("w*")):
                if not window_dir.is_dir():
                    continue
                parsed = _parse_window_name(window_dir.name)
                if not parsed:
                    continue
                w_idx, test_start, _ = parsed
                w_idx = int(w_idx) + int(offset)

                cfg = None
                cfg_path = window_dir / "config_used.yaml"
                if cfg_path.exists():
                    cfg = _load_yaml(cfg_path)
                if cfg is None:
                    cfg_path = run_dir / "config_used.yaml"
                    if cfg_path.exists():
                        cfg = _load_yaml(cfg_path)

                valid_window_days, gap_days, train_lookback_days, estimated = _get_train_params(cfg)
                if estimated:
                    estimated_count[year] += 1

                train_end = test_start - timedelta(days=valid_window_days + gap_days)
                train_start_requested = train_end - timedelta(days=train_lookback_days)
                effective_train_start = max(train_start_requested, db_min_date)
                effective_train_days = (train_end - effective_train_start).days
                if effective_train_days < 0:
                    effective_train_days = 0
                cold_start_ratio = (
                    float(effective_train_days) / float(train_lookback_days)
                    if train_lookback_days > 0
                    else float("nan")
                )

                rows.append(
                    {
                        "year": int(year),
                        "window_idx": int(w_idx),
                        "split": "design" if int(w_idx) <= 12 else "eval",
                        "test_start": test_start.isoformat(),
                        "train_end": train_end.isoformat(),
                        "train_start_requested": train_start_requested.isoformat(),
                        "effective_train_start": effective_train_start.isoformat(),
                        "effective_train_days": int(effective_train_days),
                        "cold_start_ratio": cold_start_ratio,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no windows found")

    df.to_csv(out_dir / "training_coverage_windows.csv", index=False, encoding="utf-8")

    summary = (
        df.groupby(["year", "split"], dropna=False)
        .agg(
            n_windows=("window_idx", "count"),
            median_cold_start_ratio=("cold_start_ratio", "median"),
            p10_cold_start_ratio=("cold_start_ratio", lambda s: s.quantile(0.1)),
            p90_cold_start_ratio=("cold_start_ratio", lambda s: s.quantile(0.9)),
            median_effective_train_days=("effective_train_days", "median"),
            min_effective_train_days=("effective_train_days", "min"),
        )
        .reset_index()
    )

    eval_2024 = df[(df["year"] == 2024) & (df["split"] == "eval")]
    eval_2025 = df[(df["year"] == 2025) & (df["split"] == "eval")]
    med_2024 = float(eval_2024["cold_start_ratio"].median()) if not eval_2024.empty else float("nan")
    med_2025 = float(eval_2025["cold_start_ratio"].median()) if not eval_2025.empty else float("nan")

    conclusion_lines = []
    if pd.notna(med_2024) and pd.notna(med_2025) and med_2024 < 0.9 and med_2025 >= 0.99:
        conclusion_lines.append(
            "Conclusion: 2024 eval looks cold-started (median cold_start_ratio < 0.9) "
            "while 2025 eval is near full lookback. Cross-year comparison may be invalid."
        )
        conclusion_lines.append(
            "Action: consider backfilling 2023 data or shortening lookback to align coverage."
        )
    else:
        conclusion_lines.append(
            "Conclusion: cold-start signal is not clearly dominant from coverage alone."
        )

    report_lines = []
    report_lines.append("# Training coverage report")
    report_lines.append("")
    report_lines.append("## Conclusion")
    report_lines.extend([f"- {line}" for line in conclusion_lines])
    report_lines.append("")
    report_lines.append("## Coverage summary (all splits)")
    report_lines.append(_df_to_md(summary))
    report_lines.append("")
    report_lines.append("## Eval-only summary")
    eval_summary = summary[summary["split"] == "eval"].copy()
    report_lines.append(_df_to_md(eval_summary))
    report_lines.append("")
    report_lines.append("## Notes")
    report_lines.append(f"- db_min_date={db_min_date.isoformat()}")
    report_lines.append(f"- estimated_windows_2024={estimated_count['2024']}")
    report_lines.append(f"- estimated_windows_2025={estimated_count['2025']}")
    report_lines.append(
        "- train_end is computed as test_start - (valid_window_days + gap_days)."
    )

    (out_dir / "training_coverage_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
