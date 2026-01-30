"""
Stage B0（オッズ不要）: 三連複/三連単の確率品質を大量の過去レースで評価する。

出力:
  - b0_details.csv（1RごとのNLL/hit@K）
  - b0_summary.json
  - b0_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]  # keiba/
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_import_path()
    from keiba.db.loader import get_session
    from keiba.config import get_config
    from keiba.exotics.b0_eval import run_b0_eval

    p = argparse.ArgumentParser(description="Run exotics Stage B0 evaluation")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/holdout_runs_exotics_b0"))
    p.add_argument("--buy-t-minus-minutes", type=int, default=None)
    args = p.parse_args()

    cfg = get_config()
    buy_min = int(args.buy_t_minus_minutes) if args.buy_t_minus_minutes is not None else int(cfg.backtest.buy_t_minus_minutes)

    project_root = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"b0_{args.start_date}_{args.end_date}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sess = get_session()
    summary, df = run_b0_eval(sess, start_date=args.start_date, end_date=args.end_date, buy_t_minus_minutes=buy_min)

    details_csv = run_dir / "b0_details.csv"
    df.to_csv(details_csv, index=False, encoding="utf-8")

    summary_json = run_dir / "b0_summary.json"
    summary_json.write_text(json.dumps(summary.__dict__, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report_md = run_dir / "b0_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Exotics Stage B0 report",
                "",
                f"- period: {args.start_date}..{args.end_date}",
                f"- buy_t_minus_minutes: {buy_min}",
                "",
                "## Summary",
                "",
                f"- n_races: {summary.n_races}",
                f"- n_skipped: {summary.n_skipped}",
                f"- mean_nll_trifecta: {summary.mean_nll_trifecta:.6f}",
                f"- mean_nll_trio: {summary.mean_nll_trio:.6f}",
                f"- hit@1 trifecta: {summary.hit1_trifecta:.3f}",
                f"- hit@10 trifecta: {summary.hit10_trifecta:.3f}",
                f"- hit@1 trio: {summary.hit1_trio:.3f}",
                f"- hit@10 trio: {summary.hit10_trio:.3f}",
                "",
                "## Outputs",
                "",
                f"- details: {details_csv.name}",
                f"- summary: {summary_json.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("OK")
    print("run_dir:", run_dir)
    print("details:", details_csv)
    print("summary:", summary_json)
    print("report:", report_md)


if __name__ == "__main__":
    main()


