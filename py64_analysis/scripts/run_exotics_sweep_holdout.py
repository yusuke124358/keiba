"""
戦略カタログ（YAML）を一括で回す（固定期間）。

出力:
  data/holdout_runs_exotics_sweep/<name>_<ts>/
    - config_used.yaml
    - <strategy_name>/
        - summary.json
        - report_close.md
        - report_buy.md
    - summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def main() -> None:
    _ensure_import_path()
    from keiba.db.loader import get_session
    from keiba.config import get_config
    from keiba.exotics.backtest import ExoticsBacktestEngine
    from keiba.exotics.strategy import parse_strategies_yaml

    p = argparse.ArgumentParser(description="Run exotics strategy sweep (holdout)")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--strategies", type=Path, default=Path("config/exotics_strategies.yaml"))
    p.add_argument("--name", default="exotics_sweep")
    p.add_argument("--out-root", type=Path, default=Path("data/holdout_runs_exotics_sweep"))
    p.add_argument("--max-staleness-min", type=float, default=None, help="全戦略共通で上書きしたい場合")
    args = p.parse_args()

    cfg = get_config()
    project_root = Path(__file__).resolve().parents[2]
    out_root = args.out_root if args.out_root.is_absolute() else project_root / args.out_root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_dir = out_root / f"{args.name}_{ts}"
    group_dir.mkdir(parents=True, exist_ok=True)

    # config保存
    (group_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg.model_dump(), allow_unicode=True, sort_keys=False), encoding="utf-8")

    # strategies
    d = yaml.safe_load(Path(args.strategies).read_text(encoding="utf-8"))
    strategies = parse_strategies_yaml(d)
    if not strategies:
        raise SystemExit("No strategies found in YAML")

    sess = get_session()
    eng = ExoticsBacktestEngine(sess)

    summary_rows: list[dict] = []
    for st in strategies:
        run_dir = group_dir / st.name
        run_dir.mkdir(parents=True, exist_ok=True)

        staleness = float(args.max_staleness_min) if args.max_staleness_min is not None else float(st.max_staleness_min)

        close = eng.run(
            start_date=args.start_date,
            end_date=args.end_date,
            ticket_type=st.ticket_type,
            odds_mode="close",
            max_bets_per_race=int(st.max_bets_per_race),
            ev_margin=float(st.ev_margin),
            max_staleness_min=staleness,
            missing_policy="skip",
        )
        buy = eng.run(
            start_date=args.start_date,
            end_date=args.end_date,
            ticket_type=st.ticket_type,
            odds_mode="buy",
            max_bets_per_race=int(st.max_bets_per_race),
            ev_margin=float(st.ev_margin),
            max_staleness_min=staleness,
            missing_policy="skip",
        )

        # report（簡易）
        (run_dir / "report_close.md").write_text(f"ROI={close.roi:.6f}\nmax_dd={close.max_drawdown:.6f}\n", encoding="utf-8")
        (run_dir / "report_buy.md").write_text(f"ROI={buy.roi:.6f}\nmax_dd={buy.max_drawdown:.6f}\n", encoding="utf-8")

        summary = {
            "strategy": st.__dict__,
            "period": {"start": args.start_date, "end": args.end_date},
            "close": close.__dict__,
            "buy": buy.__dict__,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        summary_rows.append(
            {
                "strategy": st.name,
                "ticket_type": st.ticket_type,
                "max_bets_per_race": st.max_bets_per_race,
                "ev_margin": st.ev_margin,
                "max_staleness_min": staleness,
                "roi_close": close.roi,
                "roi_buy": buy.roi,
                "dd_close": close.max_drawdown,
                "dd_buy": buy.max_drawdown,
                "n_bets_close": close.n_bets,
                "n_bets_buy": buy.n_bets,
                "skipped_snapshot_close": close.n_races_skipped_snapshot,
                "skipped_snapshot_buy": buy.n_races_skipped_snapshot,
                "skipped_hr_close": close.n_races_skipped_hr,
                "skipped_hr_buy": buy.n_races_skipped_hr,
            }
        )

    # summary.csv
    out_csv = group_dir / "summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print("OK")
    print("group_dir:", group_dir)
    print("summary:", out_csv)


if __name__ == "__main__":
    main()


