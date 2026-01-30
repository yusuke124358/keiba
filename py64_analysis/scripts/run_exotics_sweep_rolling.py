"""
rolling window で exotics 戦略カタログを一括実行する。

注意:
  - 既存の単勝 rolling と分離する
  - ここでは学習は行わず、DBにあるオッズ/払戻を使って backtest だけを回す
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@dataclass(frozen=True)
class WindowPlan:
    idx: int
    test_start: date
    test_end: date


def _date_range_windows(
    test_range_start: date,
    test_range_end: date,
    test_window_days: int,
    step_days: int,
) -> list[WindowPlan]:
    plans: list[WindowPlan] = []
    cur = test_range_start
    idx = 1
    while True:
        end = cur + timedelta(days=test_window_days - 1)
        if end > test_range_end:
            break
        plans.append(WindowPlan(idx=idx, test_start=cur, test_end=end))
        cur = cur + timedelta(days=step_days)
        idx += 1
    return plans


def main() -> None:
    _ensure_import_path()
    from keiba.db.loader import get_session
    from keiba.exotics.backtest import ExoticsBacktestEngine
    from keiba.exotics.strategy import parse_strategies_yaml

    p = argparse.ArgumentParser(description="Run rolling exotics strategy sweep")
    p.add_argument("--name", default="exotics_rolling")
    p.add_argument("--test-range-start", required=True)
    p.add_argument("--test-range-end", required=True)
    p.add_argument("--test-window-days", type=int, default=60)
    p.add_argument("--step-days", type=int, default=14)
    p.add_argument("--strategies", type=Path, default=Path("config/exotics_strategies.yaml"))
    p.add_argument("--out-root", type=Path, default=Path("data/holdout_runs_exotics_rolling"))
    p.add_argument("--max-windows", type=int, default=None)
    args = p.parse_args()

    test_range_start = _parse_date(args.test_range_start)
    test_range_end = _parse_date(args.test_range_end)

    plans = _date_range_windows(test_range_start, test_range_end, args.test_window_days, args.step_days)
    if args.max_windows is not None:
        plans = plans[: max(0, int(args.max_windows))]
    if not plans:
        raise SystemExit("No windows")

    d = yaml.safe_load(Path(args.strategies).read_text(encoding="utf-8"))
    strategies = parse_strategies_yaml(d)
    if not strategies:
        raise SystemExit("No strategies")

    project_root = Path(__file__).resolve().parents[2]
    out_root = args.out_root if args.out_root.is_absolute() else project_root / args.out_root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_dir = out_root / f"{args.name}_{ts}"
    group_dir.mkdir(parents=True, exist_ok=True)

    sess = get_session()
    eng = ExoticsBacktestEngine(sess)

    summary_rows: list[dict] = []
    for plan in plans:
        window_name = f"w{plan.idx:03d}_{plan.test_start:%Y%m%d}_{plan.test_end:%Y%m%d}"
        wdir = group_dir / window_name
        wdir.mkdir(parents=True, exist_ok=True)

        for st in strategies:
            sdir = wdir / st.name
            sdir.mkdir(parents=True, exist_ok=True)
            close = eng.run(
                start_date=plan.test_start.isoformat(),
                end_date=plan.test_end.isoformat(),
                ticket_type=st.ticket_type,
                odds_mode="close",
                max_bets_per_race=int(st.max_bets_per_race),
                ev_margin=float(st.ev_margin),
                max_staleness_min=float(st.max_staleness_min),
                missing_policy="skip",
            )
            buy = eng.run(
                start_date=plan.test_start.isoformat(),
                end_date=plan.test_end.isoformat(),
                ticket_type=st.ticket_type,
                odds_mode="buy",
                max_bets_per_race=int(st.max_bets_per_race),
                ev_margin=float(st.ev_margin),
                max_staleness_min=float(st.max_staleness_min),
                missing_policy="skip",
            )
            (sdir / "summary.json").write_text(
                json.dumps(
                    {
                        "window": window_name,
                        "strategy": st.__dict__,
                        "close": close.__dict__,
                        "buy": buy.__dict__,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            summary_rows.append(
                {
                    "window": window_name,
                    "start": plan.test_start.isoformat(),
                    "end": plan.test_end.isoformat(),
                    "strategy": st.name,
                    "ticket_type": st.ticket_type,
                    "roi_close": close.roi,
                    "roi_buy": buy.roi,
                    "dd_close": close.max_drawdown,
                    "dd_buy": buy.max_drawdown,
                    "n_bets_close": close.n_bets,
                    "n_bets_buy": buy.n_bets,
                }
            )

    out_csv = group_dir / "summary.csv"
    if summary_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print("OK")
    print("group_dir:", group_dir)
    print("summary:", out_csv)


if __name__ == "__main__":
    main()


