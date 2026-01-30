"""
三連複/三連単（exotics）用の単発 holdout 実行スクリプト。

目的:
  - B1（close） / C（buy）を同一期間で回して比較できる最小の実行系を用意する
  - 既存の単勝 holdout とは別スクリプトで分離する
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


def _write_report(run_dir: Path, *, name: str, result) -> None:
    report = [
        f"# Exotics backtest report ({name})",
        "",
        f"- n_bets: {result.n_bets}",
        f"- n_wins: {result.n_wins}",
        f"- n_refunds: {result.n_refunds}",
        f"- total_stake: {result.total_stake}",
        f"- total_payout: {result.total_payout}",
        f"- total_profit: {result.total_profit}",
        f"- ROI: {result.roi:.4f}",
        f"- max_drawdown: {result.max_drawdown:.4f}",
        f"- skipped_snapshot: {result.n_races_skipped_snapshot}",
        f"- skipped_hr: {result.n_races_skipped_hr}",
        "",
    ]
    (run_dir / f"report_{name}.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    _ensure_import_path()
    from keiba.db.loader import get_session
    from keiba.exotics.backtest import ExoticsBacktestEngine

    p = argparse.ArgumentParser(description="Run exotics holdout (close vs buy)")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--ticket-type", choices=["trio", "trifecta"], default="trio")
    p.add_argument("--max-bets-per-race", type=int, default=10)
    p.add_argument("--max-staleness-min", type=float, default=10.0)
    p.add_argument("--out-dir", type=Path, default=Path("data/holdout_runs_exotics"))
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"exotics_{args.ticket_type}_{args.start_date}_{args.end_date}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sess = get_session()
    eng = ExoticsBacktestEngine(sess)

    bt_close = eng.run(
        start_date=args.start_date,
        end_date=args.end_date,
        ticket_type=args.ticket_type,
        odds_mode="close",
        max_bets_per_race=args.max_bets_per_race,
        max_staleness_min=args.max_staleness_min,
        missing_policy="skip",
    )
    bt_buy = eng.run(
        start_date=args.start_date,
        end_date=args.end_date,
        ticket_type=args.ticket_type,
        odds_mode="buy",
        max_bets_per_race=args.max_bets_per_race,
        max_staleness_min=args.max_staleness_min,
        missing_policy="skip",
    )

    _write_report(run_dir, name="close", result=bt_close)
    _write_report(run_dir, name="buy", result=bt_buy)

    summary = {
        "ticket_type": args.ticket_type,
        "period": {"start": args.start_date, "end": args.end_date},
        "params": {
            "max_bets_per_race": args.max_bets_per_race,
            "max_staleness_min": args.max_staleness_min,
        },
        "close": bt_close.__dict__,
        "buy": bt_buy.__dict__,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("OK")
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()


