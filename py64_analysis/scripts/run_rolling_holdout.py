"""
rolling window で holdout を一気に回す（A案）。

特徴:
  - `run_holdout.py` をウィンドウごとに呼び出して、artifacts/report/summary を出す
  - すべての実行結果を 1つの group_dir に集約
  - 最後に summarize_holdout_runs.py で group_dir を集計して summary.csv を出す

例:
  python py64_analysis/scripts/run_rolling_holdout.py \
    --name rolling30 \
    --test-range-start 2025-12-01 --test-range-end 2025-12-28 \
    --test-window-days 30 --step-days 7 \
    --gap-days 0 \
    --estimate-closing-mult --closing-mult-quantile 0.30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


def _ensure_import_path() -> None:
    # `python py64_analysis/scripts/run_rolling_holdout.py` で実行される想定
    project_root = Path(__file__).resolve().parents[2]  # keiba/
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@dataclass(frozen=True)
class WindowPlan:
    idx: int
    train_start: date
    train_end: date
    valid_start: date | None
    valid_end: date | None
    test_start: date
    test_end: date


def _date_range_windows(
    test_range_start: date,
    test_range_end: date,
    test_window_days: int,
    step_days: int,
) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    if test_window_days <= 0:
        raise ValueError("test_window_days must be > 0")
    if step_days <= 0:
        raise ValueError("step_days must be > 0")

    cur = test_range_start
    while True:
        end = cur + timedelta(days=test_window_days - 1)
        if end > test_range_end:
            break
        windows.append((cur, end))
        cur = cur + timedelta(days=step_days)
    return windows


def _build_plans(
    db_min_date: date,
    test_windows: list[tuple[date, date]],
    gap_days: int,
    train_lookback_days: int | None,
    valid_window_days: int,
) -> list[WindowPlan]:
    plans: list[WindowPlan] = []
    for idx, (test_start, test_end) in enumerate(test_windows, start=1):
        pre_test_end = test_start - timedelta(days=gap_days + 1)
        if pre_test_end < db_min_date:
            continue

        valid_start = None
        valid_end = None

        if valid_window_days > 0:
            valid_end = pre_test_end
            valid_start = valid_end - timedelta(days=valid_window_days - 1)
            train_end = valid_start - timedelta(days=1)
        else:
            train_end = pre_test_end

        if train_end < db_min_date:
            continue

        if train_lookback_days is None:
            train_start = db_min_date
        else:
            if train_lookback_days <= 0:
                raise ValueError("train_lookback_days must be > 0 if specified")
            train_start = train_end - timedelta(days=train_lookback_days - 1)
            if train_start < db_min_date:
                train_start = db_min_date

        if train_start > train_end:
            continue

        plans.append(
            WindowPlan(
                idx=idx,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return plans


def main() -> None:
    _ensure_import_path()
    from keiba.db.loader import get_session
    from keiba.config import get_config
    from sqlalchemy import text

    p = argparse.ArgumentParser(description="Run rolling holdout windows")
    p.add_argument("--name", default="rolling")
    p.add_argument("--test-range-start", required=True)
    p.add_argument("--test-range-end", required=True)
    p.add_argument("--config", default=None, help="config path (optional)")
    p.add_argument("--test-window-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=7)
    p.add_argument("--gap-days", type=int, default=0, help="train/valid と test の間のギャップ（日数）")
    p.add_argument("--train-lookback-days", type=int, default=None, help="固定長train窓（未指定ならexpanding window）")
    p.add_argument("--valid-window-days", type=int, default=0, help="0なら明示validなし（train内部split）。>0ならtest直前のN日をvalidにする")
    p.add_argument("--estimate-closing-mult", action="store_true")
    p.add_argument("--closing-mult-quantile", type=float, default=0.30)
    p.add_argument("--initial-bankroll", type=float, default=None)
    p.add_argument("--out-root", type=Path, default=Path("data/holdout_runs"))
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--top", type=int, default=10, help="最後にROI上位N件を表示")
    args = p.parse_args()

    from keiba.utils.config_resolver import (
        git_commit,
        rel_path,
        resolve_config_path,
        save_config_origin,
        save_config_used,
    )

    resolved_config_path, config_origin = resolve_config_path(args.config)
    orig_env_config = os.environ.get("KEIBA_CONFIG_PATH")
    os.environ["KEIBA_CONFIG_PATH"] = str(resolved_config_path)

    test_range_start = _parse_date(args.test_range_start)
    test_range_end = _parse_date(args.test_range_end)

    sess = get_session()
    cfg = get_config()
    u = cfg.universe
    track_codes = list(u.track_codes or [])
    exclude_race_ids = list(u.exclude_race_ids or [])
    buy_minutes = int(cfg.backtest.buy_t_minus_minutes)

    row = sess.execute(
        text(
            """
            SELECT MIN(r.date), MAX(r.date)
            FROM fact_race r
            WHERE r.start_time IS NOT NULL
              AND (:track_codes_len = 0 OR r.track_code = ANY(:track_codes))
              AND (:require_results = FALSE OR EXISTS (
                    SELECT 1 FROM fact_result res
                    WHERE res.race_id = r.race_id
                      AND res.finish_pos IS NOT NULL
              ))
              AND (:require_ts_win = FALSE OR EXISTS (
                    SELECT 1 FROM odds_ts_win o
                    WHERE o.race_id = r.race_id
                      AND o.odds > 0
                      AND o.asof_time <= ((r.date::timestamp + r.start_time) - make_interval(mins => :buy_minutes))
              ))
              AND (:exclude_len = 0 OR NOT (r.race_id = ANY(:exclude_race_ids)))
            """
        ),
        {
            "track_codes_len": len(track_codes),
            "track_codes": track_codes,
            "require_results": bool(u.require_results),
            "require_ts_win": bool(u.require_ts_win),
            "buy_minutes": buy_minutes,
            "exclude_len": len(exclude_race_ids),
            "exclude_race_ids": exclude_race_ids,
        },
    ).fetchone()
    if not row or not row[0] or not row[1]:
        raise SystemExit("fact_race is empty or start_time is NULL for all rows")
    db_min_date = row[0]

    test_windows = _date_range_windows(
        test_range_start=test_range_start,
        test_range_end=test_range_end,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
    )

    plans = _build_plans(
        db_min_date=db_min_date,
        test_windows=test_windows,
        gap_days=args.gap_days,
        train_lookback_days=args.train_lookback_days,
        valid_window_days=args.valid_window_days,
    )

    if args.max_windows is not None:
        plans = plans[: max(0, int(args.max_windows))]

    if not plans:
        raise SystemExit("No runnable windows (check date ranges / lookbacks)")

    project_root = Path(__file__).resolve().parents[2]
    out_root = args.out_root
    if not out_root.is_absolute():
        out_root = project_root / out_root

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_dir = out_root / f"{args.name}_{ts}"
    group_dir.mkdir(parents=True, exist_ok=True)

    config_meta = save_config_used(resolved_config_path, group_dir)
    config_used_path = Path(config_meta["config_used_path"])
    origin_payload = {
        "origin": config_origin,
        "resolved_config_path": rel_path(resolved_config_path, project_root),
        "config_used_path": rel_path(config_used_path, project_root),
        "config_hash_sha256": config_meta.get("config_hash_sha256"),
        "git_commit": git_commit(project_root),
        "generated_at": ts,
    }
    save_config_origin(group_dir, origin_payload)

    run_holdout = project_root / "py64_analysis" / "scripts" / "run_holdout.py"
    summarize = project_root / "py64_analysis" / "scripts" / "summarize_holdout_runs.py"

    index: dict = {
        "name": args.name,
        "generated_at": ts,
        "group_dir": str(group_dir),
        "config_used_path": rel_path(config_used_path, project_root),
        "plans": [],
        "errors": [],
        "args": vars(args),
    }

    print("group_dir:", group_dir)
    print("windows:", len(plans))

    for plan in plans:
        window_name = f"w{plan.idx:03d}_{plan.test_start:%Y%m%d}_{plan.test_end:%Y%m%d}"
        run_dir = group_dir / window_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(run_holdout),
            "--train-start",
            plan.train_start.isoformat(),
            "--train-end",
            plan.train_end.isoformat(),
            "--test-start",
            plan.test_start.isoformat(),
            "--test-end",
            plan.test_end.isoformat(),
            "--name",
            window_name,
            "--out-dir",
            str(run_dir),
        ]

        sub_env = os.environ.copy()
        if orig_env_config is None:
            sub_env.pop("KEIBA_CONFIG_PATH", None)
            if config_origin == "cli:--config":
                cmd += ["--config", str(resolved_config_path)]

        if plan.valid_start and plan.valid_end:
            cmd += ["--valid-start", plan.valid_start.isoformat(), "--valid-end", plan.valid_end.isoformat()]

        if args.estimate_closing_mult:
            cmd += ["--estimate-closing-mult", "--closing-mult-quantile", str(args.closing_mult_quantile)]

        if args.initial_bankroll is not None:
            cmd += ["--initial-bankroll", str(args.initial_bankroll)]

        index["plans"].append(
            {
                "window_name": window_name,
                "train_start": plan.train_start.isoformat(),
                "train_end": plan.train_end.isoformat(),
                "valid_start": plan.valid_start.isoformat() if plan.valid_start else None,
                "valid_end": plan.valid_end.isoformat() if plan.valid_end else None,
                "test_start": plan.test_start.isoformat(),
                "test_end": plan.test_end.isoformat(),
                "run_dir": str(run_dir),
                "cmd": cmd,
            }
        )

        print(f"[{plan.idx}/{len(plans)}] {window_name}: train={plan.train_start}..{plan.train_end} test={plan.test_start}..{plan.test_end}")
        if args.dry_run:
            continue

        try:
            # 各windowのログを run_dir に保存（後から原因切り分けが容易）
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            with open(stdout_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
                subprocess.run(cmd, stdout=out_f, stderr=err_f, check=True, env=sub_env)
        except subprocess.CalledProcessError as e:
            err = {
                "window_name": window_name,
                "run_dir": str(run_dir),
                "returncode": e.returncode,
            }
            index["errors"].append(err)
            print("ERROR:", err)
            if args.stop_on_error:
                break

    # index.json を保存
    (group_dir / "rolling_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # 集計（group_dir 配下だけ）
    if not args.dry_run:
        summary_csv = group_dir / "summary.csv"
        subprocess.run(
            [
                sys.executable,
                str(summarize),
                "--root",
                str(group_dir),
                "--out",
                str(summary_csv),
                "--top",
                str(args.top),
            ],
            check=True,
        )
        # metrics.json (non-fatal if extraction fails)
        try:
            from keiba.eval.extract_metrics import write_metrics_json

            write_metrics_json(group_dir, run_kind="rolling_holdout")
            print("metrics:", group_dir / "metrics.json")
        except Exception as e:
            print(f"[metrics] failed: {e}")

    print("DONE")
    print("group_dir:", group_dir)


if __name__ == "__main__":
    main()

