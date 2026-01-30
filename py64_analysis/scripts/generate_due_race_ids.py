"""
forward収集用: 「今この瞬間に取るべき」レース（due races）の race_id をDBから抽出し、
`data/state/race_ids_due_<tag>.txt` に書き出す。

想定運用（ユーザー指定）:
  - buy 収集: 発走 -6分 に実行（検証側 buy_t_minus=5分を跨がないためのバッファ込み）
  - close収集: 発走 -1分 に実行
  - window_sec=90（±45秒）で遅延を吸収

使い方（例）:
  cd py64_analysis
  PYTHONPATH=src python scripts/generate_due_race_ids.py --tag buy  --t-minus-min 6 --window-sec 90
  PYTHONPATH=src python scripts/generate_due_race_ids.py --tag close --t-minus-min 1 --window-sec 90
  
  # --race-date を省略すると、自動で「次の開催日（today以降で最小のdate）」を採用
  PYTHONPATH=src python scripts/generate_due_race_ids.py --tag buy --t-minus-min 6
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]  # keiba/
    src = project_root / "py64_analysis" / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass(frozen=True)
class DueParams:
    tag: str
    t_minus_min: int
    window_sec: int
    now: datetime
    race_date: date

    @property
    def target_time(self) -> datetime:
        return self.now + timedelta(minutes=self.t_minus_min)

    @property
    def window_delta(self) -> timedelta:
        return timedelta(seconds=int(self.window_sec) // 2)

    @property
    def lo(self) -> datetime:
        return self.target_time - self.window_delta

    @property
    def hi(self) -> datetime:
        return self.target_time + self.window_delta


def _parse_yyyymmdd(s: str) -> date:
    s2 = (s or "").strip()
    if len(s2) != 8 or not s2.isdigit():
        raise ValueError(f"invalid YYYYMMDD: {s}")
    return date(int(s2[0:4]), int(s2[4:6]), int(s2[6:8]))


def _find_next_race_date(sess, today: date) -> date | None:
    """fact_race から today 以降で最小の date を取得（次の開催日）"""
    from sqlalchemy import text
    q = text(
        """
        SELECT MIN(date) AS next_date
        FROM fact_race
        WHERE date >= :today
          AND start_time IS NOT NULL
        """
    )
    row = sess.execute(q, {"today": today}).fetchone()
    if row and row[0]:
        return row[0]
    return None


def main() -> None:
    _ensure_import_path()
    from sqlalchemy import text

    from keiba.db.loader import get_session

    p = argparse.ArgumentParser(description="Generate due race_id list for exotics forward collection")
    p.add_argument("--tag", choices=["buy", "close"], required=True)
    p.add_argument("--t-minus-min", type=int, required=True, help="now + N分 が発走時刻のレースを対象にする")
    p.add_argument("--window-sec", type=int, default=90, help="探索窓の秒数（デフォルト90=±45秒）")
    p.add_argument("--race-date", type=str, default="", help="対象日 YYYYMMDD（未指定なら次の開催日を自動検出）")
    p.add_argument("--output", type=Path, default=None, help="出力ファイル（未指定なら data/state/race_ids_due_<tag>.txt）")
    p.add_argument("--now", type=str, default="", help="テスト用: 現在時刻を指定（YYYY-MM-DDTHH:MM:SS）")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if args.output is None:
        out_path = project_root / "data" / "state" / f"race_ids_due_{args.tag}.txt"
    else:
        out_path = args.output if args.output.is_absolute() else (project_root / args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now_dt = datetime.now()
    if args.now:
        now_dt = datetime.fromisoformat(args.now)

    sess = get_session()
    try:
        if args.race_date:
            race_date = _parse_yyyymmdd(args.race_date)
        else:
            # 次の開催日を自動検出
            today = date.today()
            next_date = _find_next_race_date(sess, today)
            if next_date is None:
                print(f"ERROR: No race dates found in fact_race where date >= {today.isoformat()}", file=sys.stderr)
                sys.exit(1)
            race_date = next_date
            print(f"INFO: Auto-detected next race date: {race_date.isoformat()}")

        params = DueParams(
            tag=args.tag,
            t_minus_min=int(args.t_minus_min),
            window_sec=int(args.window_sec),
            now=now_dt,
            race_date=race_date,
        )

        q = text(
            """
            SELECT race_id
            FROM fact_race
            WHERE date = :race_date
              AND start_time IS NOT NULL
              AND (date + start_time) BETWEEN :lo AND :hi
            ORDER BY race_id
            """
        )

        rows = sess.execute(q, {"race_date": params.race_date, "lo": params.lo, "hi": params.hi}).fetchall()
    finally:
        sess.close()

    race_ids = [r[0] for r in rows if r and r[0]]

    out_path.write_text("\n".join(race_ids) + ("\n" if race_ids else ""), encoding="utf-8")

    print(f"tag={params.tag}")
    print(f"race_date={params.race_date}")
    print(f"now={params.now.isoformat(sep=' ', timespec='seconds')}")
    print(f"target={params.target_time.isoformat(sep=' ', timespec='seconds')}")
    print(f"window=[{params.lo.isoformat(sep=' ', timespec='seconds')}, {params.hi.isoformat(sep=' ', timespec='seconds')}]")
    print(f"output={out_path}")
    print(f"n_races={len(race_ids)}")
    for rid in race_ids[:10]:
        print(" -", rid)
    if len(race_ids) > 10:
        print(" ...")


if __name__ == "__main__":
    main()


