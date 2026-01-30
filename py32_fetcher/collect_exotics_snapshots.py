"""
forward 収集（0B35=三連複, 0B36=三連単）の最小ユーティリティ。

設計意図:
  - まずは「buy / close の2回だけ」取得して raw JSONL を確実に貯める
  - 後からDB投入・検証に回せるよう、data_spec ごとに state を持って resume 可能にする
  - record_type の揺れがあっても parse_record が raw 退避するので、欠落しにくい

使い方（例）:
  # buy（購入想定時刻）スナップショット
  python py32_fetcher/collect_exotics_snapshots.py --tag buy --race-ids-file data/state/race_ids_today.txt

  # close（締切直前）スナップショット
  python py32_fetcher/collect_exotics_snapshots.py --tag close --race-ids-file data/state/race_ids_today.txt --resume
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルート（keiba/）をsys.pathに追加（直接実行でも import が通るように）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py32_fetcher.bulk_fetch_realtime import bulk_fetch_realtime  # noqa: E402


def _default_out_dir(tag: str) -> Path:
    day = datetime.now().strftime("%Y%m%d")
    return PROJECT_ROOT / "data" / "raw" / "exotics_snapshots" / day / tag


def _default_state_file(tag: str, data_spec: str) -> Path:
    return PROJECT_ROOT / "data" / "state" / f"rt_exotics_{tag}_{data_spec}.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Collect exotics (0B35/0B36) snapshots as JSONL")
    p.add_argument("--race-ids-file", type=Path, required=True, help="race_id一覧ファイル（1行1レース）")
    p.add_argument("--tag", default="buy", help="スナップショット種別（buy/close/任意文字列）")
    p.add_argument(
        "--specs",
        default="0B35,0B36",
        help="取得するdata_specのカンマ区切り（デフォルト: 0B35,0B36）",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="出力先（未指定なら data/raw/exotics_snapshots/<YYYYMMDD>/<tag>/）")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--sleep-sec", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--software-id",
        dest="software_id",
        default=os.environ.get("KEIBA_JV_SID") or os.environ.get("KEIBA_JV_SOFTWARE_ID") or "UNKNOWN",
        help="JVInitに渡すSid（未指定ならKEIBA_JV_SID/KEIBA_JV_SOFTWARE_ID）",
    )
    p.add_argument(
        "--use-key",
        dest="use_key",
        default=os.environ.get("KEIBA_JV_USE_KEY") or None,
        help="利用キー（JVSetServiceKey）。推奨: 環境変数KEIBA_JV_USE_KEY",
    )
    args = p.parse_args()

    tag = str(args.tag).strip() or "buy"
    output_dir = args.output_dir if args.output_dir is not None else _default_out_dir(tag)
    specs = [s.strip() for s in str(args.specs).split(",") if s.strip()]
    if not specs:
        raise SystemExit("--specs is empty")

    # 0B35/0B36の2回取得（buy/close）を想定するが、tag/specsは任意に拡張できる
    for data_spec in specs:
        state_file = _default_state_file(tag, data_spec)
        out_dir = output_dir / data_spec
        result = bulk_fetch_realtime(
            data_spec=data_spec,
            race_ids=_read_race_ids(args.race_ids_file),
            output_dir=out_dir,
            software_id=args.software_id,
            service_key=args.use_key,
            sleep_sec=float(args.sleep_sec),
            chunk_size=int(args.chunk_size),
            resume=bool(args.resume),
            state_file=state_file,
        )
        print("result:", result)


def _read_race_ids(path: Path) -> list[str]:
    race_ids: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            race_ids.append(s)
    return race_ids


if __name__ == "__main__":
    main()


