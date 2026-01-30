"""
データ取得エントリポイント

使用方法:
    # 蓄積系（レース情報）
    python run_fetch.py stored RACE
    
    # 蓄積系（時系列オッズ）
    python run_fetch.py stored 0B41
    python run_fetch.py stored 0B42
    
    # 速報系（当日オッズ）
    python run_fetch.py realtime 0B31 2024122801010101
    python run_fetch.py realtime 0B32 2024122801010101
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import re

# プロジェクトルート（keiba/）をsys.pathに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py32_fetcher.fetcher_stored import fetch_stored
from py32_fetcher.fetcher_realtime import fetch_realtime
from py32_fetcher.dataspec import STORED_SPECS, REALTIME_SPECS

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FROM_TIME_PATTERN = re.compile(r"^\d{14}$")  # YYYYMMDDHHmmss
RACE_DATE_PATTERN = re.compile(r"^\d{8}$")   # YYYYMMDD


def main():
    parser = argparse.ArgumentParser(description="JV-Link データ取得")
    parser.add_argument(
        "mode", 
        choices=["stored", "realtime"],
        help="取得モード（stored=蓄積系, realtime=速報系）"
    )
    parser.add_argument(
        "data_spec",
        help="データ種別ID（RACE, 0B41, 0B42, 0B31, 0B32）"
    )
    parser.add_argument(
        "race_key",
        nargs="?",
        help="レースキー（速報系のみ、YYYYMMDDJJKKNNRR形式）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw",
        help="出力ディレクトリ（デフォルト: keiba/data/raw）"
    )
    # JVInit は Sid（ソフトウェアID）を受け取る（利用キーとは別）
    parser.add_argument(
        "--software-id",
        dest="software_id",
        default=os.environ.get("KEIBA_JV_SID") or os.environ.get("KEIBA_JV_SOFTWARE_ID") or "UNKNOWN",
        help="JVInitに渡すSid（ソフトウェアID）。未指定ならKEIBA_JV_SID/KEIBA_JV_SOFTWARE_ID、なければUNKNOWN",
    )
    # 利用キーは JVSetServiceKey で設定される（17桁英数字。4-4-4-4-1形式も可）
    parser.add_argument(
        "--use-key",
        dest="use_key",
        default=os.environ.get("KEIBA_JV_USE_KEY") or None,
        help="利用キー（JVSetServiceKey）。推奨: 環境変数KEIBA_JV_USE_KEY（値はログに出しません）",
    )
    parser.add_argument(
        "--from-time",
        default=None,
        help="蓄積系のみ: 取得開始日時（YYYYMMDDHHmmss）。指定するとstateより優先します（スモーク用に短期間だけ取る時に便利）",
    )
    parser.add_argument(
        "--max-race-date",
        default=None,
        help="蓄積系のみ: race_id先頭8桁(YYYYMMDD)がこの日付を超えたら処理を止める（早期停止）。例: 20231231",
    )
    parser.add_argument(
        "--no-state-update",
        action="store_true",
        help="蓄積系のみ: fetch_state.json を更新しない（過年度バックフィル用。通常運用ではOFF推奨）",
    )
    parser.add_argument(
        "--option",
        type=int,
        default=1,
        help=(
            "蓄積系のみ: JVOpenの取得オプション（1=通常, 2=今週/差分, 3=セットアップ(ダイアログあり), 4=セットアップ(ダイアログなし/推奨)）。"
            "RACEで過年度を取りたい場合は 4 + --from-time 20240101000000 のように期間を切って実行するのがおすすめです。"
        ),
    )
    parser.add_argument(
        "--min-race-date",
        default=None,
        help="蓄積系のみ: race_id先頭8桁(YYYYMMDD)がこの日付未満の行はJSONLに書かない（ファイル肥大化防止）。例: 20240101",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="蓄積系のみ: ダウンロード待機タイムアウト秒（未指定なら option=3/4 は6時間、他は30分）。",
    )
    
    args = parser.parse_args()
    
    # バリデーション
    if args.mode == "stored":
        if args.data_spec not in STORED_SPECS:
            logger.error(f"Invalid data_spec for stored: {args.data_spec}")
            logger.error(f"Valid options: {STORED_SPECS}")
            sys.exit(1)
        
        if args.from_time and not FROM_TIME_PATTERN.match(args.from_time):
            logger.error(f"Invalid --from-time: {args.from_time} (expected YYYYMMDDHHmmss)")
            sys.exit(1)
        if args.option not in (1, 2, 3, 4):
            logger.error(f"Invalid --option: {args.option} (expected 1/2/3/4)")
            sys.exit(1)
        if args.min_race_date and not RACE_DATE_PATTERN.match(args.min_race_date):
            logger.error(f"Invalid --min-race-date: {args.min_race_date} (expected YYYYMMDD)")
            sys.exit(1)
        if args.max_race_date and not RACE_DATE_PATTERN.match(args.max_race_date):
            logger.error(f"Invalid --max-race-date: {args.max_race_date} (expected YYYYMMDD)")
            sys.exit(1)

        result = fetch_stored(
            args.data_spec,
            args.output_dir,
            software_id=args.software_id,
            service_key=args.use_key,
            from_time_override=args.from_time,
            option=args.option,
            min_race_date=args.min_race_date,
            max_race_date=args.max_race_date,
            no_state_update=bool(args.no_state_update),
            timeout_sec=args.timeout_sec,
        )
        
    elif args.mode == "realtime":
        if args.data_spec not in REALTIME_SPECS:
            logger.error(f"Invalid data_spec for realtime: {args.data_spec}")
            logger.error(f"Valid options: {REALTIME_SPECS}")
            sys.exit(1)
        
        if not args.race_key:
            logger.error("race_key is required for realtime mode")
            sys.exit(1)
        
        result = fetch_realtime(
            args.data_spec,
            args.race_key,
            args.output_dir,
            software_id=args.software_id,
            service_key=args.use_key,
        )
    
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()

