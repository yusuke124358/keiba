"""
蓄積系データ取得（JVOpen）

JVOpenを使用してデータを取得し、JSONLファイルに出力する
"""
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from .jvlink_client import JVLinkClient
from .parsers import parse_record
from .state import (
    get_from_time, 
    update_watermark_for_update_type, 
    update_watermark_for_kaisai_type
)
from .dataspec import FROM_TIME_TYPE

logger = logging.getLogger(__name__)

FROM_TIME_PATTERN = re.compile(r"^\d{14}$")  # YYYYMMDDHHmmss
RACE_DATE_PATTERN = re.compile(r"^\d{8}$")  # YYYYMMDD


def fetch_stored(
    data_spec: str,
    output_dir: Path,
    software_id: str = "UNKNOWN",
    service_key: str | None = None,
    from_time_override: str | None = None,
    option: int = 1,
    min_race_date: str | None = None,
    max_race_date: str | None = None,
    no_state_update: bool = False,
    timeout_sec: Optional[int] = None,
) -> dict:
    """
    蓄積系データを取得してJSONL出力
    
    Args:
        data_spec: データ種別ID（"RACE", "0B41", "0B42"）
        output_dir: 出力ディレクトリ
        software_id: JVInit に渡す Sid（ソフトウェアID）
        service_key: 利用キー（17桁英数字、または4-4-4-4-1形式）
    
    Returns:
        {
            "records": int,
            "parsed_count": int,
            "raw_count": int,
            "max_race_date": str,
            "last_file_timestamp": str,
            "output_file": str,
        }
    """
    client = JVLinkClient(software_id, service_key=service_key)
    if option not in (1, 2, 3, 4):
        raise ValueError(f"Invalid option: {option} (expected 1/2/3/4)")

    # setup(option=3/4) は「過年度データ」を含み得る（ダイアログ有無だけが違い）。
    # ただし option=3/4 でも from_time は効く（read_countが減る）ので、
    # 期間を切りたい場合は --from-time を指定するのが推奨。
    if from_time_override is not None:
        if not FROM_TIME_PATTERN.match(from_time_override):
            raise ValueError(f"Invalid from_time_override (expected YYYYMMDDHHmmss): {from_time_override}")
        from_time = from_time_override
    elif option in (3, 4):
        # setupで明示指定がない場合は全期間扱い
        from_time = "00000000000000"
    else:
        from_time = get_from_time(data_spec)

    if min_race_date is not None and not RACE_DATE_PATTERN.match(min_race_date):
        raise ValueError(f"Invalid min_race_date (expected YYYYMMDD): {min_race_date}")
    if max_race_date is not None and not RACE_DATE_PATTERN.match(max_race_date):
        raise ValueError(f"Invalid max_race_date (expected YYYYMMDD): {max_race_date}")

    # NOTE:
    #   引数 max_race_date は「早期停止の上限日（YYYYMMDD）」として使う。
    #   取得中に追跡する「見えた最大日付」と変数名を分けないと、上限がすり替わってしまうため注意。
    max_race_date_limit = max_race_date
    
    logger.info(f"Fetching {data_spec} from {from_time} (option={option})")
    
    ret_code, read_count, dl_count, last_file_ts = client.open_stored(
        data_spec, from_time, option=option
    )
    
    if ret_code != 0:
        client.close()
        raise RuntimeError(f"JVOpen failed: {ret_code}")
    
    logger.info(f"JVOpen success: read_count={read_count}, dl_count={dl_count}")
    
    # ダウンロード待機（必要な場合）
    if dl_count > 0:
        import time
        logger.info(f"Waiting for download: {dl_count} files")
        # setup(option=3) はファイル数が多く時間がかかりやすいので長めにする
        if timeout_sec is None:
            timeout_sec = 21600 if option in (3, 4) else 1800  # 6時間 / 30分
        start = time.time()
        while True:
            status = client.status()
            if status >= dl_count:
                logger.info(f"Download complete: {status} files")
                break
            elapsed = time.time() - start
            if elapsed > timeout_sec:
                logger.error(f"Download timeout after {timeout_sec}s")
                client.cancel()
                client.close()
                raise TimeoutError(f"Download timeout: {status}/{dl_count}")
            time.sleep(1)  # 1秒待機（CPU負荷軽減）
    
    output_file = output_dir / f"{data_spec}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    record_count = 0
    parsed_count = 0  # パース成功数
    raw_count = 0     # パース失敗（_raw）数
    skipped_count = 0 # min_race_dateでスキップした行数
    stopped_by_max_date = False
    max_race_date_seen = ""
    last_seen_race_date: str | None = None
    seen_unsorted_dates = False
    
    with open(output_file, "w", encoding="utf-8") as f:
        while True:
            ret, data, filename = client.read()
            
            if ret == 0:  # EOF
                logger.info("EOF reached")
                break
            if ret == -1:  # ファイル切り替わり
                logger.debug(f"File switch: {filename}")
                continue
            if ret == -3:  # ファイルダウンロード中
                # JV-Link仕様: -3 は異常ではなく「ダウンロード中」
                import time
                time.sleep(1)
                continue
            if ret in (-402, -403):
                # 仕様書: -402/-403 は「ダウンロードしたファイルが異常」
                #   JVFiledeleteで該当ファイルを削除し、JVOpenからやり直す
                logger.warning(f"JVRead returned {ret} (bad file). Deleting and retrying: {filename}")
                try:
                    if filename:
                        client.file_delete(filename)
                except Exception as e:
                    logger.warning(f"JVFiledelete failed (will still retry JVOpen): {e}")
                # JVOpenからやり直す（同一パラメータ）
                try:
                    client.close()
                except Exception:
                    pass
                client = JVLinkClient(software_id, service_key=service_key)
                ret_code2, read_count2, dl_count2, last_file_ts = client.open_stored(
                    data_spec, from_time, option=option
                )
                if ret_code2 != 0:
                    client.close()
                    raise RuntimeError(f"JVOpen failed (after retry): {ret_code2}")
                logger.info(f"JVOpen retry success: read_count={read_count2}, dl_count={dl_count2}")
                continue
            if ret < -1:  # エラー
                client.close()
                raise RuntimeError(f"JVRead error: {ret}")
            
            # パース
            record = parse_record(data, data_spec, filename)
            if record:
                # 期間フィルタ（ファイルサイズ抑制用）
                if min_race_date:
                    rd = (record.get("race_id") or "")[:8]
                    if rd and rd < min_race_date:
                        skipped_count += 1
                        continue

                # max_race_date（早期停止 / それ以降は書かない）
                if max_race_date_limit:
                    rd = (record.get("race_id") or "")[:8]
                    if rd:
                        # RACEは非レース系レコードも混ざり得るので、順序性チェックは保守的に
                        # （完全に単調でない場合は早期停止を無効化し、上限超過分はスキップのみ）
                        if last_seen_race_date and rd < last_seen_race_date:
                            seen_unsorted_dates = True
                        last_seen_race_date = rd

                        if rd > max_race_date_limit:
                            if not seen_unsorted_dates:
                                stopped_by_max_date = True
                                logger.info(
                                    f"Stopping early due to max_race_date_limit={max_race_date_limit} (rd={rd})"
                                )
                                break
                            skipped_count += 1
                            continue

                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                record_count += 1
                
                # パース成功/失敗をカウント
                if "_raw" in record:
                    raw_count += 1
                else:
                    parsed_count += 1
                    # kaisai系用：最大race_dateを追跡
                    # ★重要: パース成功したレコードのみでmax_race_dateを更新
                    #   パース失敗で_rawになったレコードでstateを進めると
                    #   その開催日を二度と取り直せなくなる
                    race_date = record.get("race_id", "")[:8]
                    if race_date > max_race_date_seen:
                        max_race_date_seen = race_date
            
            if record_count % 1000 == 0:
                logger.info(f"Processed {record_count} records (parsed={parsed_count}, raw={raw_count})")
    
    client.close()
    
    logger.info(
        f"Fetch complete: {record_count} records (parsed={parsed_count}, raw={raw_count}), "
        f"max_date_seen={max_race_date_seen}"
    )
    
    # ウォーターマーク更新
    # 重要: kaisai系は0件時にupdate関数にフォールバックしない
    if not no_state_update:
        spec_type = FROM_TIME_TYPE.get(data_spec, "update")
        if spec_type == "kaisai":
            # kaisai系: 0件でもupdate_watermark_for_kaisai_typeを呼ぶ
            # （内部で0件判定してstate更新をスキップする）
            update_watermark_for_kaisai_type(data_spec, last_file_ts, max_race_date_seen)
        else:
            # update系: 常に更新
            update_watermark_for_update_type(data_spec, last_file_ts)
    else:
        logger.info("no_state_update=True: state watermark not updated")
    
    return {
        "records": record_count,
        "parsed_count": parsed_count,
        "raw_count": raw_count,
        "skipped_count": skipped_count,
        "stopped_by_max_date": stopped_by_max_date,
        "max_race_date": max_race_date_seen,
        "max_race_date_limit": max_race_date_limit,
        "last_file_timestamp": last_file_ts,
        "output_file": str(output_file),
    }

