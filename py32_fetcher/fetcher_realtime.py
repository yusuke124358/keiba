"""
速報系データ取得（JVRTOpen）

JVRTOpenを使用してリアルタイムデータを取得する
"""
import json
import logging
from pathlib import Path
from datetime import datetime

from .jvlink_client import JVLinkClient
from .parsers import parse_record

logger = logging.getLogger(__name__)


def fetch_realtime(
    data_spec: str, 
    race_key: str, 
    output_dir: Path, 
    software_id: str = "UNKNOWN",
    service_key: str | None = None,
) -> dict:
    """
    速報系データを取得してJSONL出力
    
    Args:
        data_spec: データ種別ID（"0B31", "0B32"）
        race_key: レースキー（YYYYMMDDJJKKNNRR形式）
        output_dir: 出力ディレクトリ
        software_id: JVInit に渡す Sid（ソフトウェアID）
        service_key: 利用キー（17桁英数字、または4-4-4-4-1形式）
    
    Returns:
        {
            "records": int,
            "output_file": str,
        }
    
    Note:
        速報系はfrom_timeではなくrace_keyで直接指定するため、
        state管理は不要
    """
    client = JVLinkClient(software_id, service_key=service_key)
    
    logger.info(f"Fetching realtime {data_spec} for {race_key}")
    
    result = client.open_realtime(data_spec, race_key)
    
    # 戻り値の型はCOM実装依存（intまたはtuple）
    if isinstance(result, tuple):
        ret_code = result[0]
    else:
        ret_code = result
    
    if ret_code != 0 and ret_code < 0:
        client.close()
        raise RuntimeError(f"JVRTOpen failed: {ret_code}")
    
    output_file = output_dir / f"{data_spec}_{race_key}_{datetime.now():%H%M%S}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    record_count = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        while True:
            ret, data, filename = client.read()
            
            if ret == 0:  # EOF
                logger.info("EOF reached")
                break
            if ret == -1:  # ファイル切り替わり
                logger.debug(f"File switch: {filename}")
                continue
            if ret < -1:  # エラー
                client.close()
                raise RuntimeError(f"JVRead error: {ret}")
            
            # パース
            record = parse_record(data, data_spec, filename)
            if record:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                record_count += 1
    
    client.close()
    
    logger.info(f"Realtime fetch complete: {record_count} records")
    
    return {
        "records": record_count,
        "output_file": str(output_file),
    }

