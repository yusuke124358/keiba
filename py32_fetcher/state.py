"""
ウォーターマーク（増分更新）管理

JVOpen用のstate管理。JVRTOpen（速報系）はrace_keyで直接指定するためstate不要。
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# プロジェクトルート基準でパスを固定（実行ディレクトリに依存しない）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "data/state/fetch_state.json"
SCHEMA_VERSION = "1.0.0"  # state構造変更時にインクリメント


def load_state() -> dict:
    """stateファイルを読み込む"""
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        # schema_versionチェック（将来の移行用）
        if state.get("_schema_version") != SCHEMA_VERSION:
            logger.warning(
                f"State schema mismatch: {state.get('_schema_version')} != {SCHEMA_VERSION}"
            )
        return state
    return {"_schema_version": SCHEMA_VERSION}


def save_state(state: dict) -> None:
    """stateファイルを保存する"""
    state["_schema_version"] = SCHEMA_VERSION
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), 
        encoding="utf-8"
    )


def get_from_time(data_spec: str) -> str:
    """
    データ種別に応じたfrom_timeを返す（JVOpen用のみ）
    
    update系: last_from_time = last_file_timestamp
    kaisai系: last_from_time = 処理済み開催日の翌日
    
    Note: JVRTOpen（速報系0B31/0B32）はrace_keyで直接指定するためstate不要
    """
    state = load_state()
    spec_state = state.get(data_spec, {})
    
    # 初回は2024年1月1日から
    default = "20240101000000"
    return spec_state.get("last_from_time", default)


def update_watermark_for_update_type(data_spec: str, last_file_timestamp: str) -> None:
    """
    update系（RACE等、JVOpen蓄積系）用
    JVOpenの戻り値をそのまま使う
    """
    state = load_state()
    state[data_spec] = {
        "type": "update",
        "last_from_time": last_file_timestamp,
        "last_file_timestamp": last_file_timestamp,
        "updated_at": datetime.now().isoformat(),
    }
    save_state(state)


def update_watermark_for_kaisai_type(
    data_spec: str, 
    last_file_timestamp: str, 
    max_race_date: str
) -> None:
    """
    kaisai系（0B41, 0B42）用
    処理したrace_dateから次のfrom_timeを決定（JVOpen戻りに依存しない）
    
    Args:
        max_race_date: 処理したレコードの最大race_date (YYYYMMDD形式)
                       空の場合は更新しない（0件時の安全策）
    
    重要: 0件の場合はstateを更新しない（update関数にフォールバックしない）
    """
    if not max_race_date:
        # 0件の場合はstateを更新しない
        logger.info(f"No records for {data_spec}, state not updated")
        return
    
    state = load_state()
    
    # 次回は max_race_date の翌日から
    next_date = datetime.strptime(max_race_date, "%Y%m%d") + timedelta(days=1)
    next_from_time = next_date.strftime("%Y%m%d") + "000000"
    
    state[data_spec] = {
        "type": "kaisai",
        "last_from_time": next_from_time,
        "last_file_timestamp": last_file_timestamp,  # デバッグ用に保持
        "last_kaisai_date": max_race_date,
        "updated_at": datetime.now().isoformat(),
    }
    save_state(state)

