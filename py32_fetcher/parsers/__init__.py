"""
JV-Data パーサー

レコードをパースして正規化済みdictを返す
"""
from datetime import datetime
from typing import Optional
import logging

from .race_parser import parse_ra_race, parse_se_race_uma, parse_hr_pay
from .odds_parser import parse_o1_odds, parse_o2_odds, parse_o5_odds, parse_o6_odds

logger = logging.getLogger(__name__)


def parse_record(raw_bytes: bytes, data_spec: str, file_name: str) -> Optional[dict]:
    """
    レコードをパースして正規化済みdictを返す
    
    設計ルール:
        - パーサーがNoneを返した場合 → raw保存にフォールバック
        - 未対応レコード → raw保存
        - 必ずdictを返す（Noneは返さない、ただしバイト不足時のみNone）
    
    Returns:
        共通フィールド:
            - record_type: str ("RA", "SE", "O1", "O2", etc.)
            - race_id: str (16桁)
            - data_kubun: str (1=中間, 2=前日売最終, 3=最終, 4=確定)
            - _meta: dict (data_spec, file_name, ingested_at, parser_version)
        
        O1固有:
            - asof_time: datetime (JST)
            - win_odds: list[{horse_no, odds, popularity}]
            - place_odds: list[{horse_no, odds_low, odds_high, popularity}]
            - total_sales_win: int
            - total_sales_place: int
        
        O2固有:
            - asof_time: datetime (JST)
            - quinella_odds: list[{horse_no_1, horse_no_2, odds, popularity}]
            - total_sales: int
        
        未対応/パースエラー時:
            - _raw: hex文字列（raw保存でDB投入はしない）
    """
    if len(raw_bytes) < 27:
        return None
    
    # レコード種別（先頭2バイト）
    record_type = raw_bytes[0:2].decode("shift_jis", errors="ignore")
    
    # データ区分（3バイト目）
    data_kubun = raw_bytes[2:3].decode("shift_jis", errors="ignore")
    
    # race_id抽出（12-27バイト目）
    race_id = raw_bytes[11:27].decode("shift_jis", errors="ignore").strip()
    
    # race_date
    race_date = race_id[0:8] if len(race_id) >= 8 else None  # YYYYMMDD
    
    # 共通メタデータ
    meta = {
        "data_spec": data_spec,
        "file_name": file_name,
        "ingested_at": datetime.now().isoformat(),
        "parser_version": "1.0.0",
    }
    
    # 基本情報（raw保存用にも使う）
    base = {
        "record_type": record_type,
        "race_id": race_id,
        "data_kubun": data_kubun,
        "_meta": meta,
    }
    
    # レコード種別ごとにパース
    parsers = {
        "RA": parse_ra_race,
        "SE": parse_se_race_uma,
        "HR": parse_hr_pay,
        "O1": parse_o1_odds,
        "O2": parse_o2_odds,
        # 3連複/3連単（速報オッズ）
        "O5": parse_o5_odds,
        "O6": parse_o6_odds,
    }
    
    parser = parsers.get(record_type)
    if parser is None:
        # 0B35/0B36 は record_type の揺れがあると致命的なので、raw退避しつつ目立つログを出す
        if data_spec in ("0B35", "0B36"):
            logger.warning(f"Unhandled record_type='{record_type}' for data_spec={data_spec} (will store raw)")
        # 未対応レコードはraw保存
        base["_raw"] = raw_bytes.hex()
        return base
    
    # パーサー呼び出し
    try:
        result = parser(raw_bytes, race_id, race_date)
    except Exception as e:
        logger.warning(f"Parser error for {record_type}: {e}")
        result = None
    
    # パーサーがNone/失敗 → raw保存にフォールバック
    if result is None:
        base["_raw"] = raw_bytes.hex()
        return base
    
    # 成功：基本情報をマージ
    result.update(base)
    return result



