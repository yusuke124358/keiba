"""
JRA-VAN Data Lab データ種別ID定義

データ種別ごとの用途・取得API・state管理方式を定義
"""
from enum import Enum


class DataSpec(str, Enum):
    """データ種別ID"""
    
    # MVP対象（蓄積系 - JVOpen）
    RACE = "RACE"              # レース情報（RA/SE/HR）
    TS_ODDS_WIN = "0B41"       # 時系列オッズ（単複枠）O1
    TS_ODDS_QUINELLA = "0B42"  # 時系列オッズ（馬連）O2
    
    # MVP対象（速報系 - JVRTOpen）
    RT_ODDS_WIN = "0B31"       # 速報オッズ（単複枠）O1
    RT_ODDS_QUINELLA = "0B32"  # 速報オッズ（馬連）O2
    
    # 後続Phase（MVP外）
    # RT_ODDS_WIDE = "0B33"
    # RT_ODDS_EXACTA = "0B34"
    # RT_ODDS_TRIO = "0B35"
    # RT_ODDS_TRIFECTA = "0B36"
    
    # 参考（オッズではない）
    # RT_WEIGHT = "0B11"       # 速報馬体重
    # RT_RACE_INFO = "0B12"    # 速報レース情報（成績確定後）


# from_timeの意味（JVOpen用のみ）
# - update: 更新日時基準（JVOpenのlast_file_timestampをそのまま使う）
# - kaisai: 開催日基準（処理したrace_dateから次のfrom_timeを決定）
FROM_TIME_TYPE: dict[str, str] = {
    "RACE": "update",
    "0B41": "kaisai",
    "0B42": "kaisai",
    # 速報系（0B31/0B32）はJVRTOpenでrace_key指定のためstate不要
}


# 速報系データ種別（JVRTOpenで取得、JVSkip不可）
REALTIME_SPECS = frozenset({
    "0B31", "0B32", "0B33", "0B34", "0B35", "0B36"
})


# 蓄積系データ種別（JVOpenで取得）
STORED_SPECS = frozenset({
    "RACE", "0B41", "0B42"
})



