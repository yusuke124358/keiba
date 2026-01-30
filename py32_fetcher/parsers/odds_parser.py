"""
オッズレコード（O1/O2）パーサー

すべての時刻はJST前提
"""
from datetime import datetime
from typing import Optional


def parse_o1_odds(raw_bytes: bytes, race_id: str, race_date: str) -> Optional[dict]:
    """
    JV_O1_ODDS_TANFUKUWAKU パーサー（単勝/複勝/枠連オッズ）
    
    Args:
        raw_bytes: 生バイト列
        race_id: 16桁のレースID
        race_date: YYYYMMDD形式
    
    Returns:
        dict with:
            - asof_time: datetime (JST, 年またぎ補正済み)
            - win_odds: list[dict] (単勝オッズ)
            - place_odds: list[dict] (複勝オッズ)
            - total_sales_win: int
            - total_sales_place: int
        
        None if:
            - レコード長が不足（949バイト未満）
            - asof_time が取れない
    """
    # ★レコード長チェック（total_sales_placeまで参照するので949バイト必要）
    if len(raw_bytes) < 949:
        return None
    
    if not race_date or len(race_date) < 6:
        return None
    
    race_year = int(race_date[0:4])
    race_month = int(race_date[4:6])
    
    # 発表月日時分（HappyoTime）: 28-35バイト目 (MMDDHHMM)
    happyo_raw = raw_bytes[27:35].decode("shift_jis", errors="ignore").strip()
    asof_time = _parse_happyo_time(happyo_raw, race_year, race_month)
    
    # ★重要: asof_time が取れなければパース失敗として扱う
    #   loader側でスキップされるがstate側では成功扱いになる"ねじれ"を防ぐ
    if asof_time is None:
        return None
    
    # 単勝オッズ（28頭分、44バイト目から8バイトずつ）
    win_odds = []
    for i in range(28):
        offset = 43 + (i * 8)
        row = _parse_tansyo_info(raw_bytes, offset)
        if row:
            win_odds.append(row)
    
    # 複勝オッズ（28頭分、268バイト目から12バイトずつ）
    place_odds = []
    for i in range(28):
        offset = 267 + (i * 12)
        row = _parse_fukusyo_info(raw_bytes, offset)
        if row:
            place_odds.append(row)
    
    # 票数合計
    total_sales_win = _parse_int(raw_bytes[927:938])
    total_sales_place = _parse_int(raw_bytes[938:949])
    
    return {
        "asof_time": asof_time,
        "win_odds": win_odds,
        "place_odds": place_odds,
        "total_sales_win": total_sales_win,
        "total_sales_place": total_sales_place,
    }


def parse_o2_odds(raw_bytes: bytes, race_id: str, race_date: str) -> Optional[dict]:
    """
    JV_O2_ODDS_UMAREN パーサー（馬連オッズ）
    
    Args:
        raw_bytes: 生バイト列
        race_id: 16桁のレースID
        race_date: YYYYMMDD形式
    
    Returns:
        dict with:
            - asof_time: datetime (JST, 年またぎ補正済み)
            - quinella_odds: list[dict] (馬連オッズ)
            - total_sales: int
        
        None if:
            - レコード長が不足（2040バイト未満）
            - asof_time が取れない
    """
    # ★レコード長チェック（total_salesまで参照するので2040バイト必要）
    if len(raw_bytes) < 2040:
        return None
    
    if not race_date or len(race_date) < 6:
        return None
    
    race_year = int(race_date[0:4])
    race_month = int(race_date[4:6])
    
    # 発表月日時分（HappyoTime）: 28-35バイト目 (MMDDHHMM)
    happyo_raw = raw_bytes[27:35].decode("shift_jis", errors="ignore").strip()
    asof_time = _parse_happyo_time(happyo_raw, race_year, race_month)
    
    # ★重要: asof_time が取れなければパース失敗として扱う
    if asof_time is None:
        return None
    
    # 馬連オッズ（153組分、41バイト目から13バイトずつ）
    quinella_odds = []
    for i in range(153):
        offset = 40 + (i * 13)
        row = _parse_umaren_info(raw_bytes, offset)
        if row:
            quinella_odds.append(row)
    
    # 票数合計
    total_sales = _parse_int(raw_bytes[2029:2040])
    
    return {
        "asof_time": asof_time,
        "quinella_odds": quinella_odds,
        "total_sales": total_sales,
    }


def _parse_happyo_time(happyo_raw: str, race_year: int, race_month: int) -> Optional[datetime]:
    """
    発表月日時分をdatetimeに変換（年またぎ補正込み）
    
    補正ルール:
        - race_monthが1月で、happyo月が12月 → year = race_year - 1
        - race_monthが12月で、happyo月が1月 → year = race_year + 1
    
    すべての時刻はJST前提
    """
    if len(happyo_raw) != 8:
        return None
    
    try:
        happyo_month = int(happyo_raw[0:2])
        happyo_day = int(happyo_raw[2:4])
        happyo_hour = int(happyo_raw[4:6])
        happyo_minute = int(happyo_raw[6:8])
    except ValueError:
        return None
    
    # 年またぎ補正
    year = race_year
    if race_month == 1 and happyo_month == 12:
        year = race_year - 1
    elif race_month == 12 and happyo_month == 1:
        year = race_year + 1
    
    try:
        return datetime(year, happyo_month, happyo_day, happyo_hour, happyo_minute)
    except ValueError:
        return None


def _parse_tansyo_info(raw_bytes: bytes, offset: int) -> Optional[dict]:
    """単勝オッズ情報（8バイト）"""
    try:
        umaban = raw_bytes[offset:offset+2].decode("shift_jis").strip()
        odds_raw = raw_bytes[offset+2:offset+6].decode("shift_jis").strip()
        ninki = raw_bytes[offset+6:offset+8].decode("shift_jis").strip()
        
        if not umaban or not odds_raw:
            return None
        
        return {
            "horse_no": int(umaban),
            "odds": float(odds_raw) / 10,  # 10倍されている
            "popularity": int(ninki) if ninki else None,
        }
    except (ValueError, UnicodeDecodeError):
        return None


def _parse_fukusyo_info(raw_bytes: bytes, offset: int) -> Optional[dict]:
    """複勝オッズ情報（12バイト）"""
    try:
        umaban = raw_bytes[offset:offset+2].decode("shift_jis").strip()
        odds_low = raw_bytes[offset+2:offset+6].decode("shift_jis").strip()
        odds_high = raw_bytes[offset+6:offset+10].decode("shift_jis").strip()
        ninki = raw_bytes[offset+10:offset+12].decode("shift_jis").strip()
        
        if not umaban or not odds_low:
            return None
        
        return {
            "horse_no": int(umaban),
            "odds_low": float(odds_low) / 10,
            "odds_high": float(odds_high) / 10 if odds_high else None,
            "popularity": int(ninki) if ninki else None,
        }
    except (ValueError, UnicodeDecodeError):
        return None


def _parse_umaren_info(raw_bytes: bytes, offset: int) -> Optional[dict]:
    """馬連オッズ情報（13バイト）"""
    try:
        kumi = raw_bytes[offset:offset+4].decode("shift_jis").strip()
        odds_raw = raw_bytes[offset+4:offset+10].decode("shift_jis").strip()
        ninki = raw_bytes[offset+10:offset+13].decode("shift_jis").strip()
        
        if not kumi or len(kumi) < 2 or not odds_raw:
            return None
        
        # 組番を馬番ペアに分解（例: "0102" → horse_no_1=1, horse_no_2=2）
        horse_no_1 = int(kumi[0:2])
        horse_no_2 = int(kumi[2:4])
        
        # 正規化: horse_no_1 < horse_no_2 を保証
        if horse_no_1 > horse_no_2:
            horse_no_1, horse_no_2 = horse_no_2, horse_no_1
        
        return {
            "horse_no_1": horse_no_1,
            "horse_no_2": horse_no_2,
            "odds": float(odds_raw) / 10,
            "popularity": int(ninki) if ninki else None,
        }
    except (ValueError, UnicodeDecodeError):
        return None


def _parse_int(raw: bytes) -> Optional[int]:
    """バイト列を整数に変換"""
    try:
        s = raw.decode("shift_jis").strip()
        return int(s) if s else None
    except (ValueError, UnicodeDecodeError):
        return None


def _parse_kumi_3(kumi: str) -> Optional[tuple[int, int, int]]:
    """
    3頭組番（6桁）を (a,b,c) に変換する。

    例: "010203" -> (1,2,3)
    """
    if not kumi or len(kumi) != 6:
        return None
    if not kumi.isdigit():
        return None
    try:
        a = int(kumi[0:2])
        b = int(kumi[2:4])
        c = int(kumi[4:6])
    except ValueError:
        return None
    if a <= 0 or b <= 0 or c <= 0:
        return None
    return a, b, c


def _parse_odds_times10(s: str) -> Optional[float]:
    """
    JV-Dataのオッズ文字列（倍率×10の整数文字列）を float に変換する。

    - digits の場合: int(s)/10
    - それ以外: None（返還/発売無し/特払/登録なし等の特殊値）
    """
    if not s:
        return None
    s2 = s.strip()
    if not s2:
        return None
    if not s2.isdigit():
        return None
    try:
        return float(int(s2)) / 10.0
    except ValueError:
        return None


def _odds_status_from_raw(s: str) -> str:
    """
    オッズ欄の特殊値をステータスとして返す（解析用）。
    """
    if s is None:
        return "missing"
    raw = str(s)
    if raw.strip() == "":
        return "not_registered"
    if set(raw.strip()) == {"-"}:
        return "no_sale"
    if set(raw.strip()) == {"*"}:
        return "special_payout"
    if raw.strip().strip("0") == "":
        # "000000" / "0000000" は返還
        return "refund"
    if raw.strip().isdigit():
        return "ok"
    return "unknown"


def parse_o5_odds(raw_bytes: bytes, race_id: str, race_date: str) -> Optional[dict]:
    """
    JV_O5_ODDS_SANRENPUKU パーサー（3連複オッズ）

    JV-Data4512.xlsx（フォーマット）より:
      - レコード長: 12293 bytes
      - HappyoTime: pos=28 len=8 (MMDDHHMM)
      - <3連複オッズ>: pos=41, 816件, 15 bytes/件
          a 組番: 6
          b オッズ: 6 (倍率×10, "000000"=返還, "------"=発売無し, "******"=特払, "      "=登録なし)
          c 人気: 3
      - 3連複売上合計: pos=12281 len=11
    """
    # レコード長チェック（売上合計まで参照）
    if len(raw_bytes) < 12293:
        return None
    if not race_date or len(race_date) < 6:
        return None

    race_year = int(race_date[0:4])
    race_month = int(race_date[4:6])

    happyo_raw = raw_bytes[27:35].decode("shift_jis", errors="ignore").strip()
    asof_time = _parse_happyo_time(happyo_raw, race_year, race_month)
    if asof_time is None:
        return None

    # 発売フラグ（pos=40 len=1）: 0/1/3/7 等（意味はExcel参照）
    sale_flag = raw_bytes[39:40].decode("shift_jis", errors="ignore").strip()

    rows: list[dict] = []
    base_offset = 40  # pos=41 -> idx=40
    stride = 15
    for i in range(816):
        off = base_offset + i * stride
        kumi_raw = raw_bytes[off : off + 6].decode("shift_jis", errors="ignore")
        kumi = kumi_raw.strip()
        if not kumi:
            continue
        triple = _parse_kumi_3(kumi)
        if triple is None:
            continue
        a, b, c = triple
        # 三連複は順序なし（昇順に正規化）
        x1, x2, x3 = sorted((a, b, c))

        odds_raw = raw_bytes[off + 6 : off + 12].decode("shift_jis", errors="ignore")
        ninki_raw = raw_bytes[off + 12 : off + 15].decode("shift_jis", errors="ignore").strip()

        status = _odds_status_from_raw(odds_raw)
        # 返還/発売無し/特払/登録なし は odds=None に統一（解析や欠落率に使う）
        if status == "ok":
            odds = _parse_odds_times10(odds_raw)
        else:
            odds = None

        popularity = None
        if ninki_raw and ninki_raw.isdigit():
            try:
                popularity = int(ninki_raw)
            except ValueError:
                popularity = None

        rows.append(
            {
                "horse_no_1": x1,
                "horse_no_2": x2,
                "horse_no_3": x3,
                "odds": odds,
                "popularity": popularity,
                "odds_status": status,
            }
        )

    total_sales = _parse_int(raw_bytes[12280:12291])

    return {
        "asof_time": asof_time,
        "sale_flag": sale_flag,
        "trio_odds": rows,
        "total_sales": total_sales,
    }


def parse_o6_odds(raw_bytes: bytes, race_id: str, race_date: str) -> Optional[dict]:
    """
    JV_O6_ODDS_SANRENTAN パーサー（3連単オッズ）

    JV-Data4512.xlsx（フォーマット）より:
      - レコード長: 83285 bytes
      - HappyoTime: pos=28 len=8 (MMDDHHMM)
      - <3連単オッズ>: pos=41, 4896件, 17 bytes/件
          a 組番: 6
          b オッズ: 7 (倍率×10, "0000000"=返還, "-------"=発売無し, "*******"=特払, "       "=登録なし)
          c 人気: 4
      - 3連単売上合計: pos=83273 len=11
    """
    if len(raw_bytes) < 83285:
        return None
    if not race_date or len(race_date) < 6:
        return None

    race_year = int(race_date[0:4])
    race_month = int(race_date[4:6])

    happyo_raw = raw_bytes[27:35].decode("shift_jis", errors="ignore").strip()
    asof_time = _parse_happyo_time(happyo_raw, race_year, race_month)
    if asof_time is None:
        return None

    sale_flag = raw_bytes[39:40].decode("shift_jis", errors="ignore").strip()

    rows: list[dict] = []
    base_offset = 40
    stride = 17
    for i in range(4896):
        off = base_offset + i * stride
        kumi_raw = raw_bytes[off : off + 6].decode("shift_jis", errors="ignore")
        kumi = kumi_raw.strip()
        if not kumi:
            continue
        triple = _parse_kumi_3(kumi)
        if triple is None:
            continue
        a, b, c = triple

        odds_raw = raw_bytes[off + 6 : off + 13].decode("shift_jis", errors="ignore")
        ninki_raw = raw_bytes[off + 13 : off + 17].decode("shift_jis", errors="ignore").strip()

        status = _odds_status_from_raw(odds_raw)
        # 返還/発売無し/特払/登録なし は odds=None に統一
        if status == "ok":
            odds = _parse_odds_times10(odds_raw)
        else:
            odds = None

        popularity = None
        if ninki_raw and ninki_raw.isdigit():
            try:
                popularity = int(ninki_raw)
            except ValueError:
                popularity = None

        rows.append(
            {
                "first_no": a,
                "second_no": b,
                "third_no": c,
                "odds": odds,
                "popularity": popularity,
                "odds_status": status,
            }
        )

    total_sales = _parse_int(raw_bytes[83272:83283])

    return {
        "asof_time": asof_time,
        "sale_flag": sale_flag,
        "trifecta_odds": rows,
        "total_sales": total_sales,
    }

