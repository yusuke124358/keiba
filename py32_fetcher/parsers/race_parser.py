"""
レースレコード（RA/SE/HR）パーサー

すべての時刻はJST前提
"""
from typing import Optional


def _s(raw_bytes: bytes, pos_1based: int, size: int) -> str:
    """
    仕様書の「位置」（1-based）で切り出して decode する。
    """
    a = int(pos_1based) - 1
    b = a + int(size)
    if a < 0 or b <= 0 or b > len(raw_bytes):
        return ""
    return raw_bytes[a:b].decode("shift_jis", errors="ignore")


def _parse_int(raw: str) -> Optional[int]:
    t = (raw or "").strip()
    if not t:
        return None
    if not t.isdigit():
        return None
    try:
        return int(t)
    except ValueError:
        return None


def _parse_time_tenths(raw: str) -> Optional[float]:
    """
    99.9秒形式（3桁=0.1秒単位）を float 秒に変換。
    0/000/999/9999 系は欠損扱い。
    """
    v = _parse_int(raw)
    if v is None:
        return None
    if v <= 0:
        return None
    # 仕様上 99.9秒が上限（=999）だが、999 は欠損コードとして扱うケースもあるため欠損に寄せる
    if v >= 999:
        return None
    return float(v) / 10.0


def parse_ra_race(raw_bytes: bytes, race_id: str, race_date: str) -> Optional[dict]:
    """
    JV_RA_RACE パーサー（レース詳細）
    
    Args:
        raw_bytes: 生バイト列
        race_id: 16桁のレースID
        race_date: YYYYMMDD形式
    
    Returns:
        dict with race details
    """
    # ★最小要件: 現状参照している最大offset（889）+1 = 890バイト
    # 注意: RAのレコード長は仕様上もっと長い（ラップ等を含む）ため、
    # 追加項目は len チェック付きで条件パースする。
    if len(raw_bytes) < 890:
        return None
    
    try:
        # 基本情報
        track_code = raw_bytes[19:21].decode("shift_jis", errors="ignore").strip()
        race_no = raw_bytes[25:27].decode("shift_jis", errors="ignore").strip()
        
        # 距離（698-701バイト目）
        kyori = raw_bytes[697:701].decode("shift_jis", errors="ignore").strip()
        
        # トラックコード（706-707バイト目）
        track_cd = raw_bytes[705:707].decode("shift_jis", errors="ignore").strip()
        
        # 発走時刻（874-877バイト目）
        hasso_time = raw_bytes[873:877].decode("shift_jis", errors="ignore").strip()
        
        # 出走頭数（884-885バイト目）
        syusso_tosu = raw_bytes[883:885].decode("shift_jis", errors="ignore").strip()
        
        # 天候・馬場状態（888-890バイト目）
        tenko_cd = raw_bytes[887:888].decode("shift_jis", errors="ignore").strip()
        siba_baba_cd = raw_bytes[888:889].decode("shift_jis", errors="ignore").strip()
        dirt_baba_cd = raw_bytes[889:890].decode("shift_jis", errors="ignore").strip()
        
        out = {
            "track_code": track_code,
            "race_no": int(race_no) if race_no else None,
            "distance": int(kyori) if kyori else None,
            "track_cd": track_cd,
            "start_time": hasso_time,
            "field_size": int(syusso_tosu) if syusso_tosu else None,
            "weather_cd": tenko_cd,
            "going_turf": siba_baba_cd,
            "going_dirt": dirt_baba_cd,
        }
        
        # ===== C4-0: ペース素材（公式仕様: JV-Data490.xlsx / フォーマット）=====
        # ラップタイム（位置=891, 25個×3byte, 0.1秒）
        # 前3F/前4F/後3F/後4F（位置=970/973/976/979, 各3byte, 0.1秒）
        if len(raw_bytes) >= 981:
            # lap_times_200m: 長さ25で保持（欠損はNone）
            laps: list[Optional[float]] = []
            base_pos = 891
            for i in range(25):
                raw = _s(raw_bytes, base_pos + i * 3, 3)
                laps.append(_parse_time_tenths(raw))
            out["lap_times_200m"] = laps
            out["pace_first3f"] = _parse_time_tenths(_s(raw_bytes, 970, 3))
            out["pace_first4f"] = _parse_time_tenths(_s(raw_bytes, 973, 3))
            out["pace_last3f"] = _parse_time_tenths(_s(raw_bytes, 976, 3))
            out["pace_last4f"] = _parse_time_tenths(_s(raw_bytes, 979, 3))
        else:
            out["lap_times_200m"] = None
            out["pace_first3f"] = None
            out["pace_first4f"] = None
            out["pace_last3f"] = None
            out["pace_last4f"] = None

        return out
    except Exception:
        return None


def parse_se_race_uma(raw_bytes: bytes, race_id: str, race_date: str) -> Optional[dict]:
    """
    JV_SE_RACE_UMA パーサー（出走馬情報）
    
    Args:
        raw_bytes: 生バイト列
        race_id: 16桁のレースID
        race_date: YYYYMMDD形式
    
    Returns:
        dict with horse entry details
    """
    # ★修正: 実際に参照する最大offset（393）+1 = 394バイト
    # 以前の500は強すぎて正しいレコードを捨てる可能性があった
    if len(raw_bytes) < 394:
        return None
    
    try:
        # 枠番・馬番
        wakuban = raw_bytes[27:28].decode("shift_jis", errors="ignore").strip()
        umaban = raw_bytes[28:30].decode("shift_jis", errors="ignore").strip()
        
        # 血統登録番号
        ketto_num = raw_bytes[30:40].decode("shift_jis", errors="ignore").strip()
        
        # 馬名
        bamei = raw_bytes[40:76].decode("shift_jis", errors="ignore").strip()
        
        # 性別コード
        sex_cd = raw_bytes[78:79].decode("shift_jis", errors="ignore").strip()
        
        # 馬齢
        barei = raw_bytes[82:84].decode("shift_jis", errors="ignore").strip()
        
        # 騎手コード
        kisyu_code = raw_bytes[296:301].decode("shift_jis", errors="ignore").strip()
        
        # 負担重量
        futan = raw_bytes[288:291].decode("shift_jis", errors="ignore").strip()
        
        # 馬体重
        bataijyu = raw_bytes[324:327].decode("shift_jis", errors="ignore").strip()
        
        # 確定着順
        kakutei_jyuni = raw_bytes[334:336].decode("shift_jis", errors="ignore").strip()
        
        # 走破タイム
        time = raw_bytes[338:342].decode("shift_jis", errors="ignore").strip()
        
        # 単勝オッズ・人気
        odds = raw_bytes[359:363].decode("shift_jis", errors="ignore").strip()
        ninki = raw_bytes[363:365].decode("shift_jis", errors="ignore").strip()
        
        # 後3ハロンタイム
        harontime_l3 = raw_bytes[390:393].decode("shift_jis", errors="ignore").strip()

        # ===== C4-0: 通過順（コーナー順位）=====
        # 1C..4C（位置=352/354/356/358, 各2byte）
        pos_1c = _parse_int(_s(raw_bytes, 352, 2))
        pos_2c = _parse_int(_s(raw_bytes, 354, 2))
        pos_3c = _parse_int(_s(raw_bytes, 356, 2))
        pos_4c = _parse_int(_s(raw_bytes, 358, 2))
        # 0/00/99 などの初期値は欠損扱い
        def _clean_pos(v: Optional[int]) -> Optional[int]:
            if v is None:
                return None
            if v <= 0:
                return None
            if v >= 99:
                return None
            return v
        
        return {
            "frame_no": int(wakuban) if wakuban else None,
            "horse_no": int(umaban) if umaban else None,
            "horse_id": ketto_num,
            "horse_name": bamei,
            "sex_cd": sex_cd,
            "age": int(barei) if barei else None,
            "jockey_id": kisyu_code,
            "weight_carried": float(futan) / 10 if futan else None,
            "horse_weight": int(bataijyu) if bataijyu else None,
            "finish_pos": int(kakutei_jyuni) if kakutei_jyuni else None,
            "time": time,
            "odds": float(odds) / 10 if odds else None,
            "popularity": int(ninki) if ninki else None,
            "last_3f": float(harontime_l3) / 10 if harontime_l3 else None,
            "pos_1c": _clean_pos(pos_1c),
            "pos_2c": _clean_pos(pos_2c),
            "pos_3c": _clean_pos(pos_3c),
            "pos_4c": _clean_pos(pos_4c),
        }
    except Exception:
        return None


def parse_hr_pay(raw_bytes: bytes, race_id: str, race_date: str) -> Optional[dict]:
    """
    JV_HR_PAY パーサー（払戻情報）
    
    Args:
        raw_bytes: 生バイト列
        race_id: 16桁のレースID
        race_date: YYYYMMDD形式
    
    Returns:
        dict with payout details
    
    Note:
        JV-Data4512.xlsx（フォーマット）に基づく「最小実装」。
        - 返還馬番表（馬番01..28）
        - 3連複払戻（最大3件：同着の複数的中に対応）
        - 3連単払戻（最大6件：同着の複数的中に対応）
    """
    # レコード長（フォーマットシートより 719 bytes）
    if len(raw_bytes) < 719:
        return None

    def _s(a: int, b: int) -> str:
        return raw_bytes[a:b].decode("shift_jis", errors="ignore")

    def _parse_int(s: str) -> Optional[int]:
        t = (s or "").strip()
        if not t:
            return None
        if not t.isdigit():
            return None
        try:
            return int(t)
        except ValueError:
            return None

    def _parse_kumi3(s: str) -> Optional[tuple[int, int, int]]:
        t = (s or "").strip()
        if len(t) != 6 or (not t.isdigit()):
            return None
        try:
            a = int(t[0:2])
            b = int(t[2:4])
            c = int(t[4:6])
        except ValueError:
            return None
        if a <= 0 or b <= 0 or c <= 0:
            return None
        return a, b, c

    # 返還馬番表（pos=59 len=28, 1-based）
    refund_bitmap = _s(58, 86)
    refund_horse_nos: list[int] = []
    if refund_bitmap:
        for i, ch in enumerate(refund_bitmap[:28]):
            if ch == "1":
                refund_horse_nos.append(i + 1)

    # フラグ（任意・解析用）
    trio_pay_flag = _s(47, 48).strip()   # pos=48
    trifecta_pay_flag = _s(48, 49).strip()  # pos=49
    trio_refund_flag = _s(56, 57).strip()   # pos=57
    trifecta_refund_flag = _s(57, 58).strip()  # pos=58

    # 3連複払戻（<3連複払戻> pos=550, count=3, stride=18）
    trio_payouts: list[dict] = []
    base = 549
    stride = 18
    for i in range(3):
        off = base + i * stride
        kumi = _s(off, off + 6)
        pay = _s(off + 6, off + 15)
        pop = _s(off + 15, off + 18)
        k = _parse_kumi3(kumi)
        payout_yen = _parse_int(pay)
        popularity = _parse_int(pop)
        if k is None:
            continue
        if payout_yen is None and popularity is None:
            # 組番だけ埋まっているケースを避ける
            continue
        a, b, c = sorted(k)  # 三連複は順序なし
        trio_payouts.append(
            {
                "horse_no_1": a,
                "horse_no_2": b,
                "horse_no_3": c,
                "payout_yen": payout_yen,
                "popularity": popularity,
            }
        )

    # 3連単払戻（<3連単払戻> pos=604, count=6, stride=19）
    trifecta_payouts: list[dict] = []
    base = 603
    stride = 19
    for i in range(6):
        off = base + i * stride
        kumi = _s(off, off + 6)
        pay = _s(off + 6, off + 15)
        pop = _s(off + 15, off + 19)
        k = _parse_kumi3(kumi)
        payout_yen = _parse_int(pay)
        popularity = _parse_int(pop)
        if k is None:
            continue
        if payout_yen is None and popularity is None:
            continue
        a, b, c = k  # 三連単は順序あり
        trifecta_payouts.append(
            {
                "first_no": a,
                "second_no": b,
                "third_no": c,
                "payout_yen": payout_yen,
                "popularity": popularity,
            }
        )

    return {
        "refund_horse_nos": refund_horse_nos,
        "trio_pay_flag": trio_pay_flag or None,
        "trifecta_pay_flag": trifecta_pay_flag or None,
        "trio_refund_flag": trio_refund_flag or None,
        "trifecta_refund_flag": trifecta_refund_flag or None,
        "trio_payouts": trio_payouts,
        "trifecta_payouts": trifecta_payouts,
    }

