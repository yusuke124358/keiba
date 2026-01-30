"""
オッズパーサーのテスト
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime

# py32_fetcherをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestParseHappyoTime:
    """年またぎ補正のテスト"""
    
    def test_same_year(self):
        """同じ年の場合"""
        from py32_fetcher.parsers.odds_parser import _parse_happyo_time
        
        # 3月のレースで3月のオッズ
        result = _parse_happyo_time("03151030", race_year=2024, race_month=3)
        assert result == datetime(2024, 3, 15, 10, 30)
    
    def test_year_rollback(self):
        """年またぎ（1月のレースで12月のオッズ）"""
        from py32_fetcher.parsers.odds_parser import _parse_happyo_time
        
        # 1月のレースで12月のオッズ → 前年
        result = _parse_happyo_time("12311800", race_year=2024, race_month=1)
        assert result == datetime(2023, 12, 31, 18, 0)
    
    def test_year_forward(self):
        """年またぎ（12月のレースで1月のオッズ）"""
        from py32_fetcher.parsers.odds_parser import _parse_happyo_time
        
        # 12月のレースで1月のオッズ → 翌年
        result = _parse_happyo_time("01010900", race_year=2024, race_month=12)
        assert result == datetime(2025, 1, 1, 9, 0)
    
    def test_invalid_format(self):
        """不正なフォーマット"""
        from py32_fetcher.parsers.odds_parser import _parse_happyo_time
        
        # 短すぎる
        assert _parse_happyo_time("0315", race_year=2024, race_month=3) is None
        
        # 空
        assert _parse_happyo_time("", race_year=2024, race_month=3) is None


class TestParseO1Odds:
    """O1パーサーのテスト"""
    
    def test_returns_none_for_invalid_date(self):
        """不正な日付の場合Noneを返す"""
        from py32_fetcher.parsers.odds_parser import parse_o1_odds
        
        result = parse_o1_odds(b"", "1234567890123456", "")
        assert result is None
        
        result = parse_o1_odds(b"", "1234567890123456", None)
        assert result is None


class TestParseO5Odds:
    """O5（3連複）パーサーのテスト（最小ケース）"""

    def test_parse_single_row(self):
        from py32_fetcher.parsers.odds_parser import parse_o5_odds

        # JV-Data4512.xlsx: O5 record length 12293
        b = bytearray(b" " * 12293)
        # HappyoTime at pos=28..35 -> idx 27..35
        b[27:35] = "01011000".encode("shift_jis")  # Jan 1 10:00
        # sale_flag at pos=40 -> idx 39
        b[39:40] = b"1"
        # first combo at pos=41 -> idx 40
        b[40:46] = b"010203"
        b[46:52] = b"000123"  # 12.3倍
        b[52:55] = b"001"
        # total_sales at pos=12281 -> idx 12280..12290
        b[12280:12291] = b"00000001000"

        out = parse_o5_odds(bytes(b), "2025010101010101", "20250101")
        assert out is not None
        assert out["sale_flag"] == "1"
        assert out["asof_time"].month == 1
        assert len(out["trio_odds"]) == 1
        row = out["trio_odds"][0]
        assert (row["horse_no_1"], row["horse_no_2"], row["horse_no_3"]) == (1, 2, 3)
        assert row["odds"] == 12.3
        assert row["popularity"] == 1
        assert row["odds_status"] == "ok"


class TestParseO6Odds:
    """O6（3連単）パーサーのテスト（最小ケース）"""

    def test_parse_single_row(self):
        from py32_fetcher.parsers.odds_parser import parse_o6_odds

        # JV-Data4512.xlsx: O6 record length 83285
        b = bytearray(b" " * 83285)
        b[27:35] = "01011000".encode("shift_jis")
        b[39:40] = b"1"
        # first combo at pos=41 -> idx 40
        b[40:46] = b"010203"
        b[46:53] = b"0000123"  # 12.3倍（7桁×10）
        b[53:57] = b"0001"
        b[83272:83283] = b"00000002000"

        out = parse_o6_odds(bytes(b), "2025010101010101", "20250101")
        assert out is not None
        assert out["sale_flag"] == "1"
        assert len(out["trifecta_odds"]) == 1
        row = out["trifecta_odds"][0]
        assert (row["first_no"], row["second_no"], row["third_no"]) == (1, 2, 3)
        assert row["odds"] == 12.3
        assert row["popularity"] == 1
        assert row["odds_status"] == "ok"


