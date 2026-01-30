"""
HR（払戻）パーサーの最小テスト
"""

import sys
from pathlib import Path


# py32_fetcherをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_parse_hr_minimal_trio_trifecta_and_refund_bitmap():
    from py32_fetcher.parsers.race_parser import parse_hr_pay

    b = bytearray(b" " * 719)

    # 返還馬番表 pos=59 len=28 -> idx 58..86
    # 馬番5を返還対象にする
    bitmap = list("0" * 28)
    bitmap[4] = "1"
    b[58:86] = "".join(bitmap).encode("shift_jis")

    # 3連複払戻 pos=550 -> idx 549
    off = 549
    b[off : off + 6] = b"010203"
    b[off + 6 : off + 15] = b"000012340"  # 12,340円（100円あたり）
    b[off + 15 : off + 18] = b"001"

    # 3連単払戻 pos=604 -> idx 603
    off = 603
    b[off : off + 6] = b"010203"
    b[off + 6 : off + 15] = b"001234560"  # 1,234,560円（100円あたり）
    b[off + 15 : off + 19] = b"0001"

    out = parse_hr_pay(bytes(b), "2025010101010101", "20250101")
    assert out is not None
    assert out["refund_horse_nos"] == [5]

    assert len(out["trio_payouts"]) == 1
    t = out["trio_payouts"][0]
    assert (t["horse_no_1"], t["horse_no_2"], t["horse_no_3"]) == (1, 2, 3)
    assert t["payout_yen"] == 12340
    assert t["popularity"] == 1

    assert len(out["trifecta_payouts"]) == 1
    s = out["trifecta_payouts"][0]
    assert (s["first_no"], s["second_no"], s["third_no"]) == (1, 2, 3)
    assert s["payout_yen"] == 1234560
    assert s["popularity"] == 1


