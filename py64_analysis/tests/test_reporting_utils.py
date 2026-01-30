import math

from keiba.analysis.metrics_utils import compute_roi, profit_from_return, roi_footer, sign_mismatch
from keiba.analysis.selector_utils import is_eligible, shrink_delta


def test_profit_and_roi():
    profit = profit_from_return(110.0, 100.0)
    assert profit == 10.0
    roi = compute_roi(profit, 100.0)
    assert roi == 0.1
    assert math.isnan(compute_roi(5.0, 0.0))


def test_sign_mismatch():
    assert sign_mismatch(0.1, -0.1) is True
    assert sign_mismatch(-0.1, 0.1) is True
    assert sign_mismatch(0.1, 0.2) is False
    assert sign_mismatch(0.0, -0.1) is False


def test_roi_footer():
    assert "ROI = profit / stake" in roi_footer()


def test_selector_eligibility_and_shrink():
    eligible, reason = is_eligible(60, 100, min_valid_bets=30, min_valid_bets_ratio=0.5)
    assert eligible is True
    assert reason == "ok"
    eligible, reason = is_eligible(40, 100, min_valid_bets=30, min_valid_bets_ratio=0.5)
    assert eligible is False
    assert reason == "valid_bets_below_ratio"
    assert shrink_delta(0.1, 50, 50) == 0.05
