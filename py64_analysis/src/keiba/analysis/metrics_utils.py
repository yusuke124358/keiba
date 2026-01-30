"""Lightweight metric helpers for reporting and tests."""
from __future__ import annotations

import math
from typing import Optional


def profit_from_return(total_return: float, total_stake: float) -> float:
    """Compute profit as return - stake."""
    return float(total_return) - float(total_stake)


def compute_roi(total_profit: float, total_stake: float) -> float:
    """Compute ROI as profit/stake; return NaN if stake is non-positive."""
    if total_stake is None:
        return float("nan")
    try:
        stake_val = float(total_stake)
    except Exception:
        return float("nan")
    if stake_val <= 0.0:
        return float("nan")
    return float(total_profit) / stake_val


def sign_mismatch(pooled_roi: Optional[float], step14_roi: Optional[float]) -> bool:
    """True when pooled and step14 have opposite signs (ignore zeros/NaN)."""
    if pooled_roi is None or step14_roi is None:
        return False
    try:
        p = float(pooled_roi)
        s = float(step14_roi)
    except Exception:
        return False
    if not math.isfinite(p) or not math.isfinite(s):
        return False
    if p == 0.0 or s == 0.0:
        return False
    return (p > 0.0) != (s > 0.0)


def roi_footer() -> str:
    return "ROI definition: ROI = profit / stake, profit = return - stake."
