"""Selector helpers for valid-only gating and shrinkage."""
from __future__ import annotations

from typing import Optional, Tuple


def shrink_delta(delta: Optional[float], n: Optional[float], n0: float) -> Optional[float]:
    if delta is None or n is None:
        return None
    try:
        n_val = float(n)
        n0_val = float(n0)
    except Exception:
        return None
    denom = n_val + n0_val
    if denom <= 0.0:
        return float(delta)
    return float((n_val / denom) * float(delta))


def is_eligible(
    valid_bets: Optional[float],
    base_valid_bets: Optional[float],
    min_valid_bets: float,
    min_valid_bets_ratio: float,
) -> Tuple[bool, str]:
    if valid_bets is None or base_valid_bets is None:
        return False, "missing_valid_bets"
    try:
        v_bets = float(valid_bets)
        b_bets = float(base_valid_bets)
    except Exception:
        return False, "invalid_valid_bets"
    if v_bets < float(min_valid_bets):
        return False, "valid_bets_below_min"
    if b_bets > 0.0 and v_bets < float(min_valid_bets_ratio) * b_bets:
        return False, "valid_bets_below_ratio"
    return True, "ok"
