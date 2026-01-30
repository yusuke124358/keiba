from __future__ import annotations

import math
from typing import Tuple


def _clip_prob(p: float, lo: float, hi: float) -> float:
    if p < lo:
        return lo
    if p > hi:
        return hi
    return p


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def shrink_probability(
    p_hat: float,
    p_mkt: float | None,
    alpha: float,
    p_mkt_clip: Tuple[float, float],
) -> float:
    """Shrink p_hat toward p_mkt in logit space."""
    if p_mkt is None:
        return float(p_hat)
    try:
        p_hat_f = float(p_hat)
        p_mkt_f = float(p_mkt)
        a = float(alpha)
    except Exception:
        return float(p_hat)

    if not (math.isfinite(p_hat_f) and math.isfinite(p_mkt_f) and math.isfinite(a)):
        return float(p_hat)

    lo, hi = p_mkt_clip
    lo = float(lo)
    hi = float(hi)

    if a <= 0.0:
        return float(_clip_prob(p_mkt_f, lo, hi))
    if a >= 1.0:
        return float(_clip_prob(p_hat_f, lo, hi))

    p_hat_c = _clip_prob(p_hat_f, lo, hi)
    p_mkt_c = _clip_prob(p_mkt_f, lo, hi)

    logit_hat = _logit(p_hat_c)
    logit_mkt = _logit(p_mkt_c)
    logit_shrunk = logit_mkt + a * (logit_hat - logit_mkt)
    return float(_sigmoid(logit_shrunk))
