from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_odds_dyn_metric(pred: dict, metric: str, lookback_minutes: int) -> float | None:
    metric = str(metric or "").lower()
    try:
        lookback = int(lookback_minutes)
    except Exception:
        lookback = 10

    if metric == "odds_delta_log":
        key = f"odds_chg_{lookback}m"
        delta = pred.get(key)
        try:
            delta = float(delta) if delta is not None else None
        except Exception:
            delta = None
        if delta is None or not np.isfinite(delta):
            return None
        # odds_chg_*m is stored as ratio change: (odds_t0 / odds_t-Δ) - 1
        if delta <= -0.999999:
            return None
        return float(math.log1p(delta))

    if metric == "p_mkt_delta":
        key = f"p_mkt_chg_{lookback}m"
        val = pred.get(key)
        try:
            val = float(val) if val is not None else None
        except Exception:
            val = None
        if val is None or not np.isfinite(val):
            return None
        return float(val)

    return None


def eval_odds_dyn_filter(value: Any, threshold: Any, direction: str) -> bool | None:
    try:
        v = float(value) if value is not None else None
    except Exception:
        v = None
    try:
        thr = float(threshold) if threshold is not None else None
    except Exception:
        thr = None

    if v is None or thr is None or not np.isfinite(v) or not np.isfinite(thr):
        return None

    direction = str(direction or "exclude_high").lower()
    if direction == "exclude_low":
        return v >= thr
    return v <= thr
