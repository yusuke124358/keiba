from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class MarketBlendResult:
    p_blend: float
    ev_blend: float
    odds_band: str


def _to_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _clip_prob_array(arr: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(arr, eps, 1.0 - eps)


def odds_band(odds: float) -> str:
    if odds < 3:
        return "<3"
    if odds < 5:
        return "3-5"
    if odds < 10:
        return "5-10"
    if odds < 20:
        return "10-20"
    return "20+"


def parse_exclude_odds_band(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if str(v).strip()]
        return tuple(items)
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in ("none", "null", "nan"):
            return ()
        parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
        return tuple(parts)
    return (str(value).strip(),)


def require_market_prob(p_mkt_col: object, market_method: str) -> float:
    method = str(market_method or "").lower()
    if method != "p_mkt_col":
        raise ValueError(f"market_prob_method must be p_mkt_col (got: {market_method})")
    p_mkt = _to_float(p_mkt_col)
    if p_mkt is None:
        raise ValueError("market_method=p_mkt_col but column missing or all null")
    return p_mkt


def logit_blend_prob(
    p_model: object,
    p_mkt: object,
    blend_w: float,
    *,
    clip_eps: float = 1e-6,
) -> np.ndarray | float:
    p_model_arr = np.asarray(p_model, dtype=float)
    p_mkt_arr = np.asarray(p_mkt, dtype=float)
    p_model_c = _clip_prob_array(p_model_arr, clip_eps)
    p_mkt_c = _clip_prob_array(p_mkt_arr, clip_eps)
    logits = blend_w * np.log(p_model_c / (1.0 - p_model_c)) + (1.0 - blend_w) * np.log(p_mkt_c / (1.0 - p_mkt_c))
    out = 1.0 / (1.0 + np.exp(-logits))
    if np.isscalar(p_model) and np.isscalar(p_mkt):
        return float(out)
    return out


def compute_market_blend(
    *,
    p_model: object,
    p_mkt_col: object,
    odds: object,
    blend_w: float,
    market_method: str = "p_mkt_col",
    clip_eps: float = 1e-6,
) -> MarketBlendResult:
    p_model_f = _to_float(p_model)
    odds_f = _to_float(odds)
    if p_model_f is None:
        raise ValueError("p_model is missing or not finite")
    if odds_f is None or odds_f <= 0:
        raise ValueError("odds is missing or <= 0")

    p_mkt_f = require_market_prob(p_mkt_col, market_method)
    p_blend = logit_blend_prob(p_model_f, p_mkt_f, float(blend_w), clip_eps=clip_eps)
    if not isinstance(p_blend, float):
        p_blend = float(p_blend)
    ev_blend = p_blend * odds_f - 1.0
    return MarketBlendResult(p_blend=float(p_blend), ev_blend=float(ev_blend), odds_band=odds_band(odds_f))
