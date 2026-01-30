"""
Ticket N2: Favorite–Longshot bias 対策（odds帯ごとの確率補正）

方針（v1）:
  - train期間のみで odds帯ごとに k_band = win_rate / mean(p_cal) を推定
  - 推論: p_adj = clip(p_cal * k_band, 0, 1)
  - EV判定・stake計算は p_adj を使う

注意:
  - 係数は min_count 未満の帯では global_k にフォールバック
  - k は k_clip でクリップして暴れを抑える
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


def _extend_edges(bins: list[float]) -> list[float]:
    vals = [float(x) for x in (bins or [])]
    vals = sorted(set(vals))
    if not vals:
        return [0.0, float("inf")]
    if vals[0] > 0.0:
        vals = [0.0] + vals
    vals = vals + [float("inf")]
    return vals


def _bin_index(v: Optional[float], edges: list[float]) -> Optional[int]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2


@dataclass(frozen=True)
class OddsBandBias:
    odds_bins: list[float]  # configのbins（最後は暗黙に+inf）
    odds_edges: list[float]
    min_count: int
    lambda_shrink: int
    enforce_monotone: bool
    k_clip: tuple[float, float]
    global_k: float
    k_by_band: dict[int, float]
    n_by_band: dict[int, int]

    @classmethod
    def fit_from_training_data(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        p_cal: pd.Series,
        *,
        odds_bins: list[float],
        min_count: int,
        lambda_shrink: int = 0,
        enforce_monotone: bool = False,
        k_clip: tuple[float, float],
        odds_col: str = "odds",
    ) -> "OddsBandBias":
        # 必須列
        if X is None or X.empty or odds_col not in X.columns:
            return cls(
                odds_bins=list(odds_bins or []),
                odds_edges=_extend_edges(list(odds_bins or [])),
                min_count=int(min_count),
                lambda_shrink=int(lambda_shrink),
                enforce_monotone=bool(enforce_monotone),
                k_clip=(float(k_clip[0]), float(k_clip[1])),
                global_k=1.0,
                k_by_band={},
                n_by_band={},
            )

        odds = pd.to_numeric(X[odds_col], errors="coerce")
        yv = pd.to_numeric(y, errors="coerce")
        pv = pd.to_numeric(p_cal, errors="coerce")

        df = pd.DataFrame({"odds": odds, "y": yv, "p": pv}).dropna()
        df = df[(df["odds"] > 0) & (df["p"] >= 0) & (df["p"] <= 1)]
        edges = _extend_edges(list(odds_bins or []))

        if df.empty:
            return cls(
                odds_bins=list(odds_bins or []),
                odds_edges=edges,
                min_count=int(min_count),
                lambda_shrink=int(lambda_shrink),
                enforce_monotone=bool(enforce_monotone),
                k_clip=(float(k_clip[0]), float(k_clip[1])),
                global_k=1.0,
                k_by_band={},
                n_by_band={},
            )

        # global
        p_mean = float(df["p"].mean()) if len(df) else 0.0
        y_mean = float(df["y"].mean()) if len(df) else 0.0
        global_k = (y_mean / p_mean) if (p_mean > 0 and np.isfinite(p_mean)) else 1.0

        k_lo, k_hi = float(k_clip[0]), float(k_clip[1])
        if not np.isfinite(k_lo):
            k_lo = 0.0
        if not np.isfinite(k_hi):
            k_hi = 1e9
        if k_hi < k_lo:
            k_lo, k_hi = k_hi, k_lo

        k_by: dict[int, float] = {}
        n_by: dict[int, int] = {}

        df["band"] = df["odds"].apply(lambda v: _bin_index(v, edges))
        for band, g in df.groupby("band", dropna=False):
            if band is None or pd.isna(band):
                continue
            n = int(len(g))
            n_by[int(band)] = n
            if n < int(min_count):
                k = global_k
            else:
                p_m = float(g["p"].mean())
                y_m = float(g["y"].mean())
                k = (y_m / p_m) if (p_m > 0 and np.isfinite(p_m)) else global_k
                # Shrink（過学習抑制）
                lam = int(lambda_shrink) if lambda_shrink is not None else 0
                if lam > 0:
                    w = float(n) / float(n + lam)
                    k = 1.0 + w * (float(k) - 1.0)
            # clip
            if not np.isfinite(k):
                k = global_k
            k = float(min(max(k, k_lo), k_hi))
            k_by[int(band)] = k

        # Monotone（oddsが大きいほどkは増えない）
        if bool(enforce_monotone) and k_by:
            # band index は odds_edges の順序（小→大）
            bands_sorted = sorted(k_by.keys())
            prev = None
            for b in bands_sorted:
                cur = float(k_by[b])
                if prev is None:
                    prev = cur
                else:
                    # non-increasing: cur <= prev
                    cur = min(cur, prev)
                    k_by[b] = float(cur)
                    prev = cur

        return cls(
            odds_bins=list(odds_bins or []),
            odds_edges=edges,
            min_count=int(min_count),
            lambda_shrink=int(lambda_shrink),
            enforce_monotone=bool(enforce_monotone),
            k_clip=(k_lo, k_hi),
            global_k=float(min(max(global_k, k_lo), k_hi)),
            k_by_band=k_by,
            n_by_band=n_by,
        )

    def apply(self, p_cal: float, odds_buy: float) -> tuple[float, dict[str, Any]]:
        meta: dict[str, Any] = {"mode": "odds_band_bias"}
        try:
            p0 = float(p_cal)
        except Exception:
            return p_cal, {**meta, "used": False}
        try:
            o = float(odds_buy)
        except Exception:
            return p0, {**meta, "used": False}

        if not np.isfinite(p0) or p0 < 0:
            return p0, {**meta, "used": False}
        if not np.isfinite(o) or o <= 0:
            return float(min(max(p0, 0.0), 1.0)), {**meta, "used": False}

        band = _bin_index(o, self.odds_edges)
        k = self.k_by_band.get(int(band)) if band is not None else None
        n = self.n_by_band.get(int(band), 0) if band is not None else 0
        if k is None:
            k = float(self.global_k)
            used = "global"
        else:
            used = "band" if n >= self.min_count else "global"
            if n < self.min_count:
                k = float(self.global_k)

        p_adj = float(min(max(p0 * float(k), 0.0), 1.0))
        meta.update(
            {
                "used": used,
                "band": int(band) if band is not None else None,
                "k": float(k),
                "band_n": int(n),
                "lambda_shrink": int(self.lambda_shrink),
                "enforce_monotone": bool(self.enforce_monotone),
            }
        )
        return p_adj, meta

    def to_dict(self) -> dict[str, Any]:
        return {
            "odds_bins": [float(x) for x in (self.odds_bins or [])],
            "min_count": int(self.min_count),
            "lambda_shrink": int(self.lambda_shrink),
            "enforce_monotone": bool(self.enforce_monotone),
            "k_clip": [float(self.k_clip[0]), float(self.k_clip[1])],
            "global_k": float(self.global_k),
            "k_by_band": {str(k): float(v) for k, v in self.k_by_band.items()},
            "n_by_band": {str(k): int(v) for k, v in self.n_by_band.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OddsBandBias":
        odds_bins = [float(x) for x in (d.get("odds_bins") or [])]
        k_clip = d.get("k_clip") or [0.5, 1.1]
        k_lo, k_hi = float(k_clip[0]), float(k_clip[1])
        k_by = {int(k): float(v) for k, v in (d.get("k_by_band") or {}).items()}
        n_by = {int(k): int(v) for k, v in (d.get("n_by_band") or {}).items()}
        return cls(
            odds_bins=odds_bins,
            odds_edges=_extend_edges(odds_bins),
            min_count=int(d.get("min_count", 200)),
            lambda_shrink=int(d.get("lambda_shrink", 0)),
            enforce_monotone=bool(d.get("enforce_monotone", False)),
            k_clip=(k_lo, k_hi),
            global_k=float(d.get("global_k", 1.0)),
            k_by_band=k_by,
            n_by_band=n_by,
        )


