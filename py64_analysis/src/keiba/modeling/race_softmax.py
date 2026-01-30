"""
Race-level softmax probability generation.

p_race_i = exp(s_i / T) / sum_j exp(s_j / T)
where s_i = w * score_model + (1 - w) * score_mkt
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RaceSoftmaxFitResult:
    w: float
    t: float
    loss: float
    n_races: int
    n_rows: int


def _clip(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _build_scores(
    p_model: np.ndarray,
    p_mkt: np.ndarray,
    w: float,
    score_space: str,
    clip_eps: float,
) -> np.ndarray:
    p_model = _clip(p_model, clip_eps)
    p_mkt = _clip(p_mkt, clip_eps)
    if score_space == "logit":
        s_model = _logit(p_model)
        s_mkt = _logit(p_mkt)
    elif score_space == "log":
        s_model = np.log(p_model)
        s_mkt = np.log(p_mkt)
    else:
        raise ValueError(f"Unsupported score_space: {score_space}")
    return (w * s_model) + ((1.0 - w) * s_mkt)


def apply_race_softmax(
    df: pd.DataFrame,
    *,
    w: float,
    t: float,
    score_space: str,
    clip_eps: float,
) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=float)

    p_mkt = pd.to_numeric(df["p_mkt"], errors="coerce").astype(float)
    p_model = pd.to_numeric(df.get("p_model"), errors="coerce").astype(float)
    p_model = p_model.where(np.isfinite(p_model), p_mkt)
    p_mkt = p_mkt.where(np.isfinite(p_mkt), p_model)
    p_model = p_model.fillna(0.0)
    p_mkt = p_mkt.fillna(0.0)

    scores = _build_scores(p_model.values, p_mkt.values, float(w), score_space, float(clip_eps))
    t_val = float(t) if float(t) > 0 else 1.0
    scores = scores / t_val

    score_series = pd.Series(scores, index=df.index)

    def _softmax_series(s: pd.Series) -> pd.Series:
        v = s.to_numpy()
        v = v - np.max(v)
        exp_v = np.exp(v)
        denom = exp_v.sum()
        if denom <= 0:
            return pd.Series(np.zeros_like(v, dtype=float), index=s.index)
        return pd.Series(exp_v / denom, index=s.index)

    probs = score_series.groupby(df["race_id"], dropna=False, sort=False).transform(_softmax_series)
    return probs.astype(float)


def fit_race_softmax(
    df: pd.DataFrame,
    *,
    w_grid_step: float,
    t_grid: Iterable[float],
    score_space: str,
    clip_eps: float,
) -> RaceSoftmaxFitResult:
    if df.empty:
        return RaceSoftmaxFitResult(w=0.0, t=1.0, loss=float("nan"), n_races=0, n_rows=0)

    work = df.copy()
    work = work[(work["race_id"].notna()) & (work["y"].notna())]
    if work.empty:
        return RaceSoftmaxFitResult(w=0.0, t=1.0, loss=float("nan"), n_races=0, n_rows=0)

    work["y"] = pd.to_numeric(work["y"], errors="coerce").fillna(0).astype(int)
    work["p_mkt"] = pd.to_numeric(work["p_mkt"], errors="coerce").astype(float)
    work["p_model"] = pd.to_numeric(work["p_model"], errors="coerce").astype(float)
    work["p_model"] = work["p_model"].where(np.isfinite(work["p_model"]), work["p_mkt"])
    work["p_mkt"] = work["p_mkt"].where(np.isfinite(work["p_mkt"]), work["p_model"])
    work = work.dropna(subset=["p_mkt", "p_model"])
    if work.empty:
        return RaceSoftmaxFitResult(w=0.0, t=1.0, loss=float("nan"), n_races=0, n_rows=0)

    w_values = np.arange(0.0, 1.0 + 1e-12, float(w_grid_step)) if w_grid_step > 0 else np.array([0.0])
    t_values = [float(t) for t in t_grid if float(t) > 0]
    if not t_values:
        t_values = [1.0]

    best = None
    best_loss = float("inf")
    n_rows = int(len(work))
    n_races = int(work["race_id"].nunique())

    for w in w_values:
        scores = _build_scores(
            work["p_model"].values,
            work["p_mkt"].values,
            float(w),
            score_space,
            float(clip_eps),
        )
        for t in t_values:
            t_val = float(t)
            s = scores / t_val
            score_series = pd.Series(s, index=work.index)

            def _softmax_series(s_local: pd.Series) -> pd.Series:
                v = s_local.to_numpy()
                v = v - np.max(v)
                exp_v = np.exp(v)
                denom = exp_v.sum()
                if denom <= 0:
                    return pd.Series(np.zeros_like(v, dtype=float), index=s_local.index)
                return pd.Series(exp_v / denom, index=s_local.index)

            p_race = score_series.groupby(work["race_id"], dropna=False, sort=False).transform(_softmax_series)
            p_race = _clip(p_race.values, float(clip_eps))

            y = work["y"].values
            mask_win = y == 1
            if not mask_win.any():
                continue
            loss = -float(np.mean(np.log(p_race[mask_win])))
            if loss < best_loss:
                best_loss = loss
                best = (float(w), float(t_val))

    if best is None:
        best = (0.0, 1.0)
        best_loss = float("nan")

    return RaceSoftmaxFitResult(
        w=float(best[0]),
        t=float(best[1]),
        loss=float(best_loss),
        n_races=n_races,
        n_rows=n_rows,
    )
