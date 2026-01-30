"""
三連複/三連単の確率計算（Harville / Plackett–Luce 近似）。

入力:
  - 各馬の勝率（sum=1 を想定。sum!=1 の場合は正規化する）

出力:
  - 三連単（順序あり）: P(i,j,k)
  - 三連複（集合）: P({a,b,c}) = Σ_{perm} P(perm)
"""

from __future__ import annotations

from itertools import permutations, combinations
from typing import Iterable

import numpy as np


def normalize_probs(p: dict[int, float], *, eps: float = 1e-12) -> dict[int, float]:
    """
    p を非負にクリップして正規化する（和が0なら一様にする）。
    """
    keys = list(p.keys())
    vals = np.array([float(p[k]) for k in keys], dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.maximum(vals, 0.0)
    s = float(vals.sum())
    if s <= eps:
        if not keys:
            return {}
        u = 1.0 / float(len(keys))
        return {k: u for k in keys}
    vals = vals / s
    return {k: float(v) for k, v in zip(keys, vals)}


def pl_prob_top3_order(p: dict[int, float], i: int, j: int, k: int) -> float:
    """
    PL/Harville型の三連単確率（i→j→k）。

    P(i,j,k) = p_i * p_j/(1-p_i) * p_k/(1-p_i-p_j)
    """
    if i == j or j == k or i == k:
        return 0.0
    pi = float(p.get(i, 0.0))
    pj = float(p.get(j, 0.0))
    pk = float(p.get(k, 0.0))
    if pi <= 0 or pj <= 0 or pk <= 0:
        return 0.0
    d1 = 1.0 - pi
    d2 = 1.0 - pi - pj
    if d1 <= 0 or d2 <= 0:
        return 0.0
    return float(pi * (pj / d1) * (pk / d2))


def pl_all_top3_order_probs(p: dict[int, float]) -> dict[tuple[int, int, int], float]:
    """
    全ての (i,j,k) の三連単確率を計算する。
    """
    p2 = normalize_probs(p)
    ks = list(p2.keys())
    out: dict[tuple[int, int, int], float] = {}
    for i, j, k in permutations(ks, 3):
        out[(i, j, k)] = pl_prob_top3_order(p2, i, j, k)
    return out


def pl_all_trio_set_probs(p: dict[int, float]) -> dict[tuple[int, int, int], float]:
    """
    全ての {a,b,c}（昇順タプル）の三連複確率を計算する。
    """
    p2 = normalize_probs(p)
    ks = sorted(p2.keys())
    out: dict[tuple[int, int, int], float] = {}
    for a, b, c in combinations(ks, 3):
        s = 0.0
        for i, j, k in permutations((a, b, c), 3):
            s += pl_prob_top3_order(p2, i, j, k)
        out[(a, b, c)] = float(s)
    return out


def sum_probs(d: dict) -> float:
    return float(sum(float(v) for v in d.values()))


