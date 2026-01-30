from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _safe_edges(p: pd.Series, n_bins: int) -> tuple[list[float], list[int]]:
    v = pd.to_numeric(p, errors="coerce").clip(0.0, 1.0)
    v = v[np.isfinite(v)]
    if v.empty:
        return [0.0, 1.0], [0]

    try:
        bins = pd.qcut(v, q=int(n_bins), duplicates="drop")
    except ValueError:
        return [0.0, 1.0], [int(len(v))]

    if not hasattr(bins, "cat") or bins.cat.categories is None:
        return [0.0, 1.0], [int(len(v))]

    intervals = list(bins.cat.categories)
    if not intervals:
        return [0.0, 1.0], [int(len(v))]

    edges = [float(intervals[0].left)]
    for itv in intervals:
        edges.append(float(itv.right))

    counts = bins.value_counts().sort_index().tolist()
    if len(edges) != len(counts) + 1:
        return [0.0, 1.0], [int(len(v))]

    edges[0] = 0.0
    edges[-1] = 1.0
    return edges, [int(c) for c in counts]


def _bin_index(p: float, edges: list[float]) -> int:
    x = float(p)
    if not np.isfinite(x):
        return 0
    if x <= edges[0]:
        return 0
    if x >= edges[-1]:
        return len(edges) - 2
    idx = int(np.searchsorted(np.asarray(edges), x, side="right") - 1)
    return max(0, min(len(edges) - 2, idx))


@dataclass(frozen=True)
class UncertaintyShrink:
    n_bins: int
    bin_edges: list[float]
    bin_counts: list[int]
    n0: float
    min_mult: float

    @classmethod
    def fit_from_p_cal(
        cls,
        p_cal: pd.Series | np.ndarray,
        *,
        n_bins: int,
        n0: float,
        min_mult: float,
    ) -> "UncertaintyShrink":
        series = pd.Series(p_cal)
        edges, counts = _safe_edges(series, int(n_bins))
        return cls(
            n_bins=int(len(counts)),
            bin_edges=[float(x) for x in edges],
            bin_counts=[int(x) for x in counts],
            n0=float(n0),
            min_mult=float(min_mult),
        )

    def apply(self, p_cal: float) -> tuple[float, dict[str, Any]]:
        idx = _bin_index(float(p_cal), self.bin_edges)
        n_bin = int(self.bin_counts[idx]) if idx < len(self.bin_counts) else 0
        if n_bin <= 0:
            mult = 0.0
        else:
            mult = float(np.sqrt(n_bin / (n_bin + float(self.n0))))
        mult = float(min(1.0, max(float(self.min_mult), mult)))
        meta = {
            "bin_idx": int(idx),
            "n_bin": int(n_bin),
            "n0": float(self.n0),
            "min_mult": float(self.min_mult),
        }
        return mult, meta

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bins": int(self.n_bins),
            "bin_edges": [float(x) for x in self.bin_edges],
            "bin_counts": [int(x) for x in self.bin_counts],
            "n0": float(self.n0),
            "min_mult": float(self.min_mult),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UncertaintyShrink":
        return cls(
            n_bins=int(d.get("n_bins", 0)),
            bin_edges=[float(x) for x in (d.get("bin_edges") or [0.0, 1.0])],
            bin_counts=[int(x) for x in (d.get("bin_counts") or [0])],
            n0=float(d.get("n0", 2000)),
            min_mult=float(d.get("min_mult", 0.2)),
        )
