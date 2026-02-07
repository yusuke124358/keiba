from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def audit_numeric_signal(
    values: Sequence[object],
    *,
    min_coverage: float = 0.95,
) -> tuple[dict[str, float], bool]:
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError("min_coverage must be in [0, 1]")

    n_total = len(values)
    nonnull: list[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        nonnull.append(float(v))

    n_nonnull = len(nonnull)
    coverage = (n_nonnull / n_total) if n_total else 0.0
    stddev = statistics.pstdev(nonnull) if n_nonnull else 0.0

    metrics = {
        "feature.coverage": float(coverage),
        "feature.stddev": float(stddev),
    }
    ok = (coverage >= min_coverage) and (stddev > 0.0)
    return metrics, ok
