from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from .build_features import FeatureBuilder


def audit_feature_coverage(
    session: Session,
    race_ids: Iterable[str],
    feature_name: str,
    *,
    buy_t_minus_minutes: int,
) -> dict[str, Any]:
    race_ids_list = [str(rid) for rid in race_ids if rid]
    if not race_ids_list:
        return {
            "feature": feature_name,
            "total": 0,
            "nonnull": 0,
            "coverage": 0.0,
            "stddev": None,
        }

    q = text(
        """
        WITH bt AS (
            SELECT
                r.race_id,
                ((r.date::timestamp + r.start_time) - make_interval(mins => :buy_minutes)) AS buy_time
            FROM fact_race r
            WHERE r.race_id = ANY(:race_ids)
              AND r.start_time IS NOT NULL
        )
        SELECT f.payload
        FROM features f
        JOIN bt
          ON f.race_id = bt.race_id
         AND f.asof_time = bt.buy_time
        WHERE f.feature_version = :feature_version
        """
    )
    rows = session.execute(
        q,
        {
            "race_ids": race_ids_list,
            "buy_minutes": int(buy_t_minus_minutes),
            "feature_version": FeatureBuilder.VERSION,
        },
    ).fetchall()

    total = 0
    nonnull = 0
    values: list[float] = []
    for r in rows:
        payload = r[0] if not hasattr(r, "_mapping") else r._mapping.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        total += 1
        value = payload.get(feature_name)
        try:
            value = float(value) if value is not None and np.isfinite(value) else None
        except Exception:
            value = None
        if value is None:
            continue
        nonnull += 1
        values.append(value)

    coverage = float(nonnull) / float(total) if total else 0.0
    stddev = float(np.std(values, ddof=0)) if values else None
    return {
        "feature": feature_name,
        "total": int(total),
        "nonnull": int(nonnull),
        "coverage": coverage,
        "stddev": stddev,
    }
