from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import text
from sqlalchemy.orm import Session

QUINELLA_ODDS_MOVEMENT_COLS = [
    "q_min_odds",
    "q_mean_odds",
    "q_snap_age_min",
    "q_min_odds_chg_60m",
    "q_mean_odds_chg_60m",
]


def fetch_quinella_odds_movement_features(
    session: Session,
    race_ids: Iterable[str],
    *,
    buy_t_minus_minutes: int,
    lookback_minutes: int = 60,
) -> pd.DataFrame:
    """
    Fetch 0B42 (odds_ts_quinella) per-horse snapshot + movement features (leak-free).
    """
    race_ids = list(race_ids)
    base_cols = ["race_id", "horse_no"]
    empty = pd.DataFrame(columns=base_cols + QUINELLA_ODDS_MOVEMENT_COLS)
    if not race_ids:
        return empty
    q = text(
        """
        WITH targets AS (
            SELECT
                r.race_id,
                (
                    (r.date::timestamp + r.start_time)
                    - make_interval(mins => :buy_minutes)
                ) AS buy_time,
                (
                    (r.date::timestamp + r.start_time)
                    - make_interval(mins => :buy_minutes)
                    - make_interval(mins => :lookback_minutes)
                ) AS buy_time_lb
            FROM fact_race r
            WHERE r.race_id = ANY(:race_ids)
              AND r.start_time IS NOT NULL
        ),
        snap_t0 AS (
            SELECT t.race_id, MAX(q.asof_time) AS t0_q
            FROM targets t
            JOIN odds_ts_quinella q ON q.race_id = t.race_id
            WHERE q.asof_time <= t.buy_time
              AND q.odds > 0
            GROUP BY t.race_id
        ),
        snap_lb AS (
            SELECT t.race_id, MAX(q.asof_time) AS t_lb_q
            FROM targets t
            JOIN odds_ts_quinella q ON q.race_id = t.race_id
            WHERE q.asof_time <= t.buy_time_lb
              AND q.odds > 0
            GROUP BY t.race_id
        ),
        pairs_t0 AS (
            SELECT DISTINCT ON (q.race_id, q.asof_time, q.horse_no_1, q.horse_no_2)
                q.race_id, q.asof_time, q.horse_no_1, q.horse_no_2, q.odds
            FROM odds_ts_quinella q
            JOIN snap_t0 s ON q.race_id = s.race_id AND q.asof_time = s.t0_q
            WHERE q.odds > 0
            ORDER BY q.race_id, q.asof_time, q.horse_no_1, q.horse_no_2, q.data_kubun DESC
        ),
        pairs_lb AS (
            SELECT DISTINCT ON (q.race_id, q.asof_time, q.horse_no_1, q.horse_no_2)
                q.race_id, q.asof_time, q.horse_no_1, q.horse_no_2, q.odds
            FROM odds_ts_quinella q
            JOIN snap_lb s ON q.race_id = s.race_id AND q.asof_time = s.t_lb_q
            WHERE q.odds > 0
            ORDER BY q.race_id, q.asof_time, q.horse_no_1, q.horse_no_2, q.data_kubun DESC
        ),
        horse_t0 AS (
            SELECT
                race_id,
                horse_no,
                MIN(odds)::float AS q_min_odds,
                AVG(odds)::float AS q_mean_odds
            FROM (
                SELECT race_id, horse_no_1 AS horse_no, odds FROM pairs_t0
                UNION ALL
                SELECT race_id, horse_no_2 AS horse_no, odds FROM pairs_t0
            ) u
            GROUP BY race_id, horse_no
        ),
        horse_lb AS (
            SELECT
                race_id,
                horse_no,
                MIN(odds)::float AS q_min_odds_lb,
                AVG(odds)::float AS q_mean_odds_lb
            FROM (
                SELECT race_id, horse_no_1 AS horse_no, odds FROM pairs_lb
                UNION ALL
                SELECT race_id, horse_no_2 AS horse_no, odds FROM pairs_lb
            ) u
            GROUP BY race_id, horse_no
        )
        SELECT
            t.race_id,
            h.horse_no,
            h.q_min_odds,
            h.q_mean_odds,
            (EXTRACT(EPOCH FROM (t.buy_time - s.t0_q)) / 60.0)::float AS q_snap_age_min,
            CASE
                WHEN hl.q_min_odds_lb IS NOT NULL
                 AND hl.q_min_odds_lb > 0
                 AND h.q_min_odds IS NOT NULL
                THEN (h.q_min_odds / hl.q_min_odds_lb) - 1.0
                ELSE NULL
            END AS q_min_odds_chg_60m,
            CASE
                WHEN hl.q_mean_odds_lb IS NOT NULL
                 AND hl.q_mean_odds_lb > 0
                 AND h.q_mean_odds IS NOT NULL
                THEN (h.q_mean_odds / hl.q_mean_odds_lb) - 1.0
                ELSE NULL
            END AS q_mean_odds_chg_60m
        FROM targets t
        LEFT JOIN snap_t0 s ON t.race_id = s.race_id
        LEFT JOIN horse_t0 h ON t.race_id = h.race_id
        LEFT JOIN horse_lb hl ON t.race_id = hl.race_id AND h.horse_no = hl.horse_no
        """
    )
    try:
        rows = session.execute(
            q,
            {
                "race_ids": race_ids,
                "buy_minutes": int(buy_t_minus_minutes),
                "lookback_minutes": int(lookback_minutes),
            },
        ).fetchall()
    except Exception as exc:
        # Keep the pipeline runnable even if 0B42 is not ingested yet.
        logging.getLogger(__name__).warning(
            "Failed to fetch quinella odds movement features (0B42): %s", exc
        )
        return empty
    if not rows:
        return empty
    df = pd.DataFrame([dict(r._mapping) for r in rows])
    if "horse_no" in df.columns:
        df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    return df
