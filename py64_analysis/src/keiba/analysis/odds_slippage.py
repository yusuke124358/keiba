"""
Ticket1: 「買い時点オッズ → 締切(確定)オッズ」のズレを定量化するユーティリティ。

目的:
  - parimutuel（パリミューチュアル）では購入時点のオッズで払戻が確定しないため、
    EV判定が恒常的に楽観になりがち。
  - odds_ts_win で「買い時点スナップショット」を作り、fact_result.odds（確定）との比を集計する。

出力例:
  - ratio_final_to_buy の分位点（中央値、30%点など）
  - snap_age_min（buy_time - t_snap）の分布
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_config


@dataclass(frozen=True)
class OddsFinalToBuySummary:
    n: int
    median: float
    p30: float
    p10: float
    p50: float
    p90: float
    mean: float


def fetch_odds_final_to_buy(
    session: Session,
    start_date: str,
    end_date: str,
    buy_t_minus_minutes: int,
) -> pd.DataFrame:
    """
    指定期間の各馬について、買い時点の最新スナップショットオッズと確定オッズの比を返す。

    Returns columns:
      - race_id, horse_no
      - buy_time, t_snap, snap_age_min
      - odds_buy, odds_final, ratio_final_to_buy
    """
    # ★重要: FeatureBuilder/Backtest と同じ「レース単位の t_snap」で揃える（スナップショット混在を防ぐ）
    # さらに、同一 (race_id, horse_no, t_snap) に複数 data_kubun がある場合は
    # data_kubun DESC で決定性を担保する。
    cfg = get_config()
    u = cfg.universe
    track_codes = list(u.track_codes or [])
    exclude_race_ids = list(u.exclude_race_ids or [])
    q = text(
        """
        WITH bt AS (
          SELECT
            r.race_id,
            ((r.date::timestamp + r.start_time) - make_interval(mins => :buy_minutes)) AS buy_time
          FROM fact_race r
          WHERE r.date BETWEEN :d1 AND :d2
            AND r.start_time IS NOT NULL
            AND (:track_codes_len = 0 OR r.track_code = ANY(:track_codes))
            AND (:exclude_len = 0 OR NOT (r.race_id = ANY(:exclude_race_ids)))
        ),
        snap_t AS (
          SELECT
            o.race_id,
            MAX(o.asof_time) AS t_snap
          FROM odds_ts_win o
          JOIN bt ON o.race_id = bt.race_id
          WHERE o.asof_time <= bt.buy_time
            AND o.odds IS NOT NULL
            AND o.odds > 0
          GROUP BY o.race_id
        ),
        snap AS (
          SELECT DISTINCT ON (o.race_id, o.horse_no)
            o.race_id,
            o.horse_no,
            st.t_snap,
            o.odds::float AS odds_buy
          FROM odds_ts_win o
          JOIN snap_t st
            ON o.race_id = st.race_id
           AND o.asof_time = st.t_snap
          WHERE o.odds IS NOT NULL
            AND o.odds > 0
          ORDER BY o.race_id, o.horse_no, o.data_kubun DESC
        )
        SELECT
          s.race_id,
          s.horse_no,
          bt.buy_time,
          s.t_snap,
          EXTRACT(EPOCH FROM (bt.buy_time - s.t_snap)) / 60.0 AS snap_age_min,
          s.odds_buy,
          res.odds::float AS odds_final,
          (res.odds::float / s.odds_buy) AS ratio_final_to_buy
        FROM snap s
        JOIN bt ON bt.race_id = s.race_id
        JOIN fact_result res
          ON res.race_id = s.race_id
         AND res.horse_no = s.horse_no
         AND res.odds IS NOT NULL
         AND res.odds > 0
        """
    )

    rows = session.execute(
        q,
        {
            "d1": start_date,
            "d2": end_date,
            "buy_minutes": buy_t_minus_minutes,
            "track_codes_len": len(track_codes),
            "track_codes": track_codes,
            "exclude_len": len(exclude_race_ids),
            "exclude_race_ids": exclude_race_ids,
        },
    ).fetchall()

    df = pd.DataFrame([dict(r._mapping) for r in rows])
    if df.empty:
        return df

    # 念のため数値化
    df["odds_buy"] = pd.to_numeric(df["odds_buy"], errors="coerce")
    df["odds_final"] = pd.to_numeric(df["odds_final"], errors="coerce")
    df["ratio_final_to_buy"] = pd.to_numeric(df["ratio_final_to_buy"], errors="coerce")
    df["snap_age_min"] = pd.to_numeric(df["snap_age_min"], errors="coerce")

    df = df.dropna(subset=["odds_buy", "odds_final", "ratio_final_to_buy"])
    df = df[df["odds_buy"] > 0]
    df = df[df["odds_final"] > 0]
    df = df[df["ratio_final_to_buy"] > 0]
    return df


def summarize_ratio_final_to_buy(
    df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.10, 0.30, 0.50, 0.90),
) -> OddsFinalToBuySummary:
    """ratio_final_to_buy の要約統計を返す。"""
    if df is None or df.empty:
        return OddsFinalToBuySummary(
            n=0,
            median=float("nan"),
            p30=float("nan"),
            p10=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            mean=float("nan"),
        )

    s = pd.to_numeric(df["ratio_final_to_buy"], errors="coerce").dropna()
    if s.empty:
        return OddsFinalToBuySummary(
            n=0,
            median=float("nan"),
            p30=float("nan"),
            p10=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            mean=float("nan"),
        )

    q = s.quantile(list(quantiles)).to_dict()
    p10 = float(q.get(0.10, np.nan))
    p30 = float(q.get(0.30, np.nan))
    p50 = float(q.get(0.50, np.nan))
    p90 = float(q.get(0.90, np.nan))
    return OddsFinalToBuySummary(
        n=int(s.shape[0]),
        median=p50,
        p30=p30,
        p10=p10,
        p50=p50,
        p90=p90,
        mean=float(s.mean()),
    )


def recommend_closing_odds_multiplier(
    df: pd.DataFrame,
    quantile: float = 0.30,
    clip: Optional[tuple[float, float]] = (0.5, 1.2),
) -> float:
    """
    closing_odds_multiplier の推奨値を返す（デフォルト: 30%点）。

    - quantile を低めにすると保守的（EVが減り、bet数が減りやすい）
    - clip は異常値対策（None で無効）
    """
    if df is None or df.empty:
        return 1.0

    s = pd.to_numeric(df["ratio_final_to_buy"], errors="coerce").dropna()
    if s.empty:
        return 1.0

    v = float(s.quantile(quantile))
    if clip is not None:
        lo, hi = clip
        v = min(max(v, lo), hi)
    if not np.isfinite(v) or v <= 0:
        return 1.0
    return v


