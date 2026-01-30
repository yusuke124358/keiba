"""
Stage B0: オッズ（3連系）不要で回せる確率品質の評価。

ここでは「各馬の勝率（例: 市場p_mkt）」から PL/Harville で top3分布を作り、
実現Top3（順序/集合）に対する NLL と hit@K を出す。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import math
import pandas as pd

from sqlalchemy import text
from sqlalchemy.orm import Session

from .probability import (
    normalize_probs,
    pl_prob_top3_order,
    pl_all_top3_order_probs,
    pl_all_trio_set_probs,
)


@dataclass(frozen=True)
class B0Summary:
    n_races: int
    n_skipped: int
    mean_nll_trifecta: float
    mean_nll_trio: float
    hit1_trifecta: float
    hit10_trifecta: float
    hit1_trio: float
    hit10_trio: float


def _get_buy_time(session: Session, race_id: str, buy_t_minus_minutes: int) -> Optional[datetime]:
    row = session.execute(
        text("SELECT date, start_time FROM fact_race WHERE race_id = :race_id"),
        {"race_id": race_id},
    ).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None
    dt = datetime.combine(row[0], row[1])
    return dt - timedelta(minutes=int(buy_t_minus_minutes))


def _get_actual_top3(session: Session, race_id: str) -> Optional[tuple[int, int, int]]:
    """
    1-3着を取得。同着がある場合（同一posに複数頭）はNoneを返してスキップ扱いにする。
    """
    rows = session.execute(
        text(
            """
            SELECT horse_no, finish_pos
            FROM fact_result
            WHERE race_id = :race_id
              AND finish_pos IN (1,2,3)
            """
        ),
        {"race_id": race_id},
    ).fetchall()
    if not rows:
        return None

    # finish_pos ごとに件数を数え、1,2,3 がそれぞれ「ちょうど1件」のときだけ採用
    from collections import Counter
    pos_counts = Counter(int(r[1]) for r in rows if r[0] is not None and r[1] is not None)
    if pos_counts.get(1, 0) != 1 or pos_counts.get(2, 0) != 1 or pos_counts.get(3, 0) != 1:
        return None

    m = {int(r[1]): int(r[0]) for r in rows if r[0] is not None and r[1] is not None}
    if 1 not in m or 2 not in m or 3 not in m:
        return None
    # 3頭が別馬であることを再確認
    if len({m[1], m[2], m[3]}) != 3:
        return None
    return m[1], m[2], m[3]


def _get_market_probs(session: Session, race_id: str, asof_time: datetime) -> dict[int, float]:
    # まずスナップショット時刻を揃える
    t_snap = session.execute(
        text(
            """
            SELECT MAX(asof_time) AS t_snap
            FROM odds_ts_win
            WHERE race_id = :race_id
              AND asof_time <= :asof_time
              AND odds > 0
            """
        ),
        {"race_id": race_id, "asof_time": asof_time},
    ).scalar()

    if t_snap is None:
        return {}

    rows = session.execute(
        text(
            """
            SELECT DISTINCT ON (horse_no)
              horse_no, odds
            FROM odds_ts_win
            WHERE race_id = :race_id
              AND asof_time = :t_snap
              AND odds > 0
            ORDER BY horse_no, data_kubun DESC
            """
        ),
        {"race_id": race_id, "t_snap": t_snap},
    ).fetchall()

    p: dict[int, float] = {}
    inv_sum = 0.0
    for r in rows:
        h = int(r[0])
        o = float(r[1])
        if o <= 0:
            continue
        inv = 1.0 / o
        inv_sum += inv
        p[h] = inv
    if inv_sum > 0:
        p = {k: v / inv_sum for k, v in p.items()}
    return normalize_probs(p)


def run_b0_eval(
    session: Session,
    *,
    start_date: str,
    end_date: str,
    buy_t_minus_minutes: int,
    hit_k: tuple[int, int] = (1, 10),
) -> tuple[B0Summary, pd.DataFrame]:
    # race_ids
    race_rows = session.execute(
        text(
            """
            SELECT r.race_id
            FROM fact_race r
            WHERE r.date BETWEEN :s AND :e
              AND r.start_time IS NOT NULL
              AND EXISTS (
                SELECT 1 FROM fact_result res
                WHERE res.race_id = r.race_id
                  AND res.finish_pos IS NOT NULL
              )
            ORDER BY r.date, r.race_id
            """
        ),
        {"s": start_date, "e": end_date},
    ).fetchall()
    race_ids = [r[0] for r in race_rows]

    rows_out: list[dict] = []
    skipped = 0
    for race_id in race_ids:
        buy_time = _get_buy_time(session, race_id, buy_t_minus_minutes)
        top3 = _get_actual_top3(session, race_id)
        if buy_time is None or top3 is None:
            skipped += 1
            continue

        p = _get_market_probs(session, race_id, buy_time)
        if len(p) < 3:
            skipped += 1
            continue

        i, j, k = top3
        p_order = pl_prob_top3_order(p, i, j, k)
        p_set = 0.0
        for a, b, c in ((i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)):
            p_set += pl_prob_top3_order(p, a, b, c)

        eps = 1e-15
        nll_trifecta = -math.log(max(p_order, eps))
        nll_trio = -math.log(max(p_set, eps))

        # hit@K
        order_probs = pl_all_top3_order_probs(p)
        set_probs = pl_all_trio_set_probs(p)

        order_sorted = sorted(order_probs.items(), key=lambda kv: kv[1], reverse=True)
        set_sorted = sorted(set_probs.items(), key=lambda kv: kv[1], reverse=True)

        hit1_o = int((i, j, k) == order_sorted[0][0]) if order_sorted else 0
        hit10_o = int(any((i, j, k) == t for (t, _) in order_sorted[: hit_k[1]]))
        key_set = tuple(sorted((i, j, k)))
        hit1_s = int(key_set == set_sorted[0][0]) if set_sorted else 0
        hit10_s = int(any(key_set == t for (t, _) in set_sorted[: hit_k[1]]))

        rows_out.append(
            {
                "race_id": race_id,
                "buy_time": buy_time,
                "top3_first": i,
                "top3_second": j,
                "top3_third": k,
                "p_trifecta": p_order,
                "p_trio": p_set,
                "nll_trifecta": nll_trifecta,
                "nll_trio": nll_trio,
                "hit1_trifecta": hit1_o,
                "hit10_trifecta": hit10_o,
                "hit1_trio": hit1_s,
                "hit10_trio": hit10_s,
                "n_horses": len(p),
            }
        )

    df = pd.DataFrame(rows_out)
    if df.empty:
        summary = B0Summary(
            n_races=0,
            n_skipped=skipped,
            mean_nll_trifecta=float("nan"),
            mean_nll_trio=float("nan"),
            hit1_trifecta=float("nan"),
            hit10_trifecta=float("nan"),
            hit1_trio=float("nan"),
            hit10_trio=float("nan"),
        )
        return summary, df

    summary = B0Summary(
        n_races=int(len(df)),
        n_skipped=int(skipped),
        mean_nll_trifecta=float(df["nll_trifecta"].mean()),
        mean_nll_trio=float(df["nll_trio"].mean()),
        hit1_trifecta=float(df["hit1_trifecta"].mean()),
        hit10_trifecta=float(df["hit10_trifecta"].mean()),
        hit1_trio=float(df["hit1_trio"].mean()),
        hit10_trio=float(df["hit10_trio"].mean()),
    )
    return summary, df


