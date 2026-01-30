"""
三連複/三連単のバックテスト（B1/C向け）。

前提:
  - 0B35/0B36 のスナップショットが odds_rt_snapshot / odds_rt_trio / odds_rt_trifecta に入っている
  - HR由来の払戻が fact_payout_trio / fact_payout_trifecta に入っている
  - 返還馬番が fact_refund_horse に入っている

注意:
  - 既存の単勝 backtest とは別エンジンとして実装する（コンフリクト回避）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_config
from ..betting.sizing import calculate_stake
from .odds_snapshot import get_snapshot, MissingPolicy, TicketType
from .probability import pl_all_trio_set_probs, pl_all_top3_order_probs, normalize_probs

logger = logging.getLogger(__name__)

OddsMode = Literal["buy", "close"]  # C / B1


@dataclass
class ExoticBet:
    race_id: str
    ticket_type: TicketType
    selection: dict
    stake_yen: int
    odds_at_use: float
    p_hat: float
    ev: float
    snapshot_id: int
    asof_time: datetime
    extra: dict = field(default_factory=dict)


@dataclass
class ExoticBetResult:
    bet: ExoticBet
    result_status: str  # win/lose/refund/skip
    payout_yen: int
    profit_yen: int


@dataclass
class ExoticsBacktestResult:
    bets: list[ExoticBetResult] = field(default_factory=list)
    initial_bankroll: int = 0
    final_bankroll: int = 0
    total_stake: int = 0
    total_payout: int = 0
    total_profit: int = 0
    roi: float = 0.0
    max_drawdown: float = 0.0
    n_bets: int = 0
    n_wins: int = 0
    n_refunds: int = 0
    n_races_skipped_snapshot: int = 0
    n_races_skipped_hr: int = 0


def _get_races(session: Session, start_date: str, end_date: str) -> list[str]:
    cfg = get_config()
    u = cfg.universe
    track_codes = list(u.track_codes or [])
    exclude_race_ids = list(u.exclude_race_ids or [])
    rows = session.execute(
        text(
            """
            SELECT r.race_id
            FROM fact_race r
            WHERE r.date BETWEEN :s AND :e
              AND r.start_time IS NOT NULL
              AND (:track_codes_len = 0 OR r.track_code = ANY(:track_codes))
              AND (:exclude_len = 0 OR NOT (r.race_id = ANY(:exclude_race_ids)))
            ORDER BY r.date, r.race_id
            """
        ),
        {
            "s": start_date,
            "e": end_date,
            "track_codes_len": len(track_codes),
            "track_codes": track_codes,
            "exclude_len": len(exclude_race_ids),
            "exclude_race_ids": exclude_race_ids,
        },
    ).fetchall()
    return [r[0] for r in rows]


def _get_start_time(session: Session, race_id: str) -> Optional[datetime]:
    row = session.execute(
        text("SELECT date, start_time FROM fact_race WHERE race_id = :race_id"),
        {"race_id": race_id},
    ).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None
    return datetime.combine(row[0], row[1])


def _market_probs(session: Session, race_id: str, asof_time: datetime) -> dict[int, float]:
    # t_snap揃え
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

    inv_sum = 0.0
    p: dict[int, float] = {}
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


def _has_hr_for_race(session: Session, race_id: str, ticket_type: TicketType) -> bool:
    if ticket_type == "trio":
        q = "SELECT 1 FROM fact_payout_trio WHERE race_id = :race_id LIMIT 1"
    else:
        q = "SELECT 1 FROM fact_payout_trifecta WHERE race_id = :race_id LIMIT 1"
    r = session.execute(text(q), {"race_id": race_id}).fetchone()
    return bool(r)


def _refund_horses(session: Session, race_id: str) -> set[int]:
    rows = session.execute(
        text("SELECT horse_no FROM fact_refund_horse WHERE race_id = :race_id"),
        {"race_id": race_id},
    ).fetchall()
    return {int(r[0]) for r in rows if r and r[0] is not None}


def _payout_trio(session: Session, race_id: str, a: int, b: int, c: int) -> Optional[int]:
    row = session.execute(
        text(
            """
            SELECT payout_yen
            FROM fact_payout_trio
            WHERE race_id = :race_id
              AND horse_no_1 = :a AND horse_no_2 = :b AND horse_no_3 = :c
            """
        ),
        {"race_id": race_id, "a": a, "b": b, "c": c},
    ).fetchone()
    if not row:
        return None
    return int(row[0]) if row[0] is not None else None


def _payout_trifecta(session: Session, race_id: str, a: int, b: int, c: int) -> Optional[int]:
    row = session.execute(
        text(
            """
            SELECT payout_yen
            FROM fact_payout_trifecta
            WHERE race_id = :race_id
              AND first_no = :a AND second_no = :b AND third_no = :c
            """
        ),
        {"race_id": race_id, "a": a, "b": b, "c": c},
    ).fetchone()
    if not row:
        return None
    return int(row[0]) if row[0] is not None else None


class ExoticsBacktestEngine:
    def __init__(self, session: Session):
        self.session = session
        self.cfg = get_config()

    def run(
        self,
        *,
        start_date: str,
        end_date: str,
        ticket_type: TicketType = "trio",
        odds_mode: OddsMode = "buy",
        initial_bankroll_yen: int | None = None,
        max_bets_per_race: int = 10,
        ev_margin: float | None = None,
        max_staleness_min: float = 10.0,
        missing_policy: MissingPolicy = "skip",
        buy_t_minus_minutes: int | None = None,
    ) -> ExoticsBacktestResult:
        if initial_bankroll_yen is None:
            bankroll = int(self.cfg.betting.bankroll_yen)
        else:
            bankroll = int(initial_bankroll_yen)

        if ev_margin is None:
            ev_margin = float(self.cfg.betting.ev_margin)

        races = _get_races(self.session, start_date, end_date)
        res = ExoticsBacktestResult(initial_bankroll=bankroll)

        peak = bankroll
        dd_max = 0.0

        current_date = None
        daily_stake = 0
        daily_loss = 0
        per_day_cap = int(bankroll * float(self.cfg.betting.caps.per_day_pct))
        max_daily_loss = int(bankroll * float(self.cfg.betting.stop.max_daily_loss_pct))

        for race_id in races:
            start_dt = _get_start_time(self.session, race_id)
            if start_dt is None:
                continue

            # 日次リセット
            race_date = race_id[:8]
            if race_date != current_date:
                current_date = race_date
                daily_stake = 0
                daily_loss = 0

            if daily_loss >= max_daily_loss:
                continue

            # buy_t_minus_minutes が指定されていればそれを使う、なければconfigから
            buy_t_minus = buy_t_minus_minutes if buy_t_minus_minutes is not None else int(self.cfg.backtest.buy_t_minus_minutes)
            buy_time = start_dt - timedelta(minutes=buy_t_minus)
            t_odds = buy_time if odds_mode == "buy" else start_dt

            # HRが無いレースは決済できないのでスキップ
            if not _has_hr_for_race(self.session, race_id, ticket_type):
                res.n_races_skipped_hr += 1
                continue

            snap = get_snapshot(
                self.session,
                ticket_type=ticket_type,
                race_id=race_id,
                t=t_odds,
                max_staleness_min=float(max_staleness_min),
                missing_policy=missing_policy,
            )
            if snap is None:
                res.n_races_skipped_snapshot += 1
                continue

            # 勝率（market）
            p_win = _market_probs(self.session, race_id, buy_time)
            if len(p_win) < 3:
                res.n_races_skipped_snapshot += 1
                continue

            # 確率を一括計算
            if ticket_type == "trio":
                p_map = pl_all_trio_set_probs(p_win)
            else:
                p_map = pl_all_top3_order_probs(p_win)

            # オッズ行から候補生成
            candidates: list[ExoticBet] = []
            for r in snap.rows:
                try:
                    if ticket_type == "trio":
                        key = (int(r["horse_no_1"]), int(r["horse_no_2"]), int(r["horse_no_3"]))
                    else:
                        key = (int(r["first_no"]), int(r["second_no"]), int(r["third_no"]))
                    odds = float(r["odds"]) if r.get("odds") is not None else None
                except Exception:
                    continue

                if odds is None or odds <= 0:
                    continue
                p_hat = float(p_map.get(key, 0.0))
                if p_hat <= 0:
                    continue
                ev = p_hat * odds - 1.0
                if ev < float(ev_margin):
                    continue

                # stake（既存sizing流用）→ 100円単位に丸める
                stake_raw = calculate_stake(
                    p_hat=p_hat,
                    odds=odds,
                    bankroll=float(bankroll),
                    method=self.cfg.betting.sizing.method,
                    fraction=float(self.cfg.betting.sizing.fraction),
                    max_pct=float(self.cfg.betting.caps.per_race_pct),
                )
                # JRA馬券は100円単位なので切り捨て丸め
                stake = int((stake_raw // 100) * 100)
                if stake <= 0:
                    continue

                selection = (
                    {"horse_no_1": key[0], "horse_no_2": key[1], "horse_no_3": key[2]}
                    if ticket_type == "trio"
                    else {"first_no": key[0], "second_no": key[1], "third_no": key[2]}
                )

                candidates.append(
                    ExoticBet(
                        race_id=race_id,
                        ticket_type=ticket_type,
                        selection=selection,
                        stake_yen=int(stake),
                        odds_at_use=odds,
                        p_hat=p_hat,
                        ev=ev,
                        snapshot_id=int(snap.selection.snapshot_id),  # type: ignore[arg-type]
                        asof_time=snap.selection.asof_time or t_odds,
                        extra={
                            "odds_mode": odds_mode,
                            "asof_time": snap.selection.asof_time,
                            "age_min": snap.selection.age_min,
                            "stale": snap.selection.stale,
                            "n_rows": snap.selection.n_rows,
                        },
                    )
                )

            if not candidates:
                continue

            candidates.sort(key=lambda b: b.ev, reverse=True)
            selected = candidates[: int(max_bets_per_race)]

            # 日次予算
            remaining = max(0, per_day_cap - daily_stake)
            picked: list[ExoticBet] = []
            used = 0
            for b in selected:
                if used + b.stake_yen <= remaining:
                    picked.append(b)
                    used += b.stake_yen

            if not picked:
                continue

            refund_set = _refund_horses(self.session, race_id)

            for bet in picked:
                stake = int(bet.stake_yen)
                daily_stake += stake

                # refund
                sel_horses = set(bet.selection.values())
                if refund_set and any(int(h) in refund_set for h in sel_horses):
                    payout = stake
                    profit = 0
                    status = "refund"
                    res.n_refunds += 1
                else:
                    if ticket_type == "trio":
                        a = int(bet.selection["horse_no_1"])
                        b = int(bet.selection["horse_no_2"])
                        c = int(bet.selection["horse_no_3"])
                        payout100 = _payout_trio(self.session, race_id, a, b, c)
                    else:
                        a = int(bet.selection["first_no"])
                        b = int(bet.selection["second_no"])
                        c = int(bet.selection["third_no"])
                        payout100 = _payout_trifecta(self.session, race_id, a, b, c)

                    if payout100 is not None:
                        payout = int((stake // 100) * int(payout100))
                        profit = payout - stake
                        status = "win"
                        res.n_wins += 1
                    else:
                        payout = 0
                        profit = -stake
                        status = "lose"
                        daily_loss += stake

                bankroll += profit
                peak = max(peak, bankroll)
                dd = (peak - bankroll) / peak if peak > 0 else 0.0
                dd_max = max(dd_max, float(dd))

                res.bets.append(
                    ExoticBetResult(
                        bet=bet,
                        result_status=status,
                        payout_yen=int(payout),
                        profit_yen=int(profit),
                    )
                )

        res.final_bankroll = int(bankroll)
        res.total_stake = int(sum(b.bet.stake_yen for b in res.bets))
        res.total_payout = int(sum(b.payout_yen for b in res.bets))
        res.total_profit = int(res.total_payout - res.total_stake)
        res.n_bets = int(len(res.bets))
        res.roi = float(res.total_profit / res.total_stake) if res.total_stake > 0 else 0.0
        res.max_drawdown = float(dd_max)
        return res


