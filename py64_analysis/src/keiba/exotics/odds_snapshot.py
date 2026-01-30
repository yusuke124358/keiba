"""
三連複/三連単の速報オッズスナップショット抽出（購入時点/締切直前）。

ポイント:
  - MAX(asof_time) <= t でスナップショットを選ぶ
  - max_staleness を超える古い値は欠落扱いにできる
  - 欠落時ルール（missing_policy）を固定しておく
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session


TicketType = Literal["trio", "trifecta"]
MissingPolicy = Literal["skip", "allow_stale"]


@dataclass(frozen=True)
class SnapshotSelection:
    ok: bool
    ticket_type: TicketType
    race_id: str
    t_target: datetime
    max_staleness_min: float
    missing_policy: MissingPolicy

    snapshot_id: int | None = None
    asof_time: datetime | None = None
    fetched_at: datetime | None = None
    age_min: float | None = None
    n_rows: int | None = None

    stale: bool = False
    reason: str | None = None  # missing / stale / none


def select_snapshot(
    session: Session,
    *,
    ticket_type: TicketType,
    race_id: str,
    t: datetime,
    max_staleness_min: float = 10.0,
    missing_policy: MissingPolicy = "skip",
) -> SnapshotSelection:
    """
    指定時刻 t に対して、利用する odds_rt_snapshot を選ぶ。
    """
    row = session.execute(
        text(
            """
            SELECT id, asof_time, fetched_at, n_rows
            FROM odds_rt_snapshot
            WHERE ticket_type = :ticket_type
              AND race_id = :race_id
              AND asof_time <= :t
            ORDER BY asof_time DESC, fetched_at DESC
            LIMIT 1
            """
        ),
        {"ticket_type": ticket_type, "race_id": race_id, "t": t},
    ).fetchone()

    if not row:
        return SnapshotSelection(
            ok=False,
            ticket_type=ticket_type,
            race_id=race_id,
            t_target=t,
            max_staleness_min=float(max_staleness_min),
            missing_policy=missing_policy,
            reason="missing",
        )

    snapshot_id = int(row[0])
    asof_time = row[1]
    fetched_at = row[2]
    n_rows = row[3]

    age_min = float((t - asof_time).total_seconds() / 60.0) if t and asof_time else None
    stale = bool(age_min is not None and age_min > float(max_staleness_min))

    if stale and missing_policy == "skip":
        return SnapshotSelection(
            ok=False,
            ticket_type=ticket_type,
            race_id=race_id,
            t_target=t,
            max_staleness_min=float(max_staleness_min),
            missing_policy=missing_policy,
            snapshot_id=snapshot_id,
            asof_time=asof_time,
            fetched_at=fetched_at,
            age_min=age_min,
            n_rows=int(n_rows) if n_rows is not None else None,
            stale=True,
            reason="stale",
        )

    return SnapshotSelection(
        ok=True,
        ticket_type=ticket_type,
        race_id=race_id,
        t_target=t,
        max_staleness_min=float(max_staleness_min),
        missing_policy=missing_policy,
        snapshot_id=snapshot_id,
        asof_time=asof_time,
        fetched_at=fetched_at,
        age_min=age_min,
        n_rows=int(n_rows) if n_rows is not None else None,
        stale=stale,
        reason=None,
    )


def fetch_trio_snapshot_rows(session: Session, snapshot_id: int) -> list[dict]:
    rows = session.execute(
        text(
            """
            SELECT horse_no_1, horse_no_2, horse_no_3, odds, popularity, odds_status
            FROM odds_rt_trio
            WHERE snapshot_id = :snapshot_id
            """
        ),
        {"snapshot_id": snapshot_id},
    ).fetchall()
    return [dict(r._mapping) for r in rows]


def fetch_trifecta_snapshot_rows(session: Session, snapshot_id: int) -> list[dict]:
    rows = session.execute(
        text(
            """
            SELECT first_no, second_no, third_no, odds, popularity, odds_status
            FROM odds_rt_trifecta
            WHERE snapshot_id = :snapshot_id
            """
        ),
        {"snapshot_id": snapshot_id},
    ).fetchall()
    return [dict(r._mapping) for r in rows]


@dataclass(frozen=True)
class SnapshotData:
    selection: SnapshotSelection
    rows: list[dict]


def get_snapshot(
    session: Session,
    *,
    ticket_type: TicketType,
    race_id: str,
    t: datetime,
    max_staleness_min: float = 10.0,
    missing_policy: MissingPolicy = "skip",
) -> Optional[SnapshotData]:
    sel = select_snapshot(
        session,
        ticket_type=ticket_type,
        race_id=race_id,
        t=t,
        max_staleness_min=max_staleness_min,
        missing_policy=missing_policy,
    )
    if not sel.ok or sel.snapshot_id is None:
        return None

    if ticket_type == "trio":
        rows = fetch_trio_snapshot_rows(session, sel.snapshot_id)
    else:
        rows = fetch_trifecta_snapshot_rows(session, sel.snapshot_id)

    return SnapshotData(selection=sel, rows=rows)


