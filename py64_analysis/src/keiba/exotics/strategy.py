"""
戦略カタログ（YAML）を読み込むための最小定義。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


TicketType = Literal["trio", "trifecta"]


@dataclass(frozen=True)
class ExoticsStrategy:
    name: str
    ticket_type: TicketType = "trio"
    max_bets_per_race: int = 10
    ev_margin: float = 0.10
    max_staleness_min: float = 10.0


def parse_strategies_yaml(obj: Any) -> list[ExoticsStrategy]:
    """
    YAMLのdict/listを ExoticsStrategy の配列に変換する。
    """
    if obj is None:
        return []
    if isinstance(obj, dict) and "strategies" in obj:
        items = obj.get("strategies") or []
    else:
        items = obj
    if not isinstance(items, list):
        return []

    out: list[ExoticsStrategy] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or f"strategy_{i}")
        ticket_type = str(it.get("ticket_type") or "trio")
        if ticket_type not in ("trio", "trifecta"):
            ticket_type = "trio"
        out.append(
            ExoticsStrategy(
                name=name,
                ticket_type=ticket_type,  # type: ignore[arg-type]
                max_bets_per_race=int(it.get("max_bets_per_race", 10)),
                ev_margin=float(it.get("ev_margin", 0.10)),
                max_staleness_min=float(it.get("max_staleness_min", 10.0)),
            )
        )
    return out


