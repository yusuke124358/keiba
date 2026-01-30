from __future__ import annotations

import math
from typing import Optional


def _round_down(amount: float, unit: int) -> int:
    if unit <= 0:
        return int(amount) if amount > 0 else 0
    if amount <= 0:
        return 0
    return int(amount // unit) * unit


def clip_stake(
    stake_raw: int,
    bankroll: float,
    *,
    min_yen: int,
    max_frac_per_bet: Optional[float],
    max_yen_per_bet: Optional[int],
    remaining_race_budget: Optional[float],
    remaining_daily_budget: Optional[float],
) -> tuple[int, Optional[str]]:
    reasons: list[str] = []
    if stake_raw <= 0 or bankroll <= 0:
        return 0, None

    stake = int(stake_raw)
    bankroll_cap = _round_down(float(bankroll), min_yen)
    if bankroll_cap <= 0:
        return 0, None

    if max_frac_per_bet is not None and math.isfinite(float(max_frac_per_bet)):
        max_bet = float(bankroll) * float(max_frac_per_bet)
        if max_yen_per_bet is not None:
            max_bet = min(max_bet, float(max_yen_per_bet))
        max_bet = _round_down(max_bet, min_yen)
        if max_bet <= 0:
            return 0, "per_bet"
        if stake > max_bet:
            stake = max_bet
            reasons.append("per_bet")

    if stake > bankroll_cap:
        stake = bankroll_cap
        if "per_bet" not in reasons:
            reasons.append("per_bet")

    if remaining_race_budget is not None:
        cap = _round_down(float(remaining_race_budget), min_yen)
        if cap <= 0:
            return 0, "per_race"
        if stake > cap:
            stake = cap
            reasons.append("per_race")

    if remaining_daily_budget is not None:
        cap = _round_down(float(remaining_daily_budget), min_yen)
        if cap <= 0:
            return 0, "per_day"
        if stake > cap:
            stake = cap
            reasons.append("per_day")

    if stake < int(min_yen):
        return 0, ",".join(reasons) if reasons else None

    return stake, ",".join(reasons) if reasons else None
