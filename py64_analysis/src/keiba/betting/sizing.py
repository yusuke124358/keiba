"""
賭け金計算（Bet Sizing）

分数ケリー基準による賭け金計算
"""
from typing import Literal


def kelly_criterion(p: float, odds: float) -> float:
    """
    ケリー基準による最適賭け率
    
    f* = (p * odds - 1) / (odds - 1)
       = (p * b - q) / b
    
    where:
        p = 勝利確率
        q = 1 - p
        b = odds - 1 (純オッズ)
    
    Args:
        p: 勝利確率
        odds: オッズ（払戻倍率）
    
    Returns:
        最適賭け率（0-1）、負の場合は0
    """
    if p <= 0 or p >= 1 or odds <= 1:
        return 0
    
    q = 1 - p
    b = odds - 1
    
    f = (p * b - q) / b
    return max(0, f)


def calculate_stake(
    p_hat: float,
    odds: float,
    bankroll: float,
    method: Literal["fractional_kelly", "fixed_pct"] = "fractional_kelly",
    fraction: float = 0.2,
    max_pct: float = 0.01,
    min_stake: int = 100,
) -> int:
    """
    賭け金を計算
    
    Args:
        p_hat: 予測勝利確率
        odds: オッズ
        bankroll: 資金
        method: 計算方法
        fraction: ケリー分数（fractional_kellyの場合）
        max_pct: 最大賭け率
        min_stake: 最小賭け金
    
    Returns:
        賭け金（100円単位）
    """
    if method == "fractional_kelly":
        full_kelly = kelly_criterion(p_hat, odds)
        stake_pct = full_kelly * fraction
    elif method == "fixed_pct":
        stake_pct = fraction
    else:
        stake_pct = 0.01
    
    # 上限適用
    stake_pct = min(stake_pct, max_pct)
    
    # 金額計算
    stake = bankroll * stake_pct
    
    # 100円単位に丸め
    stake = int(stake / 100) * 100
    
    # 最小賭け金
    if stake < min_stake:
        return 0
    
    return stake



