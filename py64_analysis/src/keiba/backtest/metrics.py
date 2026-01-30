"""
バックテストメトリクス計算
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from .engine import BacktestResult


@dataclass
class BacktestMetrics:
    """バックテストメトリクス"""
    # 基本
    n_bets: int
    n_wins: int
    win_rate: float
    
    # 損益
    total_stake: float
    total_payout: float
    total_profit: float
    roi: float
    
    # リスク
    max_drawdown: float
    min_bankroll: float
    max_drawdown_bankroll: float
    ending_bankroll: float
    log_growth: Optional[float]
    sharpe_ratio: Optional[float]
    
    # 期間別
    monthly_roi: Optional[pd.Series]
    popularity_roi: Optional[pd.DataFrame]


def calculate_metrics(result: BacktestResult) -> BacktestMetrics:
    """メトリクス計算"""
    n_bets = result.n_bets
    n_wins = result.n_wins
    
    win_rate = n_wins / n_bets if n_bets > 0 else 0
    
    # Sharpe Ratio（簡易版）
    if n_bets > 1:
        profits = [b.profit for b in result.bets]
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)
        sharpe = mean_profit / std_profit if std_profit > 0 else 0
    else:
        sharpe = None
    
    # 月次ROI
    monthly_roi = None
    if result.bets:
        df = pd.DataFrame([{
            "date": b.bet.asof_time.date(),
            "stake": b.bet.stake,
            "profit": b.profit,
        } for b in result.bets])
        
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
        monthly = df.groupby("month").agg({
            "stake": "sum",
            "profit": "sum",
        })
        monthly_roi = monthly["profit"] / monthly["stake"]
    
    # 人気帯別ROI
    popularity_roi = None
    if result.bets:
        df = pd.DataFrame([{
            "odds": b.bet.odds_at_buy,
            "stake": b.bet.stake,
            "profit": b.profit,
        } for b in result.bets])
        
        # オッズ帯分類
        bins = [0, 3, 6, 10, 20, 50, float("inf")]
        labels = ["1-3", "3-6", "6-10", "10-20", "20-50", "50+"]
        df["odds_band"] = pd.cut(df["odds"], bins=bins, labels=labels)
        
        by_odds = df.groupby("odds_band", observed=True).agg({
            "stake": "sum",
            "profit": "sum",
        })
        by_odds["roi"] = by_odds["profit"] / by_odds["stake"]
        by_odds["n_bets"] = df.groupby("odds_band", observed=True).size()
        popularity_roi = by_odds
    
    return BacktestMetrics(
        n_bets=n_bets,
        n_wins=n_wins,
        win_rate=win_rate,
        total_stake=result.total_stake,
        total_payout=result.total_payout,
        total_profit=result.total_profit,
        roi=result.roi,
        max_drawdown=result.max_drawdown,
        min_bankroll=result.min_bankroll,
        max_drawdown_bankroll=result.max_drawdown_bankroll,
        ending_bankroll=result.final_bankroll,
        log_growth=result.log_growth,
        sharpe_ratio=sharpe,
        monthly_roi=monthly_roi,
        popularity_roi=popularity_roi,
    )



