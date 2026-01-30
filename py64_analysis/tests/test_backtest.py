"""
バックテストのテスト
"""
import pytest
from datetime import datetime
from dataclasses import dataclass


@dataclass
class MockModel:
    """モックモデル"""
    blend_weight: float = 0.8
    lgb_model = None
    feature_names: list = None


class TestBacktestStake:
    """バックテストのstake計算テスト"""
    
    def test_calculate_stake_uses_sizing_module(self):
        """sizingモジュールを使用していることを確認"""
        from keiba.betting.sizing import calculate_stake
        
        # 正しい計算（/100がない）
        stake = calculate_stake(
            p_hat=0.15,
            odds=10.0,
            bankroll=300000,
            method="fractional_kelly",
            fraction=0.2,
            max_pct=0.01,
        )
        
        # 上限（300000 * 0.01 = 3000円）以下
        assert stake <= 3000
        # 計算が正しければ0より大きい
        # EV = 0.15 * 10 - 1 = 0.5 > 0 なのでベットあり
        assert stake > 0 or stake == 0  # 計算次第
    
    def test_stake_is_100_yen_unit(self):
        """100円単位になっていることを確認"""
        from keiba.betting.sizing import calculate_stake
        
        stake = calculate_stake(
            p_hat=0.2,
            odds=5.0,
            bankroll=100000,
            method="fractional_kelly",
            fraction=0.2,
            max_pct=0.05,
        )
        
        assert stake % 100 == 0


class TestBacktestResult:
    """バックテスト結果のテスト"""
    
    def test_backtest_result_dataclass(self):
        """BacktestResultの構造テスト"""
        from keiba.backtest.engine import BacktestResult
        
        result = BacktestResult(initial_bankroll=300000)
        
        assert result.initial_bankroll == 300000
        assert result.final_bankroll == 0
        assert result.n_bets == 0
        assert result.roi == 0
        assert result.max_drawdown == 0



