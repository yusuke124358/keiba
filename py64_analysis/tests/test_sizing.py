"""
賭け金計算のテスト
"""
import pytest


class TestKellyCriterion:
    """ケリー基準のテスト"""
    
    def test_positive_edge(self):
        """正のエッジがある場合"""
        from keiba.betting.sizing import kelly_criterion
        
        # 勝率50%、オッズ3.0倍 → エッジあり
        f = kelly_criterion(p=0.5, odds=3.0)
        assert f > 0
        assert f < 1
    
    def test_no_edge(self):
        """エッジがない場合（期待値=1）"""
        from keiba.betting.sizing import kelly_criterion
        
        # 勝率33.3%、オッズ3.0倍 → エッジなし
        f = kelly_criterion(p=1/3, odds=3.0)
        assert f == pytest.approx(0, abs=0.01)
    
    def test_negative_edge(self):
        """負のエッジの場合"""
        from keiba.betting.sizing import kelly_criterion
        
        # 勝率20%、オッズ3.0倍 → 負のエッジ
        f = kelly_criterion(p=0.2, odds=3.0)
        assert f == 0
    
    def test_invalid_inputs(self):
        """不正な入力"""
        from keiba.betting.sizing import kelly_criterion
        
        assert kelly_criterion(p=0, odds=3.0) == 0
        assert kelly_criterion(p=1, odds=3.0) == 0
        assert kelly_criterion(p=0.5, odds=1) == 0
        assert kelly_criterion(p=0.5, odds=0.5) == 0


class TestCalculateStake:
    """賭け金計算のテスト"""
    
    def test_fractional_kelly(self):
        """分数ケリーの計算"""
        from keiba.betting.sizing import calculate_stake
        
        stake = calculate_stake(
            p_hat=0.5,
            odds=3.0,
            bankroll=100000,
            method="fractional_kelly",
            fraction=0.2,
            max_pct=0.05,
        )
        
        # 100円単位
        assert stake % 100 == 0
        # 上限以下
        assert stake <= 100000 * 0.05
    
    def test_respects_max_pct(self):
        """上限が適用される"""
        from keiba.betting.sizing import calculate_stake
        
        stake = calculate_stake(
            p_hat=0.9,  # 高確率 → 大きな賭け率
            odds=5.0,
            bankroll=100000,
            method="fractional_kelly",
            fraction=1.0,  # フルケリー
            max_pct=0.01,  # 1%上限
        )
        
        assert stake <= 1000  # 100000 * 0.01
    
    def test_min_stake(self):
        """最小賭け金未満は0"""
        from keiba.betting.sizing import calculate_stake
        
        stake = calculate_stake(
            p_hat=0.1,
            odds=2.0,
            bankroll=1000,  # 少額
            method="fractional_kelly",
            fraction=0.1,
            max_pct=0.01,
            min_stake=100,
        )
        
        # 計算結果が100円未満なら0
        assert stake == 0 or stake >= 100



