"""
ベッティングモジュール

Note:
    重いモジュール（SQLAlchemy依存のpolicy等）は明示的にimportしてください。
    例: from keiba.betting.policy import BettingPolicy
"""

# 軽量なsizingのみ直接import可能
from .sizing import calculate_stake, kelly_criterion

__all__ = [
    # sizing（軽量、直接import可）
    "calculate_stake", 
    "kelly_criterion",
    # policy（重い、明示import推奨）
    # "BettingPolicy",
    # "generate_bet_signals",
]

