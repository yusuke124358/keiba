"""
バックテストモジュール

Note:
    このモジュールの __init__ は軽量設計です。
    SQLAlchemy/pandas 依存のモジュールは明示的にimportしてください。
    
    例:
        from keiba.backtest.engine import BacktestEngine, run_backtest
        from keiba.backtest.report import generate_report
        from keiba.backtest.metrics import calculate_metrics
"""

# ★軽量設計: engine/metrics/report は SQLAlchemy/pandas 依存があるため
#   ここでは import しない（利用側で明示import）
#   これにより `import keiba.backtest` だけで重い依存が起動しない

__all__ = [
    # すべて明示import推奨
    # "BacktestEngine",
    # "run_backtest", 
    # "generate_report",
    # "calculate_metrics",
]

