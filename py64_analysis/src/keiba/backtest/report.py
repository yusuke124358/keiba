"""
バックテストレポート生成
"""
import logging
from pathlib import Path
from datetime import datetime

from jinja2 import Template

# ★重要: matplotlib.use() は pyplot import より前に呼ぶ
#   ヘッドレス環境でバックエンドが効かない問題を防ぐ
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import BacktestResult
from .metrics import BacktestMetrics, calculate_metrics
logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """
# バックテストレポート

生成日時: {{ generated_at }}

## サマリ

| 指標 | 値 |
|------|-----|
| ベット数 | {{ metrics.n_bets }} |
| 勝利数 | {{ metrics.n_wins }} |
| 勝率 | {{ "%.1f"|format(metrics.win_rate * 100) }}% |
| 総賭け金 | ¥{{ "{:,.0f}".format(metrics.total_stake) }} |
| 総払戻 | ¥{{ "{:,.0f}".format(metrics.total_payout) }} |
| 総損益 | ¥{{ "{:,.0f}".format(metrics.total_profit) }} |
| ROI | {{ "%.2f"|format(metrics.roi * 100) }}% |
| Ending bankroll | ¥{{ "{:,.0f}".format(metrics.ending_bankroll) }} |
| Min bankroll | ¥{{ "{:,.0f}".format(metrics.min_bankroll) }} |
| Log growth | {{ "%.6f"|format(metrics.log_growth) if metrics.log_growth is not none else "n/a" }} |
| 最大DD | {{ "%.2f"|format(metrics.max_drawdown * 100) }}% |
| MaxDD (bankroll) | {{ "%.2f"|format(metrics.max_drawdown_bankroll * 100) }}% |
{% if metrics.sharpe_ratio %}| Sharpe | {{ "%.2f"|format(metrics.sharpe_ratio) }} |{% endif %}

{% if pred_quality %}
## 確率品質（{{ pred_quality_period }}）

| 指標 | market(p_mkt) | model(p_model) | blend | calibrated{% if pred_quality.calibration_method %}({{ pred_quality.calibration_method }}){% endif %} |
|------|--------------:|---------------:|------:|-----------------------------------------------:|
| LogLoss | {{ "%.4f"|format(pred_quality.logloss_market) }} | {{ "%.4f"|format(pred_quality.logloss_model) }} | {{ "%.4f"|format(pred_quality.logloss_blend) }} | {{ "%.4f"|format(pred_quality.logloss_calibrated) }} |
| Brier | {{ "%.4f"|format(pred_quality.brier_market) }} | {{ "%.4f"|format(pred_quality.brier_model) }} | {{ "%.4f"|format(pred_quality.brier_blend) }} | {{ "%.4f"|format(pred_quality.brier_calibrated) }} |
| N(頭) | {{ pred_quality.n_samples }} |  |  |  |

{% if reliability_image %}
![校正プロット]({{ reliability_image }})
{% endif %}

{% endif %}

## 資金曲線

![資金曲線](equity_curve.png)

## 月次ROI

{% if metrics.monthly_roi is not none %}
| 月 | ROI |
|-----|-----|
{% for month, roi in monthly_roi_items %}| {{ month }} | {{ "%.2f"|format(roi * 100) }}% |
{% endfor %}
{% endif %}

## 人気帯別ROI

{% if metrics.popularity_roi is not none %}
| オッズ帯 | ベット数 | ROI |
|----------|----------|-----|
{% for row in popularity_roi_items %}| {{ row.odds_band }} | {{ row.n_bets }} | {{ "%.2f"|format(row.roi * 100) }}% |
{% endfor %}
{% endif %}
"""


def generate_report(
    result: BacktestResult,
    output_dir: Path,
    name: str = "backtest",
    pred_quality=None,
    pred_quality_period: str = "",
    reliability_image: Path | str | None = None,
) -> Path:
    """
    レポート生成
    
    Args:
        result: バックテスト結果
        output_dir: 出力ディレクトリ
        name: レポート名
    
    Returns:
        レポートファイルパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    
    # メトリクス計算
    metrics = calculate_metrics(result)
    
    # 資金曲線プロット
    _plot_equity_curve(result, output_dir / "equity_curve.png")
    
    # テンプレートレンダリング
    template = Template(REPORT_TEMPLATE)
    
    # 月次ROIをリストに変換
    monthly_roi_items = []
    if metrics.monthly_roi is not None:
        monthly_roi_items = [(str(k), v) for k, v in metrics.monthly_roi.items()]
    
    # 人気帯別ROIをリストに変換
    popularity_roi_items = []
    if metrics.popularity_roi is not None:
        for idx, row in metrics.popularity_roi.iterrows():
            popularity_roi_items.append({
                "odds_band": str(idx),
                "n_bets": int(row.get("n_bets", 0)),
                "roi": float(row.get("roi", 0)),
            })
    
    # 画像パスを安全に（output_dir配下なら相対化、Windowsパスはposix化）
    reliability_image_str = None
    if reliability_image:
        p = reliability_image if isinstance(reliability_image, Path) else Path(str(reliability_image))
        # 相対パスで渡された場合は output_dir からの相対として解釈する（存在すればそれを優先）
        if not p.is_absolute():
            candidate = (output_dir / p)
            if candidate.exists():
                p = candidate
        if p.is_absolute():
            p = p.resolve()
            try:
                reliability_image_str = p.relative_to(output_dir).as_posix()
            except ValueError:
                reliability_image_str = p.as_posix()
        else:
            reliability_image_str = p.as_posix()

    content = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metrics=metrics,
        monthly_roi_items=monthly_roi_items,
        popularity_roi_items=popularity_roi_items,
        pred_quality=pred_quality,
        pred_quality_period=pred_quality_period,
        reliability_image=reliability_image_str,
    )
    
    # 保存
    report_path = output_dir / f"{name}.md"
    report_path.write_text(content, encoding="utf-8")
    
    logger.info(f"Report saved to {report_path}")
    return report_path


def _plot_equity_curve(result: BacktestResult, output_path: Path) -> None:
    """資金曲線をプロット"""
    if not result.bets:
        return
    
    # 累積損益
    cumulative = [result.initial_bankroll]
    for bet_result in result.bets:
        cumulative.append(cumulative[-1] + bet_result.profit)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative, linewidth=1.5)
    ax.axhline(y=result.initial_bankroll, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Bet #")
    ax.set_ylabel("Bankroll (¥)")
    ax.set_title("Equity Curve")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
