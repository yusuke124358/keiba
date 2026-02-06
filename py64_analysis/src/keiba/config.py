"""
設定管理

config/config.yaml を読み込み、Pydanticで型安全に管理する

パス解決ルール:
    1. 環境変数 KEIBA_CONFIG_PATH があればそれを使用
    2. なければ cwd から親ディレクトリを遡って config/config.yaml を探索
    3. 見つからなければデフォルト設定を使用
"""

import os
from datetime import date
from pathlib import Path
from typing import Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


def find_project_root() -> Optional[Path]:
    """
    プロジェクトルート（keiba/）を探す

    config/config.yaml が存在するディレクトリをプロジェクトルートとみなす
    """
    # 環境変数優先
    env_path = os.environ.get("KEIBA_PROJECT_ROOT")
    if env_path:
        root = Path(env_path)
        if root.exists():
            return root

    # カレントディレクトリから親へ遡って探索
    current = Path.cwd()
    for _ in range(10):  # 最大10階層まで
        config_path = current / "config" / "config.yaml"
        if config_path.exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    # このファイルの位置から推測（py64_analysis/src/keiba/config.py）
    module_path = Path(__file__).resolve()
    for _ in range(10):
        config_path = module_path / "config" / "config.yaml"
        if config_path.exists():
            return module_path
        if module_path.parent == module_path:
            break
        module_path = module_path.parent

    return None


# プロジェクトルート（起動時に一度だけ解決）
PROJECT_ROOT: Optional[Path] = find_project_root()


class PayoutRateDefault(BaseModel):
    """デフォルト払戻率"""

    win: float = 0.80
    place: float = 0.80
    bracket_quinella: float = 0.775
    quinella: float = 0.775
    wide: float = 0.775
    exacta: float = 0.75
    trifecta_box: float = 0.75
    trifecta: float = 0.725
    win5: float = 0.70


class PayoutRateOverride(BaseModel):
    """特定日の払戻率上書き"""

    dates: list[str]
    all_types: Optional[float] = None
    win: Optional[float] = None
    place: Optional[float] = None


class PayoutRateConfig(BaseModel):
    """払戻率設定"""

    default: PayoutRateDefault = Field(default_factory=PayoutRateDefault)
    overrides: list[PayoutRateOverride] = Field(default_factory=list)

    def get_rate(self, ticket_type: str, race_date: date) -> float:
        """指定日・券種の払戻率を取得"""
        date_str = race_date.strftime("%Y-%m-%d")

        # 上書きチェック
        for override in self.overrides:
            if date_str in override.dates:
                if override.all_types is not None:
                    return override.all_types
                rate = getattr(override, ticket_type, None)
                if rate is not None:
                    return rate

        # デフォルト
        return getattr(self.default, ticket_type, 0.80)


class DatabaseConfig(BaseModel):
    """データベース設定"""

    url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/keiba"


class OddsTimeseriesConfig(BaseModel):
    """時系列オッズ設定"""

    supported_types: list[str] = Field(default=["win", "place", "bracket_quinella", "quinella"])
    interval_minutes: list[int] = Field(default=[5, 10])
    include_total_sales: bool = True


class SlippageConfig(BaseModel):
    """スリッページ設定"""

    enabled_if_no_ts_odds: bool = True
    odds_multiplier: float = 0.95


class BacktestConfig(BaseModel):
    """バックテスト設定"""

    buy_t_minus_minutes: int = 10
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)


class SizingConfig(BaseModel):
    """賭け金計算設定"""

    method: str = "fractional_kelly"
    fraction: float = 0.2


class CapsConfig(BaseModel):
    """上限設定"""

    per_race_pct: float = 0.01
    per_day_pct: float = 0.03


class StakeConfig(BaseModel):
    """Stake clip 設定"""

    enabled: bool = False
    max_frac_per_bet: Optional[float] = Field(default=None, gt=0)
    max_frac_per_race: Optional[float] = Field(default=None, gt=0)
    max_frac_per_day: Optional[float] = Field(default=None, gt=0)
    max_yen_per_bet: Optional[int] = Field(default=None, gt=0)
    min_yen: int = 100


class UncertaintyConfig(BaseModel):
    """Stake uncertainty 設定"""

    enabled: bool = False
    n_bins: int = 10
    n0: int = 2000
    min_mult: float = 0.2


class StopConfig(BaseModel):
    """停止条件"""

    max_daily_loss_pct: float = 0.03


class SlippageTableConfig(BaseModel):
    """
    TicketB: 馬別スリッページ補正（テーブル型）

    目的:
      - r = odds_final / odds_buy の分位点（例: q=0.30）を
        odds帯×TSボラ（×snap_age）で見積もり、
        odds_effective = odds_buy * r_hat としてEV/賭け金計算に使う。

    注意:
      - 窓ごとに推定する想定（train〜validまで。testは参照しない）
      - 有効化すると closing_odds_multiplier は原則 1.0 に固定して二重計上を避ける
    """

    enabled: bool = False
    # r の分位点（保守度）。0.30 がデフォルト（従来の closing_mult 推定と整合）
    quantile: float = Field(default=0.30, ge=0.0, le=1.0)
    # bin内の最小サンプル数。満たさない場合はフォールバックを使う
    min_count: int = Field(default=200, ge=0)
    # odds帯（下限の列挙）。最後は +inf に自動拡張される
    odds_bins: list[float] = Field(default_factory=lambda: [1.0, 3.0, 5.0, 10.0, 20.0])
    # TSボラ帯（分位点）。例: [0.5, 0.8] -> 3分割
    ts_vol_quantiles: list[float] = Field(default_factory=lambda: [0.5, 0.8])
    # snap_age をbinに含めるか（featuresに snap_age_min が入っている前提）
    use_snap_age: bool = False
    # snap_age帯（分）。最後は +inf に自動拡張される
    snap_age_bins: list[float] = Field(default_factory=lambda: [0.0, 1.0, 3.0, 5.0])


class OddsBandBiasConfig(BaseModel):
    """
    Ticket N2: Favorite–Longshot bias 対策（odds帯ごとの確率補正）

    方針（v1）:
      - odds帯ごとに k_band = win_rate / mean(p_cal) を train期間のみで推定
      - 推論: p_adj = clip(p_cal * k_band, 0, 1)
      - EV判定・stake計算は p_adj を使う
    """

    enabled: bool = False
    # odds帯（下限の列挙）。最後は +inf に自動拡張される
    odds_bins: list[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    # bin内の最小サンプル数（満たさないbinはglobal係数にフォールバック）
    min_count: int = Field(default=200, ge=0)
    # Post-N3 / Ticket N2’:
    # 係数の過学習を抑えるためのShrink（n/(n+lambda)）。0なら無効（従来どおり）。
    lambda_shrink: int = Field(default=0, ge=0)
    # Post-N3 / Ticket N2’:
    # oddsが大きいほど k が増えない（= non-increasing）制約をかける
    enforce_monotone: bool = False
    # 係数の暴れ防止（kをclip）
    k_clip: tuple[float, float] = (0.50, 1.10)


class BlendSegmentedConfig(BaseModel):
    """segment別ブレンド設定"""

    enabled: bool = False
    segment_by: str = "surface"
    min_count: int = 200
    n0: int = 500
    w_min: float = 0.0
    w_max: float = 1.0
    grid_step: float = 0.02
    loss: str = "logloss"


class BlendConfig(BaseModel):
    """ブレンド設定"""

    segmented: BlendSegmentedConfig = Field(default_factory=BlendSegmentedConfig)


class RaceSoftmaxFitConfig(BaseModel):
    """race softmax fit 設定"""

    enabled: bool = True
    w_grid_step: float = 0.02
    t_grid: list[float] = Field(
        default_factory=lambda: [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0]
    )
    loss: str = "race_logloss"


class RaceSoftmaxApplyConfig(BaseModel):
    """race softmax apply 設定"""

    w_default: float = 0.2
    t_default: float = 1.0


class RaceSoftmaxSelectorConfig(BaseModel):
    """race softmax selector 設定 (valid-based)"""

    enabled: bool = False
    metric: str = "valid_roi"
    min_valid_bets: int = 30
    min_valid_bets_ratio: float = 0.50
    min_delta_valid_roi: float = 0.02
    min_delta_valid_logloss: float = 0.001
    fallback: str = "baseline"
    save_decision: bool = True


class RaceSoftmaxConfig(BaseModel):
    """race-level softmax 設定"""

    enabled: bool = False
    score_space: str = "logit"
    clip_eps: float = 1e-6
    selector: RaceSoftmaxSelectorConfig = Field(default_factory=RaceSoftmaxSelectorConfig)
    fit: RaceSoftmaxFitConfig = Field(default_factory=RaceSoftmaxFitConfig)
    apply: RaceSoftmaxApplyConfig = Field(default_factory=RaceSoftmaxApplyConfig)


class DailyTopNSelectionConfig(BaseModel):
    """Daily top-N selection config."""

    enabled: bool = False
    n: int = 5
    metric: str = "ev"
    min_value: Optional[float] = None
    min_ev: Optional[float] = None
    scope: str = "date"
    tie_break: str = "score_desc_then_odds_asc"


class SelectionConfig(BaseModel):
    """Bet selection mode config."""

    mode: str = "ev_threshold"
    daily_top_n: DailyTopNSelectionConfig = Field(default_factory=DailyTopNSelectionConfig)


class RaceCostFilterSelectorConfig(BaseModel):
    """Race cost cap selector config (valid-only)."""

    enabled: bool = False

    @staticmethod
    def _default_candidate_caps() -> list[Optional[float]]:
        # Keep the default as a list containing a single "no-cap" candidate.
        return [None]

    candidate_caps: list[Optional[float]] = Field(default_factory=_default_candidate_caps)
    min_valid_n_bets: int = 80
    min_valid_bets_ratio: float = 0.0
    min_delta_valid_roi: float = 0.02
    min_delta_valid_roi_shrunk: Optional[float] = None
    n0: float = 0.0
    tie_breaker: str = "prefer_baseline"
    selected_cap_value: Optional[float] = None
    selected_from: Optional[str] = None


class RaceCostFilterConfig(BaseModel):
    """Race-level cost filter (overround / takeout)."""

    enabled: bool = False
    metric: str = "takeout_implied"
    cap_mode: str = "fixed"
    cap_fit_scope: str = "train"
    cap_fit_population: str = "all_races"
    train_quantile: float = 0.85
    max_takeout_implied: Optional[float] = None
    max_overround_sum_inv: Optional[float] = None
    reject_if_missing: bool = True
    selector: RaceCostFilterSelectorConfig = Field(default_factory=RaceCostFilterSelectorConfig)
    scope: str = "race"


class TakeoutEvMarginConfig(BaseModel):
    """Takeout-aware EV margin config (soft filter)."""

    enabled: bool = False
    ref_takeout: float = 0.215
    slope: float = 0.0


class StakeOddsDampConfig(BaseModel):
    """Stake damping by odds (post-selection)."""

    enabled: bool = False
    ref_odds: float = Field(default=0.0, ge=0.0)
    power: float = Field(default=1.0, gt=0.0)
    min_mult: float = Field(default=0.0, ge=0.0)


class BettingConfig(BaseModel):
    """ベッティング設定"""

    ev_margin: float = 0.10
    enable_market_blend: bool = False
    market_prob_method: str = "p_mkt_col"
    t_ev: float = 0.10
    odds_cap: Optional[float] = Field(default=None, gt=0)
    exclude_odds_band: Optional[str] = None
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    race_cost_filter: RaceCostFilterConfig = Field(default_factory=RaceCostFilterConfig)
    takeout_ev_margin: TakeoutEvMarginConfig = Field(default_factory=TakeoutEvMarginConfig)
    # Ticket1: 買い時点オッズ→締切オッズのズレを保守的に見積もる倍率
    # 例: 0.97 なら「締切でオッズが平均3%悪化する」想定でEV計算・stake計算に使う
    closing_odds_multiplier: float = Field(default=1.0, gt=0)
    # Ticket G2: EV上側カット（過大EVによるdecile9/10壊滅を抑止）
    ev_upper_cap: dict = Field(
        default_factory=lambda: {
            "enabled": False,
            "quantile": 0.95,  # train期間のbet候補EV分布のquantile（例: 0.90, 0.95）
        }
    )
    # Ticket N6: overlay shrink (logit residual) for p_hat toward p_mkt
    # alpha=1.0 keeps p_hat, alpha=0.0 uses p_mkt
    overlay_shrink_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # TicketB: 馬別スリッページ補正（テーブル型）。enabled=True の場合はこちらが優先。
    slippage_table: SlippageTableConfig = Field(default_factory=SlippageTableConfig)
    # Ticket N2: odds帯バイアス補正（favorite–longshot bias）
    odds_band_bias: OddsBandBiasConfig = Field(default_factory=OddsBandBiasConfig)
    # Step4: TSボラ上限（log(odds) std in last 60m）による回避フィルタ
    # None の場合は無効（baselineはこれで固定できる）
    log_odds_std_60m_max: Optional[float] = Field(default=None, gt=0)
    # 上の閾値が有効なとき、特徴量が欠損（None/NaN）の場合の扱い
    # 推奨: True（欠損は弾く）
    reject_if_log_odds_std_60m_missing: bool = True
    # 勝ち筋探索用: ベット対象のオッズ帯を制限（Noneなら制限なし）
    min_odds: Optional[float] = Field(default=None, gt=0)
    max_odds: Optional[float] = Field(default=None, gt=0)
    # ???????E????????odds_at_buy?
    min_buy_odds: Optional[float] = Field(default=None, gt=0)
    max_buy_odds: Optional[float] = Field(default=None, gt=0)
    odds_floor_min_odds: float = Field(default=0.0, ge=0.0)
    stake_odds_damp: StakeOddsDampConfig = Field(default_factory=StakeOddsDampConfig)
    # EV/overlay tail cap (post-hoc quantile filter)
    ev_cap_quantile: Optional[float] = Field(default=None, gt=0, lt=1)
    overlay_abs_cap_quantile: Optional[float] = Field(default=None, gt=0, lt=1)
    reject_if_overlay_missing: bool = True
    # Odds dynamics filter (train/valid-fitted quantile threshold)
    odds_dynamics_filter: "OddsDynamicsFilterConfig" = Field(
        default_factory=lambda: OddsDynamicsFilterConfig()
    )
    # Odds dynamics EV margin (post-cap soft filter)
    odds_dyn_ev_margin: "OddsDynamicsEvMarginConfig" = Field(
        default_factory=lambda: OddsDynamicsEvMarginConfig()
    )
    max_bets_per_race: int = 1
    bankroll_yen: int = 300000
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    stake: StakeConfig = Field(default_factory=StakeConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    caps: CapsConfig = Field(default_factory=CapsConfig)
    stop: StopConfig = Field(default_factory=StopConfig)


class ModelConfig(BaseModel):
    """モデル設定"""

    type: str = "lightgbm"
    blend_weight_w: float = 0.8
    blend: BlendConfig = Field(default_factory=BlendConfig)
    race_softmax: RaceSoftmaxConfig = Field(default_factory=RaceSoftmaxConfig)
    calibration: str = "isotonic"
    market_prob_mode: str = "raw"
    # rolling比較の再現性向上（LightGBMの乱数seed）
    seed: int = 42
    # Option C0: 市場確率 p_mkt を prior として固定し、残差（logit）だけを学習する
    use_market_offset: bool = False
    # p_mkt の clip（logit計算の発散防止）
    p_mkt_clip: tuple[float, float] = (1e-4, 1.0 - 1e-4)
    # Ticket G1: 市場からの乖離（logit残差）を上限制限してdecile10壊滅を潰す
    residual_cap: dict = Field(
        default_factory=lambda: {
            "enabled": False,
            "quantile": 0.99,
            "p_clip": [1e-4, 1.0 - 1e-4],
            "apply_stage": "pre_calibration",  # pre_calibration or post_calibration
        }
    )
    # EXP-018: train per-distance bucket models + route inference by distance
    distance_bucket_models: dict = Field(
        default_factory=lambda: {
            "enabled": False,
            "min_train_samples": 5000,
            "min_valid_samples": 2000,
        }
    )


class OddsDynamicsFilterConfig(BaseModel):
    """Odds dynamics post-cap filter config."""

    enabled: bool = False
    lookback_minutes: int = 5
    metric: str = "odds_delta_log"  # odds_delta_log or p_mkt_delta
    direction: str = "exclude_high"  # exclude_high or exclude_low
    threshold_mode: str = "quantile"
    quantile: float = 0.95
    fit_scope: str = "train_valid"
    fit_population: str = "candidates_with_bets_under_baseline_gating"
    threshold: Optional[float] = None


class OddsDynamicsEvMarginConfig(BaseModel):
    """Odds dynamics EV margin config (post-cap soft margin)."""

    enabled: bool = False
    metric: str = "odds_delta_log"
    lookback_minutes: int = 5
    direction: str = "high"  # high or low
    ref: Optional[float] = None
    slope: float = 0.0
    fit_scope: str = "train_valid"
    fit_population: str = "candidates_with_bets_under_baseline_gating"


class PaceHistoryFeatureConfig(BaseModel):
    """Pace history feature flags."""

    enabled: bool = False


class FeaturesConfig(BaseModel):
    """Feature selection flags."""

    pace_history: PaceHistoryFeatureConfig = Field(default_factory=PaceHistoryFeatureConfig)


class CVConfig(BaseModel):
    """クロスバリデーション設定"""

    scheme: str = "walk_forward"
    train_years: int = 4
    test_months: int = 12


class DataSourceConfig(BaseModel):
    """データソース設定"""

    provider: str = "jra_van_datalab"
    storage_raw_dir: str = "data/raw"
    storage_processed_dir: str = "data/processed"


class UniverseConfig(BaseModel):
    """学習/検証に使うレース母集団（race_universe）"""

    # 中央（01..10）をデフォルトにする
    track_codes: list[str] = Field(default_factory=lambda: [f"{i:02d}" for i in range(1, 11)])
    # 結果があるレースのみ（ラベル汚染防止）
    require_results: bool = True
    # 0B41（odds_ts_win）があるレースのみ（p_mkt/TS特徴量の前提）
    require_ts_win: bool = True
    # 恒久的に除外したい race_id（16桁）
    exclude_race_ids: list[str] = Field(default_factory=list)


class Config(BaseModel):
    """全体設定"""

    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    payout_rate: PayoutRateConfig = Field(default_factory=PayoutRateConfig)
    odds_timeseries: OddsTimeseriesConfig = Field(default_factory=OddsTimeseriesConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    betting: BettingConfig = Field(default_factory=BettingConfig)


def load_config(config_path: Optional[Path | str] = None) -> Config:
    """
    設定ファイルを読み込む

    Args:
        config_path: 設定ファイルパス（Noneなら自動探索）

    探索順序:
        1. 引数で指定されたパス
        2. 環境変数 KEIBA_CONFIG_PATH
        3. PROJECT_ROOT / config / config.yaml
        4. デフォルト設定
    """
    # 1. 引数指定
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return Config.model_validate(data)

    # 2. 環境変数
    env_config = os.environ.get("KEIBA_CONFIG_PATH")
    if env_config:
        path = Path(env_config)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return Config.model_validate(data)

    # 3. プロジェクトルートから
    if PROJECT_ROOT:
        path = PROJECT_ROOT / "config" / "config.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return Config.model_validate(data)

    # 4. デフォルト設定を返す
    return Config()


def get_data_path(relative_path: str) -> Path:
    """
    データパスを取得（プロジェクトルート基準）

    Args:
        relative_path: 相対パス（例: "data/raw"）

    Returns:
        絶対パス
    """
    if PROJECT_ROOT:
        return PROJECT_ROOT / relative_path
    return Path(relative_path)


# シングルトン的に使う場合
_config: Optional[Config] = None


def get_config() -> Config:
    """設定を取得（遅延読み込み）"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """設定をリセット（テスト用）"""
    global _config
    _config = None
