"""
バックテストエンジン

時系列オッズを考慮した購入時点シミュレーション
"""
import logging
import math
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from itertools import groupby

import pandas as pd
import numpy as np

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_config
from ..features.build_features import FeatureBuilder
from ..analysis.slippage_table import SlippageTable
from ..betting.odds_band_bias import OddsBandBias
from ..betting.odds_dynamics import compute_odds_dyn_metric, eval_odds_dyn_filter
from ..betting.overlay_shrink import shrink_probability
from ..betting.market_blend import compute_market_blend, parse_exclude_odds_band
from ..betting.sizing import calculate_stake
from ..betting.stake_clip import clip_stake
from ..betting.uncertainty_shrink import UncertaintyShrink
from ..modeling.race_softmax import apply_race_softmax

logger = logging.getLogger(__name__)


def _surface_label(val) -> str:
    if val is None:
        return "unknown"
    s = str(val).strip().lower()
    if s in ("1", "turf", "grass", "shiba", "\u829d"):
        return "turf"
    if s in ("2", "dirt", "dar", "\u30c0", "\u30c0\u30fc\u30c8"):
        return "dirt"
    if s in ("3", "jump", "\u969c", "\u969c\u5bb3"):
        return "jump"
    return "unknown"


@dataclass
class Bet:
    """ベット情報"""
    race_id: str
    horse_no: int
    ticket_type: str
    stake: float
    odds_at_buy: float
    # Ticket1: EV/賭け金計算に使った「有効オッズ」
    # - 時系列オッズがある場合: odds_at_buy * closing_odds_multiplier
    # - 時系列オッズがない場合: odds_at_buy はslippage後の値（推定）、さらに closing_odds_multiplier を適用
    odds_effective: float
    p_hat: float
    ev: float
    asof_time: datetime
    # 解析・デバッグ用（任意）：特徴量やスナップショット由来の追加情報
    extra: dict = field(default_factory=dict)


@dataclass
class BetResult:
    """ベット結果"""
    bet: Bet
    finish_pos: int
    is_win: bool
    payout: float
    profit: float
    # 決済に使った確定オッズ（fact_result.odds があればそれ、なければ odds_at_buy）
    odds_final: float | None = None


@dataclass
class BacktestResult:
    """バックテスト結果"""
    bets: list[BetResult] = field(default_factory=list)
    initial_bankroll: float = 0
    final_bankroll: float = 0
    min_bankroll: float = 0
    total_stake: float = 0
    total_payout: float = 0
    total_profit: float = 0
    n_bets: int = 0
    n_wins: int = 0
    roi: float = 0
    max_drawdown: float = 0
    log_growth: float | None = None
    max_drawdown_bankroll: float = 0
    odds_floor_min_odds: float = 0.0
    odds_floor_filtered_bets: int = 0
    odds_floor_filtered_stake: float = 0.0
    stake_odds_damp_ref_odds: float = 0.0
    stake_odds_damp_power: float = 1.0
    stake_odds_damp_min_mult: float = 0.0
    stake_odds_damp_mean_mult: float = 1.0
    stake_odds_damp_low_odds_stake_before: float = 0.0
    stake_odds_damp_low_odds_stake_after: float = 0.0


class BacktestEngine:
    """バックテストエンジン"""
    
    def __init__(
        self,
        session: Session,
        slippage_table: SlippageTable | None = None,
        odds_band_bias: OddsBandBias | None = None,
        ev_upper_cap: float | None = None,
        uncertainty_shrink: UncertaintyShrink | None = None,
    ):
        self.session = session
        self.config = get_config()
        self.slippage_table = slippage_table
        self.odds_band_bias = odds_band_bias
        self.ev_upper_cap = ev_upper_cap  # Ticket G2: EV上限値
        self.uncertainty_shrink = uncertainty_shrink
        self._surface_cache: dict[str, str] = {}
        self._odds_floor_filtered_bets = 0
        self._odds_floor_filtered_stake = 0.0
        self._odds_floor_min_odds = 0.0
        self._stake_damp_ref_odds = 0.0
        self._stake_damp_power = 1.0
        self._stake_damp_min_mult = 0.0
        self._stake_damp_mult_sum = 0.0
        self._stake_damp_n = 0
        self._stake_damp_low_odds_stake_before = 0.0
        self._stake_damp_low_odds_stake_after = 0.0
    
    def run(
        self,
        start_date: str,
        end_date: str,
        model,
        initial_bankroll: float = 300000,
    ) -> BacktestResult:
        """
        バックテスト実行
        
        Args:
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
            model: 学習済みモデル（WinProbabilityModel）
            initial_bankroll: 初期資金
        """
        logger.info(f"Running backtest: {start_date} to {end_date}")

        self._stake_damp_ref_odds = 0.0
        self._stake_damp_power = 1.0
        self._stake_damp_min_mult = 0.0
        self._stake_damp_mult_sum = 0.0
        self._stake_damp_n = 0
        self._stake_damp_low_odds_stake_before = 0.0
        self._stake_damp_low_odds_stake_after = 0.0

        # レース一覧取得
        races = self._get_races(start_date, end_date)
        logger.info(f"Found {len(races)} races")
        
        result = BacktestResult(initial_bankroll=initial_bankroll)
        bankroll = initial_bankroll
        bankroll_decision = initial_bankroll
        peak_bankroll = initial_bankroll
        min_bankroll = initial_bankroll
        log_growths: list[float] = []
        
        # ★日次制約用: 日ごとの stake/loss を追跡
        stake_cfg = getattr(self.config.betting, "stake", None)
        stake_enabled = bool(stake_cfg and getattr(stake_cfg, "enabled", False))
        uncert_cfg = getattr(self.config.betting, "uncertainty", None)
        uncert_enabled = bool(
            uncert_cfg and getattr(uncert_cfg, "enabled", False) and self.uncertainty_shrink is not None
        )
        use_new_stake = stake_enabled or uncert_enabled
        min_yen = int(getattr(stake_cfg, "min_yen", 100)) if stake_cfg is not None else 100

        def _per_day_cap_for(bankroll_value: float) -> Optional[float]:
            if stake_cfg is not None and bool(getattr(stake_cfg, "enabled", False)):
                frac = getattr(stake_cfg, "max_frac_per_day", None)
                if frac is None:
                    return None
                return float(bankroll_value) * float(frac)
            return float(initial_bankroll) * float(self.config.betting.caps.per_day_pct)

        stop_cfg = getattr(self.config.betting, "stop", None)
        max_daily_loss = initial_bankroll * float(getattr(stop_cfg, "max_daily_loss_pct", 0.0) or 0.0)
        max_daily_profit = None
        try:
            max_daily_profit_pct = float(getattr(stop_cfg, "max_daily_profit_pct", None))
        except Exception:
            max_daily_profit_pct = None
        if max_daily_profit_pct is not None and max_daily_profit_pct > 0:
            max_daily_profit = initial_bankroll * max_daily_profit_pct

        def _profit_before_damp(bet: Bet, profit_actual: float) -> float:
            extra = bet.extra or {}
            stake_before = extra.get("stake_before")
            try:
                stake_before_val = float(stake_before) if stake_before is not None else None
            except Exception:
                stake_before_val = None
            stake_after_val = float(bet.stake)
            if stake_before_val is None or not np.isfinite(stake_before_val) or stake_before_val <= 0:
                stake_before_val = stake_after_val
            if stake_after_val > 0 and np.isfinite(stake_after_val):
                scale = stake_before_val / stake_after_val
            else:
                scale = 1.0
            return float(profit_actual) * float(scale)


        sel_mode, daily_cfg = self._resolve_selection_mode()
        daily_topn_stats: list[dict] = []
        if sel_mode == "daily_top_n":
            for race_date, day_races in groupby(races, key=lambda r: r[:8]):
                daily_loss = 0.0
                daily_profit = 0.0
                per_day_cap = _per_day_cap_for(bankroll_decision)
                bankroll_for_day = bankroll_decision
                daily_candidates: list[dict] = []

                for race_id in day_races:
                    buy_time = self._get_buy_time(race_id)
                    if not buy_time:
                        continue

                    predictions = self._predict_race(race_id, buy_time, model)
                    if not predictions:
                        continue

                    daily_candidates.extend(self._build_candidates(race_id, predictions, buy_time))

                selected_candidates, day_stats = self._apply_daily_top_n(daily_candidates, daily_cfg, race_date)
                daily_topn_stats.extend(day_stats)

                daily_bets = self._build_bets_from_candidates(
                    selected_candidates,
                    bankroll_for_day,
                    remaining_daily_budget=None,
                    selection_mode=sel_mode,
                )

                if not use_new_stake and per_day_cap is not None:
                    daily_bets = sorted(daily_bets, key=lambda b: b.ev, reverse=True)
                    kept = []
                    used = 0
                    for bet in daily_bets:
                        if used + bet.stake <= per_day_cap:
                            kept.append(bet)
                            used += bet.stake
                    daily_bets = kept

                if per_day_cap is not None:
                    self._scale_bets_to_cap(daily_bets, per_day_cap, min_yen, reason="per_day")

                daily_bets = [b for b in daily_bets if b.stake >= min_yen]
                daily_bets = self._apply_stake_odds_damp(daily_bets, min_yen=min_yen)
                daily_bets.sort(key=lambda b: (b.asof_time, b.race_id, -b.ev))

                for bet in daily_bets:
                    if daily_loss >= max_daily_loss:
                        break
                    if max_daily_profit is not None and daily_profit >= max_daily_profit:
                        break

                    bet_result = self._settle_bet(bet)
                    result.bets.append(bet_result)

                    bankroll_before = bankroll
                    bankroll += bet_result.profit
                    min_bankroll = min(min_bankroll, bankroll)
                    if bankroll_before > 0 and bankroll > 0:
                        log_growths.append(math.log(bankroll / bankroll_before))
                    peak_bankroll = max(peak_bankroll, bankroll)

                    profit_before = _profit_before_damp(bet, bet_result.profit)
                    bankroll_decision += profit_before
                    if profit_before < 0:
                        daily_loss += abs(profit_before)
                    elif profit_before > 0:
                        daily_profit += profit_before

                    dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                    result.max_drawdown = max(result.max_drawdown, dd)
        elif use_new_stake:
            for race_date, day_races in groupby(races, key=lambda r: r[:8]):
                daily_loss = 0.0
                daily_profit = 0.0
                per_day_cap = _per_day_cap_for(bankroll_decision)
                bankroll_for_day = bankroll_decision
                daily_bets: list[Bet] = []

                for race_id in day_races:
                    # ???????E?????E
                    buy_time = self._get_buy_time(race_id)
                    if not buy_time:
                        continue

                    # ??????E???E?Esof_time?????????E?E
                    predictions = self._predict_race(race_id, buy_time, model)
                    if not predictions:
                        continue

                    # ?????????per-day cap??????????
                    bets = self._generate_bets(
                        race_id,
                        predictions,
                        buy_time,
                        bankroll_for_day,
                        remaining_daily_budget=None,
                    )
                    daily_bets.extend(bets)

                if per_day_cap is not None:
                    self._scale_bets_to_cap(daily_bets, per_day_cap, min_yen, reason="per_day")

                daily_bets = [b for b in daily_bets if b.stake >= min_yen]
                daily_bets = self._apply_stake_odds_damp(daily_bets, min_yen=min_yen)
                daily_bets.sort(key=lambda b: (b.asof_time, b.race_id, -b.ev))

                for bet in daily_bets:
                    # ?E??????????E??
                    if daily_loss >= max_daily_loss:
                        break
                    if max_daily_profit is not None and daily_profit >= max_daily_profit:
                        break

                    bet_result = self._settle_bet(bet)
                    result.bets.append(bet_result)

                    bankroll_before = bankroll
                    bankroll += bet_result.profit
                    min_bankroll = min(min_bankroll, bankroll)
                    if bankroll_before > 0 and bankroll > 0:
                        log_growths.append(math.log(bankroll / bankroll_before))
                    peak_bankroll = max(peak_bankroll, bankroll)

                    profit_before = _profit_before_damp(bet, bet_result.profit)
                    bankroll_decision += profit_before
                    if profit_before < 0:
                        daily_loss += abs(profit_before)
                    elif profit_before > 0:
                        daily_profit += profit_before

                    # ????????E
                    dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                    result.max_drawdown = max(result.max_drawdown, dd)
        else:
            current_date = None
            daily_stake = 0
            daily_loss = 0.0
            daily_profit = 0.0
            per_day_cap = _per_day_cap_for(bankroll_decision)

            for race_id in races:
                # ???????E???E?????E????E???E?E
                race_date = race_id[:8]  # YYYYMMDD
                if race_date != current_date:
                    current_date = race_date
                    daily_stake = 0
                    daily_loss = 0.0
                    daily_profit = 0.0
                    per_day_cap = _per_day_cap_for(bankroll_decision)
                
                # ?E??????????E??
                if daily_loss >= max_daily_loss:
                    continue
                if max_daily_profit is not None and daily_profit >= max_daily_profit:
                    continue
                
                # ???????E?????E
                buy_time = self._get_buy_time(race_id)
                if not buy_time:
                    continue
                
                # ??????E???E?Esof_time?????????E?E
                predictions = self._predict_race(race_id, buy_time, model)
                if not predictions:
                    continue
                
                # ?E????????E?E???????E
                remaining_daily_budget = None
                if per_day_cap is not None:
                    remaining_daily_budget = max(0, per_day_cap - daily_stake)
                bets = self._generate_bets(
                    race_id, predictions, buy_time, bankroll_decision,
                    remaining_daily_budget=remaining_daily_budget
                )
                bets = self._apply_stake_odds_damp(bets, min_yen=min_yen)

                # ????E
                for bet in bets:
                    bet_result = self._settle_bet(bet)
                    result.bets.append(bet_result)
                    
                    bankroll_before = bankroll
                    bankroll += bet_result.profit
                    min_bankroll = min(min_bankroll, bankroll)
                    if bankroll_before > 0 and bankroll > 0:
                        log_growths.append(math.log(bankroll / bankroll_before))
                    peak_bankroll = max(peak_bankroll, bankroll)
                    
                    # ?E???????
                    # Use pre-damp stake for daily budget to keep selection invariant.
                    extra = bet.extra or {}
                    stake_before = extra.get("stake_before")
                    try:
                        stake_before = float(stake_before) if stake_before is not None else None
                    except Exception:
                        stake_before = None
                    stake_for_budget = stake_before if stake_before is not None and stake_before > 0 else bet.stake
                    daily_stake += stake_for_budget
                    profit_before = _profit_before_damp(bet, bet_result.profit)
                    bankroll_decision += profit_before
                    if profit_before < 0:
                        daily_loss += abs(profit_before)
                    elif profit_before > 0:
                        daily_profit += profit_before
                    
                    # ????????E
                    dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                    result.max_drawdown = max(result.max_drawdown, dd)


        if daily_topn_stats:
            result.daily_topn_stats = daily_topn_stats

        result.final_bankroll = bankroll
        result.n_bets = len(result.bets)
        result.n_wins = sum(1 for b in result.bets if b.is_win)
        result.total_stake = sum(b.bet.stake for b in result.bets)
        result.total_payout = sum(b.payout for b in result.bets)
        result.total_profit = result.total_payout - result.total_stake
        result.roi = result.total_profit / result.total_stake if result.total_stake > 0 else 0
        result.log_growth = float(np.mean(log_growths)) if log_growths else None
        result.max_drawdown_bankroll = result.max_drawdown
        result.min_bankroll = min_bankroll

        self._apply_odds_floor_postprocess(result)
        result.odds_floor_min_odds = float(self._odds_floor_min_odds or 0.0)
        result.odds_floor_filtered_bets = int(self._odds_floor_filtered_bets)
        result.odds_floor_filtered_stake = float(self._odds_floor_filtered_stake)
        result.stake_odds_damp_ref_odds = float(self._stake_damp_ref_odds)
        result.stake_odds_damp_power = float(self._stake_damp_power)
        result.stake_odds_damp_min_mult = float(self._stake_damp_min_mult)
        result.stake_odds_damp_mean_mult = (
            float(self._stake_damp_mult_sum / self._stake_damp_n) if self._stake_damp_n else 1.0
        )
        result.stake_odds_damp_low_odds_stake_before = float(self._stake_damp_low_odds_stake_before)
        result.stake_odds_damp_low_odds_stake_after = float(self._stake_damp_low_odds_stake_after)
        
        logger.info(f"Backtest complete: ROI={result.roi:.2%}, N={result.n_bets}")
        return result
    
    def _get_races(self, start_date: str, end_date: str) -> list[str]:
        """期間内のレースID一覧"""
        u = self.config.universe
        track_codes = list(u.track_codes or [])
        exclude_race_ids = list(u.exclude_race_ids or [])
        buy_minutes = int(self.config.backtest.buy_t_minus_minutes)
        query = text("""
            SELECT r.race_id 
            FROM fact_race r
            WHERE r.date BETWEEN :start_date AND :end_date
              AND r.start_time IS NOT NULL
              AND (:track_codes_len = 0 OR r.track_code = ANY(:track_codes))
              AND (:require_results = FALSE OR EXISTS (
                    SELECT 1 FROM fact_result res
                    WHERE res.race_id = r.race_id
                      AND res.finish_pos IS NOT NULL
              ))
              AND (:require_ts_win = FALSE OR EXISTS (
                    SELECT 1 FROM odds_ts_win o
                    WHERE o.race_id = r.race_id
                      AND o.odds > 0
                      AND o.asof_time <= ((r.date::timestamp + r.start_time) - make_interval(mins => :buy_minutes))
              ))
              AND (:exclude_len = 0 OR NOT (r.race_id = ANY(:exclude_race_ids)))
            ORDER BY r.date, r.race_id
        """)
        results = self.session.execute(
            query,
            {
                "start_date": start_date,
                "end_date": end_date,
                "track_codes_len": len(track_codes),
                "track_codes": track_codes,
                "require_results": bool(u.require_results),
                "require_ts_win": bool(u.require_ts_win),
                "buy_minutes": buy_minutes,
                "exclude_len": len(exclude_race_ids),
                "exclude_race_ids": exclude_race_ids,
            },
        ).fetchall()
        return [r[0] for r in results]
    
    def _get_buy_time(self, race_id: str) -> Optional[datetime]:
        """購入時点を決定"""
        # 発走時刻からN分前
        query = text("""
            SELECT date, start_time 
            FROM fact_race 
            WHERE race_id = :race_id
        """)
        result = self.session.execute(query, {"race_id": race_id}).fetchone()
        
        if not result or not result[1]:
            return None
        
        race_date = result[0]
        start_time = result[1]
        
        dt = datetime.combine(race_date, start_time)
        buy_time = dt - timedelta(minutes=self.config.backtest.buy_t_minus_minutes)
        return buy_time

    def _get_race_surface_segment(self, race_id: str) -> str:
        if race_id in self._surface_cache:
            return self._surface_cache[race_id]
        query = text("SELECT surface FROM fact_race WHERE race_id = :race_id")
        row = self.session.execute(query, {"race_id": race_id}).fetchone()
        seg = _surface_label(row[0]) if row else "unknown"
        self._surface_cache[race_id] = seg
        return seg
    
    def _predict_race(
        self, 
        race_id: str, 
        asof_time: datetime, 
        model
    ) -> list[dict]:
        """
        レースの予測を実行
        
        asof_time以前の情報のみを使用してリークを防止
        """
        # 特徴量取得（asof_time条件でリーク防止）
        query = text("""
            SELECT f.horse_id, f.payload
            FROM features f
            WHERE f.race_id = :race_id
              AND f.feature_version = :feature_version
              AND f.asof_time <= :asof_time
            ORDER BY f.asof_time DESC
        """)
        results = self.session.execute(
            query, {"race_id": race_id, "asof_time": asof_time, "feature_version": FeatureBuilder.VERSION}
        ).fetchall()
        
        if not results:
            # 特徴量がない場合、オッズからp_mktを直接取得
            return self._get_odds_based_predictions(race_id, asof_time, model)
        
        # horse_idごとに最新の特徴量を取得
        seen_horses = set()
        predictions = []
        surface_segment = self._get_race_surface_segment(race_id)
        for r in results:
            horse_id = r[0]
            if horse_id in seen_horses:
                continue
            seen_horses.add(horse_id)
            
            payload = r[1] or {}

            p_mkt, p_raw, p_race, overround, takeout = self._select_market_prob(payload)
            odds = payload.get("odds")
            horse_no = payload.get("horse_no", 1)
            
            if not p_mkt or not odds or p_mkt <= 0 or odds <= 0:
                continue
            
            # モデルを使って予測（G1: residual_metaも取得）
            p_hat, residual_meta, p_model = self._apply_model(model, payload, p_mkt, surface_segment)

            # 解析用: 時系列オッズ特徴量を必要な範囲だけ持ち回す
            ts_keys = (
                "odds_chg_5m",
                "odds_chg_10m",
                "odds_chg_30m",
                "odds_chg_60m",
                "p_mkt_chg_5m",
                "p_mkt_chg_10m",
                "p_mkt_chg_30m",
                "p_mkt_chg_60m",
                "log_odds_slope_60m",
                "log_odds_std_60m",
                "n_pts_60m",
                "snap_age_min",
            )
            ts_feat = {k: payload.get(k) for k in ts_keys}
            
            # Ticket G1: residual_metaをpayloadに追加
            if residual_meta:
                resid_val = residual_meta.get("resid")
                resid_cap_val = residual_meta.get("resid_cap")
                cap_val = residual_meta.get("cap_value")
                p_hat_capped_val = residual_meta.get("p_hat_capped")
                if resid_val is not None:
                    ts_feat["resid"] = float(resid_val[0]) if isinstance(resid_val, np.ndarray) else float(resid_val)
                if resid_cap_val is not None:
                    ts_feat["resid_cap"] = float(resid_cap_val[0]) if isinstance(resid_cap_val, np.ndarray) else float(resid_cap_val)
                if cap_val is not None:
                    ts_feat["cap_value"] = float(cap_val)
                if p_hat_capped_val is not None:
                    ts_feat["p_hat_capped"] = float(p_hat_capped_val[0]) if isinstance(p_hat_capped_val, np.ndarray) else float(p_hat_capped_val)
            
            predictions.append({
                "horse_id": horse_id,
                "horse_no": horse_no,
                "p_hat": p_hat,
                "p_model": p_model,
                "odds": odds,
                "p_mkt": p_mkt,
                "p_mkt_raw": p_raw,
                "p_mkt_race": p_race,
                "overround_sum_inv": overround,
                "takeout_implied": takeout,
                "segblend_segment": surface_segment,
                "segblend_w_used": residual_meta.get("segblend_w_used") if residual_meta else None,
                "segblend_w_global": residual_meta.get("segblend_w_global") if residual_meta else None,
                "has_ts_odds": True,  # 特徴量がある = 時系列オッズを使って生成された
                **ts_feat,
            })
        
        self._apply_race_softmax(predictions, model, race_id)
        return predictions
    
    def _get_odds_based_predictions(
        self, 
        race_id: str, 
        asof_time: datetime, 
        model
    ) -> list[dict]:
        """
        オッズから直接予測を生成（特徴量がない場合のフォールバック）
        
        ★重要: FeatureBuilder と同様に、同一 t_snap で揃えて
                スナップショットの混在を防ぐ
        """
        # まずasof_time以前の最新スナップショット時刻（t_snap）を取得
        snap_query = text("""
            SELECT MAX(asof_time) as t_snap
            FROM odds_ts_win
            WHERE race_id = :race_id
              AND asof_time <= :asof_time
              AND odds > 0
        """)
        snap_result = self.session.execute(
            snap_query, {"race_id": race_id, "asof_time": asof_time}
        ).fetchone()
        
        if not snap_result or not snap_result[0]:
            return []
        
        t_snap = snap_result[0]
        snap_age_min = float((asof_time - t_snap).total_seconds() / 60.0) if asof_time and t_snap else None
        surface_segment = self._get_race_surface_segment(race_id)
        
        # t_snap時点の全馬オッズを取得（同一スナップショット）
        query = text("""
            SELECT DISTINCT ON (horse_no)
              horse_no, odds, popularity, data_kubun
            FROM odds_ts_win
            WHERE race_id = :race_id
              AND asof_time = :t_snap
              AND odds > 0
            ORDER BY horse_no, data_kubun DESC
        """)
        results = self.session.execute(
            query, {"race_id": race_id, "t_snap": t_snap}
        ).fetchall()
        
        if not results:
            return []
        
        df = pd.DataFrame([dict(r._mapping) for r in results])
        # Numeric(Decimal) → float に揃える（演算の安定化）
        df["odds"] = df["odds"].astype(float)
        
        # 市場確率を計算（同一スナップショットなので整合性がある）
        df["inv_odds"] = 1.0 / df["odds"]
        total_inv = df["inv_odds"].sum()
        if total_inv > 0:
            df["p_mkt_race"] = df["inv_odds"] / total_inv
        else:
            df["p_mkt_race"] = np.nan
        df["p_mkt_raw"] = df["p_mkt_race"]
        overround_sum_inv = float(total_inv) if total_inv > 0 else None
        takeout_implied = (1.0 - (1.0 / overround_sum_inv)) if (overround_sum_inv is not None and overround_sum_inv > 0) else None
        
        predictions = []
        for _, row in df.iterrows():
            odds = float(row["odds"])
            p_raw = row.get("p_mkt_raw")
            p_race = row.get("p_mkt_race")
            try:
                p_raw = float(p_raw) if p_raw is not None and np.isfinite(p_raw) else None
            except Exception:
                p_raw = None
            try:
                p_race = float(p_race) if p_race is not None and np.isfinite(p_race) else None
            except Exception:
                p_race = None
            p_mkt = p_race if self._market_prob_mode() == "race_norm" else p_raw
            if p_mkt is None or not np.isfinite(p_mkt):
                p_mkt = 0.1
            horse_no = int(row["horse_no"])
            
            # モデルのブレンド予測（特徴量なしなので市場確率ベース）
            p_hat = model.blend_weight * p_mkt + (1 - model.blend_weight) * p_mkt
            p_model = float(p_mkt)
            
            predictions.append({
                "horse_id": None,
                "horse_no": horse_no,
                "p_hat": p_hat,
                "p_model": p_model,
                "odds": odds,
                "p_mkt": p_mkt,
                "p_mkt_raw": p_raw,
                "p_mkt_race": p_race,
                "overround_sum_inv": overround_sum_inv,
                "takeout_implied": takeout_implied,
                "segblend_segment": surface_segment,
                "segblend_w_used": model.get_blend_weight_for_segment(surface_segment),
                "segblend_w_global": float(model.blend_weight),
                "has_ts_odds": False,  # 特徴量なしフォールバック = slippage適用対象
                "snap_age_min": snap_age_min,
            })
        
        self._apply_race_softmax(predictions, model, race_id)
        return predictions

    def _apply_race_softmax(self, predictions: list[dict], model, race_id: str) -> None:
        rs_cfg = getattr(self.config.model, "race_softmax", None)
        if not rs_cfg or not bool(getattr(rs_cfg, "enabled", False)):
            return
        if not predictions:
            return

        selector_cfg = getattr(rs_cfg, "selector", None)
        selector_enabled = bool(selector_cfg and getattr(selector_cfg, "enabled", False))
        if selector_enabled:
            selector_meta = getattr(model, "race_softmax_selector", None)
            chosen = selector_meta.get("chosen") if isinstance(selector_meta, dict) else None
            if chosen != "softmax":
                for pred in predictions:
                    pred["race_softmax_selected"] = "baseline"
                    pred["race_softmax_enabled"] = False
                return

        w = None
        t = None
        if getattr(model, "race_softmax_params", None):
            params = model.race_softmax_params
            w = params.get("w")
            t = params.get("T")
        if w is None:
            w = float(getattr(getattr(rs_cfg, "apply", None), "w_default", 0.2))
        if t is None:
            t = float(getattr(getattr(rs_cfg, "apply", None), "t_default", 1.0))

        rows = []
        for pred in predictions:
            rows.append(
                {
                    "race_id": race_id,
                    "p_model": pred.get("p_model"),
                    "p_mkt": pred.get("p_mkt"),
                }
            )
        df = pd.DataFrame(rows)
        if df["p_mkt"].isna().all():
            logger.warning("race_softmax enabled but p_mkt is missing; skip")
            for pred in predictions:
                pred["race_softmax_selected"] = "baseline"
                pred["race_softmax_enabled"] = False
            return

        p_race = apply_race_softmax(
            df,
            w=float(w),
            t=float(t),
            score_space=str(getattr(rs_cfg, "score_space", "logit")),
            clip_eps=float(getattr(rs_cfg, "clip_eps", 1e-6)),
        )

        p_vals = np.asarray(p_race).reshape(-1).tolist()
        if len(p_vals) != len(predictions):
            logger.warning(
                "race_softmax produced length mismatch: preds=%s probs=%s; skip",
                len(predictions),
                len(p_vals),
            )
            for pred in predictions:
                pred["race_softmax_selected"] = "baseline"
                pred["race_softmax_enabled"] = False
            return
        for pred, p_val in zip(predictions, p_vals):
            pred["p_used"] = float(p_val)
            pred["race_softmax_w"] = float(w)
            pred["race_softmax_T"] = float(t)
            pred["race_softmax_enabled"] = True
            pred["race_softmax_selected"] = "softmax"
    
    def _apply_model(self, model, payload: dict, p_mkt: float, segment: Optional[str]) -> tuple[float, dict, float]:
        """
        モデルを適用して予測確率を計算
        
        特徴量がある場合はモデルで予測、なければ市場確率ベース
        """
        # モデルにlgb_modelがある場合は使用
        if hasattr(model, 'lgb_model') and model.lgb_model is not None:
            try:
                # 特徴量をDataFrameに変換
                feature_cols = model.feature_names
                X = pd.DataFrame([{c: payload.get(c, 0) for c in feature_cols}])
                # ★重要: 1行だけのDataFrameだと None 混入で dtype=object になりやすい
                # LightGBMは数値dtypeのみ受け付けるため、強制的に数値化して0埋めする
                X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
                p_mkt_series = pd.Series([p_mkt])
                
                # ブレンド予測（G1: residual_metaを取得）
                result = model.predict(X, p_mkt_series, return_residual_meta=True, segments=[segment] if segment else None)
                if isinstance(result, tuple):
                    p_hat, residual_meta = result
                    p_hat = p_hat[0]
                    if model.use_market_offset:
                        p_model = float(p_hat)
                    else:
                        p_model = float(model.lgb_model.predict(X)[0])
                    return float(p_hat), residual_meta, float(p_model)
                else:
                    p_hat = result[0]
                    if model.use_market_offset:
                        p_model = float(p_hat)
                    else:
                        p_model = float(model.lgb_model.predict(X)[0])
                    return float(p_hat), {}, float(p_model)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
        
        # フォールバック：市場確率をそのまま使用
        return float(model.blend_weight * p_mkt + (1 - model.blend_weight) * p_mkt), {}, float(p_mkt)
    

    def _resolve_selection_mode(self) -> tuple[str, object | None]:
        sel_cfg = getattr(self.config.betting, "selection", None)
        mode = getattr(sel_cfg, "mode", "ev_threshold") if sel_cfg is not None else "ev_threshold"
        daily_cfg = getattr(sel_cfg, "daily_top_n", None) if sel_cfg is not None else None
        if daily_cfg is not None and bool(getattr(daily_cfg, "enabled", False)):
            mode = "daily_top_n"
        return mode, daily_cfg

    def _market_prob_mode(self) -> str:
        mode = str(getattr(self.config.model, "market_prob_mode", "raw") or "raw").lower()
        if mode not in ("raw", "race_norm"):
            mode = "raw"
        return mode

    def _select_market_prob(self, payload: dict) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        p_raw = payload.get("p_mkt_raw")
        p_race = payload.get("p_mkt_race")
        if p_raw is None:
            p_raw = payload.get("p_mkt")
        if p_race is None:
            p_race = payload.get("p_mkt")
        try:
            p_raw = float(p_raw) if p_raw is not None else None
        except Exception:
            p_raw = None
        try:
            p_race = float(p_race) if p_race is not None else None
        except Exception:
            p_race = None

        overround = payload.get("overround_sum_inv")
        takeout = payload.get("takeout_implied")
        try:
            overround = float(overround) if overround is not None else None
        except Exception:
            overround = None
        try:
            takeout = float(takeout) if takeout is not None else None
        except Exception:
            takeout = None

        p_mkt = p_race if self._market_prob_mode() == "race_norm" else p_raw
        return p_mkt, p_raw, p_race, overround, takeout

    def _race_cost_filter_meta(self) -> tuple[str, str, float | None]:
        cfg = getattr(self.config.betting, "race_cost_filter", None)
        metric = str(getattr(cfg, "metric", "takeout_implied") or "takeout_implied").lower() if cfg else "takeout_implied"
        if metric not in ("takeout_implied", "overround_sum_inv"):
            metric = "takeout_implied"
        cap_mode = str(getattr(cfg, "cap_mode", "fixed") or "fixed").lower() if cfg else "fixed"
        if cap_mode not in ("fixed", "train_quantile"):
            cap_mode = "fixed"

        cap_value = None
        if cfg:
            if cap_mode == "train_quantile":
                cap_value = (
                    getattr(cfg, "max_takeout_implied", None)
                    if metric == "takeout_implied"
                    else getattr(cfg, "max_overround_sum_inv", None)
                )
            else:
                cap_value = getattr(cfg, "max_takeout_implied", None)
                if cap_value is None:
                    cap_value = getattr(cfg, "max_overround_sum_inv", None)
        try:
            cap_value = float(cap_value) if cap_value is not None and np.isfinite(cap_value) else None
        except Exception:
            cap_value = None
        return cap_mode, metric, cap_value


    def _race_cost_filter_extra_meta(self) -> tuple[str, float | None, str | None]:
        cfg = getattr(self.config.betting, "race_cost_filter", None)
        cap_mode, _, cap_value = self._race_cost_filter_meta()
        selector_cfg = getattr(cfg, "selector", None) if cfg else None
        selected_from = None
        if selector_cfg is not None and bool(getattr(selector_cfg, "enabled", False)):
            selected_from = getattr(selector_cfg, "selected_from", None)
        mode_label = "none"
        if selector_cfg is not None and bool(getattr(selector_cfg, "enabled", False)) and selected_from is not None:
            mode_label = "selected"
        elif cfg is not None and bool(getattr(cfg, "enabled", False)) and cap_value is not None:
            mode_label = "fixed"
        return mode_label, cap_value, selected_from

    def _race_cost_filter(self, pred: dict) -> tuple[bool, str | None]:
        cfg = getattr(self.config.betting, "race_cost_filter", None)
        if not cfg or not bool(getattr(cfg, "enabled", False)):
            return True, None

        cap_mode, metric, cap_value = self._race_cost_filter_meta()
        reject_missing = bool(getattr(cfg, "reject_if_missing", True))

        overround = pred.get("overround_sum_inv")
        takeout = pred.get("takeout_implied")
        try:
            overround = float(overround) if overround is not None and np.isfinite(overround) else None
        except Exception:
            overround = None
        if overround is not None and overround <= 0:
            overround = None
        try:
            takeout = float(takeout) if takeout is not None and np.isfinite(takeout) else None
        except Exception:
            takeout = None

        if cap_mode == "train_quantile":
            if metric == "takeout_implied":
                if takeout is None:
                    return (False, "missing") if reject_missing else (True, None)
                if cap_value is not None and takeout > cap_value:
                    return False, "takeout_gt_cap"
            else:
                if overround is None:
                    return (False, "missing") if reject_missing else (True, None)
                if cap_value is not None and overround > cap_value:
                    return False, "overround_gt_cap"
            return True, None

        max_takeout = getattr(cfg, "max_takeout_implied", None)
        max_overround = getattr(cfg, "max_overround_sum_inv", None)
        try:
            max_takeout = float(max_takeout) if max_takeout is not None else None
        except Exception:
            max_takeout = None
        try:
            max_overround = float(max_overround) if max_overround is not None else None
        except Exception:
            max_overround = None

        if overround is None and takeout is None and reject_missing:
            return False, "missing"

        if max_takeout is not None:
            if takeout is None:
                if reject_missing:
                    return False, "missing"
            elif takeout > max_takeout:
                return False, "takeout_gt_cap"

        if max_overround is not None:
            if overround is None:
                if reject_missing:
                    return False, "missing"
            elif overround > max_overround:
                return False, "overround_gt_cap"

        return True, None

    def _apply_odds_dynamics_filter(self, bets: list["Bet"]) -> list["Bet"]:
        cfg = getattr(self.config.betting, "odds_dynamics_filter", None)
        enabled = bool(cfg and getattr(cfg, "enabled", False))
        metric = str(getattr(cfg, "metric", "odds_delta_log") or "odds_delta_log") if cfg else "odds_delta_log"
        direction = str(getattr(cfg, "direction", "exclude_high") or "exclude_high") if cfg else "exclude_high"
        lookback = getattr(cfg, "lookback_minutes", None) if cfg else None
        threshold = getattr(cfg, "threshold", None) if cfg else None
        try:
            threshold_val = float(threshold) if threshold is not None and np.isfinite(threshold) else None
        except Exception:
            threshold_val = None

        filtered: list[Bet] = []
        for bet in bets:
            extra = bet.extra or {}
            value = extra.get("odds_dyn_metric_value")
            passed = eval_odds_dyn_filter(value, threshold_val, direction) if enabled else None
            passed_flag = passed
            if passed is None:
                passed = True
            extra.update({
                "odds_dyn_metric": metric,
                "odds_dyn_lookback_min": lookback,
                "odds_dyn_threshold": threshold_val,
                "passed_odds_dyn_filter": passed_flag if enabled and threshold_val is not None else None,
                "odds_dyn_filter_enabled": enabled,
            })
            bet.extra = extra
            if enabled and threshold_val is not None and passed is False:
                continue
            filtered.append(bet)
        return filtered

    def _apply_odds_dyn_ev_margin(self, bets: list["Bet"]) -> list["Bet"]:
        cfg = getattr(self.config.betting, "odds_dyn_ev_margin", None)
        enabled = bool(cfg and getattr(cfg, "enabled", False))
        metric = str(getattr(cfg, "metric", "odds_delta_log") or "odds_delta_log") if cfg else "odds_delta_log"
        direction = str(getattr(cfg, "direction", "high") or "high") if cfg else "high"
        lookback = getattr(cfg, "lookback_minutes", None) if cfg else None
        try:
            ref = float(getattr(cfg, "ref", None))
        except Exception:
            ref = None
        try:
            slope = float(getattr(cfg, "slope", 0.0))
        except Exception:
            slope = 0.0
        enabled_eff = bool(enabled and ref is not None and slope is not None and slope > 0)

        filtered: list[Bet] = []
        for bet in bets:
            extra = bet.extra or {}
            score = extra.get("odds_dyn_ev_score")
            margin = None
            min_ev_eff_odm = None
            passed = None
            try:
                score_val = float(score) if score is not None else None
            except Exception:
                score_val = None

            if score_val is not None and ref is not None and np.isfinite(score_val):
                if direction == "low":
                    margin = slope * max(0.0, ref - score_val)
                else:
                    margin = slope * max(0.0, score_val - ref)
                base_min = extra.get("min_ev_eff")
                if base_min is None:
                    base_min = extra.get("base_min_ev")
                try:
                    base_min = float(base_min) if base_min is not None else 0.0
                except Exception:
                    base_min = 0.0
                min_ev_eff_odm = base_min + (margin or 0.0)
                if enabled_eff:
                    passed = bet.ev >= min_ev_eff_odm

            extra.update({
                "odds_dyn_ev_metric": metric,
                "odds_dyn_ev_lookback_min": lookback,
                "odds_dyn_ev_ref": ref,
                "odds_dyn_ev_slope": slope,
                "odds_dyn_ev_margin": margin,
                "min_ev_eff_odds_dyn": min_ev_eff_odm,
                "passed_odds_dyn_ev_margin": passed if enabled_eff else None,
                "odds_dyn_ev_margin_enabled": enabled_eff,
            })
            bet.extra = extra
            if enabled_eff and passed is False:
                continue
            filtered.append(bet)
        return filtered

    def _apply_odds_floor(self, bets: list["Bet"]) -> list["Bet"]:
        min_odds = getattr(self.config.betting, "odds_floor_min_odds", 0.0)
        try:
            min_odds_val = float(min_odds) if min_odds is not None else 0.0
        except Exception:
            min_odds_val = 0.0
        enabled = bool(min_odds_val and min_odds_val > 0)
        filtered: list[Bet] = []
        for bet in bets:
            odds_val = bet.odds_at_buy
            passed = None
            if enabled and odds_val is not None:
                passed = bool(odds_val >= min_odds_val)
            bet.extra = bet.extra or {}
            bet.extra.update({
                "odds_floor_min_odds": float(min_odds_val) if enabled else 0.0,
                "odds_floor_odds_used": float(odds_val) if odds_val is not None else None,
                "passed_odds_floor": passed if enabled else None,
            })
            if enabled and passed is False:
                self._odds_floor_filtered_bets += 1
                self._odds_floor_filtered_stake += float(bet.stake)
                continue
            filtered.append(bet)
        self._odds_floor_min_odds = float(min_odds_val) if enabled else 0.0
        return filtered

    def _apply_stake_odds_damp(self, bets: list["Bet"], *, min_yen: int) -> list["Bet"]:
        cfg = getattr(self.config.betting, "stake_odds_damp", None)
        enabled = bool(cfg and getattr(cfg, "enabled", False))
        try:
            ref_odds = float(getattr(cfg, "ref_odds", 0.0))
        except Exception:
            ref_odds = 0.0
        try:
            power = float(getattr(cfg, "power", 1.0))
        except Exception:
            power = 1.0
        if not np.isfinite(power) or power <= 0:
            power = 1.0
        try:
            min_mult = float(getattr(cfg, "min_mult", 0.0))
        except Exception:
            min_mult = 0.0
        if not np.isfinite(min_mult):
            min_mult = 0.0
        min_mult = min(1.0, max(0.0, min_mult))
        enabled_eff = bool(enabled and ref_odds and ref_odds > 0)
        unit = int(min_yen) if min_yen and min_yen > 0 else 1
        if unit <= 0:
            unit = 1

        for bet in bets:
            odds_val = bet.odds_at_buy
            base_stake = float(bet.stake)
            odds_num = None
            ratio = None
            mult_raw = 1.0
            mult = 1.0
            stake_after = base_stake
            if odds_val is not None:
                try:
                    odds_num = float(odds_val)
                except Exception:
                    odds_num = None
                if odds_num is not None and (not np.isfinite(odds_num) or odds_num <= 0):
                    odds_num = None
            if enabled_eff and odds_num is not None:
                ratio = float(odds_num) / float(ref_odds)
                if not np.isfinite(ratio) or ratio <= 0:
                    ratio = None
                if ratio is not None:
                    if ratio >= 1.0:
                        mult_raw = 1.0
                        mult = 1.0
                        stake_after = base_stake
                    else:
                        mult_raw = float(ratio) ** float(power)
                        if not np.isfinite(mult_raw) or mult_raw <= 0:
                            mult_raw = 0.0
                        mult = max(float(min_mult), float(mult_raw))
                        mult = min(1.0, float(mult))
                        if not np.isfinite(mult) or mult <= 0:
                            mult = max(float(min_mult), 1e-9)
                            mult = min(1.0, float(mult))
                        stake_raw = base_stake * float(mult)
                        stake_after = int((stake_raw // unit) * unit) if unit > 0 else stake_raw
                        if base_stake > 0:
                            stake_after = max(float(unit), float(stake_after))
                        if stake_after > base_stake:
                            stake_after = base_stake
            bet.stake = float(stake_after)
            extra = bet.extra or {}
            extra.update({
                "stake_before": base_stake,
                "stake_mult": float(mult),
                "stake_after": float(stake_after),
                "stake_damp_ref_odds": float(ref_odds) if enabled_eff else 0.0,
                "stake_damp_power": float(power) if enabled_eff else 1.0,
                "stake_damp_min_mult": float(min_mult) if enabled_eff else 0.0,
                "stake_damp_enabled": enabled_eff,
                "stake_damp_odds": float(odds_num) if odds_num is not None else None,
                "stake_damp_ratio": float(ratio) if ratio is not None and np.isfinite(ratio) else None,
                "stake_damp_mult_raw": float(mult_raw),
                "stake_damp_mult_clamped": float(mult),
                "stake_damp_floor_unit": int(unit),
            })
            bet.extra = extra

            if enabled_eff and odds_val is not None:
                try:
                    odds_num = float(odds_val)
                except Exception:
                    odds_num = None
                if odds_num is not None and odds_num < ref_odds:
                    self._stake_damp_low_odds_stake_before += base_stake
                    self._stake_damp_low_odds_stake_after += float(stake_after)
            self._stake_damp_mult_sum += float(mult)
            self._stake_damp_n += 1

        self._stake_damp_ref_odds = float(ref_odds) if enabled_eff else 0.0
        self._stake_damp_power = float(power) if enabled_eff else 1.0
        self._stake_damp_min_mult = float(min_mult) if enabled_eff else 0.0
        return bets

    def _apply_odds_floor_postprocess(self, result: "BacktestResult") -> None:
        min_odds = getattr(self.config.betting, "odds_floor_min_odds", 0.0)
        try:
            min_odds_val = float(min_odds) if min_odds is not None else 0.0
        except Exception:
            min_odds_val = 0.0
        enabled = bool(min_odds_val and min_odds_val > 0)
        if not enabled:
            self._odds_floor_min_odds = 0.0
            return

        filtered: list[BetResult] = []
        filtered_bets = 0
        filtered_stake = 0.0
        for br in result.bets:
            odds_val = br.bet.odds_at_buy
            passed = None
            if odds_val is not None:
                passed = bool(odds_val >= min_odds_val)
            extra = br.bet.extra or {}
            extra.update({
                "odds_floor_min_odds": float(min_odds_val),
                "odds_floor_odds_used": float(odds_val) if odds_val is not None else None,
                "passed_odds_floor": passed,
            })
            br.bet.extra = extra
            if passed is False:
                filtered_bets += 1
                filtered_stake += float(br.bet.stake)
                continue
            filtered.append(br)

        result.bets = filtered
        result.n_bets = len(filtered)
        result.n_wins = sum(1 for b in filtered if b.is_win)
        result.total_stake = sum(b.bet.stake for b in filtered)
        result.total_payout = sum(b.payout for b in filtered)
        result.total_profit = result.total_payout - result.total_stake
        result.roi = result.total_profit / result.total_stake if result.total_stake > 0 else 0.0

        bankroll = result.initial_bankroll
        peak_bankroll = bankroll
        min_bankroll = bankroll
        log_growths: list[float] = []
        max_dd = 0.0
        for bet_result in filtered:
            bankroll_before = bankroll
            bankroll += bet_result.profit
            min_bankroll = min(min_bankroll, bankroll)
            if bankroll_before > 0 and bankroll > 0:
                log_growths.append(math.log(bankroll / bankroll_before))
            peak_bankroll = max(peak_bankroll, bankroll)
            dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0
            max_dd = max(max_dd, dd)

        result.final_bankroll = bankroll
        result.min_bankroll = min_bankroll
        result.max_drawdown = max_dd
        result.max_drawdown_bankroll = max_dd
        result.log_growth = float(np.mean(log_growths)) if log_growths else None

        self._odds_floor_min_odds = float(min_odds_val)
        self._odds_floor_filtered_bets = int(filtered_bets)
        self._odds_floor_filtered_stake = float(filtered_stake)

    @staticmethod
    def _resolve_daily_top_n_params(daily_cfg) -> tuple[str, float | None, str, str]:
        metric = str(getattr(daily_cfg, "metric", "ev") or "ev").lower()
        if metric not in ("ev", "overlay"):
            metric = "ev"
        min_value = getattr(daily_cfg, "min_value", None)
        if min_value is None:
            legacy = getattr(daily_cfg, "min_ev", None)
            if legacy is not None:
                min_value = float(legacy)
        else:
            try:
                min_value = float(min_value)
            except Exception:
                min_value = None
        scope = str(getattr(daily_cfg, "scope", "date") or "date")
        tie_break = str(getattr(daily_cfg, "tie_break", "score_desc_then_odds_asc") or "score_desc_then_odds_asc")
        return metric, min_value, scope, tie_break

    @staticmethod
    def _daily_topn_score(cand: dict, metric: str) -> float | None:
        if metric == "overlay":
            val = cand.get("overlay")
        else:
            val = cand.get("ev")
        if val is None or not np.isfinite(val):
            return None
        return float(val)

    @staticmethod
    def _daily_topn_sort_key(cand: dict, tie_break: str) -> tuple:
        odds = cand.get("odds_at_buy")
        if odds is not None and np.isfinite(odds):
            odds_key = float(odds)
        else:
            odds_key = float("inf")

        if str(tie_break) == "ev_desc_then_overlay_desc_then_odds_asc":
            ev = cand.get("ev")
            if ev is not None and np.isfinite(ev):
                ev_key = -float(ev)
            else:
                ev_key = float("inf")
            overlay = cand.get("overlay")
            if overlay is not None and np.isfinite(overlay):
                overlay_key = -float(overlay)
            else:
                overlay_key = float("inf")
            return (ev_key, overlay_key, odds_key, str(cand.get("race_id") or ""), int(cand.get("horse_no") or 0))

        score = cand.get("daily_score")
        if score is not None and np.isfinite(score):
            score_key = -float(score)
        else:
            score_key = float("inf")
        return (score_key, odds_key, str(cand.get("race_id") or ""), int(cand.get("horse_no") or 0))

    def _apply_daily_top_n(
        self,
        candidates: list[dict],
        daily_cfg,
        race_date: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        stats: list[dict] = []
        if daily_cfg is None:
            return candidates, stats

        n = int(getattr(daily_cfg, "n", 0) or 0)
        metric, min_value, scope, tie_break = self._resolve_daily_top_n_params(daily_cfg)

        if not candidates:
            if race_date is not None:
                stats.append({
                    "race_date": race_date,
                    "candidates_before": 0,
                    "candidates_after_min": 0,
                    "selected": 0,
                    "n": n,
                    "metric": metric,
                    "min_value": min_value,
                })
            return [], stats

        groups: dict[object, list[dict]] = {}
        if str(scope) != "date":
            scope = "date"
        for cand in candidates:
            day_key = cand.get("asof_time").date() if cand.get("asof_time") else None
            groups.setdefault(day_key, []).append(cand)

        selected: list[dict] = []
        for day_key, group in groups.items():
            candidates_before = len(group)
            eligible = []
            for cand in group:
                score = self._daily_topn_score(cand, metric)
                if score is None:
                    continue
                cand["daily_score"] = score
                if min_value is not None and score < float(min_value):
                    continue
                eligible.append(cand)
            candidates_after = len(eligible)
            group_sorted = sorted(eligible, key=lambda c: self._daily_topn_sort_key(c, tie_break))
            pick = group_sorted[:n] if n > 0 else []
            for rank, cand in enumerate(pick, start=1):
                cand["daily_rank_in_day"] = rank
                cand["daily_candidates_before"] = candidates_before
                cand["daily_candidates_after_min"] = candidates_after
                cand["daily_selected"] = True
                cand["daily_top_n_n"] = n
                cand["daily_top_n_metric"] = metric
                cand["daily_top_n_min_value"] = min_value
                selected.append(cand)
            stats.append({
                "race_date": str(day_key) if day_key is not None else (race_date or "unknown"),
                "candidates_before": candidates_before,
                "candidates_after_min": candidates_after,
                "selected": len(pick),
                "n": n,
                "metric": metric,
                "min_value": min_value,
            })

        if race_date is not None:
            try:
                race_day = datetime.strptime(race_date, "%Y%m%d").date()
            except Exception:
                race_day = None
            if race_day is not None and race_day not in groups:
                stats.append({
                    "race_date": race_date,
                    "candidates_before": 0,
                    "candidates_after_min": 0,
                    "selected": 0,
                    "n": n,
                    "metric": metric,
                    "min_value": min_value,
                })

        return selected, stats

    def _build_candidates(
        self,
        race_id: str,
        predictions: list[dict],
        asof_time: datetime,
    ) -> list[dict]:
        min_odds = self.config.betting.min_odds
        max_odds = self.config.betting.max_odds
        min_buy_odds = getattr(self.config.betting, "min_buy_odds", None)
        max_buy_odds = getattr(self.config.betting, "max_buy_odds", None)
        ts_vol_cap = self.config.betting.log_odds_std_60m_max
        reject_ts_missing = bool(self.config.betting.reject_if_log_odds_std_60m_missing)
        rs_cfg = getattr(self.config.model, "race_softmax", None)
        race_softmax_enabled = bool(rs_cfg and getattr(rs_cfg, "enabled", False))
        race_cost_cap_mode, race_cost_cap_value, race_cost_cap_selected_from = self._race_cost_filter_extra_meta()
        market_blend_enabled = bool(getattr(self.config.betting, "enable_market_blend", False))
        market_method = str(getattr(self.config.betting, "market_prob_method", "p_mkt_col") or "p_mkt_col")
        t_ev = float(getattr(self.config.betting, "t_ev", self.config.betting.ev_margin))
        odds_cap = getattr(self.config.betting, "odds_cap", None)
        exclude_bands = parse_exclude_odds_band(getattr(self.config.betting, "exclude_odds_band", None))
        blend_w = float(getattr(self.config.betting, "market_blend_w", 1.0))
        try:
            odds_cap_val = float(odds_cap) if odds_cap is not None else None
        except Exception:
            odds_cap_val = None
        if not math.isfinite(blend_w):
            blend_w = 1.0
        if market_blend_enabled and market_method.lower() != "p_mkt_col":
            raise ValueError(f"market_prob_method must be p_mkt_col (got: {market_method})")

        candidates = []
        for pred in predictions:
            odds_at_buy = pred["odds"]
            race_softmax_enabled_pred = bool(pred.get("race_softmax_enabled", race_softmax_enabled))
            p_hat_raw = pred["p_hat"]
            p_hat_pre_softmax = None
            if race_softmax_enabled_pred and pred.get("p_used") is not None:
                p_hat_pre_softmax = p_hat_raw
                p_hat_raw = float(pred.get("p_used"))
            p_mkt = pred.get("p_mkt")
            p_mkt_raw = pred.get("p_mkt_raw")
            p_mkt_race = pred.get("p_mkt_race")
            overround_sum_inv = pred.get("overround_sum_inv")
            takeout_implied = pred.get("takeout_implied")
            race_cost_passed, race_cost_reason = self._race_cost_filter(pred)

            if odds_at_buy <= 0 or p_hat_raw <= 0:
                continue

            if not race_cost_passed:
                continue

            if min_buy_odds is not None and odds_at_buy < float(min_buy_odds):
                continue
            if max_buy_odds is not None and odds_at_buy > float(max_buy_odds):
                continue

            if ts_vol_cap is not None:
                v = pred.get("log_odds_std_60m")
                try:
                    v_num = float(v) if v is not None else None
                except Exception:
                    v_num = None

                if v_num is None or not np.isfinite(v_num):
                    if reject_ts_missing:
                        continue
                else:
                    if v_num > float(ts_vol_cap):
                        continue

            if min_odds is not None and odds_at_buy < float(min_odds):
                continue
            if max_odds is not None and odds_at_buy > float(max_odds):
                continue

            p_hat = float(p_hat_raw)
            p_hat_shrunk = None
            shrink_alpha = self.config.betting.overlay_shrink_alpha
            bias_meta = None
            if not race_softmax_enabled:
                if shrink_alpha is not None:
                    p_hat_shrunk = shrink_probability(p_hat_raw, p_mkt, shrink_alpha, self.config.model.p_mkt_clip)
                    p_hat = float(p_hat_shrunk)

                if self.odds_band_bias is not None and bool(self.config.betting.odds_band_bias.enabled):
                    p_hat, bias_meta = self.odds_band_bias.apply(p_cal=p_hat, odds_buy=odds_at_buy)

            if self.slippage_table is None:
                if not pred.get("has_ts_odds", True):
                    if self.config.backtest.slippage.enabled_if_no_ts_odds:
                        odds_at_buy = odds_at_buy * self.config.backtest.slippage.odds_multiplier

            closing_mult = float(self.config.betting.closing_odds_multiplier)
            slippage_meta = None
            if self.slippage_table is not None:
                odds_effective, slippage_meta = self.slippage_table.effective_odds(
                    odds_buy=odds_at_buy,
                    ts_vol=pred.get("log_odds_std_60m"),
                    snap_age_min=pred.get("snap_age_min"),
                )
            else:
                odds_effective = odds_at_buy * closing_mult if closing_mult > 0 else odds_at_buy
            p_hat_pre_blend = None
            p_blend = None
            ev_blend = None
            odds_band_blend = None
            market_blend_used = False
            if market_blend_enabled:
                p_hat_pre_blend = float(p_hat)
                blend_res = compute_market_blend(
                    p_model=pred.get("p_model"),
                    p_mkt_col=p_mkt,
                    odds=odds_effective,
                    blend_w=blend_w,
                    market_method=market_method,
                )
                p_blend = blend_res.p_blend
                ev_blend = blend_res.ev_blend
                odds_band_blend = blend_res.odds_band
                if odds_cap_val is not None and odds_at_buy > odds_cap_val:
                    continue
                if exclude_bands and odds_band_blend in exclude_bands:
                    continue
                p_hat = float(p_blend)
                ev = float(ev_blend)
                market_blend_used = True
            else:
                ev = p_hat * odds_effective - 1
            overlay_logit = None
            overlay = None
            if p_mkt is not None:
                try:
                    p_mkt_f = float(p_mkt)
                    p_used_f = float(p_hat)
                    p_mkt_f = min(max(p_mkt_f, 1e-6), 1.0 - 1e-6)
                    p_used_f = min(max(p_used_f, 1e-6), 1.0 - 1e-6)
                    overlay_logit = math.log(p_used_f / (1.0 - p_used_f)) - math.log(p_mkt_f / (1.0 - p_mkt_f))
                    overlay = float(p_used_f) - float(p_mkt_f)
                except Exception:
                    overlay_logit = None
                    overlay = None

            odds_dyn_cfg = getattr(self.config.betting, "odds_dynamics_filter", None)
            odds_dyn_metric_value = None
            if odds_dyn_cfg is not None:
                odds_dyn_metric_value = compute_odds_dyn_metric(
                    pred,
                    getattr(odds_dyn_cfg, "metric", "odds_delta_log"),
                    getattr(odds_dyn_cfg, "lookback_minutes", 10),
                )
            odds_dyn_ev_cfg = getattr(self.config.betting, "odds_dyn_ev_margin", None)
            odds_dyn_ev_score = None
            if odds_dyn_ev_cfg is not None:
                odds_dyn_ev_score = compute_odds_dyn_metric(
                    pred,
                    getattr(odds_dyn_ev_cfg, "metric", "odds_delta_log"),
                    getattr(odds_dyn_ev_cfg, "lookback_minutes", 5),
                )

            candidates.append({
                "race_id": race_id,
                "horse_no": pred.get("horse_no"),
                "asof_time": asof_time,
                "pred": pred,
                "odds_at_buy": odds_at_buy,
                "p_hat_raw": p_hat_raw,
                "p_hat_pre_softmax": p_hat_pre_softmax,
                "p_hat": p_hat,
                "p_hat_shrunk": p_hat_shrunk,
                "shrink_alpha": shrink_alpha,
                "bias_meta": bias_meta,
                "slippage_meta": slippage_meta,
                "odds_effective": odds_effective,
                "ev": ev,
                "market_blend_enabled": market_blend_used,
                "market_prob_method": market_method,
                "market_blend_w": float(blend_w) if market_blend_used else None,
                "p_blend": p_blend,
                "ev_blend": ev_blend,
                "odds_band_blend": odds_band_blend,
                "t_ev": float(t_ev) if market_blend_used else None,
                "odds_cap": float(odds_cap_val) if (market_blend_used and odds_cap_val is not None) else None,
                "exclude_odds_band": ",".join(exclude_bands) if exclude_bands else None,
                "p_hat_pre_blend": p_hat_pre_blend,
                "overlay_logit": overlay_logit,
                "overlay": overlay,
                "odds_dyn_metric_value": odds_dyn_metric_value,
                "odds_dyn_ev_score": odds_dyn_ev_score,
                "p_mkt": p_mkt,
                "p_mkt_raw": p_mkt_raw,
                "p_mkt_race": p_mkt_race,
                "overround_sum_inv": overround_sum_inv,
                "takeout_implied": takeout_implied,
                "race_cost_cap_mode": race_cost_cap_mode,
                "race_cost_cap_value": race_cost_cap_value,
                "race_cost_cap_selected_from": race_cost_cap_selected_from,
                "race_cost_filter_passed": race_cost_passed,
                "race_cost_filter_reason": race_cost_reason,
                "race_softmax_w": pred.get("race_softmax_w"),
                "race_softmax_T": pred.get("race_softmax_T"),
                "race_softmax_enabled": pred.get("race_softmax_enabled", False),
                "race_softmax_enabled_pred": race_softmax_enabled_pred,
                "closing_mult": closing_mult,
            })

        return candidates

    def _build_bets_from_candidates(
        self,
        candidates: list[dict],
        bankroll: float,
        *,
        remaining_daily_budget: Optional[float],
        selection_mode: str,
    ) -> list[Bet]:
        bets = []
        ev_margin = self.config.betting.ev_margin
        if bool(getattr(self.config.betting, "enable_market_blend", False)):
            ev_margin = float(getattr(self.config.betting, "t_ev", ev_margin))
        takeout_ev_cfg = getattr(self.config.betting, "takeout_ev_margin", None)
        takeout_ev_enabled = bool(takeout_ev_cfg and getattr(takeout_ev_cfg, "enabled", False))
        takeout_ref = float(getattr(takeout_ev_cfg, "ref_takeout", 0.215)) if takeout_ev_cfg is not None else 0.215
        takeout_slope = float(getattr(takeout_ev_cfg, "slope", 0.0)) if takeout_ev_cfg is not None else 0.0
        min_buy_odds = getattr(self.config.betting, "min_buy_odds", None)
        max_buy_odds = getattr(self.config.betting, "max_buy_odds", None)
        stake_cfg = getattr(self.config.betting, "stake", None)
        stake_enabled = bool(stake_cfg and getattr(stake_cfg, "enabled", False))
        uncert_cfg = getattr(self.config.betting, "uncertainty", None)
        uncert_enabled = bool(
            uncert_cfg and getattr(uncert_cfg, "enabled", False) and self.uncertainty_shrink is not None
        )
        use_new_stake = stake_enabled or uncert_enabled
        min_yen = int(getattr(stake_cfg, "min_yen", 100)) if stake_cfg is not None else 100

        ev_cap_q = getattr(self.config.betting, "ev_cap_quantile", None)
        ov_cap_q = getattr(self.config.betting, "overlay_abs_cap_quantile", None)
        reject_overlay_missing = bool(getattr(self.config.betting, "reject_if_overlay_missing", True))
        max_bets_per_race = int(self.config.betting.max_bets_per_race)

        ev_cap_thr = None
        if ev_cap_q is not None and candidates:
            ev_vals = [c["ev"] for c in candidates if c.get("ev") is not None and np.isfinite(c["ev"])]
            if ev_vals:
                ev_cap_thr = float(np.quantile(ev_vals, float(ev_cap_q)))

        ov_cap_thr = None
        if ov_cap_q is not None and candidates:
            ov_vals = [
                abs(c["overlay_logit"])
                for c in candidates
                if c.get("overlay_logit") is not None and np.isfinite(c["overlay_logit"])
            ]
            if ov_vals:
                ov_cap_thr = float(np.quantile(ov_vals, float(ov_cap_q)))

        for cand in candidates:
            pred = cand["pred"]
            odds_at_buy = cand["odds_at_buy"]
            p_hat_raw = cand["p_hat_raw"]
            p_hat_pre_softmax = cand["p_hat_pre_softmax"]
            p_hat = cand["p_hat"]
            p_hat_shrunk = cand["p_hat_shrunk"]
            shrink_alpha = cand["shrink_alpha"]
            bias_meta = cand["bias_meta"]
            slippage_meta = cand["slippage_meta"]
            odds_effective = cand["odds_effective"]
            ev = cand["ev"]
            overlay_logit = cand["overlay_logit"]
            p_mkt = cand["p_mkt"]
            p_mkt_raw = cand.get("p_mkt_raw")
            p_mkt_race = cand.get("p_mkt_race")
            overround_sum_inv = cand.get("overround_sum_inv")
            takeout_implied = cand.get("takeout_implied")
            race_cost_passed = cand.get("race_cost_filter_passed")
            race_cost_reason = cand.get("race_cost_filter_reason")
            race_softmax_w = cand.get("race_softmax_w")
            race_softmax_T = cand.get("race_softmax_T")
            race_softmax_enabled = cand.get("race_softmax_enabled")
            race_softmax_enabled_pred = cand.get("race_softmax_enabled_pred")
            closing_mult = cand["closing_mult"]
            odds_dyn_ev_score = cand.get("odds_dyn_ev_score")

            base_min_ev = float(ev_margin)
            min_ev_eff = base_min_ev
            if takeout_ev_enabled and takeout_slope and takeout_implied is not None:
                try:
                    takeout_val = float(takeout_implied)
                except Exception:
                    takeout_val = None
                if takeout_val is not None and np.isfinite(takeout_val):
                    min_ev_eff = base_min_ev + takeout_slope * max(0.0, takeout_val - takeout_ref)
            passed_takeout_ev_margin = True
            if selection_mode == "ev_threshold":
                passed_takeout_ev_margin = ev >= min_ev_eff

            odds_dyn_ev_cfg = getattr(self.config.betting, "odds_dyn_ev_margin", None)
            odds_dyn_ev_enabled = bool(odds_dyn_ev_cfg and getattr(odds_dyn_ev_cfg, "enabled", False))
            odds_dyn_ev_metric = str(getattr(odds_dyn_ev_cfg, "metric", "odds_delta_log") or "odds_delta_log")
            odds_dyn_ev_lookback = int(getattr(odds_dyn_ev_cfg, "lookback_minutes", 5) or 5) if odds_dyn_ev_cfg else 5
            odds_dyn_ev_direction = str(getattr(odds_dyn_ev_cfg, "direction", "high") or "high")
            try:
                odds_dyn_ev_ref = float(getattr(odds_dyn_ev_cfg, "ref", None))
            except Exception:
                odds_dyn_ev_ref = None
            try:
                odds_dyn_ev_slope = float(getattr(odds_dyn_ev_cfg, "slope", 0.0))
            except Exception:
                odds_dyn_ev_slope = 0.0
            odds_dyn_ev_margin = None
            min_ev_eff_odds_dyn = None
            passed_odds_dyn_ev_margin = None
            odds_dyn_ev_enabled_eff = bool(
                odds_dyn_ev_enabled
                and odds_dyn_ev_ref is not None
                and odds_dyn_ev_slope is not None
                and odds_dyn_ev_slope > 0
            )
            if odds_dyn_ev_score is not None and odds_dyn_ev_ref is not None:
                if odds_dyn_ev_direction == "low":
                    odds_dyn_ev_margin = odds_dyn_ev_slope * max(0.0, odds_dyn_ev_ref - odds_dyn_ev_score)
                else:
                    odds_dyn_ev_margin = odds_dyn_ev_slope * max(0.0, odds_dyn_ev_score - odds_dyn_ev_ref)
                min_ev_eff_odds_dyn = min_ev_eff + (odds_dyn_ev_margin or 0.0)
                if odds_dyn_ev_enabled_eff:
                    passed_odds_dyn_ev_margin = ev >= min_ev_eff_odds_dyn

            if reject_overlay_missing and overlay_logit is None:
                continue

            ev_cap_passed = True
            if ev_cap_q is not None and ev_cap_thr is not None:
                ev_cap_passed = ev <= ev_cap_thr
            if not ev_cap_passed:
                continue

            ov_cap_passed = True
            if ov_cap_q is not None and ov_cap_thr is not None:
                if overlay_logit is None:
                    ov_cap_passed = False if reject_overlay_missing else None
                else:
                    ov_cap_passed = abs(overlay_logit) <= ov_cap_thr
            if ov_cap_passed is False:
                continue

            if selection_mode == "ev_threshold":
                if ev < min_ev_eff:
                    continue

            if self.ev_upper_cap is not None and ev > self.ev_upper_cap:
                continue

            max_pct_raw = self.config.betting.caps.per_race_pct
            stake_raw = calculate_stake(
                p_hat=p_hat,
                odds=odds_effective,
                bankroll=bankroll,
                method=self.config.betting.sizing.method,
                fraction=self.config.betting.sizing.fraction,
                max_pct=1.0 if stake_enabled else max_pct_raw,
                min_stake=min_yen,
            )

            if stake_raw < min_yen:
                continue

            bet = Bet(
                race_id=cand.get("race_id"),
                horse_no=pred["horse_no"],
                ticket_type="win",
                stake=stake_raw,
                odds_at_buy=odds_at_buy,
                odds_effective=odds_effective,
                p_hat=p_hat,
                ev=ev,
                asof_time=cand.get("asof_time"),
                extra={
                    "p_mkt": p_mkt,
                    "p_mkt_raw": p_mkt_raw,
                    "p_mkt_race": p_mkt_race,
                    "overround_sum_inv": overround_sum_inv,
                    "takeout_implied": takeout_implied,
                    "base_min_ev": base_min_ev,
                    "market_blend_enabled": cand.get("market_blend_enabled"),
                    "market_prob_method": cand.get("market_prob_method"),
                    "market_blend_w": cand.get("market_blend_w"),
                    "p_blend": cand.get("p_blend"),
                    "ev_blend": cand.get("ev_blend"),
                    "odds_band_blend": cand.get("odds_band_blend"),
                    "t_ev": cand.get("t_ev"),
                    "odds_cap": cand.get("odds_cap"),
                    "exclude_odds_band": cand.get("exclude_odds_band"),
                    "p_hat_pre_blend": cand.get("p_hat_pre_blend"),
                    "takeout_ref": float(takeout_ref),
                    "takeout_slope": float(takeout_slope),
                    "min_ev_eff": float(min_ev_eff),
                    "passed_takeout_ev_margin": bool(passed_takeout_ev_margin),
                    "odds_dyn_ev_metric": odds_dyn_ev_metric,
                    "odds_dyn_ev_lookback_min": odds_dyn_ev_lookback,
                    "odds_dyn_ev_score": odds_dyn_ev_score,
                    "odds_dyn_ev_ref": odds_dyn_ev_ref,
                    "odds_dyn_ev_slope": odds_dyn_ev_slope,
                    "odds_dyn_ev_margin": odds_dyn_ev_margin,
                    "min_ev_eff_odds_dyn": min_ev_eff_odds_dyn,
                    "passed_odds_dyn_ev_margin": passed_odds_dyn_ev_margin if odds_dyn_ev_enabled_eff else None,
                    "odds_dyn_ev_margin_enabled": odds_dyn_ev_enabled_eff,
                    "race_cost_cap_mode": cand.get("race_cost_cap_mode"),
                    "race_cost_cap_value": cand.get("race_cost_cap_value"),
                    "race_cost_cap_selected_from": cand.get("race_cost_cap_selected_from"),
                    "race_cost_filter_passed": race_cost_passed,
                    "race_cost_filter_reason": race_cost_reason,
                    "race_softmax_w": race_softmax_w,
                    "race_softmax_T": race_softmax_T,
                    "race_softmax_enabled": bool(race_softmax_enabled_pred),
                    "rsx_selected": pred.get(
                        "race_softmax_selected",
                        "softmax" if race_softmax_enabled_pred else "baseline",
                    ),
                    "rsx_w": float(race_softmax_w) if race_softmax_w is not None else None,
                    "rsx_T": float(race_softmax_T) if race_softmax_T is not None else None,
                    "p_hat_pre_softmax": float(p_hat_pre_softmax) if p_hat_pre_softmax is not None else None,
                    "segblend_segment": pred.get("segblend_segment"),
                    "segblend_w_used": pred.get("segblend_w_used"),
                    "segblend_w_global": pred.get("segblend_w_global"),
                    "has_ts_odds": bool(pred.get("has_ts_odds", True)),
                    "odds_window_passed": True,
                    "min_buy_odds": float(min_buy_odds) if min_buy_odds is not None else None,
                    "max_buy_odds": float(max_buy_odds) if max_buy_odds is not None else None,
                    "ev_cap_q": float(ev_cap_q) if ev_cap_q is not None else None,
                    "ev_cap_thr": float(ev_cap_thr) if ev_cap_thr is not None else None,
                    "ev_cap_passed": bool(ev_cap_passed),
                    "ov_cap_q": float(ov_cap_q) if ov_cap_q is not None else None,
                    "ov_cap_thr": float(ov_cap_thr) if ov_cap_thr is not None else None,
                    "ov_cap_passed": ov_cap_passed,
                    "closing_odds_multiplier": closing_mult,
                    "slippage_meta": slippage_meta,
                    "p_hat_raw": float(p_hat_raw),
                    "p_hat_shrunk": float(p_hat_shrunk) if p_hat_shrunk is not None else None,
                    "overlay_shrink_alpha": float(shrink_alpha) if shrink_alpha is not None else None,
                    "p_hat_adj": float(p_hat),
                    "bias_meta": bias_meta,
                    "ts_vol_cap": float(self.config.betting.log_odds_std_60m_max) if self.config.betting.log_odds_std_60m_max is not None else None,
                    "ts_vol_cap_passed": True if self.config.betting.log_odds_std_60m_max is not None else None,
                    "odds_chg_5m": pred.get("odds_chg_5m"),
                    "odds_chg_10m": pred.get("odds_chg_10m"),
                    "odds_chg_30m": pred.get("odds_chg_30m"),
                    "odds_chg_60m": pred.get("odds_chg_60m"),
                    "p_mkt_chg_5m": pred.get("p_mkt_chg_5m"),
                    "p_mkt_chg_10m": pred.get("p_mkt_chg_10m"),
                    "p_mkt_chg_30m": pred.get("p_mkt_chg_30m"),
                    "p_mkt_chg_60m": pred.get("p_mkt_chg_60m"),
                    "log_odds_slope_60m": pred.get("log_odds_slope_60m"),
                    "log_odds_std_60m": pred.get("log_odds_std_60m"),
                    "n_pts_60m": pred.get("n_pts_60m"),
                    "odds_dyn_metric_value": cand.get("odds_dyn_metric_value"),
                    "snap_age_min": pred.get("snap_age_min"),
                    "resid": pred.get("resid"),
                    "resid_cap": pred.get("resid_cap"),
                    "cap_value": pred.get("cap_value"),
                    "p_hat_capped": pred.get("p_hat_capped"),
                    "selection_mode": selection_mode,
                    "daily_top_n_n": cand.get("daily_top_n_n"),
                    "daily_top_n_metric": cand.get("daily_top_n_metric"),
                    "daily_top_n_min_value": cand.get("daily_top_n_min_value"),
                    "daily_rank_in_day": cand.get("daily_rank_in_day"),
                    "daily_candidates_before": cand.get("daily_candidates_before"),
                    "daily_candidates_after_min": cand.get("daily_candidates_after_min"),
                    "daily_selected": cand.get("daily_selected"),
                },
            )
            bet.extra["stake_raw"] = int(stake_raw)
            if not use_new_stake:
                bet.extra.update({
                    "stake_clipped": int(stake_raw),
                    "clip_reason": None,
                    "uncert_mult": None,
                    "uncert_n_bin": None,
                })
            bets.append(bet)

        # per-race max
        if max_bets_per_race > 0:
            grouped: dict[str, list[Bet]] = {}
            for bet in bets:
                grouped.setdefault(bet.race_id, []).append(bet)
            reduced = []
            for race_id, race_bets in grouped.items():
                race_bets.sort(key=lambda b: b.ev, reverse=True)
                reduced.extend(race_bets[:max_bets_per_race])
            bets = reduced

        bets = sorted(bets, key=lambda b: b.ev, reverse=True)

        if not use_new_stake:
            if remaining_daily_budget is not None and remaining_daily_budget > 0:
                selected = []
                used = 0
                for bet in bets:
                    if used + bet.stake <= remaining_daily_budget:
                        selected.append(bet)
                        used += bet.stake
                bets = selected
            bets = self._apply_odds_dynamics_filter(bets)
            bets = self._apply_odds_dyn_ev_margin(bets)
            return bets

        max_frac_per_bet = getattr(stake_cfg, "max_frac_per_bet", None) if stake_cfg is not None else None
        if max_frac_per_bet is None:
            max_frac_per_bet = self.config.betting.caps.per_race_pct
        max_frac_per_race = getattr(stake_cfg, "max_frac_per_race", None) if stake_cfg is not None else None
        if max_frac_per_race is None:
            max_frac_per_race = max_frac_per_bet
        max_yen_per_bet = getattr(stake_cfg, "max_yen_per_bet", None) if stake_cfg is not None else None

        selected = []
        for bet in bets:
            extra = bet.extra or {}
            stake_raw = int(extra.get("stake_raw", bet.stake))
            stake_clipped, clip_reason = clip_stake(
                stake_raw,
                bankroll,
                min_yen=min_yen,
                max_frac_per_bet=max_frac_per_bet,
                max_yen_per_bet=max_yen_per_bet,
                remaining_race_budget=None,
                remaining_daily_budget=None,
            )
            if stake_clipped < min_yen:
                continue

            stake_final = stake_clipped
            uncert_mult = None
            uncert_n_bin = None
            if uncert_enabled and self.uncertainty_shrink is not None:
                p_for_uncert = extra.get("p_hat_raw", bet.p_hat)
                try:
                    p_for_uncert = float(p_for_uncert)
                except Exception:
                    p_for_uncert = float(bet.p_hat)
                uncert_mult, meta = self.uncertainty_shrink.apply(p_for_uncert)
                stake_final = int((stake_clipped * float(uncert_mult)) // min_yen * min_yen)
                uncert_n_bin = meta.get("n_bin") if isinstance(meta, dict) else None
                if stake_final < min_yen:
                    continue

            bet.stake = stake_final
            extra.update({
                "stake_raw": stake_raw,
                "stake_clipped": stake_clipped,
                "clip_reason": clip_reason,
                "uncert_mult": float(uncert_mult) if uncert_mult is not None else None,
                "uncert_n_bin": int(uncert_n_bin) if uncert_n_bin is not None else None,
            })
            bet.extra = extra
            selected.append(bet)

        if max_frac_per_race is not None:
            race_cap = float(bankroll) * float(max_frac_per_race)
            self._scale_bets_to_cap(selected, race_cap, min_yen, reason="per_race")

        selected = self._apply_odds_dynamics_filter(selected)
        selected = self._apply_odds_dyn_ev_margin(selected)
        selected = [b for b in selected if b.stake >= min_yen]
        return selected

    def _generate_bets(
        self,
        race_id: str,
        predictions: list[dict],
        asof_time: datetime,
        bankroll: float,
        remaining_daily_budget: Optional[float] = None,
    ) -> list[Bet]:
        """
        ベット生成
        
        Args:
            remaining_daily_budget: 日次残り予算（Noneなら制限なし）
        """
        bets = []
        ev_margin = self.config.betting.ev_margin
        if bool(getattr(self.config.betting, "enable_market_blend", False)):
            ev_margin = float(getattr(self.config.betting, "t_ev", ev_margin))
        takeout_ev_cfg = getattr(self.config.betting, "takeout_ev_margin", None)
        takeout_ev_enabled = bool(takeout_ev_cfg and getattr(takeout_ev_cfg, "enabled", False))
        takeout_ref = float(getattr(takeout_ev_cfg, "ref_takeout", 0.215)) if takeout_ev_cfg is not None else 0.215
        takeout_slope = float(getattr(takeout_ev_cfg, "slope", 0.0)) if takeout_ev_cfg is not None else 0.0
        min_odds = self.config.betting.min_odds
        max_odds = self.config.betting.max_odds
        min_buy_odds = getattr(self.config.betting, "min_buy_odds", None)
        max_buy_odds = getattr(self.config.betting, "max_buy_odds", None)
        ts_vol_cap = self.config.betting.log_odds_std_60m_max
        reject_ts_missing = bool(self.config.betting.reject_if_log_odds_std_60m_missing)
        stake_cfg = getattr(self.config.betting, "stake", None)
        stake_enabled = bool(stake_cfg and getattr(stake_cfg, "enabled", False))
        uncert_cfg = getattr(self.config.betting, "uncertainty", None)
        uncert_enabled = bool(
            uncert_cfg and getattr(uncert_cfg, "enabled", False) and self.uncertainty_shrink is not None
        )
        use_new_stake = stake_enabled or uncert_enabled
        min_yen = int(getattr(stake_cfg, "min_yen", 100)) if stake_cfg is not None else 100
        
        ev_cap_q = getattr(self.config.betting, "ev_cap_quantile", None)
        ov_cap_q = getattr(self.config.betting, "overlay_abs_cap_quantile", None)
        reject_overlay_missing = bool(getattr(self.config.betting, "reject_if_overlay_missing", True))
        rs_cfg = getattr(self.config.model, "race_softmax", None)
        race_softmax_enabled = bool(rs_cfg and getattr(rs_cfg, "enabled", False))
        race_cost_cap_mode, race_cost_cap_value, race_cost_cap_selected_from = self._race_cost_filter_extra_meta()
        market_blend_enabled = bool(getattr(self.config.betting, "enable_market_blend", False))
        market_method = str(getattr(self.config.betting, "market_prob_method", "p_mkt_col") or "p_mkt_col")
        t_ev = float(getattr(self.config.betting, "t_ev", self.config.betting.ev_margin))
        odds_cap = getattr(self.config.betting, "odds_cap", None)
        exclude_bands = parse_exclude_odds_band(getattr(self.config.betting, "exclude_odds_band", None))
        blend_w = float(getattr(self.config.betting, "market_blend_w", 1.0))
        try:
            odds_cap_val = float(odds_cap) if odds_cap is not None else None
        except Exception:
            odds_cap_val = None
        if not math.isfinite(blend_w):
            blend_w = 1.0
        if market_blend_enabled and market_method.lower() != "p_mkt_col":
            raise ValueError(f"market_prob_method must be p_mkt_col (got: {market_method})")

        candidates = []
        for pred in predictions:
            odds_at_buy = pred["odds"]
            race_softmax_enabled_pred = bool(pred.get("race_softmax_enabled", race_softmax_enabled))
            p_hat_raw = pred["p_hat"]
            p_hat_pre_softmax = None
            if race_softmax_enabled_pred and pred.get("p_used") is not None:
                p_hat_pre_softmax = p_hat_raw
                p_hat_raw = float(pred.get("p_used"))
            p_mkt = pred.get("p_mkt")
            p_mkt_raw = pred.get("p_mkt_raw")
            p_mkt_race = pred.get("p_mkt_race")
            overround_sum_inv = pred.get("overround_sum_inv")
            takeout_implied = pred.get("takeout_implied")
            race_cost_passed, race_cost_reason = self._race_cost_filter(pred)

            if odds_at_buy <= 0 or p_hat_raw <= 0:
                continue

            if not race_cost_passed:
                continue

            if min_buy_odds is not None and odds_at_buy < float(min_buy_odds):
                continue
            if max_buy_odds is not None and odds_at_buy > float(max_buy_odds):
                continue

            # Step4: TS?????????EV/stake???????
            if ts_vol_cap is not None:
                v = pred.get("log_odds_std_60m")
                try:
                    v_num = float(v) if v is not None else None
                except Exception:
                    v_num = None

                if v_num is None or not np.isfinite(v_num):
                    if reject_ts_missing:
                        continue
                else:
                    if v_num > float(ts_vol_cap):
                        continue

            # Ticket N2: odds????????p_adj?
            # Ticket N6: overlay shrink (logit residual)
            p_hat = float(p_hat_raw)
            p_hat_shrunk = None
            shrink_alpha = self.config.betting.overlay_shrink_alpha
            bias_meta = None
            if not race_softmax_enabled:
                if shrink_alpha is not None:
                    p_hat_shrunk = shrink_probability(p_hat_raw, p_mkt, shrink_alpha, self.config.model.p_mkt_clip)
                    p_hat = float(p_hat_shrunk)

                # Ticket N2: odds bias (p_adj)
                if self.odds_band_bias is not None and bool(self.config.betting.odds_band_bias.enabled):
                    p_hat, bias_meta = self.odds_band_bias.apply(p_cal=p_hat, odds_buy=odds_at_buy)

            # slippage???TS????
            if self.slippage_table is None:
                if not pred.get("has_ts_odds", True):
                    if self.config.backtest.slippage.enabled_if_no_ts_odds:
                        odds_at_buy = odds_at_buy * self.config.backtest.slippage.odds_multiplier

            # odds??????buy???
            if min_odds is not None and odds_at_buy < float(min_odds):
                continue
            if max_odds is not None and odds_at_buy > float(max_odds):
                continue

            # Ticket1/TicketB: EV/?????????????
            closing_mult = float(self.config.betting.closing_odds_multiplier)
            slippage_meta = None
            if self.slippage_table is not None:
                odds_effective, slippage_meta = self.slippage_table.effective_odds(
                    odds_buy=odds_at_buy,
                    ts_vol=pred.get("log_odds_std_60m"),
                    snap_age_min=pred.get("snap_age_min"),
                )
            else:
                odds_effective = odds_at_buy * closing_mult if closing_mult > 0 else odds_at_buy
            p_hat_pre_blend = None
            p_blend = None
            ev_blend = None
            odds_band_blend = None
            market_blend_used = False
            if market_blend_enabled:
                p_hat_pre_blend = float(p_hat)
                blend_res = compute_market_blend(
                    p_model=pred.get("p_model"),
                    p_mkt_col=p_mkt,
                    odds=odds_effective,
                    blend_w=blend_w,
                    market_method=market_method,
                )
                p_blend = blend_res.p_blend
                ev_blend = blend_res.ev_blend
                odds_band_blend = blend_res.odds_band
                if odds_cap_val is not None and odds_at_buy > odds_cap_val:
                    continue
                if exclude_bands and odds_band_blend in exclude_bands:
                    continue
                p_hat = float(p_blend)
                ev = float(ev_blend)
                market_blend_used = True
            else:
                ev = p_hat * odds_effective - 1
            overlay_logit = None
            if p_mkt is not None:
                try:
                    p_mkt_f = float(p_mkt)
                    p_used_f = float(p_hat)
                    p_mkt_f = min(max(p_mkt_f, 1e-6), 1.0 - 1e-6)
                    p_used_f = min(max(p_used_f, 1e-6), 1.0 - 1e-6)
                    overlay_logit = math.log(p_used_f / (1.0 - p_used_f)) - math.log(p_mkt_f / (1.0 - p_mkt_f))
                except Exception:
                    overlay_logit = None

            odds_dyn_cfg = getattr(self.config.betting, "odds_dynamics_filter", None)
            odds_dyn_metric_value = None
            if odds_dyn_cfg is not None:
                odds_dyn_metric_value = compute_odds_dyn_metric(
                    pred,
                    getattr(odds_dyn_cfg, "metric", "odds_delta_log"),
                    getattr(odds_dyn_cfg, "lookback_minutes", 10),
                )
            odds_dyn_ev_cfg = getattr(self.config.betting, "odds_dyn_ev_margin", None)
            odds_dyn_ev_score = None
            if odds_dyn_ev_cfg is not None:
                odds_dyn_ev_score = compute_odds_dyn_metric(
                    pred,
                    getattr(odds_dyn_ev_cfg, "metric", "odds_delta_log"),
                    getattr(odds_dyn_ev_cfg, "lookback_minutes", 5),
                )

            candidates.append({
                "pred": pred,
                "odds_at_buy": odds_at_buy,
                "p_hat_raw": p_hat_raw,
                "p_hat_pre_softmax": p_hat_pre_softmax,
                "p_hat": p_hat,
                "p_hat_shrunk": p_hat_shrunk,
                "shrink_alpha": shrink_alpha,
                "bias_meta": bias_meta,
                "slippage_meta": slippage_meta,
                "odds_effective": odds_effective,
                "ev": ev,
                "market_blend_enabled": market_blend_used,
                "market_prob_method": market_method,
                "market_blend_w": float(blend_w) if market_blend_used else None,
                "p_blend": p_blend,
                "ev_blend": ev_blend,
                "odds_band_blend": odds_band_blend,
                "t_ev": float(t_ev) if market_blend_used else None,
                "odds_cap": float(odds_cap_val) if (market_blend_used and odds_cap_val is not None) else None,
                "exclude_odds_band": ",".join(exclude_bands) if exclude_bands else None,
                "p_hat_pre_blend": p_hat_pre_blend,
                "overlay_logit": overlay_logit,
                "odds_dyn_metric_value": odds_dyn_metric_value,
                "odds_dyn_ev_score": odds_dyn_ev_score,
                "p_mkt": p_mkt,
                "p_mkt_raw": p_mkt_raw,
                "p_mkt_race": p_mkt_race,
                "overround_sum_inv": overround_sum_inv,
                "takeout_implied": takeout_implied,
                "race_cost_cap_mode": race_cost_cap_mode,
                "race_cost_cap_value": race_cost_cap_value,
                "race_cost_cap_selected_from": race_cost_cap_selected_from,
                "race_cost_filter_passed": race_cost_passed,
                "race_cost_filter_reason": race_cost_reason,
                "race_softmax_w": pred.get("race_softmax_w"),
                "race_softmax_T": pred.get("race_softmax_T"),
                "race_softmax_enabled": pred.get("race_softmax_enabled", False),
                "closing_mult": closing_mult,
            })

        ev_cap_thr = None
        if ev_cap_q is not None and candidates:
            ev_vals = [c["ev"] for c in candidates if c.get("ev") is not None and np.isfinite(c["ev"])]
            if ev_vals:
                ev_cap_thr = float(np.quantile(ev_vals, float(ev_cap_q)))

        ov_cap_thr = None
        if ov_cap_q is not None and candidates:
            ov_vals = [
                abs(c["overlay_logit"])
                for c in candidates
                if c.get("overlay_logit") is not None and np.isfinite(c["overlay_logit"])
            ]
            if ov_vals:
                ov_cap_thr = float(np.quantile(ov_vals, float(ov_cap_q)))

        for cand in candidates:
            pred = cand["pred"]
            odds_at_buy = cand["odds_at_buy"]
            p_hat_raw = cand["p_hat_raw"]
            p_hat_pre_softmax = cand["p_hat_pre_softmax"]
            p_hat = cand["p_hat"]
            p_hat_shrunk = cand["p_hat_shrunk"]
            shrink_alpha = cand["shrink_alpha"]
            bias_meta = cand["bias_meta"]
            slippage_meta = cand["slippage_meta"]
            odds_effective = cand["odds_effective"]
            ev = cand["ev"]
            overlay_logit = cand["overlay_logit"]
            p_mkt = cand["p_mkt"]
            p_mkt_raw = cand.get("p_mkt_raw")
            p_mkt_race = cand.get("p_mkt_race")
            overround_sum_inv = cand.get("overround_sum_inv")
            takeout_implied = cand.get("takeout_implied")
            race_cost_passed = cand.get("race_cost_filter_passed")
            race_cost_reason = cand.get("race_cost_filter_reason")
            race_softmax_w = cand.get("race_softmax_w")
            race_softmax_T = cand.get("race_softmax_T")
            race_softmax_enabled = cand.get("race_softmax_enabled")
            closing_mult = cand["closing_mult"]
            odds_dyn_ev_score = cand.get("odds_dyn_ev_score")

            base_min_ev = float(ev_margin)
            min_ev_eff = base_min_ev
            if takeout_ev_enabled and takeout_slope and takeout_implied is not None:
                try:
                    takeout_val = float(takeout_implied)
                except Exception:
                    takeout_val = None
                if takeout_val is not None and np.isfinite(takeout_val):
                    min_ev_eff = base_min_ev + takeout_slope * max(0.0, takeout_val - takeout_ref)
            passed_takeout_ev_margin = ev >= min_ev_eff

            odds_dyn_ev_cfg = getattr(self.config.betting, "odds_dyn_ev_margin", None)
            odds_dyn_ev_enabled = bool(odds_dyn_ev_cfg and getattr(odds_dyn_ev_cfg, "enabled", False))
            odds_dyn_ev_metric = str(getattr(odds_dyn_ev_cfg, "metric", "odds_delta_log") or "odds_delta_log")
            odds_dyn_ev_lookback = int(getattr(odds_dyn_ev_cfg, "lookback_minutes", 5) or 5) if odds_dyn_ev_cfg else 5
            odds_dyn_ev_direction = str(getattr(odds_dyn_ev_cfg, "direction", "high") or "high")
            try:
                odds_dyn_ev_ref = float(getattr(odds_dyn_ev_cfg, "ref", None))
            except Exception:
                odds_dyn_ev_ref = None
            try:
                odds_dyn_ev_slope = float(getattr(odds_dyn_ev_cfg, "slope", 0.0))
            except Exception:
                odds_dyn_ev_slope = 0.0
            odds_dyn_ev_margin = None
            min_ev_eff_odds_dyn = None
            passed_odds_dyn_ev_margin = None
            odds_dyn_ev_enabled_eff = bool(
                odds_dyn_ev_enabled
                and odds_dyn_ev_ref is not None
                and odds_dyn_ev_slope is not None
                and odds_dyn_ev_slope > 0
            )
            if odds_dyn_ev_score is not None and odds_dyn_ev_ref is not None:
                if odds_dyn_ev_direction == "low":
                    odds_dyn_ev_margin = odds_dyn_ev_slope * max(0.0, odds_dyn_ev_ref - odds_dyn_ev_score)
                else:
                    odds_dyn_ev_margin = odds_dyn_ev_slope * max(0.0, odds_dyn_ev_score - odds_dyn_ev_ref)
                min_ev_eff_odds_dyn = min_ev_eff + (odds_dyn_ev_margin or 0.0)
                if odds_dyn_ev_enabled_eff:
                    passed_odds_dyn_ev_margin = ev >= min_ev_eff_odds_dyn

            if reject_overlay_missing and overlay_logit is None:
                continue

            ev_cap_passed = True
            if ev_cap_q is not None and ev_cap_thr is not None:
                ev_cap_passed = ev <= ev_cap_thr
            if not ev_cap_passed:
                continue

            ov_cap_passed = True
            if ov_cap_q is not None and ov_cap_thr is not None:
                if overlay_logit is None:
                    ov_cap_passed = False if reject_overlay_missing else None
                else:
                    ov_cap_passed = abs(overlay_logit) <= ov_cap_thr
            if ov_cap_passed is False:
                continue

            if ev < min_ev_eff:
                continue

            # Ticket G2: EV?????????EV????
            if self.ev_upper_cap is not None and ev > self.ev_upper_cap:
                continue

            # calculate_stake?????
            max_pct_raw = self.config.betting.caps.per_race_pct
            stake_raw = calculate_stake(
                p_hat=p_hat,
                odds=odds_effective,
                bankroll=bankroll,
                method=self.config.betting.sizing.method,
                fraction=self.config.betting.sizing.fraction,
                max_pct=1.0 if stake_enabled else max_pct_raw,
                min_stake=min_yen,
            )

            if stake_raw < min_yen:
                continue

            bet = Bet(
                race_id=race_id,
                horse_no=pred["horse_no"],
                ticket_type="win",
                stake=stake_raw,
                odds_at_buy=odds_at_buy,
                odds_effective=odds_effective,
                p_hat=p_hat,
                ev=ev,
                asof_time=asof_time,
                extra={
                    "p_mkt": p_mkt,
                    "p_mkt_raw": p_mkt_raw,
                    "p_mkt_race": p_mkt_race,
                    "overround_sum_inv": overround_sum_inv,
                    "takeout_implied": takeout_implied,
                    "base_min_ev": base_min_ev,
                    "market_blend_enabled": cand.get("market_blend_enabled"),
                    "market_prob_method": cand.get("market_prob_method"),
                    "market_blend_w": cand.get("market_blend_w"),
                    "p_blend": cand.get("p_blend"),
                    "ev_blend": cand.get("ev_blend"),
                    "odds_band_blend": cand.get("odds_band_blend"),
                    "t_ev": cand.get("t_ev"),
                    "odds_cap": cand.get("odds_cap"),
                    "exclude_odds_band": cand.get("exclude_odds_band"),
                    "p_hat_pre_blend": cand.get("p_hat_pre_blend"),
                    "takeout_ref": float(takeout_ref),
                    "takeout_slope": float(takeout_slope),
                    "min_ev_eff": float(min_ev_eff),
                    "passed_takeout_ev_margin": bool(passed_takeout_ev_margin),
                    "odds_dyn_ev_metric": odds_dyn_ev_metric,
                    "odds_dyn_ev_lookback_min": odds_dyn_ev_lookback,
                    "odds_dyn_ev_score": odds_dyn_ev_score,
                    "odds_dyn_ev_ref": odds_dyn_ev_ref,
                    "odds_dyn_ev_slope": odds_dyn_ev_slope,
                    "odds_dyn_ev_margin": odds_dyn_ev_margin,
                    "min_ev_eff_odds_dyn": min_ev_eff_odds_dyn,
                    "passed_odds_dyn_ev_margin": passed_odds_dyn_ev_margin if odds_dyn_ev_enabled_eff else None,
                    "odds_dyn_ev_margin_enabled": odds_dyn_ev_enabled_eff,
                    "race_cost_cap_mode": cand.get("race_cost_cap_mode"),
                    "race_cost_cap_value": cand.get("race_cost_cap_value"),
                    "race_cost_cap_selected_from": cand.get("race_cost_cap_selected_from"),
                    "race_cost_filter_passed": race_cost_passed,
                    "race_cost_filter_reason": race_cost_reason,
                    "race_softmax_w": race_softmax_w,
                    "race_softmax_T": race_softmax_T,
                    "race_softmax_enabled": bool(race_softmax_enabled_pred),
                    "rsx_selected": pred.get(
                        "race_softmax_selected",
                        "softmax" if race_softmax_enabled_pred else "baseline",
                    ),
                    "rsx_w": float(race_softmax_w) if race_softmax_w is not None else None,
                    "rsx_T": float(race_softmax_T) if race_softmax_T is not None else None,
                    "p_hat_pre_softmax": float(p_hat_pre_softmax) if p_hat_pre_softmax is not None else None,
                    "segblend_segment": pred.get("segblend_segment"),
                    "segblend_w_used": pred.get("segblend_w_used"),
                    "segblend_w_global": pred.get("segblend_w_global"),
                    "has_ts_odds": bool(pred.get("has_ts_odds", True)),
                    "odds_window_passed": True,
                    "min_buy_odds": float(min_buy_odds) if min_buy_odds is not None else None,
                    "max_buy_odds": float(max_buy_odds) if max_buy_odds is not None else None,
                    "ev_cap_q": float(ev_cap_q) if ev_cap_q is not None else None,
                    "ev_cap_thr": float(ev_cap_thr) if ev_cap_thr is not None else None,
                    "ev_cap_passed": bool(ev_cap_passed),
                    "ov_cap_q": float(ov_cap_q) if ov_cap_q is not None else None,
                    "ov_cap_thr": float(ov_cap_thr) if ov_cap_thr is not None else None,
                    "ov_cap_passed": ov_cap_passed,
                    "closing_odds_multiplier": closing_mult,
                    "slippage_meta": slippage_meta,
                    "p_hat_raw": float(p_hat_raw),
                    "p_hat_shrunk": float(p_hat_shrunk) if p_hat_shrunk is not None else None,
                    "overlay_shrink_alpha": float(shrink_alpha) if shrink_alpha is not None else None,
                    "p_hat_adj": float(p_hat),
                    "bias_meta": bias_meta,
                    "ts_vol_cap": float(ts_vol_cap) if ts_vol_cap is not None else None,
                    "ts_vol_cap_passed": True if ts_vol_cap is not None else None,
                    "odds_chg_5m": pred.get("odds_chg_5m"),
                    "odds_chg_10m": pred.get("odds_chg_10m"),
                    "odds_chg_30m": pred.get("odds_chg_30m"),
                    "odds_chg_60m": pred.get("odds_chg_60m"),
                    "p_mkt_chg_5m": pred.get("p_mkt_chg_5m"),
                    "p_mkt_chg_10m": pred.get("p_mkt_chg_10m"),
                    "p_mkt_chg_30m": pred.get("p_mkt_chg_30m"),
                    "p_mkt_chg_60m": pred.get("p_mkt_chg_60m"),
                    "log_odds_slope_60m": pred.get("log_odds_slope_60m"),
                    "log_odds_std_60m": pred.get("log_odds_std_60m"),
                    "n_pts_60m": pred.get("n_pts_60m"),
                    "odds_dyn_metric_value": cand.get("odds_dyn_metric_value"),
                    "snap_age_min": pred.get("snap_age_min"),
                    "resid": pred.get("resid"),
                    "resid_cap": pred.get("resid_cap"),
                    "cap_value": pred.get("cap_value"),
                    "p_hat_capped": pred.get("p_hat_capped"),
                },
            )
            bet.extra["stake_raw"] = int(stake_raw)
            if not use_new_stake:
                bet.extra.update({
                    "stake_clipped": int(stake_raw),
                    "clip_reason": None,
                    "uncert_mult": None,
                    "uncert_n_bin": None,
                })
            bets.append(bet)

        bets = sorted(bets, key=lambda b: b.ev, reverse=True)
        bets = bets[:self.config.betting.max_bets_per_race]
        
        # ★日次残り予算を考慮
        if not use_new_stake:
            if remaining_daily_budget is not None and remaining_daily_budget > 0:
                selected = []
                used = 0
                for bet in bets:
                    if used + bet.stake <= remaining_daily_budget:
                        selected.append(bet)
                        used += bet.stake
                bets = selected
            bets = self._apply_odds_dynamics_filter(bets)
            bets = self._apply_odds_dyn_ev_margin(bets)
            return bets

        max_frac_per_bet = getattr(stake_cfg, "max_frac_per_bet", None) if stake_cfg is not None else None
        if max_frac_per_bet is None:
            max_frac_per_bet = self.config.betting.caps.per_race_pct
        max_frac_per_race = getattr(stake_cfg, "max_frac_per_race", None) if stake_cfg is not None else None
        if max_frac_per_race is None:
            max_frac_per_race = max_frac_per_bet
        max_yen_per_bet = getattr(stake_cfg, "max_yen_per_bet", None) if stake_cfg is not None else None


        selected = []
        for bet in bets:
            extra = bet.extra or {}
            stake_raw = int(extra.get("stake_raw", bet.stake))
            stake_clipped, clip_reason = clip_stake(
                stake_raw,
                bankroll,
                min_yen=min_yen,
                max_frac_per_bet=max_frac_per_bet,
                max_yen_per_bet=max_yen_per_bet,
                remaining_race_budget=None,
                remaining_daily_budget=None,
            )
            if stake_clipped < min_yen:
                continue

            stake_final = stake_clipped
            uncert_mult = None
            uncert_n_bin = None
            if uncert_enabled and self.uncertainty_shrink is not None:
                p_for_uncert = extra.get("p_hat_raw", bet.p_hat)
                try:
                    p_for_uncert = float(p_for_uncert)
                except Exception:
                    p_for_uncert = float(bet.p_hat)
                uncert_mult, meta = self.uncertainty_shrink.apply(p_for_uncert)
                stake_final = int((stake_clipped * float(uncert_mult)) // min_yen * min_yen)
                uncert_n_bin = meta.get("n_bin") if isinstance(meta, dict) else None
                if stake_final < min_yen:
                    continue

            bet.stake = stake_final
            extra.update({
                "stake_raw": stake_raw,
                "stake_clipped": stake_clipped,
                "clip_reason": clip_reason,
                "uncert_mult": float(uncert_mult) if uncert_mult is not None else None,
                "uncert_n_bin": int(uncert_n_bin) if uncert_n_bin is not None else None,
            })
            bet.extra = extra
            selected.append(bet)

        if max_frac_per_race is not None:
            race_cap = float(bankroll) * float(max_frac_per_race)
            self._scale_bets_to_cap(selected, race_cap, min_yen, reason="per_race")

        selected = self._apply_odds_dynamics_filter(selected)
        selected = self._apply_odds_dyn_ev_margin(selected)
        selected = [b for b in selected if b.stake >= min_yen]
        return selected
    

    @staticmethod
    def _append_clip_reason(existing: Optional[str], reason: str) -> Optional[str]:
        if not reason:
            return existing
        if not existing:
            return reason
        parts = [p for p in existing.split(",") if p]
        if reason in parts:
            return existing
        parts.append(reason)
        return ",".join(parts)

    def _scale_bets_to_cap(
        self,
        bets: list[Bet],
        cap: Optional[float],
        min_yen: int,
        *,
        reason: str,
    ) -> None:
        if cap is None:
            return
        total = sum(b.stake for b in bets if b.stake >= min_yen)
        if total <= 0 or total <= cap:
            return
        scale = float(cap) / float(total)
        for bet in bets:
            if bet.stake < min_yen:
                continue
            scaled = int((bet.stake * scale) // min_yen * min_yen)
            if scaled < min_yen:
                scaled = 0
            if scaled != bet.stake:
                bet.stake = scaled
                extra = bet.extra or {}
                extra["clip_reason"] = self._append_clip_reason(extra.get("clip_reason"), reason)
                bet.extra = extra

    def _settle_bet(self, bet: Bet) -> BetResult:
        """
        ベット決済
        
        重要: 確定オッズ（fact_result.odds）を優先して使用
              購入時オッズで固定すると評価が過大になりやすい
        """
        query = text("""
            SELECT finish_pos, odds
            FROM fact_result
            WHERE race_id = :race_id AND horse_no = :horse_no
        """)
        result = self.session.execute(
            query, {"race_id": bet.race_id, "horse_no": bet.horse_no}
        ).fetchone()
        
        if not result:
            return BetResult(
                bet=bet,
                finish_pos=99,
                is_win=False,
                payout=0,
                profit=-bet.stake,
                odds_final=None,
            )
        
        finish_pos = result[0] or 99
        final_odds = result[1]  # 確定オッズ
        
        # ★重要: 確定オッズを優先、なければ購入時オッズをフォールバック
        odds_for_settlement = float(final_odds) if final_odds else bet.odds_at_buy
        
        is_win = finish_pos == 1
        
        if is_win:
            payout = bet.stake * odds_for_settlement
        else:
            payout = 0
        
        return BetResult(
            bet=bet,
            finish_pos=finish_pos,
            is_win=is_win,
            payout=payout,
            profit=payout - bet.stake,
            odds_final=odds_for_settlement,
        )


def run_backtest(
    session: Session,
    start_date: str,
    end_date: str,
    model,
    initial_bankroll: float = 300000,
    slippage_table: SlippageTable | None = None,
    odds_band_bias: OddsBandBias | None = None,
    ev_upper_cap: float | None = None,
    uncertainty_shrink: UncertaintyShrink | None = None,
) -> BacktestResult:
    """バックテスト実行（ショートカット）"""
    engine = BacktestEngine(
        session,
        slippage_table=slippage_table,
        odds_band_bias=odds_band_bias,
        ev_upper_cap=ev_upper_cap,
        uncertainty_shrink=uncertainty_shrink,
    )
    return engine.run(start_date, end_date, model, initial_bankroll)
