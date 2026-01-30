"""
ベッティングポリシー

買い目生成・CSV出力（半自動運用用）
"""
import logging
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import uuid
import numpy as np
import pandas as pd

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_config
from ..db.models import BetSignal
from .sizing import calculate_stake
from .stake_clip import clip_stake
from .uncertainty_shrink import UncertaintyShrink
from ..analysis.slippage_table import SlippageTable
from .odds_band_bias import OddsBandBias
from .overlay_shrink import shrink_probability
from .odds_dynamics import compute_odds_dyn_metric, eval_odds_dyn_filter
from .market_blend import compute_market_blend, parse_exclude_odds_band
from ..modeling.race_softmax import apply_race_softmax

logger = logging.getLogger(__name__)


@dataclass
class BetCandidate:
    """買い目候補"""
    race_id: str
    race_name: str
    horse_no: int
    horse_name: str
    ticket_type: str
    odds: float
    p_hat: float
    ev: float
    stake_yen: int
    asof_time: datetime
    reason: str
    odds_window_passed: Optional[bool] = None
    min_buy_odds: Optional[float] = None
    max_buy_odds: Optional[float] = None
    ev_cap_q: Optional[float] = None
    ev_cap_thr: Optional[float] = None
    ev_cap_passed: Optional[bool] = None
    ov_cap_q: Optional[float] = None
    ov_cap_thr: Optional[float] = None
    ov_cap_passed: Optional[bool] = None
    p_hat_raw: Optional[float] = None
    race_softmax_enabled: Optional[bool] = None
    race_softmax_w: Optional[float] = None
    race_softmax_T: Optional[float] = None
    p_hat_pre_softmax: Optional[float] = None
    segblend_segment: Optional[str] = None
    segblend_w_used: Optional[float] = None
    segblend_w_global: Optional[float] = None
    stake_before: Optional[float] = None
    stake_mult: Optional[float] = None
    stake_after: Optional[float] = None
    stake_damp_ref_odds: Optional[float] = None
    stake_damp_power: Optional[float] = None
    stake_damp_min_mult: Optional[float] = None
    stake_damp_odds: Optional[float] = None
    stake_damp_ratio: Optional[float] = None
    stake_damp_mult_raw: Optional[float] = None
    stake_damp_mult_clamped: Optional[float] = None
    stake_damp_floor_unit: Optional[int] = None
    stake_damp_enabled: Optional[bool] = None
    stake_raw: Optional[int] = None
    stake_clipped: Optional[int] = None
    clip_reason: Optional[str] = None
    uncert_mult: Optional[float] = None
    uncert_n_bin: Optional[int] = None
    selection_mode: Optional[str] = None
    daily_top_n_n: Optional[int] = None
    daily_top_n_metric: Optional[str] = None
    daily_top_n_min_value: Optional[float] = None
    daily_rank_in_day: Optional[int] = None
    daily_candidates_before: Optional[int] = None
    daily_candidates_after_min: Optional[int] = None
    daily_selected: Optional[bool] = None
    odds_dyn_metric: Optional[str] = None
    odds_dyn_lookback_min: Optional[int] = None
    odds_dyn_metric_value: Optional[float] = None
    odds_dyn_threshold: Optional[float] = None
    passed_odds_dyn_filter: Optional[bool] = None
    odds_dyn_filter_enabled: Optional[bool] = None
    odds_dyn_ev_metric: Optional[str] = None
    odds_dyn_ev_lookback_min: Optional[int] = None
    odds_dyn_ev_score: Optional[float] = None
    odds_dyn_ev_ref: Optional[float] = None
    odds_dyn_ev_slope: Optional[float] = None
    odds_dyn_ev_margin: Optional[float] = None
    min_ev_eff_odds_dyn: Optional[float] = None
    passed_odds_dyn_ev_margin: Optional[bool] = None
    odds_dyn_ev_margin_enabled: Optional[bool] = None
    odds_floor_min_odds: Optional[float] = None
    odds_floor_odds_used: Optional[float] = None
    passed_odds_floor: Optional[bool] = None
    overround_sum_inv: Optional[float] = None
    takeout_implied: Optional[float] = None
    base_min_ev: Optional[float] = None
    takeout_ref: Optional[float] = None
    takeout_slope: Optional[float] = None
    min_ev_eff: Optional[float] = None
    passed_takeout_ev_margin: Optional[bool] = None
    race_cost_cap_mode: Optional[str] = None
    race_cost_cap_value: Optional[float] = None
    race_cost_cap_selected_from: Optional[str] = None
    race_cost_filter_passed: Optional[bool] = None
    race_cost_filter_reason: Optional[str] = None
    market_blend_enabled: Optional[bool] = None
    market_prob_method: Optional[str] = None
    market_blend_w: Optional[float] = None
    p_blend: Optional[float] = None
    ev_blend: Optional[float] = None
    odds_band_blend: Optional[str] = None
    t_ev: Optional[float] = None
    odds_cap: Optional[float] = None
    exclude_odds_band: Optional[str] = None
    p_hat_pre_blend: Optional[float] = None


class BettingPolicy:
    """
    ベッティングポリシー
    
    注意:
        daily_loss による stop-loss 機能は現在未実装です。
        check_stop_loss() は daily_loss をチェックしますが、
        daily_loss を更新する処理（決済取り込み連携）は実装されていません。
        将来的に BetSettlement と連携して実装する予定です。
    """
    
    def __init__(
        self,
        session: Session,
        bankroll: float,
        slippage_table: SlippageTable | None = None,
        odds_band_bias: OddsBandBias | None = None,
        ev_upper_cap: float | None = None,
        uncertainty_shrink: UncertaintyShrink | None = None,
    ):
        self.session = session
        self.config = get_config()
        self.bankroll = bankroll
        self.slippage_table = slippage_table
        self.odds_band_bias = odds_band_bias
        self.ev_upper_cap = ev_upper_cap  # Ticket G2: EV上限値
        self.uncertainty_shrink = uncertainty_shrink
        self.daily_stake = 0
        # TODO: 未実装 - 決済取り込みと連携して更新する必要がある
        self.daily_loss = 0
    

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

    def _select_market_prob(self, pred: dict) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        p_raw = pred.get("p_mkt_raw")
        p_race = pred.get("p_mkt_race")
        if p_raw is None:
            p_raw = pred.get("p_mkt")
        if p_race is None:
            p_race = pred.get("p_mkt")
        try:
            p_raw = float(p_raw) if p_raw is not None else None
        except Exception:
            p_raw = None
        try:
            p_race = float(p_race) if p_race is not None else None
        except Exception:
            p_race = None

        overround = pred.get("overround_sum_inv")
        takeout = pred.get("takeout_implied")
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

    @staticmethod
    def _resolve_daily_top_n_params(daily_cfg) -> tuple[str, float | None, str]:
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
        tie_break = str(getattr(daily_cfg, "tie_break", "score_desc_then_odds_asc") or "score_desc_then_odds_asc")
        return metric, min_value, tie_break

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
        odds = cand.get("odds")
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

    def _apply_daily_top_n(self, candidates: list[dict], daily_cfg) -> list[dict]:
        if daily_cfg is None:
            return candidates
        n = int(getattr(daily_cfg, "n", 0) or 0)
        metric, min_value, tie_break = self._resolve_daily_top_n_params(daily_cfg)

        if not candidates:
            return []

        groups: dict[object, list[dict]] = {}
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

        return selected

    def generate_candidates(
        self,
        race_id: str,
        predictions: list[dict],
        asof_time: datetime,
    ) -> list[BetCandidate]:
        """
        買い目候補を生成
        
        Args:
            race_id: レースID
            predictions: 予測結果リスト
            asof_time: 購入時点
        """
        candidates = []
        ev_margin = self.config.betting.ev_margin
        market_blend_enabled = bool(getattr(self.config.betting, "enable_market_blend", False))
        market_method = str(getattr(self.config.betting, "market_prob_method", "p_mkt_col") or "p_mkt_col")
        t_ev = float(getattr(self.config.betting, "t_ev", ev_margin))
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
        if market_blend_enabled:
            ev_margin = t_ev
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

        sel_mode, daily_cfg = self._resolve_selection_mode()
        
        # レース情報取得
        race_info = self._get_race_info(race_id)
        race_name = race_info.get("race_name", race_id) if race_info else race_id
        
        ev_cap_q = getattr(self.config.betting, "ev_cap_quantile", None)
        ov_cap_q = getattr(self.config.betting, "overlay_abs_cap_quantile", None)
        reject_overlay_missing = bool(getattr(self.config.betting, "reject_if_overlay_missing", True))
        rs_cfg = getattr(self.config.model, "race_softmax", None)
        race_softmax_enabled = bool(rs_cfg and getattr(rs_cfg, "enabled", False))

        for pred in predictions:
            p_mkt, p_raw, p_race, overround, takeout = self._select_market_prob(pred)
            pred["p_mkt"] = p_mkt
            pred["p_mkt_raw"] = p_raw
            pred["p_mkt_race"] = p_race
            pred["overround_sum_inv"] = overround
            pred["takeout_implied"] = takeout

        if race_softmax_enabled and predictions:
            if not any("p_used" in p for p in predictions):
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
                else:
                    w = float(getattr(getattr(rs_cfg, "apply", None), "w_default", 0.2))
                    t = float(getattr(getattr(rs_cfg, "apply", None), "t_default", 1.0))
                    p_race = apply_race_softmax(
                        df,
                        w=w,
                        t=t,
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
                    else:
                        for pred, p_val in zip(predictions, p_vals):
                            pred["p_used"] = float(p_val)
                            pred["race_softmax_w"] = w
                            pred["race_softmax_T"] = t
                            pred["race_softmax_enabled"] = True

        race_cost_cap_mode, race_cost_cap_value, race_cost_cap_selected_from = self._race_cost_filter_extra_meta()
        cand_meta = []
        for pred in predictions:
            odds = pred.get("odds", 0)
            p_hat_raw = pred.get("p_hat", 0)
            p_hat_pre_softmax = None
            if race_softmax_enabled and pred.get("p_used") is not None:
                p_hat_pre_softmax = p_hat_raw
                p_hat_raw = float(pred.get("p_used"))
            p_mkt = pred.get("p_mkt")
            overround_sum_inv = pred.get("overround_sum_inv")
            takeout_implied = pred.get("takeout_implied")
            race_cost_passed, race_cost_reason = self._race_cost_filter(pred)
            horse_no = pred.get("horse_no", 0)
            horse_name = pred.get("horse_name", f"??{horse_no}")

            if odds <= 0 or p_hat_raw <= 0:
                continue

            if not race_cost_passed:
                continue

            if min_buy_odds is not None and odds < float(min_buy_odds):
                continue
            if max_buy_odds is not None and odds > float(max_buy_odds):
                continue

            # Step4: TS?????????EV???????
            if ts_vol_cap is not None:
                v = pred.get("log_odds_std_60m")
                try:
                    v_num = float(v) if v is not None else None
                except Exception:
                    v_num = None
                if v_num is None:
                    if reject_ts_missing:
                        continue
                else:
                    if v_num > float(ts_vol_cap):
                        continue

            # odds??????buy???
            if min_odds is not None and odds < float(min_odds):
                continue
            if max_odds is not None and odds > float(max_odds):
                continue

            # Ticket N6: overlay shrink (logit residual)
            p_hat = float(p_hat_raw)
            p_hat_shrunk = None
            shrink_alpha = self.config.betting.overlay_shrink_alpha
            if not race_softmax_enabled:
                if shrink_alpha is not None:
                    p_hat_shrunk = shrink_probability(p_hat_raw, p_mkt, shrink_alpha, self.config.model.p_mkt_clip)
                    p_hat = float(p_hat_shrunk)

                # Ticket N2: odds bias (p_adj)
                if self.odds_band_bias is not None and bool(self.config.betting.odds_band_bias.enabled):
                    p_hat, _ = self.odds_band_bias.apply(p_cal=p_hat, odds_buy=odds)

            # Ticket1/TicketB: EV/?????????????
            closing_mult = float(self.config.betting.closing_odds_multiplier)
            if self.slippage_table is not None:
                odds_effective, _ = self.slippage_table.effective_odds(
                    odds_buy=odds,
                    ts_vol=pred.get("log_odds_std_60m"),
                    snap_age_min=pred.get("snap_age_min"),
                )
            else:
                odds_effective = odds * closing_mult if closing_mult > 0 else odds
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
                if odds_cap_val is not None and odds > odds_cap_val:
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
            if sel_mode == "ev_threshold":
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
            odds_dyn_ev_score = compute_odds_dyn_metric(pred, odds_dyn_ev_metric, odds_dyn_ev_lookback)
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

            cand_meta.append({
                "pred": pred,
                "odds": odds,
                "p_hat_raw": p_hat_raw,
                "p_hat_pre_softmax": p_hat_pre_softmax,
                "p_hat": p_hat,
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
                "odds_effective": odds_effective,
                "p_mkt": p_mkt,
                "overround_sum_inv": overround_sum_inv,
                "takeout_implied": takeout_implied,
                "base_min_ev": base_min_ev,
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
                "passed_odds_dyn_ev_margin": passed_odds_dyn_ev_margin,
                "odds_dyn_ev_margin_enabled": odds_dyn_ev_enabled_eff,
                "race_cost_cap_mode": race_cost_cap_mode,
                "race_cost_cap_value": race_cost_cap_value,
                "race_cost_cap_selected_from": race_cost_cap_selected_from,
                "race_cost_filter_passed": race_cost_passed,
                "race_cost_filter_reason": race_cost_reason,
                "race_softmax_w": pred.get("race_softmax_w"),
                "race_softmax_T": pred.get("race_softmax_T"),
                "race_softmax_enabled": pred.get("race_softmax_enabled", False),
                "segblend_segment": pred.get("segblend_segment"),
                "segblend_w_used": pred.get("segblend_w_used"),
                "segblend_w_global": pred.get("segblend_w_global"),
                "odds_dyn_metric_value": compute_odds_dyn_metric(
                    pred,
                    getattr(self.config.betting.odds_dynamics_filter, "metric", "odds_delta_log"),
                    getattr(self.config.betting.odds_dynamics_filter, "lookback_minutes", 10),
                ),
                "horse_no": horse_no,
                "horse_name": horse_name,
                "closing_mult": closing_mult,
            })

        if sel_mode == "daily_top_n":
            cand_meta = self._apply_daily_top_n(cand_meta, daily_cfg)

        ev_cap_thr = None
        if ev_cap_q is not None and cand_meta:
            ev_vals = [c["ev"] for c in cand_meta if c.get("ev") is not None]
            if ev_vals:
                ev_cap_thr = float(np.quantile(ev_vals, float(ev_cap_q)))

        ov_cap_thr = None
        if ov_cap_q is not None and cand_meta:
            ov_vals = [
                abs(c["overlay_logit"])
                for c in cand_meta
                if c.get("overlay_logit") is not None
            ]
            if ov_vals:
                ov_cap_thr = float(np.quantile(ov_vals, float(ov_cap_q)))

        for cand in cand_meta:
            pred = cand["pred"]
            odds = cand["odds"]
            odds_effective = cand["odds_effective"]
            p_hat_raw = cand["p_hat_raw"]
            p_hat_pre_softmax = cand["p_hat_pre_softmax"]
            p_hat = cand["p_hat"]
            ev = cand["ev"]
            overlay_logit = cand["overlay_logit"]
            p_mkt = cand["p_mkt"]
            base_min_ev = cand.get("base_min_ev")
            takeout_ref = cand.get("takeout_ref")
            takeout_slope = cand.get("takeout_slope")
            min_ev_eff = cand.get("min_ev_eff", ev_margin)
            passed_takeout_ev_margin = cand.get("passed_takeout_ev_margin")
            race_softmax_w = cand.get("race_softmax_w")
            race_softmax_T = cand.get("race_softmax_T")
            race_softmax_enabled = cand.get("race_softmax_enabled")
            segblend_segment = cand.get("segblend_segment")
            segblend_w_used = cand.get("segblend_w_used")
            segblend_w_global = cand.get("segblend_w_global")
            horse_no = cand["horse_no"]
            horse_name = cand["horse_name"]
            odds_dyn_ev_metric = cand.get("odds_dyn_ev_metric")
            odds_dyn_ev_lookback_min = cand.get("odds_dyn_ev_lookback_min")
            odds_dyn_ev_score = cand.get("odds_dyn_ev_score")
            odds_dyn_ev_ref = cand.get("odds_dyn_ev_ref")
            odds_dyn_ev_slope = cand.get("odds_dyn_ev_slope")
            odds_dyn_ev_margin = cand.get("odds_dyn_ev_margin")
            min_ev_eff_odds_dyn = cand.get("min_ev_eff_odds_dyn")
            passed_odds_dyn_ev_margin = cand.get("passed_odds_dyn_ev_margin")
            odds_dyn_ev_margin_enabled = cand.get("odds_dyn_ev_margin_enabled")

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

            if sel_mode == "ev_threshold" and ev < min_ev_eff:
                continue

            if self.ev_upper_cap is not None and ev > self.ev_upper_cap:
                continue

            max_pct_raw = self.config.betting.caps.per_race_pct
            stake_raw = calculate_stake(
                p_hat=p_hat,
                odds=odds_effective,
                bankroll=self.bankroll,
                method=self.config.betting.sizing.method,
                fraction=self.config.betting.sizing.fraction,
                max_pct=1.0 if stake_enabled else max_pct_raw,
                min_stake=min_yen,
            )
            if stake_raw < min_yen:
                continue

            candidates.append(
                BetCandidate(
                    race_id=race_id,
                    race_name=race_name,
                    horse_no=horse_no,
                    horse_name=horse_name,
                    ticket_type="win",
                    odds=odds,
                    p_hat=p_hat,
                    ev=ev,
                    stake_yen=stake_raw,
                    asof_time=asof_time,
                    reason=("EV>=margin" if sel_mode == "ev_threshold" else "daily_top_n"),
                    odds_window_passed=True,
                    min_buy_odds=float(min_buy_odds) if min_buy_odds is not None else None,
                    max_buy_odds=float(max_buy_odds) if max_buy_odds is not None else None,
                    ev_cap_q=float(ev_cap_q) if ev_cap_q is not None else None,
                    ev_cap_thr=float(ev_cap_thr) if ev_cap_thr is not None else None,
                    ev_cap_passed=bool(ev_cap_passed),
                    ov_cap_q=float(ov_cap_q) if ov_cap_q is not None else None,
                    ov_cap_thr=float(ov_cap_thr) if ov_cap_thr is not None else None,
                    ov_cap_passed=ov_cap_passed,
                    p_hat_raw=float(p_hat_raw),
                    race_softmax_enabled=bool(race_softmax_enabled),
                    race_softmax_w=float(race_softmax_w) if race_softmax_w is not None else None,
                    race_softmax_T=float(race_softmax_T) if race_softmax_T is not None else None,
                    p_hat_pre_softmax=float(p_hat_pre_softmax) if p_hat_pre_softmax is not None else None,
                    segblend_segment=str(segblend_segment) if segblend_segment is not None else None,
                    segblend_w_used=float(segblend_w_used) if segblend_w_used is not None else None,
                    segblend_w_global=float(segblend_w_global) if segblend_w_global is not None else None,
                    stake_raw=int(stake_raw),
                    selection_mode=sel_mode,
                    daily_top_n_n=cand.get("daily_top_n_n"),
                    daily_top_n_metric=cand.get("daily_top_n_metric"),
                    daily_top_n_min_value=cand.get("daily_top_n_min_value"),
                    daily_rank_in_day=cand.get("daily_rank_in_day"),
                    daily_candidates_before=cand.get("daily_candidates_before"),
                    daily_candidates_after_min=cand.get("daily_candidates_after_min"),
                    daily_selected=cand.get("daily_selected"),
                    odds_dyn_metric_value=cand.get("odds_dyn_metric_value"),
                    odds_dyn_ev_metric=str(odds_dyn_ev_metric) if odds_dyn_ev_metric is not None else None,
                    odds_dyn_ev_lookback_min=int(odds_dyn_ev_lookback_min) if odds_dyn_ev_lookback_min is not None else None,
                    odds_dyn_ev_score=float(odds_dyn_ev_score) if odds_dyn_ev_score is not None else None,
                    odds_dyn_ev_ref=float(odds_dyn_ev_ref) if odds_dyn_ev_ref is not None else None,
                    odds_dyn_ev_slope=float(odds_dyn_ev_slope) if odds_dyn_ev_slope is not None else None,
                    odds_dyn_ev_margin=float(odds_dyn_ev_margin) if odds_dyn_ev_margin is not None else None,
                    min_ev_eff_odds_dyn=float(min_ev_eff_odds_dyn) if min_ev_eff_odds_dyn is not None else None,
                    passed_odds_dyn_ev_margin=bool(passed_odds_dyn_ev_margin) if passed_odds_dyn_ev_margin is not None else None,
                    odds_dyn_ev_margin_enabled=bool(odds_dyn_ev_margin_enabled) if odds_dyn_ev_margin_enabled is not None else None,
                    overround_sum_inv=cand.get("overround_sum_inv"),
                    takeout_implied=cand.get("takeout_implied"),
                    base_min_ev=float(base_min_ev) if base_min_ev is not None else None,
                    takeout_ref=float(takeout_ref) if takeout_ref is not None else None,
                    takeout_slope=float(takeout_slope) if takeout_slope is not None else None,
                    min_ev_eff=float(min_ev_eff) if min_ev_eff is not None else None,
                    passed_takeout_ev_margin=bool(passed_takeout_ev_margin) if passed_takeout_ev_margin is not None else None,
                    race_cost_cap_mode=cand.get("race_cost_cap_mode"),
                    race_cost_cap_value=cand.get("race_cost_cap_value"),
                    race_cost_cap_selected_from=cand.get("race_cost_cap_selected_from"),
                    race_cost_filter_passed=cand.get("race_cost_filter_passed"),
                    race_cost_filter_reason=cand.get("race_cost_filter_reason"),
                    market_blend_enabled=cand.get("market_blend_enabled"),
                    market_prob_method=cand.get("market_prob_method"),
                    market_blend_w=cand.get("market_blend_w"),
                    p_blend=cand.get("p_blend"),
                    ev_blend=cand.get("ev_blend"),
                    odds_band_blend=cand.get("odds_band_blend"),
                    t_ev=cand.get("t_ev"),
                    odds_cap=cand.get("odds_cap"),
                    exclude_odds_band=cand.get("exclude_odds_band"),
                    p_hat_pre_blend=cand.get("p_hat_pre_blend"),
                )
            )
        candidates = sorted(candidates, key=lambda c: c.ev, reverse=True)
        selected = candidates[:self.config.betting.max_bets_per_race]
        
        # ★重要: 日次上限が複数レースで効くように累積を更新
        if not use_new_stake:
            selected = self._apply_odds_dynamics_filter(selected)
            selected = self._apply_odds_floor(selected)
            self.daily_stake += sum(c.stake_yen for c in selected)
            return selected

        max_frac_per_bet = getattr(stake_cfg, "max_frac_per_bet", None) if stake_cfg is not None else None
        if max_frac_per_bet is None:
            max_frac_per_bet = self.config.betting.caps.per_race_pct
        max_frac_per_race = getattr(stake_cfg, "max_frac_per_race", None) if stake_cfg is not None else None
        if max_frac_per_race is None:
            max_frac_per_race = max_frac_per_bet
        max_yen_per_bet = getattr(stake_cfg, "max_yen_per_bet", None) if stake_cfg is not None else None

        per_day_cap = None
        if stake_cfg is not None and stake_enabled:
            frac = getattr(stake_cfg, "max_frac_per_day", None)
            if frac is not None:
                per_day_cap = float(self.bankroll) * float(frac)
        else:
            per_day_cap = float(self.bankroll) * float(self.config.betting.caps.per_day_pct)

        remaining_daily_budget = None
        if per_day_cap is not None:
            remaining_daily_budget = max(0, per_day_cap - self.daily_stake)

        remaining_race_budget = None
        if max_frac_per_race is not None:
            remaining_race_budget = float(self.bankroll) * float(max_frac_per_race)

        adjusted: list[BetCandidate] = []
        for cand in selected:
            stake_raw = int(cand.stake_raw or cand.stake_yen)
            stake_clipped, clip_reason = clip_stake(
                stake_raw,
                self.bankroll,
                min_yen=min_yen,
                max_frac_per_bet=max_frac_per_bet,
                max_yen_per_bet=max_yen_per_bet,
                remaining_race_budget=remaining_race_budget,
                remaining_daily_budget=remaining_daily_budget,
            )
            if stake_clipped < min_yen:
                continue

            stake_final = stake_clipped
            uncert_mult = None
            uncert_n_bin = None
            if uncert_enabled and self.uncertainty_shrink is not None:
                p_for_uncert = cand.p_hat_raw if cand.p_hat_raw is not None else cand.p_hat
                try:
                    p_for_uncert = float(p_for_uncert)
                except Exception:
                    p_for_uncert = float(cand.p_hat)
                uncert_mult, meta = self.uncertainty_shrink.apply(p_for_uncert)
                stake_final = int((stake_clipped * float(uncert_mult)) // min_yen * min_yen)
                uncert_n_bin = meta.get("n_bin") if isinstance(meta, dict) else None
                if stake_final < min_yen:
                    continue

            cand.stake_yen = stake_final
            cand.stake_raw = stake_raw
            cand.stake_clipped = stake_clipped
            cand.clip_reason = clip_reason
            cand.uncert_mult = float(uncert_mult) if uncert_mult is not None else None
            cand.uncert_n_bin = int(uncert_n_bin) if uncert_n_bin is not None else None
            adjusted.append(cand)

            if remaining_race_budget is not None:
                remaining_race_budget = max(0, remaining_race_budget - stake_final)
            if remaining_daily_budget is not None:
                remaining_daily_budget = max(0, remaining_daily_budget - stake_final)

        adjusted = self._apply_odds_dynamics_filter(adjusted)
        adjusted = self._apply_odds_dyn_ev_margin(adjusted)
        adjusted = self._apply_odds_floor(adjusted)
        adjusted = self._apply_stake_odds_damp(adjusted, min_yen=min_yen)
        # Use pre-damp stake for daily budget to keep selection invariant.
        stake_for_budget = 0.0
        for cand in adjusted:
            base = cand.stake_before if cand.stake_before is not None else cand.stake_yen
            try:
                base_val = float(base)
            except Exception:
                base_val = float(cand.stake_yen)
            if base_val > 0:
                stake_for_budget += base_val
        self.daily_stake += stake_for_budget
        return adjusted

    def _apply_odds_dynamics_filter(self, candidates: list[BetCandidate]) -> list[BetCandidate]:
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

        filtered: list[BetCandidate] = []
        for cand in candidates:
            value = cand.odds_dyn_metric_value
            passed = eval_odds_dyn_filter(value, threshold_val, direction) if enabled else None
            passed_flag = passed
            if passed is None:
                passed = True
            cand.odds_dyn_metric = metric
            cand.odds_dyn_lookback_min = lookback
            cand.odds_dyn_threshold = threshold_val
            cand.passed_odds_dyn_filter = passed_flag if enabled and threshold_val is not None else None
            cand.odds_dyn_filter_enabled = enabled
            if enabled and threshold_val is not None and passed is False:
                continue
            filtered.append(cand)
        return filtered

    def _apply_odds_dyn_ev_margin(self, candidates: list[BetCandidate]) -> list[BetCandidate]:
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

        filtered: list[BetCandidate] = []
        for cand in candidates:
            score = cand.odds_dyn_ev_score
            margin = None
            min_ev_eff_odm = None
            passed = None
            if score is not None and ref is not None:
                if direction == "low":
                    margin = slope * max(0.0, ref - score)
                else:
                    margin = slope * max(0.0, score - ref)
                min_ev_eff_odm = (cand.min_ev_eff or cand.base_min_ev or 0.0) + (margin or 0.0)
                if enabled_eff:
                    passed = cand.ev >= min_ev_eff_odm

            cand.odds_dyn_ev_metric = metric
            cand.odds_dyn_ev_lookback_min = lookback
            cand.odds_dyn_ev_ref = ref
            cand.odds_dyn_ev_slope = slope
            cand.odds_dyn_ev_margin = margin
            cand.min_ev_eff_odds_dyn = min_ev_eff_odm
            cand.passed_odds_dyn_ev_margin = passed if enabled_eff else None
            cand.odds_dyn_ev_margin_enabled = enabled_eff

            if enabled_eff and passed is False:
                continue
            filtered.append(cand)
        return filtered

    def _apply_odds_floor(self, candidates: list[BetCandidate]) -> list[BetCandidate]:
        min_odds = getattr(self.config.betting, "odds_floor_min_odds", 0.0)
        try:
            min_odds_val = float(min_odds) if min_odds is not None else 0.0
        except Exception:
            min_odds_val = 0.0
        enabled = bool(min_odds_val and min_odds_val > 0)
        filtered: list[BetCandidate] = []
        for cand in candidates:
            odds_val = cand.odds
            passed = None
            if enabled and odds_val is not None:
                passed = bool(odds_val >= min_odds_val)
            cand.odds_floor_min_odds = float(min_odds_val) if enabled else 0.0
            cand.odds_floor_odds_used = float(odds_val) if odds_val is not None else None
            cand.passed_odds_floor = passed if enabled else None
            if enabled and passed is False:
                continue
            filtered.append(cand)
        return filtered

    def _apply_stake_odds_damp(self, candidates: list[BetCandidate], *, min_yen: int) -> list[BetCandidate]:
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

        adjusted: list[BetCandidate] = []
        for cand in candidates:
            base_stake = float(cand.stake_yen)
            odds_val = cand.odds
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
            cand.stake_before = base_stake
            cand.stake_mult = float(mult)
            cand.stake_after = float(stake_after)
            cand.stake_damp_ref_odds = float(ref_odds) if enabled_eff else 0.0
            cand.stake_damp_power = float(power) if enabled_eff else 1.0
            cand.stake_damp_min_mult = float(min_mult) if enabled_eff else 0.0
            cand.stake_damp_odds = float(odds_num) if odds_num is not None else None
            cand.stake_damp_ratio = float(ratio) if ratio is not None and np.isfinite(ratio) else None
            cand.stake_damp_mult_raw = float(mult_raw)
            cand.stake_damp_mult_clamped = float(mult)
            cand.stake_damp_floor_unit = int(unit)
            cand.stake_damp_enabled = enabled_eff
            cand.stake_yen = int(stake_after)
            adjusted.append(cand)
        return adjusted

    def _get_race_info(self, race_id: str) -> Optional[dict]:
        """レース情報取得"""
        query = text("""
            SELECT race_id, date, track_code, race_no
            FROM fact_race
            WHERE race_id = :race_id
        """)
        result = self.session.execute(query, {"race_id": race_id}).fetchone()
        if result:
            return {
                "race_name": f"{result[2]}場{result[3]}R",
            }
        return None
    
    def check_stop_loss(self) -> bool:
        """損失停止条件チェック"""
        max_loss = self.bankroll * self.config.betting.stop.max_daily_loss_pct
        return self.daily_loss >= max_loss
    
    def save_signals(self, candidates: list[BetCandidate], run_id: str) -> int:
        """シグナルをDBに保存"""
        count = 0
        closing_mult = float(self.config.betting.closing_odds_multiplier)
        ts_vol_cap = self.config.betting.log_odds_std_60m_max
        for cand in candidates:
            # 実運用では slippage_table を外部から注入する（なければglobalのみ）
            slippage_meta = None
            if self.slippage_table is not None:
                odds_effective, slippage_meta = self.slippage_table.effective_odds(
                    odds_buy=cand.odds,
                    ts_vol=None,
                    snap_age_min=None,
                )
            else:
                odds_effective = cand.odds * closing_mult if closing_mult > 0 else cand.odds
            signal = BetSignal(
                signal_id=str(uuid.uuid4()),
                run_id=run_id,
                race_id=cand.race_id,
                ticket_type=cand.ticket_type,
                selection={"horse_no": cand.horse_no},
                asof_time=cand.asof_time,
                p_hat=cand.p_hat,
                odds_snapshot=cand.odds,
                ev=cand.ev,
                stake_yen=cand.stake_yen,
                # Ticket1: 追跡性向上（DBスキーマは変えずreason JSONに入れる）
                reason={
                    "text": cand.reason,
                    "closing_odds_multiplier": closing_mult,
                    "odds_effective": odds_effective,
                    "slippage_meta": slippage_meta,
                    "stake_before": cand.stake_before,
                    "stake_after": cand.stake_after,
                    "stake_damp_ref_odds": cand.stake_damp_ref_odds,
                    "stake_damp_power": cand.stake_damp_power,
                    "stake_damp_min_mult": cand.stake_damp_min_mult,
                    "stake_damp_odds": cand.stake_damp_odds,
                    "stake_damp_ratio": cand.stake_damp_ratio,
                    "stake_damp_mult_raw": cand.stake_damp_mult_raw,
                    "stake_damp_mult_clamped": cand.stake_damp_mult_clamped,
                    "stake_damp_floor_unit": cand.stake_damp_floor_unit,
                    "stake_damp_enabled": cand.stake_damp_enabled,
                    # Ticket N2: odds帯バイアス補正（運用時は外から注入したmodelで計算する）
                    "odds_band_bias_enabled": bool(self.config.betting.odds_band_bias.enabled),
                    "overlay_shrink_alpha": float(self.config.betting.overlay_shrink_alpha) if self.config.betting.overlay_shrink_alpha is not None else None,
                    "ts_vol_cap": float(ts_vol_cap) if ts_vol_cap is not None else None,
                    "segblend_segment": cand.segblend_segment,
                    "segblend_w_used": cand.segblend_w_used,
                    "segblend_w_global": cand.segblend_w_global,
                    "race_softmax_enabled": cand.race_softmax_enabled,
                    "race_softmax_w": cand.race_softmax_w,
                    "race_softmax_T": cand.race_softmax_T,
                    "selection_mode": cand.selection_mode,
                    "daily_top_n_n": cand.daily_top_n_n,
                    "daily_top_n_metric": cand.daily_top_n_metric,
                    "daily_top_n_min_value": cand.daily_top_n_min_value,
                    "daily_rank_in_day": cand.daily_rank_in_day,
                    "daily_candidates_before": cand.daily_candidates_before,
                    "daily_candidates_after_min": cand.daily_candidates_after_min,
                    "overround_sum_inv": cand.overround_sum_inv,
                    "takeout_implied": cand.takeout_implied,
                    "base_min_ev": cand.base_min_ev,
                    "takeout_ref": cand.takeout_ref,
                    "takeout_slope": cand.takeout_slope,
                    "min_ev_eff": cand.min_ev_eff,
                    "passed_takeout_ev_margin": cand.passed_takeout_ev_margin,
                    "market_blend_enabled": cand.market_blend_enabled,
                    "market_prob_method": cand.market_prob_method,
                    "market_blend_w": cand.market_blend_w,
                    "p_blend": cand.p_blend,
                    "ev_blend": cand.ev_blend,
                    "odds_band_blend": cand.odds_band_blend,
                    "t_ev": cand.t_ev,
                    "odds_cap": cand.odds_cap,
                    "exclude_odds_band": cand.exclude_odds_band,
                    "p_hat_pre_blend": cand.p_hat_pre_blend,
                    "odds_dyn_ev_metric": cand.odds_dyn_ev_metric,
                    "odds_dyn_ev_lookback_min": cand.odds_dyn_ev_lookback_min,
                    "odds_dyn_ev_score": cand.odds_dyn_ev_score,
                    "odds_dyn_ev_ref": cand.odds_dyn_ev_ref,
                    "odds_dyn_ev_slope": cand.odds_dyn_ev_slope,
                    "odds_dyn_ev_margin": cand.odds_dyn_ev_margin,
                    "min_ev_eff_odds_dyn": cand.min_ev_eff_odds_dyn,
                    "passed_odds_dyn_ev_margin": cand.passed_odds_dyn_ev_margin,
                    "odds_dyn_ev_margin_enabled": cand.odds_dyn_ev_margin_enabled,
                    "race_cost_cap_mode": cand.race_cost_cap_mode,
                    "race_cost_cap_value": cand.race_cost_cap_value,
                    "race_cost_cap_selected_from": cand.race_cost_cap_selected_from,
                    "race_cost_filter_passed": cand.race_cost_filter_passed,
                    "race_cost_filter_reason": cand.race_cost_filter_reason,
                },
            )
            self.session.add(signal)
            count += 1
        
        self.session.commit()
        return count


def generate_bet_signals(
    session: Session,
    race_ids: list[str],
    predictions_by_race: dict[str, list[dict]],
    bankroll: float,
    uncertainty_shrink: UncertaintyShrink | None = None,
    output_path: Optional[Path] = None,
) -> list[BetCandidate]:
    """
    買い目シグナルを生成
    
    Args:
        session: DBセッション
        race_ids: レースIDリスト
        predictions_by_race: レースIDごとの予測結果
        bankroll: 資金
        output_path: CSV出力先
    
    Returns:
        買い目候補リスト
    """
    policy = BettingPolicy(session, bankroll, uncertainty_shrink=uncertainty_shrink)
    asof_time = datetime.now()
    
    all_candidates = []
    
    for race_id in race_ids:
        if policy.check_stop_loss():
            logger.warning("Stop loss triggered")
            break
        
        predictions = predictions_by_race.get(race_id, [])
        candidates = policy.generate_candidates(race_id, predictions, asof_time)
        all_candidates.extend(candidates)
    
    # CSV出力
    if output_path and all_candidates:
        _export_csv(all_candidates, output_path)
    
    # DB保存
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy.save_signals(all_candidates, run_id)
    
    logger.info(f"Generated {len(all_candidates)} bet signals")
    return all_candidates


def _export_csv(candidates: list[BetCandidate], output_path: Path) -> None:
    """CSV出力"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # ヘッダー
        writer.writerow([
            "生成時刻", "レースID", "レース名", "馬番", "馬名",
            "券種", "オッズ", "予測確率", "期待値", "賭け金",
            "理由", "注意事項",
            "takeout_implied", "base_min_ev", "takeout_ref", "takeout_slope", "min_ev_eff", "passed_takeout_ev_margin",
        ])
        
        # データ
        for c in candidates:
            writer.writerow([
                c.asof_time.strftime("%Y-%m-%d %H:%M:%S"),
                c.race_id,
                c.race_name,
                c.horse_no,
                c.horse_name,
                c.ticket_type,
                f"{c.odds:.1f}",
                f"{c.p_hat:.4f}",
                f"{c.ev:.4f}",
                c.stake_yen,
                c.reason,
                "※投票は公式手段で行ってください。成立した投票の取消・変更はできません。",
                c.takeout_implied,
                c.base_min_ev,
                c.takeout_ref,
                c.takeout_slope,
                c.min_ev_eff,
                c.passed_takeout_ev_margin,
            ])
    
    logger.info(f"Exported {len(candidates)} candidates to {output_path}")

