"""
モデル学習

市場オッズベースのブレンドモデル:
    p_hat = w * p_mkt + (1-w) * p_model
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_config
from ..features.build_features import FeatureBuilder
from ..features.history_features import distance_bucket
from ..features.odds_movement import (
    QUINELLA_ODDS_MOVEMENT_COLS,
    fetch_quinella_odds_movement_features,
)
from .race_softmax import fit_race_softmax

logger = logging.getLogger(__name__)


class SurfaceDispatchBooster:
    """
    Booster-like wrapper that dispatches predictions to turf/dirt models.

    Downstream code sometimes calls `model.lgb_model.predict(X)` directly; this wrapper
    keeps that working while enabling per-surface models.
    """

    def __init__(self, turf: lgb.Booster, dirt: lgb.Booster):
        self._turf = turf
        self._dirt = dirt
        # Used only for logging/summary; choose a stable representative.
        self.best_iteration = max(
            int(getattr(turf, "best_iteration", 0) or 0),
            int(getattr(dirt, "best_iteration", 0) or 0),
        )

    def predict(self, X, **kwargs):
        n = len(X)
        if n <= 0:
            return np.asarray([], dtype=float)

        is_turf = None
        try:
            if isinstance(X, pd.DataFrame) and "is_turf" in X.columns:
                is_turf = pd.to_numeric(X["is_turf"], errors="coerce").fillna(0).astype(int).values
        except Exception:
            is_turf = None

        if is_turf is None:
            return np.asarray(self._dirt.predict(X, **kwargs), dtype=float)

        mask_turf = is_turf == 1
        out = np.empty(n, dtype=float)
        if mask_turf.any():
            out[mask_turf] = np.asarray(
                self._turf.predict(X.iloc[mask_turf], **kwargs), dtype=float
            )
        if (~mask_turf).any():
            out[~mask_turf] = np.asarray(
                self._dirt.predict(X.iloc[~mask_turf], **kwargs), dtype=float
            )
        return out


def _clip_prob_series(p: pd.Series, lo: float, hi: float) -> pd.Series:
    return p.clip(lower=lo, upper=hi)


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


def _surface_segments_from_race_ids(
    session: Session,
    race_ids: pd.Series,
    fallback_is_turf: Optional[pd.Series] = None,
) -> pd.Series:
    if race_ids is None or len(race_ids) == 0:
        return pd.Series([], dtype=str)

    segments = pd.Series(["unknown"] * len(race_ids), index=race_ids.index, dtype=str)
    unique_ids = sorted(set(race_ids.astype(str).tolist()))
    if unique_ids:
        try:
            rows = session.execute(
                text("SELECT race_id, surface FROM fact_race WHERE race_id = ANY(:race_ids)"),
                {"race_ids": unique_ids},
            ).fetchall()
            mapping = {str(r[0]): _surface_label(r[1]) for r in rows}
            segments = race_ids.astype(str).map(mapping).fillna("unknown")
        except Exception as e:
            logger.warning(f"Surface lookup failed; fallback to is_turf if available: {e}")

    if fallback_is_turf is not None and "unknown" in segments.values:
        vals = pd.to_numeric(fallback_is_turf, errors="coerce")
        mask = segments == "unknown"
        segments.loc[mask & (vals == 1)] = "turf"
        segments.loc[mask & (vals == 0)] = "dirt"
    return segments.astype(str)


def _align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X2 = X.reindex(columns=feature_names, fill_value=0.0)
    return X2.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def _grid_search_blend_weight(
    y: np.ndarray,
    p_mkt: np.ndarray,
    p_model: np.ndarray,
    *,
    w_min: float,
    w_max: float,
    grid_step: float,
    default_w: float,
) -> tuple[float, float]:
    from sklearn.metrics import log_loss  # type: ignore[import-untyped]

    if grid_step <= 0:
        return float(default_w), float("nan")
    if w_max < w_min:
        w_min, w_max = w_max, w_min

    mask = np.isfinite(y) & np.isfinite(p_mkt) & np.isfinite(p_model)
    if mask.sum() <= 0:
        return float(default_w), float("nan")

    y2 = y[mask].astype(int)
    p_mkt2 = p_mkt[mask].astype(float)
    p_model2 = p_model[mask].astype(float)

    w_values = np.arange(float(w_min), float(w_max) + 1e-12, float(grid_step))
    if w_values.size == 0:
        return float(default_w), float("nan")

    best_w = float(default_w)
    best_loss = float("inf")
    for w in w_values:
        p_blend = w * p_mkt2 + (1.0 - w) * p_model2
        p_blend = np.clip(p_blend, 1e-6, 1.0 - 1e-6)
        loss = float(log_loss(y2, p_blend, labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_w = float(w)
    return best_w, best_loss


def _logit_series(p: pd.Series) -> np.ndarray:
    v = p.values.astype(float)
    return np.log(v / (1.0 - v))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class WinProbabilityModel:
    """勝利確率予測モデル"""

    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.config = get_config()
        self.lgb_model: lgb.Booster | SurfaceDispatchBooster | None = None
        self.blend_weight = self.config.model.blend_weight_w
        self.blend_segmented: Optional[dict] = None
        self.race_softmax_params: Optional[dict] = None
        self.calibrator: Any | None = None
        self.use_market_offset = bool(getattr(self.config.model, "use_market_offset", False))
        self.p_mkt_clip = tuple(getattr(self.config.model, "p_mkt_clip", (1e-4, 1.0 - 1e-4)))
        # Ticket G1: Residual Cap
        residual_cap_cfg = getattr(self.config.model, "residual_cap", {})
        self.residual_cap_enabled = (
            bool(residual_cap_cfg.get("enabled", False))
            if isinstance(residual_cap_cfg, dict)
            else False
        )
        self.residual_cap_quantile = (
            float(residual_cap_cfg.get("quantile", 0.99))
            if isinstance(residual_cap_cfg, dict)
            else 0.99
        )
        self.residual_cap_p_clip = (
            tuple(residual_cap_cfg.get("p_clip", [1e-4, 1.0 - 1e-4]))
            if isinstance(residual_cap_cfg, dict)
            else (1e-4, 1.0 - 1e-4)
        )
        self.residual_cap_apply_stage = (
            str(residual_cap_cfg.get("apply_stage", "pre_calibration"))
            if isinstance(residual_cap_cfg, dict)
            else "pre_calibration"
        )
        self.residual_cap_value: Optional[float] = None  # fit()で計算して保存
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        p_mkt: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        p_mkt_valid: Optional[pd.Series] = None,
    ) -> dict:
        """
        モデル学習

        Args:
            X: 訓練特徴量DataFrame
            y: 訓練ターゲット（1着=1, else=0）
            p_mkt: 訓練市場確率
            X_valid: 検証特徴量（Noneなら訓練データの最後20%を使用）
            y_valid: 検証ターゲット
            p_mkt_valid: 検証市場確率

        Returns:
            学習結果メトリクス（検証データ上で評価）
        """
        from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

        self.feature_names = X.columns.tolist()

        # 検証データが指定されていない場合、訓練データの最後20%を使用
        if X_valid is None:
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            p_mkt_train, p_mkt_val = p_mkt.iloc[:split_idx], p_mkt.iloc[split_idx:]
        else:
            X_train, X_val = X, X_valid
            y_train, y_val = y, y_valid
            p_mkt_train, p_mkt_val = p_mkt, p_mkt_valid

        # LightGBM用データセット
        if self.use_market_offset:
            lo, hi = float(self.p_mkt_clip[0]), float(self.p_mkt_clip[1])
            init_train = _logit_series(_clip_prob_series(p_mkt_train.astype(float), lo, hi))
            init_val = _logit_series(_clip_prob_series(p_mkt_val.astype(float), lo, hi))
            train_data = lgb.Dataset(X_train, label=y_train, init_score=init_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data, init_score=init_val)
        else:
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            # rolling比較の再現性向上
            "seed": int(self.config.model.seed),
            "data_random_seed": int(self.config.model.seed),
            "feature_fraction_seed": int(self.config.model.seed),
            "bagging_seed": int(self.config.model.seed),
        }
        if self.use_market_offset:
            # priorをinit_scoreで与えるので、平均ラベルからの自動初期化はOFF（残差の解釈を安定化）
            params["boost_from_average"] = False

        # ★重要: 検証データで early stopping（過学習検出）
        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50, verbose=False)],
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
        )

        # ★重要: メトリクスは検証データ上で評価（訓練データでの評価は過大評価になりがち）
        if self.use_market_offset:
            lo, hi = float(self.p_mkt_clip[0]), float(self.p_mkt_clip[1])
            init_val = _logit_series(_clip_prob_series(p_mkt_val.astype(float), lo, hi))
            init_train = _logit_series(_clip_prob_series(p_mkt_train.astype(float), lo, hi))

            resid_val = self.lgb_model.predict(X_val, raw_score=True)
            resid_train = self.lgb_model.predict(X_train, raw_score=True)

            p_model_val = _sigmoid(init_val + resid_val)
            p_model_train = _sigmoid(init_train + resid_train)

            # Market-offset時は市場をprior固定するため、追加ブレンドはしない（二重カウント回避）
            p_blend_val = p_model_val
            p_blend_train = p_model_train
        else:
            p_model_val = np.asarray(self.lgb_model.predict(X_val), dtype=float)
            p_blend_val = self.blend_weight * p_mkt_val + (1 - self.blend_weight) * p_model_val

            # 訓練データ上のメトリクスも参考に（過学習チェック用）
            p_model_train = np.asarray(self.lgb_model.predict(X_train), dtype=float)
            p_blend_train = (
                self.blend_weight * p_mkt_train + (1 - self.blend_weight) * p_model_train
            )

        # ★保険: スモーク等で検証側に片ラベルしか無い場合でも落ちないように labels を固定
        # （log_loss: y_true contains only one label を回避）
        metrics = {
            # 検証データ（これが本当のパフォーマンス）
            "valid_logloss_model": log_loss(y_val, p_model_val, labels=[0, 1]),
            "valid_logloss_blend": log_loss(y_val, p_blend_val, labels=[0, 1]),
            "valid_logloss_market": log_loss(y_val, p_mkt_val, labels=[0, 1]),
            "valid_brier_model": brier_score_loss(y_val, p_model_val),
            "valid_brier_blend": brier_score_loss(y_val, p_blend_val),
            "valid_brier_market": brier_score_loss(y_val, p_mkt_val),
            # 訓練データ（過学習チェック用）
            "train_logloss_blend": log_loss(y_train, p_blend_train, labels=[0, 1]),
            "train_brier_blend": brier_score_loss(y_train, p_blend_train),
            # サンプル数
            "n_train": len(y_train),
            "n_valid": len(y_val),
            "n_features": len(self.feature_names),
            "best_iteration": self.lgb_model.best_iteration,
        }

        # ★校正用に検証データの予測結果を保持（train_model() からアクセス可能に）
        self._last_val_predictions = {
            "p_blend": p_blend_val,
            "y_true": y_val.values,
        }

        # Ticket G1: Residual Cap計算（train期間の全候補から）
        if self.residual_cap_enabled and self.use_market_offset:
            # train期間の全候補のresidを計算
            resid_train_all = self.lgb_model.predict(X_train, raw_score=True)
            # |resid|のquantileからcapを算出
            abs_resid = np.abs(resid_train_all)
            self.residual_cap_value = float(np.quantile(abs_resid, self.residual_cap_quantile))
            logger.info(
                f"Residual cap computed: quantile={self.residual_cap_quantile}, "
                f"cap={self.residual_cap_value:.4f}"
            )

        logger.info(
            f"Training complete: valid_brier={metrics['valid_brier_blend']:.4f}, "
            f"train_brier={metrics['train_brier_blend']:.4f}"
        )
        return metrics

    def _resolve_blend_weights(self, segments: Optional[Sequence[str]], n: int) -> np.ndarray:
        w_global = float(self.blend_weight)
        if segments is None or not self.blend_segmented or not self.blend_segmented.get("enabled"):
            return np.full(n, w_global, dtype=float)

        if isinstance(segments, pd.Series):
            seg_list = segments.tolist()
        elif isinstance(segments, str):
            seg_list = [segments] * n
        else:
            seg_list = list(segments)
        if len(seg_list) == 0:
            return np.full(n, w_global, dtype=float)
        if len(seg_list) != n:
            logger.warning("Segment length mismatch; fallback to global blend weight")
            return np.full(n, w_global, dtype=float)

        seg_weights = {}
        segments_meta = (
            self.blend_segmented.get("segments", {})
            if isinstance(self.blend_segmented, dict)
            else {}
        )
        for key, meta in segments_meta.items():
            if isinstance(meta, dict) and "w" in meta:
                try:
                    seg_weights[str(key)] = float(meta["w"])
                except Exception:
                    continue

        weights = []
        for seg in seg_list:
            key = "unknown" if seg is None else str(seg)
            weights.append(seg_weights.get(key, w_global))
        return np.asarray(weights, dtype=float)

    def get_blend_weight_for_segment(self, segment: Optional[str]) -> float:
        w_global = float(self.blend_weight)
        if not self.blend_segmented or not self.blend_segmented.get("enabled"):
            return w_global
        segments_meta = (
            self.blend_segmented.get("segments", {})
            if isinstance(self.blend_segmented, dict)
            else {}
        )
        if segment is None:
            return w_global
        meta = segments_meta.get(str(segment))
        if isinstance(meta, dict) and "w" in meta:
            try:
                return float(meta["w"])
            except Exception:
                return w_global
        return w_global

    def predict(
        self,
        X: pd.DataFrame,
        p_mkt: pd.Series,
        calibrate: bool = True,
        return_residual_meta: bool = False,
        segments: Optional[Sequence[str]] = None,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """
        確率予測

        Args:
            X: 特徴量
            p_mkt: 市場確率
            calibrate: 校正を適用するか（校正器がある場合）
            return_residual_meta: 残差メタデータを返すか（G1追跡用）

        Returns:
            ブレンド確率（校正済みの場合はそれ）、または (確率, メタデータ) のタプル
        """
        if self.lgb_model is None:
            raise ValueError("Model not trained")

        residual_meta = {}

        # If surface segments are provided, use them to drive turf/dirt dispatch (even if the
        # feature payload has stale/missing `is_turf`). This is a no-op for non-surface models.
        X_dispatch = X
        if (
            isinstance(self.lgb_model, SurfaceDispatchBooster)
            and segments is not None
            and isinstance(X, pd.DataFrame)
        ):
            try:
                if isinstance(segments, pd.Series):
                    seg_list = segments.tolist()
                elif isinstance(segments, str):
                    seg_list = [segments] * len(X)
                else:
                    seg_list = list(segments)
                if len(seg_list) == len(X) and len(seg_list) > 0:
                    X_dispatch = X.copy()
                    X_dispatch["is_turf"] = [
                        1 if (s is not None and str(s).strip().lower() == "turf") else 0
                        for s in seg_list
                    ]
            except Exception:
                X_dispatch = X

        if self.use_market_offset:
            lo, hi = float(self.p_mkt_clip[0]), float(self.p_mkt_clip[1])
            init = _logit_series(_clip_prob_series(p_mkt.astype(float), lo, hi))
            resid = self.lgb_model.predict(X_dispatch, raw_score=True)

            # Ticket G1: Residual Cap適用
            if self.residual_cap_enabled and self.residual_cap_value is not None:
                resid_capped = np.clip(resid, -self.residual_cap_value, self.residual_cap_value)
                if return_residual_meta:
                    residual_meta = {
                        "resid": resid,
                        "resid_cap": resid_capped,
                        "cap_value": self.residual_cap_value,
                    }
                # apply_stageに応じて適用
                if self.residual_cap_apply_stage == "pre_calibration":
                    p_blend_raw = _sigmoid(init + resid_capped)
                else:
                    # post_calibrationの場合は一旦通常通り計算してからcap適用
                    p_blend_raw = _sigmoid(init + resid)
                    # ここでcapを適用するには、p_blend_rawから逆算する必要があるが、
                    # 簡易的にresid_cappedを使う
                    p_blend_raw = _sigmoid(init + resid_capped)
            else:
                p_blend_raw = _sigmoid(init + resid)
                if return_residual_meta:
                    residual_meta = {
                        "resid": resid,
                        "resid_cap": None,
                        "cap_value": None,
                    }
        else:
            p_model = self.lgb_model.predict(X_dispatch)
            p_mkt_arr = np.asarray(p_mkt, dtype=float)
            w_arr = self._resolve_blend_weights(segments, len(p_mkt_arr))
            p_blend_raw = w_arr * p_mkt_arr + (1 - w_arr) * p_model
            if return_residual_meta and len(p_mkt_arr) == 1:
                seg_val = None
                if segments:
                    seg_val = segments[0] if not isinstance(segments, str) else segments
                residual_meta["segblend_segment"] = seg_val
                residual_meta["segblend_w_used"] = float(w_arr[0])
                residual_meta["segblend_w_global"] = float(self.blend_weight)

        # 校正器がある場合は適用
        p_blend = p_blend_raw
        if calibrate and hasattr(self, "calibrator") and self.calibrator is not None:
            p_blend = self.calibrator.transform(p_blend_raw)
            if return_residual_meta:
                residual_meta["p_hat_capped"] = p_blend_raw
                residual_meta["p_hat_final"] = p_blend
        elif return_residual_meta:
            residual_meta["p_hat_capped"] = p_blend_raw
            residual_meta["p_hat_final"] = p_blend

        # Optional post-process: shrink final probability toward market-implied p_mkt.
        # This is applied only for the "final" path (calibrate=True) so p_blend_raw
        # / calibrate=False outputs keep their original semantics.
        if calibrate:
            shrink_alpha = getattr(self.config.model, "market_shrink_alpha", None)
            if shrink_alpha is not None:
                try:
                    a = float(shrink_alpha)
                except Exception:
                    a = None
                if a is not None and np.isfinite(a):
                    lo, hi = float(self.p_mkt_clip[0]), float(self.p_mkt_clip[1])
                    p_hat_arr = np.asarray(p_blend, dtype=float)
                    p_mkt_arr = np.asarray(p_mkt, dtype=float)
                    mask = np.isfinite(p_hat_arr) & np.isfinite(p_mkt_arr)
                    p_shrunk = p_hat_arr.copy()
                    if mask.any():
                        if a <= 0.0:
                            p_shrunk[mask] = np.clip(p_mkt_arr[mask], lo, hi)
                        elif a >= 1.0:
                            p_shrunk[mask] = np.clip(p_hat_arr[mask], lo, hi)
                        else:
                            hat_c = np.clip(p_hat_arr[mask], lo, hi)
                            mkt_c = np.clip(p_mkt_arr[mask], lo, hi)
                            logit_hat = np.log(hat_c / (1.0 - hat_c))
                            logit_mkt = np.log(mkt_c / (1.0 - mkt_c))
                            logit_shrunk = logit_mkt + a * (logit_hat - logit_mkt)
                            p_shrunk[mask] = 1.0 / (1.0 + np.exp(-logit_shrunk))
                    p_blend = p_shrunk
                    if return_residual_meta:
                        residual_meta["market_shrink_alpha"] = float(a)
                        residual_meta["p_hat_pre_market_shrink"] = p_hat_arr
                        residual_meta["p_hat_final"] = p_blend
        if return_residual_meta:
            return p_blend, residual_meta
        return p_blend

    def save(self, path: Path) -> None:
        """モデルを保存（校正器含む）"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "version": self.version,
                    "lgb_model": self.lgb_model,
                    "blend_weight": self.blend_weight,
                    "blend_segmented": self.blend_segmented,
                    "race_softmax_params": self.race_softmax_params,
                    "use_market_offset": self.use_market_offset,
                    "p_mkt_clip": self.p_mkt_clip,
                    "feature_names": self.feature_names,
                    "calibrator": getattr(self, "calibrator", None),
                    # Ticket G1: Residual Cap
                    "residual_cap_enabled": self.residual_cap_enabled,
                    "residual_cap_value": self.residual_cap_value,
                    "residual_cap_quantile": self.residual_cap_quantile,
                    "residual_cap_p_clip": self.residual_cap_p_clip,
                    "residual_cap_apply_stage": self.residual_cap_apply_stage,
                },
                f,
            )
        logger.info(f"Model saved to {path}")

        # Ticket G1: Residual Cap設定をJSONでも保存（読み取り容易）
        if self.residual_cap_enabled and self.residual_cap_value is not None:
            cap_json_path = path.parent / "residual_cap.json"
            cap_json_path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "quantile": self.residual_cap_quantile,
                        "cap_value": self.residual_cap_value,
                        "p_clip": list(self.residual_cap_p_clip),
                        "apply_stage": self.residual_cap_apply_stage,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            logger.info(f"Residual cap saved to {cap_json_path}")

    @classmethod
    def load(cls, path: Path) -> "WinProbabilityModel":
        """モデルを読み込み（校正器含む）"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Allow loading wrapper objects saved via pickle.dump(model_obj, ...).
        if not isinstance(data, dict):
            return data

        model = cls(version=data["version"])
        model.lgb_model = data["lgb_model"]
        model.blend_weight = data["blend_weight"]
        model.blend_segmented = data.get("blend_segmented")
        model.race_softmax_params = data.get("race_softmax_params")
        model.use_market_offset = bool(data.get("use_market_offset", False))
        model.p_mkt_clip = tuple(data.get("p_mkt_clip", (1e-4, 1.0 - 1e-4)))
        model.feature_names = data["feature_names"]
        model.calibrator = data.get("calibrator")

        # Ticket G1: Residual Cap復元
        model.residual_cap_enabled = bool(data.get("residual_cap_enabled", False))
        model.residual_cap_value = data.get("residual_cap_value")
        model.residual_cap_quantile = float(data.get("residual_cap_quantile", 0.99))
        model.residual_cap_p_clip = tuple(data.get("residual_cap_p_clip", (1e-4, 1.0 - 1e-4)))
        model.residual_cap_apply_stage = str(
            data.get("residual_cap_apply_stage", "pre_calibration")
        )

        return model


class _DistanceBucketBoosterProxy:
    def __init__(self, parent: "DistanceBucketWinProbabilityModel"):
        self._parent = parent

    def predict(self, X: pd.DataFrame, raw_score: bool = False):
        return self._parent._predict_p_model(X, raw_score=raw_score)


class DistanceBucketWinProbabilityModel:
    """Per-distance bucket wrapper around WinProbabilityModel."""

    def __init__(
        self,
        *,
        fallback_model: WinProbabilityModel,
        bucket_models: dict[int, WinProbabilityModel],
        bucket_meta: Optional[dict] = None,
    ):
        self.fallback_model = fallback_model
        self.bucket_models = bucket_models
        self.bucket_meta = bucket_meta or {}

    def __getattr__(self, name: str):
        # Delegate config/knobs/attrs (feature_names, calibrator, blend_weight, etc.)
        return getattr(self.fallback_model, name)

    @property
    def lgb_model(self):
        return _DistanceBucketBoosterProxy(self)

    def _bucket_series(self, X: pd.DataFrame) -> pd.Series:
        if X is None or len(X) == 0 or "distance" not in X.columns:
            return pd.Series(
                [None] * (0 if X is None else len(X)), index=getattr(X, "index", None), dtype=object
            )
        d = pd.to_numeric(X["distance"], errors="coerce")
        d = d.where(d > 0)
        return d.apply(distance_bucket)

    def _predict_p_model(self, X: pd.DataFrame, *, raw_score: bool = False) -> np.ndarray:
        if X is None or len(X) == 0:
            return np.asarray([], dtype=float)
        buckets = self._bucket_series(X)
        out = np.zeros(len(X), dtype=float)

        for b in sorted(set(buckets.dropna().tolist())):
            mask = buckets == b
            m = self.bucket_models.get(int(b)) if b is not None else None
            booster = (
                getattr(m, "lgb_model", None)
                if m is not None
                else getattr(self.fallback_model, "lgb_model", None)
            )
            if booster is None:
                continue
            out[mask.values] = booster.predict(X.loc[mask], raw_score=raw_score)

        unknown = buckets.isna()
        if unknown.any():
            booster = getattr(self.fallback_model, "lgb_model", None)
            if booster is not None:
                out[unknown.values] = booster.predict(X.loc[unknown], raw_score=raw_score)
        return out

    def predict(
        self,
        X: pd.DataFrame,
        p_mkt: pd.Series,
        calibrate: bool = True,
        return_residual_meta: bool = False,
        segments: Optional[Sequence[str]] = None,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        if X is None or len(X) == 0:
            empty = np.asarray([], dtype=float)
            return (empty, {}) if return_residual_meta else empty
        if return_residual_meta and len(X) != 1:
            raise ValueError("return_residual_meta is only supported for single-row predictions")

        p_mkt_s = (
            pd.to_numeric(pd.Series(p_mkt, index=X.index), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        buckets = self._bucket_series(X)
        cal = getattr(self.fallback_model, "calibrator", None)

        if return_residual_meta:
            b = buckets.iloc[0] if len(buckets) else None
            if b is not None:
                ib = int(b)
                m = self.bucket_models[ib] if ib in self.bucket_models else self.fallback_model
            else:
                m = self.fallback_model
            p_raw, meta = m.predict(
                X, p_mkt_s, calibrate=False, return_residual_meta=True, segments=segments
            )
            p_out = cal.transform(p_raw) if (calibrate and cal is not None) else p_raw
            meta = dict(meta or {})
            meta["distance_bucket"] = int(b) if b is not None else None
            meta["p_hat_final"] = p_out
            return p_out, meta

        out = np.zeros(len(X), dtype=float)

        # Unknown distance -> fallback.
        unknown = buckets.isna()
        if unknown.any():
            seg_sub = (
                segments.reindex(unknown.index)[unknown]
                if isinstance(segments, pd.Series)
                else None
            )
            p_raw = self.fallback_model.predict(
                X.loc[unknown], p_mkt_s.loc[unknown], calibrate=False, segments=seg_sub
            )
            out[unknown.values] = cal.transform(p_raw) if (calibrate and cal is not None) else p_raw

        for b in sorted(set(buckets.dropna().tolist())):
            mask = buckets == b
            if b is not None:
                ib = int(b)
                m = self.bucket_models[ib] if ib in self.bucket_models else self.fallback_model
            else:
                m = self.fallback_model
            seg_sub = (
                segments.reindex(mask.index)[mask] if isinstance(segments, pd.Series) else None
            )
            p_raw = m.predict(X.loc[mask], p_mkt_s.loc[mask], calibrate=False, segments=seg_sub)
            out[mask.values] = cal.transform(p_raw) if (calibrate and cal is not None) else p_raw
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"DistanceBucketWinProbabilityModel saved to {path}")


def prepare_training_data(
    session: Session,
    min_date: str,
    max_date: str,
    buy_t_minus_minutes: Optional[int] = None,
) -> tuple:
    """
    学習データを準備（リーク防止版）

    重要なリーク防止策:
        1. 各レースの購入時点（発走-N分）を計算
        2. features.asof_time <= 購入時点 の条件で絞る
        3. (race_id, horse_id)ごとに最新1件のみを使用（DISTINCT ON）

    Args:
        session: DBセッション
        min_date: 開始日（YYYY-MM-DD）
        max_date: 終了日（YYYY-MM-DD）
        buy_t_minus_minutes: 発走何分前を購入時点とするか

    Returns:
        (X, y, p_mkt, race_ids)
    """
    # ★config一元化: buy_t_minus_minutes が None なら config から取得
    if buy_t_minus_minutes is None:
        buy_t_minus_minutes = get_config().backtest.buy_t_minus_minutes

    cfg = get_config()
    u = cfg.universe
    track_codes = list(u.track_codes or [])
    exclude_race_ids = list(u.exclude_race_ids or [])

    # PostgreSQLのDISTINCT ONを使用して、(race_id, horse_id)ごとに
    # 購入時点以前の最新特徴量のみを取得
    #
    # 注意: date::timestamp + start_time で正しい timestamp を作成
    #       INTERVAL は make_interval() で動的に生成
    query = text("""
        WITH buy_times AS (
            -- 各レースの購入時点を計算
            -- date + start_time を timestamp に変換し、買い付け時刻を算出
            SELECT
                r.race_id,
                r.date,
                (
                    (r.date::timestamp + r.start_time)
                    - make_interval(mins => :buy_minutes)
                ) AS buy_time
            FROM fact_race r
            WHERE r.date BETWEEN :min_date AND :max_date
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
                      AND o.asof_time <= (
                            (r.date::timestamp + r.start_time)
                            - make_interval(mins => :buy_minutes)
                      )
              ))
              AND (:exclude_len = 0 OR NOT (r.race_id = ANY(:exclude_race_ids)))
        ),
        latest_features AS (
            -- (race_id, horse_id)ごとに購入時点以前の最新特徴量
            SELECT DISTINCT ON (f.race_id, f.horse_id)
                f.race_id,
                f.horse_id,
                f.payload,
                f.asof_time,
                bt.date
            FROM features f
            JOIN buy_times bt ON f.race_id = bt.race_id
             WHERE f.feature_version = :feature_version
               AND f.asof_time <= bt.buy_time
             ORDER BY f.race_id, f.horse_id, f.asof_time DESC
        )
        SELECT
            lf.race_id,
            lf.horse_id,
            lf.payload,
            lf.date,
            CASE WHEN res.finish_pos = 1 THEN 1 ELSE 0 END as is_winner
        FROM latest_features lf
        -- ★重要: 結果未投入レースが混ざると is_winner=0 扱いで静かに汚れるため、
        --         結果がある行だけに限定
        JOIN fact_result res ON lf.race_id = res.race_id
            AND lf.horse_id = res.horse_id
            AND res.finish_pos IS NOT NULL
        ORDER BY lf.date, lf.race_id, lf.horse_id
    """)

    results = session.execute(
        query,
        {
            "min_date": min_date,
            "max_date": max_date,
            "buy_minutes": buy_t_minus_minutes,
            "feature_version": FeatureBuilder.VERSION,
            "track_codes_len": len(track_codes),
            "track_codes": track_codes,
            "require_results": bool(u.require_results),
            "require_ts_win": bool(u.require_ts_win),
            "exclude_len": len(exclude_race_ids),
            "exclude_race_ids": exclude_race_ids,
        },
    ).fetchall()

    if not results:
        return None, None, None, None

    # DataFrameに変換
    rows = []
    for r in results:
        row = dict(r._mapping)
        payload = row.pop("payload") or {}
        row.update(payload)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Normalize `is_turf` using fact_race.surface (robust to DB type: int/str).
    # This column is used for surface-dispatching models.
    try:
        seg = _surface_segments_from_race_ids(session, df["race_id"], df.get("is_turf"))
        df["is_turf"] = (seg == "turf").astype(int)
    except Exception as e:
        logger.warning(f"Failed to normalize is_turf from fact_race.surface: {e}")
    # 0B42 (quinella) odds movement features at buy_time (leak-free, snapshot-based).
    q_df = fetch_quinella_odds_movement_features(
        session,
        df["race_id"].unique().tolist(),
        buy_t_minus_minutes=buy_t_minus_minutes,
        lookback_minutes=60,
    )
    if not q_df.empty:
        df = df.merge(q_df, on=["race_id", "horse_no"], how="left")
    else:
        # Ensure columns exist so feature selection stays stable (filled to 0.0 later).
        for c in QUINELLA_ODDS_MOVEMENT_COLS:
            if c not in df.columns:
                df[c] = None
    # 特徴量カラム
    feature_cols = [
        "odds",
        "log_odds",
        "p_mkt",
        "odds_rank",
        "is_favorite",
        # 時系列オッズ特徴量（直近）
        "snap_age_min",
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
        # 0B42: 馬連オッズ（スナップショット + 60分変化）
        *QUINELLA_ODDS_MOVEMENT_COLS,
        "n_races",
        "win_rate",
        "place_rate",
        "avg_finish_pos",
        "avg_last_3f",
        "days_since_last",
        "field_size",
        "distance",
        "is_turf",
        "frame_no",
        "horse_no",
        "horse_no_pct",
        "weight_carried",
    ]

    # Option C1: 履歴特徴量（存在するものだけ使用される）
    feature_cols += [
        "horse_starts_365",
        "horse_wins_365",
        "horse_places_365",
        "horse_avg_finish_365",
        "horse_last_finish",
        "horse_last2_finish",
        "horse_last3_finish",
        "horse_days_since_last",
        "horse_win_rate_last5",
        "horse_starts_dist_bin",
        "horse_win_rate_dist_bin",
        "jockey_starts_365",
        "jockey_win_rate_365",
        "jockey_place_rate_365",
        "trainer_starts_365",
        "trainer_win_rate_365",
        "trainer_place_rate_365",
        "horse_jockey_starts_365",
        "horse_jockey_win_rate_365",
    ]

    pace_feature_cols = [
        "horse_past_first3f_p20",
        "horse_past_last3f_p50",
        "horse_past_pace_diff_p50",
        "horse_past_lap_slope_p50",
        "horse_past_n_races_pace",
        "race_expected_first3f",
        "race_expected_last3f",
        "race_expected_pace_diff",
        "race_expected_pace_pressure",
    ]
    feature_cols += pace_feature_cols
    pace_cfg = getattr(cfg, "features", None)
    pace_enabled = False
    if pace_cfg is not None:
        pace_hist = getattr(pace_cfg, "pace_history", None)
        if pace_hist is not None:
            pace_enabled = bool(getattr(pace_hist, "enabled", False))
    if not pace_enabled:
        feature_cols = [c for c in feature_cols if c not in pace_feature_cols]

    # 存在するカラムのみ
    available_cols = [c for c in feature_cols if c in df.columns]

    # 数値化（1行/少数行でも dtype=object になってLightGBMで落ちるのを防ぐ）
    X = df[available_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["is_winner"]

    market_mode = str(
        getattr(getattr(cfg, "model", None), "market_prob_mode", "raw") or "raw"
    ).lower()
    if market_mode not in ("raw", "race_norm"):
        market_mode = "raw"
    if market_mode == "race_norm":
        p_mkt_col = "p_mkt_race" if "p_mkt_race" in df.columns else "p_mkt"
    else:
        p_mkt_col = "p_mkt_raw" if "p_mkt_raw" in df.columns else "p_mkt"

    p_mkt_raw = pd.to_numeric(df.get(p_mkt_col), errors="coerce")
    p_mkt_mean = float(p_mkt_raw.mean()) if p_mkt_raw is not None else 0.0
    if np.isnan(p_mkt_mean):
        p_mkt_mean = 0.0
    p_mkt = p_mkt_raw.fillna(p_mkt_mean)

    logger.info(f"Prepared {len(df)} training samples (leak-free)")
    return X, y, p_mkt, df["race_id"]


def train_model(
    session: Session,
    train_start: str,
    train_end: str,
    valid_start: Optional[str] = None,
    valid_end: Optional[str] = None,
    model_path: Optional[Path] = None,
    calibrator_path: Optional[Path] = None,
    buy_t_minus_minutes: Optional[int] = None,
) -> tuple[WinProbabilityModel | DistanceBucketWinProbabilityModel, dict]:
    """
    モデルを学習（検証データ分離 + 校正統合版）

    Args:
        session: DBセッション
        train_start: 訓練開始日（YYYY-MM-DD）
        train_end: 訓練終了日（YYYY-MM-DD）
        valid_start: 検証開始日（Noneなら訓練データの最後20%を使用）
        valid_end: 検証終了日
        model_path: モデル保存先パス
        calibrator_path: 校正器保存先パス
        buy_t_minus_minutes: 購入時点（Noneならconfigから取得）

    Returns:
        (model, metrics)
        metricsには校正後の評価も含まれる
    """
    from .calibrate import ProbabilityCalibrator

    config = get_config()
    if buy_t_minus_minutes is None:
        buy_t_minus_minutes = config.backtest.buy_t_minus_minutes

    logger.info(
        f"Training model: train={train_start}~{train_end}, "
        f"valid={valid_start}~{valid_end}, buy_t_minus={buy_t_minus_minutes}"
    )

    # 訓練データ準備
    X_train, y_train, p_mkt_train, race_ids_train = prepare_training_data(
        session, train_start, train_end, buy_t_minus_minutes
    )

    if X_train is None or len(X_train) == 0:
        raise ValueError("No training data found")

    # 検証データ準備（明示的に指定された場合）
    X_valid, y_valid, p_mkt_valid, race_ids_valid = None, None, None, None
    if valid_start and valid_end:
        X_valid, y_valid, p_mkt_valid, race_ids_valid = prepare_training_data(
            session, valid_start, valid_end, buy_t_minus_minutes
        )
        if X_valid is None or len(X_valid) == 0:
            logger.warning("No validation data found, falling back to train split")
            X_valid, y_valid, p_mkt_valid = None, None, None

    # Surface split (turf vs dirt): train 2 models and dispatch by race surface
    # at inference/backtest.
    # Enabled via env `KEIBA_SURFACE_SPLIT_MODEL=1` or auto-enabled for this run_id.
    surface_split_enabled = False
    env_flag = os.environ.get("KEIBA_SURFACE_SPLIT_MODEL")
    if env_flag is not None:
        surface_split_enabled = str(env_flag).strip().lower() in ("1", "true", "yes", "y", "on")
    else:
        run_name = None
        if model_path is not None:
            try:
                run_name = model_path.parent.parent.name
            except Exception:
                run_name = None
        surface_split_enabled = run_name == "exp_20260206_080929"

    model: Optional[WinProbabilityModel] = None
    metrics: dict = {}
    if surface_split_enabled and race_ids_train is not None:
        seg_train = _surface_segments_from_race_ids(session, race_ids_train, X_train.get("is_turf"))
        mask_turf_tr = seg_train.astype(str) == "turf"
        mask_dirt_tr = ~mask_turf_tr  # treat non-turf as dirt (incl. jump/unknown)
        n_turf_tr = int(mask_turf_tr.sum())
        n_dirt_tr = int(mask_dirt_tr.sum())

        # Avoid degenerate splits; fall back to the single-model path.
        if n_turf_tr >= 500 and n_dirt_tr >= 500:
            seg_valid = None
            mask_turf_va = None
            if (
                X_valid is not None
                and y_valid is not None
                and p_mkt_valid is not None
                and race_ids_valid is not None
            ):
                seg_valid = _surface_segments_from_race_ids(
                    session, race_ids_valid, X_valid.get("is_turf")
                )
                mask_turf_va = seg_valid.astype(str) == "turf"

            model_turf = WinProbabilityModel()
            turf_fit = model_turf.fit(
                X_train[mask_turf_tr],
                y_train[mask_turf_tr],
                p_mkt_train[mask_turf_tr],
                X_valid[mask_turf_va]
                if (
                    X_valid is not None and mask_turf_va is not None and int(mask_turf_va.sum()) > 0
                )
                else None,
                y_valid[mask_turf_va]
                if (
                    y_valid is not None and mask_turf_va is not None and int(mask_turf_va.sum()) > 0
                )
                else None,
                p_mkt_valid[mask_turf_va]
                if (
                    p_mkt_valid is not None
                    and mask_turf_va is not None
                    and int(mask_turf_va.sum()) > 0
                )
                else None,
            )

            model_dirt = WinProbabilityModel()
            mask_dirt_va = None if mask_turf_va is None else ~mask_turf_va
            dirt_fit = model_dirt.fit(
                X_train[mask_dirt_tr],
                y_train[mask_dirt_tr],
                p_mkt_train[mask_dirt_tr],
                X_valid[mask_dirt_va]
                if (
                    X_valid is not None and mask_dirt_va is not None and int(mask_dirt_va.sum()) > 0
                )
                else None,
                y_valid[mask_dirt_va]
                if (
                    y_valid is not None and mask_dirt_va is not None and int(mask_dirt_va.sum()) > 0
                )
                else None,
                p_mkt_valid[mask_dirt_va]
                if (
                    p_mkt_valid is not None
                    and mask_dirt_va is not None
                    and int(mask_dirt_va.sum()) > 0
                )
                else None,
            )

            if model_turf.feature_names != model_dirt.feature_names:
                raise ValueError("Surface-split models produced different feature_names")

            model = WinProbabilityModel()
            model.feature_names = list(model_turf.feature_names)
            turf_booster = model_turf.lgb_model
            dirt_booster = model_dirt.lgb_model
            if turf_booster is None or dirt_booster is None:
                raise ValueError("Surface-split models failed to train")
            if not isinstance(turf_booster, lgb.Booster) or not isinstance(
                dirt_booster, lgb.Booster
            ):
                raise ValueError("Surface-split models did not produce raw Boosters")
            model.lgb_model = SurfaceDispatchBooster(turf_booster, dirt_booster)
            metrics = {
                "surface_split_enabled": True,
                "n_train": int(len(y_train)),
                "n_train_turf": n_turf_tr,
                "n_train_dirt": n_dirt_tr,
                "turf_fit": turf_fit,
                "dirt_fit": dirt_fit,
            }
        else:
            surface_split_enabled = False
            metrics["surface_split_fallback_reason"] = (
                f"insufficient_rows(n_turf={n_turf_tr}, n_dirt={n_dirt_tr})"
            )

    if model is None:
        # Default single-model path.
        model = WinProbabilityModel()
        metrics = model.fit(X_train, y_train, p_mkt_train, X_valid, y_valid, p_mkt_valid)
        metrics["surface_split_enabled"] = False

    # Ticket: segment別 blend weight（validのみで推定）
    blend_seg_cfg = None
    try:
        blend_seg_cfg = getattr(getattr(config.model, "blend", None), "segmented", None)
    except Exception:
        blend_seg_cfg = None

    segments_valid = None
    if (
        blend_seg_cfg is not None
        and bool(getattr(blend_seg_cfg, "enabled", False))
        and not model.use_market_offset
        and X_valid is not None
        and y_valid is not None
        and p_mkt_valid is not None
    ):
        if str(getattr(blend_seg_cfg, "segment_by", "surface")).lower() != "surface":
            raise ValueError("Only segment_by='surface' is supported")
        loss_name = str(getattr(blend_seg_cfg, "loss", "logloss")).lower()
        if loss_name != "logloss":
            raise ValueError("Only loss='logloss' is supported")

        X_val2 = _align_features(X_valid, model.feature_names)
        p_model_val = (
            model.lgb_model.predict(X_val2)
            if model.lgb_model is not None
            else np.zeros(len(X_val2))
        )
        y_val_arr = y_valid.values.astype(int)
        p_mkt_val_arr = pd.to_numeric(p_mkt_valid, errors="coerce").fillna(0.0).astype(float).values

        fallback_is_turf = X_valid.get("is_turf") if isinstance(X_valid, pd.DataFrame) else None
        if race_ids_valid is not None:
            segments_valid = _surface_segments_from_race_ids(
                session, race_ids_valid, fallback_is_turf
            )
        else:
            segments_valid = pd.Series(["unknown"] * len(X_val2))

        w_min = float(getattr(blend_seg_cfg, "w_min", 0.0))
        w_max = float(getattr(blend_seg_cfg, "w_max", 1.0))
        grid_step = float(getattr(blend_seg_cfg, "grid_step", 0.02))
        w_global, loss_global = _grid_search_blend_weight(
            y_val_arr,
            p_mkt_val_arr,
            np.asarray(p_model_val, dtype=float),
            w_min=w_min,
            w_max=w_max,
            grid_step=grid_step,
            default_w=float(model.blend_weight),
        )

        min_count = int(getattr(blend_seg_cfg, "min_count", 200))
        n0 = float(getattr(blend_seg_cfg, "n0", 500))
        segments_meta = {}
        for seg in sorted(set(segments_valid.astype(str).tolist())):
            mask = segments_valid.astype(str) == seg
            n_seg = int(mask.sum())
            if n_seg >= min_count:
                w_raw, loss_seg = _grid_search_blend_weight(
                    y_val_arr[mask.values],
                    p_mkt_val_arr[mask.values],
                    np.asarray(p_model_val, dtype=float)[mask.values],
                    w_min=w_min,
                    w_max=w_max,
                    grid_step=grid_step,
                    default_w=w_global,
                )
                w_shrunk = (n_seg / (n_seg + n0)) * w_raw + (n0 / (n_seg + n0)) * w_global
                w_final = min(max(float(w_shrunk), w_min), w_max)
                segments_meta[seg] = {
                    "n": n_seg,
                    "w_raw": float(w_raw),
                    "w": float(w_final),
                    "loss": float(loss_seg) if loss_seg == loss_seg else None,
                }
            else:
                segments_meta[seg] = {
                    "n": n_seg,
                    "w_raw": None,
                    "w": float(w_global),
                    "fallback": True,
                }

        model.blend_weight = float(w_global)
        model.blend_segmented = {
            "enabled": True,
            "segment_by": "surface",
            "w_global": float(w_global),
            "loss_global": float(loss_global) if loss_global == loss_global else None,
            "n0": float(n0),
            "min_count": int(min_count),
            "w_min": float(w_min),
            "w_max": float(w_max),
            "grid_step": float(grid_step),
            "loss": str(getattr(blend_seg_cfg, "loss", "logloss")),
            "segments": segments_meta,
        }

        if model_path is not None:
            blend_json_path = model_path.parent / "blend_weights.json"
            blend_json_path.write_text(
                json.dumps(model.blend_segmented, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # race softmax params (valid only)
    rs_cfg = getattr(config.model, "race_softmax", None)
    if (
        rs_cfg is not None
        and bool(getattr(rs_cfg, "enabled", False))
        and X_valid is not None
        and y_valid is not None
        and p_mkt_valid is not None
    ):
        if race_ids_valid is None:
            logger.warning("race_softmax enabled but race_ids_valid is missing; skip fit")
        else:
            fit_cfg = getattr(rs_cfg, "fit", None)
            if fit_cfg is None or bool(getattr(fit_cfg, "enabled", True)):
                X_val2 = _align_features(X_valid, model.feature_names)
                if model.use_market_offset:
                    p_model_val = model.predict(X_val2, p_mkt_valid, calibrate=False)
                else:
                    p_model_val = (
                        model.lgb_model.predict(X_val2)
                        if model.lgb_model is not None
                        else np.zeros(len(X_val2))
                    )
                df_valid = pd.DataFrame(
                    {
                        "race_id": race_ids_valid.values,
                        "y": y_valid.values,
                        "p_model": np.asarray(p_model_val, dtype=float),
                        "p_mkt": pd.to_numeric(p_mkt_valid, errors="coerce")
                        .fillna(0.0)
                        .astype(float)
                        .values,
                    }
                )
                rs_fit = fit_race_softmax(
                    df_valid,
                    w_grid_step=float(getattr(fit_cfg, "w_grid_step", 0.02)),
                    t_grid=list(getattr(fit_cfg, "t_grid", [1.0])),
                    score_space=str(getattr(rs_cfg, "score_space", "logit")),
                    clip_eps=float(getattr(rs_cfg, "clip_eps", 1e-6)),
                )
                model.race_softmax_params = {
                    "enabled": True,
                    "w": float(rs_fit.w),
                    "T": float(rs_fit.t),
                    "loss": float(rs_fit.loss) if rs_fit.loss == rs_fit.loss else None,
                    "n_races": int(rs_fit.n_races),
                    "n_rows": int(rs_fit.n_rows),
                    "score_space": str(getattr(rs_cfg, "score_space", "logit")),
                    "clip_eps": float(getattr(rs_cfg, "clip_eps", 1e-6)),
                }
                if model_path is not None:
                    rs_path = model_path.parent / "race_softmax_params.json"
                    rs_path.write_text(
                        json.dumps(model.race_softmax_params, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

    # ★校正（検証データ上で校正器をfit）
    # 明示的な検証データがある場合はそれを使用、なければ内部splitの結果を使用
    from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

    if X_valid is not None and len(X_valid) > 0:
        # 明示的な検証データを使用
        if y_valid is None:
            model.calibrator = None
            if model_path:
                model.save(model_path)
            return model, metrics
        X_valid2 = _align_features(X_valid, model.feature_names)
        pred_out = model.predict(X_valid2, p_mkt_valid, calibrate=False, segments=segments_valid)
        if isinstance(pred_out, tuple):
            pred_out = pred_out[0]
        p_blend_valid = np.asarray(pred_out, dtype=float)
        y_valid_arr = y_valid.values
    elif hasattr(model, "_last_val_predictions") and model._last_val_predictions:
        # ★内部splitした検証データを使用（これがP6の修正ポイント）
        p_blend_valid = np.asarray(model._last_val_predictions["p_blend"], dtype=float)
        y_valid_arr = model._last_val_predictions["y_true"]
        logger.info("Using internal train/valid split for calibration")
    else:
        # 検証データがない場合は校正スキップ
        model.calibrator = None
        if model_path:
            model.save(model_path)
        return model, metrics

    # 校正器をfit
    calibrator = ProbabilityCalibrator(method=config.model.calibration)
    p_blend_valid_arr = p_blend_valid
    calibrator.fit(p_blend_valid_arr, y_valid_arr)

    # 校正後の評価
    p_calibrated = calibrator.transform(p_blend_valid_arr)
    metrics["valid_brier_calibrated"] = brier_score_loss(y_valid_arr, p_calibrated)
    metrics["valid_logloss_calibrated"] = log_loss(y_valid_arr, p_calibrated)
    metrics["calibration_method"] = config.model.calibration

    # 校正器を保存
    if calibrator_path:
        calibrator.save(calibrator_path)
        logger.info(f"Calibrator saved to {calibrator_path}")

    # モデルに校正器を付与（推論時に使えるように）
    model.calibrator = calibrator

    # Distance Bucket Models: train per-distance models and route inference by distance bucket.
    dist_cfg = getattr(getattr(config, "model", None), "distance_bucket_models", None)
    enabled = bool(dist_cfg.get("enabled", False)) if isinstance(dist_cfg, dict) else False
    final_model: WinProbabilityModel | DistanceBucketWinProbabilityModel = model
    if enabled:
        min_train = (
            int(dist_cfg.get("min_train_samples", 5000)) if isinstance(dist_cfg, dict) else 5000
        )
        dist = (
            pd.to_numeric(X_train.get("distance"), errors="coerce") if X_train is not None else None
        )
        buckets_train = (
            dist.where(dist > 0).apply(distance_bucket)
            if dist is not None
            else pd.Series([], dtype=object)
        )

        bucket_models: dict[int, WinProbabilityModel] = {}
        n_train_by_bucket: dict[str, int] = {}
        for b in sorted(set(buckets_train.dropna().tolist())):
            mask = buckets_train == b
            n = int(mask.sum())
            n_train_by_bucket[str(b)] = n
            if n < min_train:
                continue
            m = WinProbabilityModel()
            m.fit(X_train.loc[mask], y_train.loc[mask], p_mkt_train.loc[mask])
            bucket_models[int(b)] = m

        bucket_meta = {
            "enabled": True,
            "min_train_samples": min_train,
            "n_train_by_bucket": n_train_by_bucket,
            "trained_buckets": sorted(bucket_models.keys()),
        }
        metrics["distance_bucket_models"] = bucket_meta
        final_model = DistanceBucketWinProbabilityModel(
            fallback_model=model,
            bucket_models=bucket_models,
            bucket_meta=bucket_meta,
        )

    if model_path:
        final_model.save(model_path)

    return final_model, metrics
