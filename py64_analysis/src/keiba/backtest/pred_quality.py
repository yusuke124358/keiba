"""
Ticket2: backtest期間の「確率の品質」を可視化するユーティリティ。

ROIだけ見ると迷走しやすいので、以下を同じレポートに載せられる形にする:
  - logloss / brier（market / model / blend / calibrated）
  - reliability diagram（校正曲線）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_config
from ..modeling.train import prepare_training_data, WinProbabilityModel
from ..modeling.race_softmax import apply_race_softmax


@dataclass(frozen=True)
class PredictionQuality:
    n_samples: int
    logloss_market: float
    logloss_model: float
    logloss_blend: float
    logloss_calibrated: float
    brier_market: float
    brier_model: float
    brier_blend: float
    brier_calibrated: float
    calibration_method: Optional[str]


def _ensure_numeric_frame(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """LightGBM用に、列を揃えて数値化する（欠損は0埋め）。"""
    X2 = X.reindex(columns=feature_names, fill_value=0.0)
    X2 = X2.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return X2


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
        except Exception:
            pass

    if fallback_is_turf is not None and "unknown" in segments.values:
        vals = pd.to_numeric(fallback_is_turf, errors="coerce")
        mask = segments == "unknown"
        segments.loc[mask & (vals == 1)] = "turf"
        segments.loc[mask & (vals == 0)] = "dirt"
    return segments.astype(str)


def compute_prediction_quality(
    session: Session,
    start_date: str,
    end_date: str,
    model: WinProbabilityModel,
    buy_t_minus_minutes: Optional[int] = None,
    n_bins: int = 10,
    reliability_columns: tuple[str, ...] = ("p_mkt", "p_blend", "p_cal"),
) -> tuple[PredictionQuality, pd.DataFrame, pd.DataFrame]:
    """
    backtest期間（start_date..end_date）における確率品質を算出する。

    Returns:
      - PredictionQuality（集計）
      - df_pred（1行=1頭: y_true, p_mkt, p_model, p_blend, p_cal）
      - reliability_df（ビン集計）
    """
    from sklearn.metrics import log_loss, brier_score_loss

    cfg = get_config()
    if buy_t_minus_minutes is None:
        buy_t_minus_minutes = cfg.backtest.buy_t_minus_minutes

    X, y, p_mkt, race_ids = prepare_training_data(
        session, start_date, end_date, buy_t_minus_minutes
    )
    if X is None or len(X) == 0:
        raise ValueError("No evaluation data found")

    # 市場確率（数値化）
    p_mkt = pd.to_numeric(p_mkt, errors="coerce").fillna(0.0).astype(float)

    # モデル列を揃える
    X2 = _ensure_numeric_frame(X, model.feature_names)

    # 予測（校正前）
    if getattr(model, "use_market_offset", False):
        p_model = model.predict(X2, p_mkt, calibrate=False)
    else:
        p_model = model.lgb_model.predict(X2) if model.lgb_model is not None else np.zeros(len(X2))
    p_model = np.asarray(p_model, dtype=float)

    segments = None
    if getattr(model, "blend_segmented", None) and model.blend_segmented.get("enabled"):
        segments = _surface_segments_from_race_ids(session, race_ids, X.get("is_turf"))

    rs_cfg = getattr(cfg.model, "race_softmax", None)
    race_softmax_enabled = bool(rs_cfg and getattr(rs_cfg, "enabled", False))

    if race_softmax_enabled and race_ids is not None:
        w = None
        t = None
        if getattr(model, "race_softmax_params", None):
            w = model.race_softmax_params.get("w")
            t = model.race_softmax_params.get("T")
        if w is None:
            w = float(getattr(getattr(rs_cfg, "apply", None), "w_default", 0.2))
        if t is None:
            t = float(getattr(getattr(rs_cfg, "apply", None), "t_default", 1.0))

        df_rs = pd.DataFrame(
            {
                "race_id": race_ids.values,
                "p_model": p_model,
                "p_mkt": p_mkt.values,
            }
        )
        p_blend = apply_race_softmax(
            df_rs,
            w=float(w),
            t=float(t),
            score_space=str(getattr(rs_cfg, "score_space", "logit")),
            clip_eps=float(getattr(rs_cfg, "clip_eps", 1e-6)),
        ).values
        p_cal = p_blend
        calibration_method = None
    else:
        p_blend = model.predict(X2, p_mkt, calibrate=False, segments=segments)

        # 校正（あれば）
        if getattr(model, "calibrator", None) is not None:
            p_cal = model.predict(X2, p_mkt, calibrate=True, segments=segments)
            calibration_method = cfg.model.calibration
        else:
            p_cal = p_blend
            calibration_method = None

    y_true = y.values.astype(int)

    # ★保険: labels固定（少数データでも落ちない）
    ll_mkt = float(log_loss(y_true, p_mkt.values, labels=[0, 1]))
    ll_model = float(log_loss(y_true, p_model, labels=[0, 1]))
    ll_blend = float(log_loss(y_true, p_blend, labels=[0, 1]))
    ll_cal = float(log_loss(y_true, p_cal, labels=[0, 1]))

    b_mkt = float(brier_score_loss(y_true, p_mkt.values))
    b_model = float(brier_score_loss(y_true, p_model))
    b_blend = float(brier_score_loss(y_true, p_blend))
    b_cal = float(brier_score_loss(y_true, p_cal))

    pq = PredictionQuality(
        n_samples=int(len(y_true)),
        logloss_market=ll_mkt,
        logloss_model=ll_model,
        logloss_blend=ll_blend,
        logloss_calibrated=ll_cal,
        brier_market=b_mkt,
        brier_model=b_model,
        brier_blend=b_blend,
        brier_calibrated=b_cal,
        calibration_method=calibration_method,
    )

    df_pred = pd.DataFrame(
        {
            "y_true": y_true,
            "p_mkt": p_mkt.values,
            "p_model": p_model,
            "p_blend": p_blend,
            "p_cal": np.asarray(p_cal, dtype=float),
        }
    )

    reliability_df = build_reliability_table(
        df_pred,
        n_bins=n_bins,
        columns=reliability_columns,
    )

    return pq, df_pred, reliability_df


def build_reliability_table(
    df_pred: pd.DataFrame,
    n_bins: int = 10,
    columns: tuple[str, ...] = ("p_cal",),
) -> pd.DataFrame:
    """
    reliability（分位ビニングで 予測平均 vs 実測勝率）を作る。
    """
    if df_pred.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    y = df_pred["y_true"].astype(int)

    for col in columns:
        p = pd.to_numeric(df_pred[col], errors="coerce").clip(0.0, 1.0)
        # 分位でビニング（同値が多い場合のエラーを避けるため duplicates='drop'）
        try:
            bins = pd.qcut(p, q=n_bins, duplicates="drop")
        except ValueError:
            # データが少なすぎる等でqcutできない場合は単一bin扱い
            bins = pd.Series(["all"] * len(p))

        g = pd.DataFrame({"bin": bins.astype(str), "p": p, "y": y})
        agg = g.groupby("bin", dropna=False).agg(
            n=("y", "size"),
            p_mean=("p", "mean"),
            win_rate=("y", "mean"),
        )
        agg = agg.reset_index()
        agg["series"] = col
        out_rows.append(agg)

    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def plot_reliability(
    reliability_df: pd.DataFrame,
    output_path,
    title: str = "Reliability Diagram",
) -> None:
    """reliability plot を保存する（matplotlib）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if reliability_df is None or reliability_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="ideal")

    for series, g in reliability_df.groupby("series"):
        g2 = g.sort_values("p_mean")
        ax.plot(g2["p_mean"], g2["win_rate"], marker="o", linewidth=1.5, label=series)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical win rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
