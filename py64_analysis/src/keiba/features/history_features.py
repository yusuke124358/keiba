"""
履歴特徴量（Option C1）

ポイント:
- リーク防止が最優先: 必ず `race_dt < asof_time` で未来を除外してから集計する
- まずは DB から取得した past results を DataFrame にして、ここで特徴量に落とす
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def distance_bin(distance: Optional[float]) -> Optional[int]:
    """
    距離を粗いbinに丸める（最小セット）
    例: 1200/1400/1600/1800/2000/2400/2400+
    """
    if distance is None or (isinstance(distance, float) and np.isnan(distance)):
        return None
    d = int(distance)
    if d <= 1300:
        return 1200
    if d <= 1500:
        return 1400
    if d <= 1700:
        return 1600
    if d <= 1900:
        return 1800
    if d <= 2100:
        return 2000
    if d <= 2500:
        return 2400
    return 2800


def compute_horse_history_features(
    past_results: pd.DataFrame,
    *,
    asof_time: datetime,
    target_distance: Optional[float],
) -> dict:
    """
    past_results 必須列:
      - race_dt (datetime)
      - finish_pos (numeric)
      - distance (numeric, optional)
      - time_sec (numeric, optional)
      - last_3f (numeric, optional)
      - pos_1c/pos_2c/pos_3c/pos_4c (numeric, optional)
      - field_size (numeric, optional)
      - pace_first3f/pace_last3f (numeric, optional)
    """
    def _safe_z(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        n = int(s.notna().sum())
        if n < 2:
            # 分散が作れない場合はすべてNaN扱い（＝特徴量なし）
            return pd.Series([np.nan] * len(s), index=s.index)
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series([np.nan] * len(s), index=s.index)
        return (s - mu) / sd

    def _trend_last3(z: pd.Series) -> Optional[float]:
        # 直近3点のトレンド（+なら改善方向と解釈できるように、zの定義に依存）
        z3 = z.head(3).dropna()
        if len(z3) < 2:
            return None
        x = np.arange(len(z3), dtype=float)  # 0=最新
        y = z3.values.astype(float)
        a, _b = np.polyfit(x, y, 1)
        return float(a)

    if past_results is None or len(past_results) == 0:
        return {
            "horse_starts_365": 0,
            "horse_wins_365": 0,
            "horse_places_365": 0,
            "horse_avg_finish_365": None,
            "horse_last_finish": None,
            "horse_last2_finish": None,
            "horse_last3_finish": None,
            "horse_days_since_last": None,
            "horse_win_rate_last5": None,
            "horse_starts_dist_bin": 0,
            "horse_win_rate_dist_bin": None,
            # C3: perf
            "horse_speed_z_last": None,
            "horse_speed_z_best3": None,
            "horse_speed_z_mean3": None,
            "horse_speed_z_trend3": None,
            "horse_last3f_z_last": None,
            "horse_last3f_z_best3": None,
            "horse_last3f_z_mean3": None,
            "horse_last3f_z_trend3": None,
            # C4: pace/position
            "horse_pos_4c_pct_last": None,
            "horse_pos_4c_pct_mean3": None,
            "horse_pos_gain_last": None,
            "horse_pos_gain_mean3": None,
            "horse_pace_diff_last": None,
            "horse_pace_diff_mean3": None,
            # 既存互換
            "n_races": 0,
            "win_rate": None,
            "place_rate": None,
            "avg_finish_pos": None,
            "avg_last_3f": None,
            "days_since_last": None,
        }

    df = past_results.copy()
    df = df[df["race_dt"] < asof_time]
    if len(df) == 0:
        return {
            "horse_starts_365": 0,
            "horse_wins_365": 0,
            "horse_places_365": 0,
            "horse_avg_finish_365": None,
            "horse_last_finish": None,
            "horse_last2_finish": None,
            "horse_last3_finish": None,
            "horse_days_since_last": None,
            "horse_win_rate_last5": None,
            "horse_starts_dist_bin": 0,
            "horse_win_rate_dist_bin": None,
            # C3: perf
            "horse_speed_z_last": None,
            "horse_speed_z_best3": None,
            "horse_speed_z_mean3": None,
            "horse_speed_z_trend3": None,
            "horse_last3f_z_last": None,
            "horse_last3f_z_best3": None,
            "horse_last3f_z_mean3": None,
            "horse_last3f_z_trend3": None,
            # C4: pace/position
            "horse_pos_4c_pct_last": None,
            "horse_pos_4c_pct_mean3": None,
            "horse_pos_gain_last": None,
            "horse_pos_gain_mean3": None,
            "horse_pace_diff_last": None,
            "horse_pace_diff_mean3": None,
            # 既存互換
            "n_races": 0,
            "win_rate": None,
            "place_rate": None,
            "avg_finish_pos": None,
            "avg_last_3f": None,
            "days_since_last": None,
        }

    df = df.sort_values("race_dt", ascending=False)
    # 数値化
    df["finish_pos"] = pd.to_numeric(df.get("finish_pos"), errors="coerce")
    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df.get("distance"), errors="coerce")
    if "time_sec" in df.columns:
        df["time_sec"] = pd.to_numeric(df.get("time_sec"), errors="coerce")
    if "last_3f" in df.columns:
        df["last_3f"] = pd.to_numeric(df.get("last_3f"), errors="coerce")
    # C4: position (SE)
    for c in ("pos_1c", "pos_2c", "pos_3c", "pos_4c"):
        if c in df.columns:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
    if "field_size" in df.columns:
        df["field_size"] = pd.to_numeric(df.get("field_size"), errors="coerce")
    # C4: pace (RA)
    for c in ("pace_first3f", "pace_last3f"):
        if c in df.columns:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")

    cutoff = asof_time - timedelta(days=365)
    df365 = df[df["race_dt"] >= cutoff]

    starts_365 = int(len(df365))
    wins_365 = int((df365["finish_pos"] == 1).sum()) if starts_365 > 0 else 0
    places_365 = int((df365["finish_pos"] <= 3).sum()) if starts_365 > 0 else 0
    avg_finish_365 = float(df365["finish_pos"].mean()) if starts_365 > 0 else None

    last_dt = df["race_dt"].iloc[0]
    days_since_last = int((asof_time - last_dt).days) if pd.notna(last_dt) else None

    last_finish = int(df["finish_pos"].iloc[0]) if pd.notna(df["finish_pos"].iloc[0]) else None
    last2_finish = int(df["finish_pos"].iloc[1]) if len(df) >= 2 and pd.notna(df["finish_pos"].iloc[1]) else None
    last3_finish = int(df["finish_pos"].iloc[2]) if len(df) >= 3 and pd.notna(df["finish_pos"].iloc[2]) else None

    last5 = df.head(5)
    win_rate_last5 = float((last5["finish_pos"] == 1).mean()) if len(last5) > 0 else None

    # 距離bin適性（過去365日の中で、target_distanceと同binの成績）
    tgt_bin = distance_bin(target_distance)
    starts_dist_bin = 0
    win_rate_dist_bin = None
    if tgt_bin is not None and "distance" in df365.columns:
        dist_bins = df365["distance"].apply(distance_bin)
        sel = df365[dist_bins == tgt_bin]
        starts_dist_bin = int(len(sel))
        if starts_dist_bin > 0:
            win_rate_dist_bin = float((sel["finish_pos"] == 1).mean())

    # 既存互換（旧実装の名前）
    n_recent = min(5, len(df))
    recent = df.head(n_recent)
    win_rate_recent = float((recent["finish_pos"] == 1).mean()) if n_recent > 0 else None
    place_rate_recent = float((recent["finish_pos"] <= 3).mean()) if n_recent > 0 else None
    avg_finish_recent = float(recent["finish_pos"].mean()) if n_recent > 0 else None
    avg_last3f_recent = (
        float(recent["last_3f"].mean()) if ("last_3f" in recent.columns and n_recent > 0) else None
    )

    # ===== C3: speed/last3f（直近N走ベースのz化） =====
    # sec/100m（距離で正規化）を作って、遅い=悪い を 速い=良い に向けて符号反転したzを作る
    speed_z_last = None
    speed_z_best3 = None
    speed_z_mean3 = None
    speed_z_trend3 = None
    last3f_z_last = None
    last3f_z_best3 = None
    last3f_z_mean3 = None
    last3f_z_trend3 = None

    if ("time_sec" in df.columns) and ("distance" in df.columns):
        sec_per_100m = df["time_sec"] / df["distance"] * 100.0
        z = _safe_z(sec_per_100m)
        speed_z = (-z)  # 小さいほど速い -> zを反転して「大きいほど速い」
        if speed_z.notna().any():
            speed_z_last = float(speed_z.iloc[0]) if pd.notna(speed_z.iloc[0]) else None
            best3 = speed_z.head(3).dropna()
            if len(best3) > 0:
                speed_z_best3 = float(best3.max())
                speed_z_mean3 = float(best3.mean())
            speed_z_trend3 = _trend_last3(speed_z)

    if "last_3f" in df.columns:
        z3f = _safe_z(df["last_3f"])
        last3f_z = (-z3f)  # 小さいほど速い上がり -> 反転
        if last3f_z.notna().any():
            last3f_z_last = float(last3f_z.iloc[0]) if pd.notna(last3f_z.iloc[0]) else None
            best3f = last3f_z.head(3).dropna()
            if len(best3f) > 0:
                last3f_z_best3 = float(best3f.max())
                last3f_z_mean3 = float(best3f.mean())
            last3f_z_trend3 = _trend_last3(last3f_z)

    # ===== C4: position / pace =====
    # pos_gain: 1C→4Cで何頭抜いたか（+が前進）
    pos_gain_last = None
    pos_gain_mean3 = None
    pos_4c_pct_last = None
    pos_4c_pct_mean3 = None
    pace_diff_last = None
    pace_diff_mean3 = None

    if ("pos_1c" in df.columns) and ("pos_4c" in df.columns):
        df["pos_gain"] = df["pos_1c"] - df["pos_4c"]
        g = df["pos_gain"].head(3).dropna()
        if len(g) > 0:
            pos_gain_last = float(g.iloc[0]) if pd.notna(g.iloc[0]) else None
            pos_gain_mean3 = float(g.mean())

    if ("pos_4c" in df.columns) and ("field_size" in df.columns):
        df["pos_4c_pct"] = df["pos_4c"] / df["field_size"]
        p = df["pos_4c_pct"].head(3).dropna()
        if len(p) > 0:
            pos_4c_pct_last = float(p.iloc[0]) if pd.notna(p.iloc[0]) else None
            pos_4c_pct_mean3 = float(p.mean())

    if ("pace_first3f" in df.columns) and ("pace_last3f" in df.columns):
        df["pace_diff"] = df["pace_first3f"] - df["pace_last3f"]
        pdiff = df["pace_diff"].head(3).dropna()
        if len(pdiff) > 0:
            pace_diff_last = float(pdiff.iloc[0]) if pd.notna(pdiff.iloc[0]) else None
            pace_diff_mean3 = float(pdiff.mean())

    return {
        "horse_starts_365": starts_365,
        "horse_wins_365": wins_365,
        "horse_places_365": places_365,
        "horse_avg_finish_365": avg_finish_365,
        "horse_last_finish": last_finish,
        "horse_last2_finish": last2_finish,
        "horse_last3_finish": last3_finish,
        "horse_days_since_last": days_since_last,
        "horse_win_rate_last5": win_rate_last5,
        "horse_starts_dist_bin": starts_dist_bin,
        "horse_win_rate_dist_bin": win_rate_dist_bin,
        # C3: perf
        "horse_speed_z_last": speed_z_last,
        "horse_speed_z_best3": speed_z_best3,
        "horse_speed_z_mean3": speed_z_mean3,
        "horse_speed_z_trend3": speed_z_trend3,
        "horse_last3f_z_last": last3f_z_last,
        "horse_last3f_z_best3": last3f_z_best3,
        "horse_last3f_z_mean3": last3f_z_mean3,
        "horse_last3f_z_trend3": last3f_z_trend3,
        # C4: position / pace
        "horse_pos_4c_pct_last": pos_4c_pct_last,
        "horse_pos_4c_pct_mean3": pos_4c_pct_mean3,
        "horse_pos_gain_last": pos_gain_last,
        "horse_pos_gain_mean3": pos_gain_mean3,
        "horse_pace_diff_last": pace_diff_last,
        "horse_pace_diff_mean3": pace_diff_mean3,
        # 既存互換
        "n_races": int(len(df)),
        "win_rate": win_rate_recent,
        "place_rate": place_rate_recent,
        "avg_finish_pos": avg_finish_recent,
        "avg_last_3f": avg_last3f_recent,
        "days_since_last": days_since_last,
    }


