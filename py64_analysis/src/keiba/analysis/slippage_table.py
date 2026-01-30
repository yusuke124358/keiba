"""
TicketB: 馬別スリッページ推定（テーブル型）

目的:
  - train（またはvalidまで）で r = odds_final / odds_buy を推定し、
    testのEV/賭け金計算に使う odds_effective = odds_buy * r_hat に置換する。

設計:
  - odds帯 × TSボラ帯（× snap_age帯）で r の分位点（例: q=0.30）をテーブル化
  - binが疎な場合はフォールバック:
      3D -> 2D(odds×vol) -> 1D(odds) -> overall
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_config
from ..features.build_features import FeatureBuilder


def _extend_edges(edges: list[float]) -> list[float]:
    """[a,b,c] を [0,a,b,c,inf] の形に正規化する（単調増加であること）。"""
    if edges is None:
        return [0.0, float("inf")]
    vals = [float(x) for x in edges]
    vals = sorted(set(vals))
    if not vals:
        return [0.0, float("inf")]
    if vals[0] > 0.0:
        vals = [0.0] + vals
    if not np.isfinite(vals[-1]):
        return vals
    vals = vals + [float("inf")]
    return vals


def _bin_index(v: Optional[float], edges: list[float]) -> Optional[int]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    # edges: [e0,e1,...,inf] で left-closed/right-open を採用
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    # x==inf は最後に落とす
    return len(edges) - 2


def _quantile_cutpoints(s: pd.Series, qs: list[float]) -> list[float]:
    """分位点のcutpointを返す。空/全欠損なら []。"""
    if s is None:
        return []
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return []
    pts = []
    for q in qs:
        try:
            pts.append(float(x.quantile(float(q))))
        except Exception:
            continue
    # 近すぎるcutpointは潰す（同一値だとbinが壊れる）
    out: list[float] = []
    for p in pts:
        if not out or (p > out[-1] + 1e-12):
            out.append(p)
    return out


@dataclass(frozen=True)
class SlippageTable:
    """bin×分位の r_hat テーブル（r = odds_final / odds_buy）"""

    quantile: float
    min_count: int
    odds_edges: list[float]
    ts_vol_edges: list[float]  # cutpoints（[q50,q80] 等）。edgesは [-inf, *cut, inf] で構成
    use_snap_age: bool
    snap_age_edges: list[float]

    # maps
    overall_r_hat: float
    odds_r_hat: dict[int, float]
    odds_r_n: dict[int, int]
    odds_vol_r_hat: dict[tuple[int, int], float]
    odds_vol_r_n: dict[tuple[int, int], int]
    odds_vol_age_r_hat: dict[tuple[int, int, int], float]
    odds_vol_age_r_n: dict[tuple[int, int, int], int]

    def effective_odds(self, odds_buy: float, ts_vol: Optional[float] = None, snap_age_min: Optional[float] = None) -> tuple[float, dict]:
        """
        odds_effective を返す（meta付き）。
        フォールバック順: 3D -> 2D -> 1D -> overall
        """
        o_bin = _bin_index(odds_buy, self.odds_edges)
        v_bin = self._ts_vol_bin(ts_vol)
        a_bin = _bin_index(snap_age_min, self.snap_age_edges) if self.use_snap_age else None

        meta: dict[str, Any] = {
            "mode": "slippage_table",
            "quantile": float(self.quantile),
            "min_count": int(self.min_count),
            "odds_bin": int(o_bin) if o_bin is not None else None,
            "ts_vol_bin": int(v_bin) if v_bin is not None else None,
            "snap_age_bin": int(a_bin) if a_bin is not None else None,
        }

        r_hat = None
        used = None

        if self.use_snap_age and o_bin is not None and v_bin is not None and a_bin is not None:
            key3 = (o_bin, v_bin, a_bin)
            n3 = int(self.odds_vol_age_r_n.get(key3, 0))
            if n3 >= self.min_count and key3 in self.odds_vol_age_r_hat:
                r_hat = float(self.odds_vol_age_r_hat[key3])
                used = "odds_vol_age"
                meta["bin_n"] = n3

        if r_hat is None and o_bin is not None and v_bin is not None:
            key2 = (o_bin, v_bin)
            n2 = int(self.odds_vol_r_n.get(key2, 0))
            if n2 >= self.min_count and key2 in self.odds_vol_r_hat:
                r_hat = float(self.odds_vol_r_hat[key2])
                used = "odds_vol"
                meta["bin_n"] = n2

        if r_hat is None and o_bin is not None:
            n1 = int(self.odds_r_n.get(o_bin, 0))
            if n1 >= self.min_count and o_bin in self.odds_r_hat:
                r_hat = float(self.odds_r_hat[o_bin])
                used = "odds"
                meta["bin_n"] = n1

        if r_hat is None:
            r_hat = float(self.overall_r_hat)
            used = "overall"
            meta["bin_n"] = None

        meta["used"] = used
        meta["r_hat"] = float(r_hat)

        try:
            odds_eff = float(odds_buy) * float(r_hat)
        except Exception:
            odds_eff = float(odds_buy)
        return odds_eff, meta

    def _ts_vol_bin(self, ts_vol: Optional[float]) -> Optional[int]:
        if ts_vol is None:
            return None
        try:
            x = float(ts_vol)
        except Exception:
            return None
        if not np.isfinite(x):
            return None
        # edges: [-inf, c1, c2, inf] 相当。cutpointsが空なら 0 固定。
        if not self.ts_vol_edges:
            return 0
        # 例: [c1,c2] -> 3bin
        for i, c in enumerate(self.ts_vol_edges):
            if x < c:
                return i
        return len(self.ts_vol_edges)

    def to_dict(self) -> dict:
        def _json_float(v: float) -> float | None:
            try:
                x = float(v)
            except Exception:
                return None
            if not np.isfinite(x):
                return None
            return float(x)

        return {
            "quantile": float(self.quantile),
            "min_count": int(self.min_count),
            # JSON互換のため、inf は null に落とす（最後のbinは暗黙に +inf）
            "odds_edges": [_json_float(x) for x in self.odds_edges],
            "ts_vol_edges": [_json_float(x) for x in self.ts_vol_edges],
            "use_snap_age": bool(self.use_snap_age),
            "snap_age_edges": [_json_float(x) for x in self.snap_age_edges],
            "overall_r_hat": float(self.overall_r_hat),
            "odds_r_hat": {str(k): float(v) for k, v in self.odds_r_hat.items()},
            "odds_r_n": {str(k): int(v) for k, v in self.odds_r_n.items()},
            "odds_vol_r_hat": {f"{k0},{k1}": float(v) for (k0, k1), v in self.odds_vol_r_hat.items()},
            "odds_vol_r_n": {f"{k0},{k1}": int(v) for (k0, k1), v in self.odds_vol_r_n.items()},
            "odds_vol_age_r_hat": {f"{k0},{k1},{k2}": float(v) for (k0, k1, k2), v in self.odds_vol_age_r_hat.items()},
            "odds_vol_age_r_n": {f"{k0},{k1},{k2}": int(v) for (k0, k1, k2), v in self.odds_vol_age_r_n.items()},
        }


def fit_slippage_table(
    df: pd.DataFrame,
    *,
    quantile: float,
    min_count: int,
    odds_bins: list[float],
    ts_vol_quantiles: list[float],
    use_snap_age: bool,
    snap_age_bins: list[float],
) -> SlippageTable:
    """
    学習データ（1行=1馬）から slippage table を作る。

    df required columns:
      - odds_buy
      - ratio_final_to_buy
      - log_odds_std_60m (optional; NaN可)
      - snap_age_min (optional; NaN可)
    """
    if df is None or df.empty:
        return SlippageTable(
            quantile=float(quantile),
            min_count=int(min_count),
            odds_edges=_extend_edges(odds_bins),
            ts_vol_edges=[],
            use_snap_age=bool(use_snap_age),
            snap_age_edges=_extend_edges(snap_age_bins),
            overall_r_hat=1.0,
            odds_r_hat={},
            odds_r_n={},
            odds_vol_r_hat={},
            odds_vol_r_n={},
            odds_vol_age_r_hat={},
            odds_vol_age_r_n={},
        )

    q = float(quantile)
    q = min(max(q, 0.0), 1.0)

    odds_edges = _extend_edges(odds_bins)
    snap_age_edges = _extend_edges(snap_age_bins)

    s_ratio = pd.to_numeric(df["ratio_final_to_buy"], errors="coerce")
    s_odds = pd.to_numeric(df["odds_buy"], errors="coerce")
    s_vol = pd.to_numeric(df.get("log_odds_std_60m"), errors="coerce")
    s_age = pd.to_numeric(df.get("snap_age_min"), errors="coerce")

    # 基本フィルタ
    work = pd.DataFrame(
        {
            "ratio": s_ratio,
            "odds": s_odds,
            "vol": s_vol,
            "age": s_age,
        }
    )
    work = work.dropna(subset=["ratio", "odds"])
    work = work[(work["ratio"] > 0) & (work["odds"] > 0)]
    if work.empty:
        return SlippageTable(
            quantile=q,
            min_count=int(min_count),
            odds_edges=odds_edges,
            ts_vol_edges=[],
            use_snap_age=bool(use_snap_age),
            snap_age_edges=snap_age_edges,
            overall_r_hat=1.0,
            odds_r_hat={},
            odds_r_n={},
            odds_vol_r_hat={},
            odds_vol_r_n={},
            odds_vol_age_r_hat={},
            odds_vol_age_r_n={},
        )

    ts_cut = _quantile_cutpoints(work["vol"], [float(x) for x in (ts_vol_quantiles or [])])

    # bin列を作る
    work["odds_bin"] = work["odds"].apply(lambda x: _bin_index(x, odds_edges))
    work["vol_bin"] = work["vol"].apply(lambda x: None if pd.isna(x) else _bin_index(x, [-float("inf")] + ts_cut + [float("inf")]))
    # vol_binは _bin_index が edges前提なので、ここは手抜き: ts_cut を使って自前分類
    def _vol_bin(v):
        if v is None or pd.isna(v):
            return None
        vv = float(v)
        if not ts_cut:
            return 0
        for i, c in enumerate(ts_cut):
            if vv < c:
                return i
        return len(ts_cut)
    work["vol_bin"] = work["vol"].apply(_vol_bin)

    if use_snap_age:
        work["age_bin"] = work["age"].apply(lambda x: _bin_index(x, snap_age_edges))
    else:
        work["age_bin"] = None

    # overall
    overall_r = float(work["ratio"].quantile(q))

    # odds 1D
    odds_r_hat: dict[int, float] = {}
    odds_r_n: dict[int, int] = {}
    for k, g in work.groupby("odds_bin"):
        if k is None:
            continue
        s = g["ratio"]
        n = int(len(s))
        odds_r_n[int(k)] = n
        odds_r_hat[int(k)] = float(s.quantile(q)) if n else float("nan")

    # odds×vol 2D（vol_bin None は除外：フォールバックは odds側で拾う）
    odds_vol_r_hat: dict[tuple[int, int], float] = {}
    odds_vol_r_n: dict[tuple[int, int], int] = {}
    g2 = work.dropna(subset=["odds_bin", "vol_bin"])
    if not g2.empty:
        for (ob, vb), gg in g2.groupby(["odds_bin", "vol_bin"]):
            key = (int(ob), int(vb))
            n = int(len(gg))
            odds_vol_r_n[key] = n
            odds_vol_r_hat[key] = float(gg["ratio"].quantile(q)) if n else float("nan")

    # odds×vol×age 3D
    odds_vol_age_r_hat: dict[tuple[int, int, int], float] = {}
    odds_vol_age_r_n: dict[tuple[int, int, int], int] = {}
    if use_snap_age:
        g3 = work.dropna(subset=["odds_bin", "vol_bin", "age_bin"])
        if not g3.empty:
            for (ob, vb, ab), gg in g3.groupby(["odds_bin", "vol_bin", "age_bin"]):
                key = (int(ob), int(vb), int(ab))
                n = int(len(gg))
                odds_vol_age_r_n[key] = n
                odds_vol_age_r_hat[key] = float(gg["ratio"].quantile(q)) if n else float("nan")

    return SlippageTable(
        quantile=q,
        min_count=int(min_count),
        odds_edges=odds_edges,
        ts_vol_edges=ts_cut,
        use_snap_age=bool(use_snap_age),
        snap_age_edges=snap_age_edges,
        overall_r_hat=overall_r,
        odds_r_hat=odds_r_hat,
        odds_r_n=odds_r_n,
        odds_vol_r_hat=odds_vol_r_hat,
        odds_vol_r_n=odds_vol_r_n,
        odds_vol_age_r_hat=odds_vol_age_r_hat,
        odds_vol_age_r_n=odds_vol_age_r_n,
    )


def fetch_slippage_feature_snapshot(
    session: Session,
    start_date: str,
    end_date: str,
    buy_t_minus_minutes: int,
) -> pd.DataFrame:
    """
    指定期間の各馬について、buy_timeスナップショット時点の特徴量（TSボラ等）を features から取る。

    Returns columns:
      - race_id, horse_no
      - log_odds_std_60m, n_pts_60m
    """
    cfg = get_config()
    u = cfg.universe
    track_codes = list(u.track_codes or [])
    exclude_race_ids = list(u.exclude_race_ids or [])

    q = text(
        """
        WITH bt AS (
          SELECT
            r.race_id,
            ((r.date::timestamp + r.start_time) - make_interval(mins => :buy_minutes)) AS buy_time
          FROM fact_race r
          WHERE r.date BETWEEN :d1 AND :d2
            AND r.start_time IS NOT NULL
            AND (:track_codes_len = 0 OR r.track_code = ANY(:track_codes))
            AND (:exclude_len = 0 OR NOT (r.race_id = ANY(:exclude_race_ids)))
        )
        SELECT
          f.race_id,
          (f.payload->>'horse_no')::int AS horse_no,
          (f.payload->>'log_odds_std_60m')::double precision AS log_odds_std_60m,
          (f.payload->>'n_pts_60m')::int AS n_pts_60m
        FROM features f
        JOIN bt
          ON f.race_id = bt.race_id
         AND f.asof_time = bt.buy_time
        WHERE f.feature_version = :feature_version
          AND (f.payload ? 'horse_no')
        """
    )

    rows = session.execute(
        q,
        {
            "d1": start_date,
            "d2": end_date,
            "buy_minutes": int(buy_t_minus_minutes),
            "feature_version": FeatureBuilder.VERSION,
            "track_codes_len": len(track_codes),
            "track_codes": track_codes,
            "exclude_len": len(exclude_race_ids),
            "exclude_race_ids": exclude_race_ids,
        },
    ).fetchall()

    df = pd.DataFrame([dict(r._mapping) for r in rows])
    if df.empty:
        return df
    # 重複がある場合は潰す（念のため）
    df = df.dropna(subset=["race_id", "horse_no"]).drop_duplicates(subset=["race_id", "horse_no"])
    return df


