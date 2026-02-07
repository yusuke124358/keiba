"""
特徴量生成パイプライン

リーク防止: asof_time より未来のデータは使用しない
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_config
from ..db.models import Features
from ..db.pace_utils import lap_stats, sum_first_n, sum_last_n

logger = logging.getLogger(__name__)


def _json_safe(value):
    """
    JSONBに格納できるように型を正規化する
    - Decimal -> float
    - numpy scalar -> Python scalar
    - Infinity/NaN -> None (PostgreSQL JSONBでは無効)
    - dict/list は再帰的に処理
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        val = float(value)
        # Infinity/NaNをNoneに変換
        if not (val == val) or val == float("inf") or val == float("-inf"):
            return None
        return val
    if isinstance(value, float):
        # Infinity/NaNをNoneに変換
        if not (value == value) or value == float("inf") or value == float("-inf"):
            return None
        return value
    if isinstance(value, np.generic):
        val = value.item()
        # Infinity/NaNをNoneに変換
        if isinstance(val, float) and (
            not (val == val) or val == float("inf") or val == float("-inf")
        ):
            return None
        return val
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class FeatureBuilder:
    """特徴量ビルダー"""

    # Option C1: 履歴特徴量を拡張したためバージョン更新
    # Option C4: ペース/通過順特徴量を追加（1.4.0）
    VERSION = "1.6.2"

    def __init__(self, session: Session):
        self.session = session
        self.config = get_config()
        # race_id + asof_time ごとに市場特徴量（全馬分）をキャッシュ
        # build_for_race() 内で同じ race_id/asof_time が何度も呼ばれるため、DBクエリ削減に効く
        self._market_cache: dict[tuple[str, datetime], dict[int, dict]] = {}

    def build_for_race(self, race_id: str, asof_time: datetime) -> list[dict]:
        """
        レースの特徴量を生成

        Args:
            race_id: レースID
            asof_time: 特徴量計算時点（これより未来のデータは使わない）

        Returns:
            list of {horse_id, features}
        """
        # レース基本情報
        race_info = self._get_race_info(race_id)
        if not race_info:
            return []

        # 出走馬リスト
        entries = self._get_entries(race_id)
        if not entries:
            return []

        # C4: レースレベルの予測ペース特徴量（全出走馬から推定）
        pace_history_by_horse, race_expected_pace = self._build_pace_history_features(
            race_info, entries, asof_time
        )
        # EXP-030: track x distance conditional win-rate encodings (leak-safe: race_dt < asof_time).
        jockey_td, trainer_td = self._build_jt_track_dist_win_rate_maps(
            race_info=race_info,
            entries=entries,
            asof_time=asof_time,
        )

        results = []
        for entry in entries:
            features = {}

            # 市場特徴量（オッズベース）
            market_features = self._build_market_features(race_id, entry["horse_no"], asof_time)
            features.update(market_features)

            # 過去走特徴量
            history_features = self._build_history_features(
                horse_id=entry.get("horse_id"),
                jockey_id=entry.get("jockey_id"),
                trainer_id=entry.get("trainer_id"),
                asof_time=asof_time,
                target_distance=race_info.get("distance"),
            )
            features.update(history_features)
            jid = entry.get("jockey_id")
            tid = entry.get("trainer_id")
            features["jockey_win_rate_track_dist"] = (
                jockey_td.get(str(jid)) if jid is not None else None
            )
            features["trainer_win_rate_track_dist"] = (
                trainer_td.get(str(tid)) if tid is not None else None
            )

            pace_history = pace_history_by_horse.get(str(entry["horse_id"]))
            if pace_history:
                features.update(pace_history)

            # 構造特徴量
            struct_features = self._build_struct_features(race_info, entry)
            features.update(struct_features)

            # C4: レースレベルの予測ペース特徴量（全馬共通）
            if race_expected_pace:
                features.update(race_expected_pace)

            results.append(
                {
                    "horse_id": entry["horse_id"],
                    "horse_no": entry["horse_no"],
                    "features": features,
                }
            )

        return results

    def _estimate_race_pace_from_entries(
        self,
        race_id: str,
        entries: list[dict],
        asof_time: datetime,
        target_distance: Optional[float],
    ) -> dict:
        """
        C4: 出走馬の過去走からレースレベルの予測ペースを推定

        注意: 当該レースの実測値は使わない（リーク防止）
        出走馬全員の過去走から集約して「予測ペース」を計算

        Returns:
            {
                "race_expected_first3f": float | None,
                "race_expected_last3f": float | None,
                "race_expected_pace_diff": float | None,
                "race_expected_pace_pressure": int | None,
            }
        """
        if not entries:
            return {
                "race_expected_first3f": None,
                "race_expected_last3f": None,
                "race_expected_pace_diff": None,
                "race_expected_pace_pressure": None,
            }

        # 出走馬全員の過去走からペース情報を取得
        horse_ids = [e.get("horse_id") for e in entries if e.get("horse_id")]
        if not horse_ids:
            return {
                "race_expected_first3f": None,
                "race_expected_last3f": None,
                "race_expected_pace_diff": None,
                "race_expected_pace_pressure": None,
            }

        query = text("""
            SELECT
                r.pace_first3f,
                r.pace_last3f,
                res.pos_1c,
                res.pos_4c,
                r.field_size
            FROM fact_result res
            JOIN fact_race r ON res.race_id = r.race_id
            WHERE res.horse_id = ANY(:horse_ids)
              AND r.start_time IS NOT NULL
              AND (r.date::timestamp + r.start_time) < :asof_time
              AND r.pace_first3f IS NOT NULL
              AND r.pace_last3f IS NOT NULL
            ORDER BY (r.date::timestamp + r.start_time) DESC
            LIMIT 200
        """)
        rows = self.session.execute(
            query, {"horse_ids": horse_ids, "asof_time": asof_time}
        ).fetchall()

        if not rows:
            return {
                "race_expected_first3f": None,
                "race_expected_last3f": None,
                "race_expected_pace_diff": None,
                "race_expected_pace_pressure": None,
            }

        df = pd.DataFrame([dict(r._mapping) for r in rows])
        df["pace_first3f"] = pd.to_numeric(df["pace_first3f"], errors="coerce")
        df["pace_last3f"] = pd.to_numeric(df["pace_last3f"], errors="coerce")
        df["pos_1c"] = pd.to_numeric(df["pos_1c"], errors="coerce")
        df["pos_4c"] = pd.to_numeric(df["pos_4c"], errors="coerce")
        df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce")

        # 前3F: 速い馬（20%分位点）が作るペースを想定
        first3f_valid = df["pace_first3f"].dropna()
        expected_first3f = None
        if len(first3f_valid) >= 3:
            expected_first3f = float(np.quantile(first3f_valid, 0.20))

        # 後3F: 中央値（一般的な上がり）
        last3f_valid = df["pace_last3f"].dropna()
        expected_last3f = None
        if len(last3f_valid) >= 3:
            expected_last3f = float(np.median(last3f_valid))

        # ペース差
        expected_pace_diff = None
        if expected_first3f is not None and expected_last3f is not None:
            expected_pace_diff = expected_first3f - expected_last3f

        # ペースプレッシャー: 先行型（1Cで前1/3以内）の馬の割合
        pace_pressure = None
        if "pos_1c" in df.columns and "field_size" in df.columns:
            df["pos_1c_pct"] = df["pos_1c"] / df["field_size"]
            front_runners = df["pos_1c_pct"].dropna()
            if len(front_runners) >= 3:
                # 前1/3以内の馬の割合
                pace_pressure = int((front_runners <= 0.33).sum())

        return {
            "race_expected_first3f": expected_first3f,
            "race_expected_last3f": expected_last3f,
            "race_expected_pace_diff": expected_pace_diff,
            "race_expected_pace_pressure": pace_pressure,
        }

    def _build_pace_history_features(
        self,
        race_info: dict,
        entries: list[dict],
        asof_time: datetime,
    ) -> tuple[dict[str, dict], dict]:
        race_date = race_info.get("date")
        empty_race = {
            "race_expected_first3f": None,
            "race_expected_last3f": None,
            "race_expected_pace_diff": None,
            "race_expected_pace_pressure": None,
        }
        if not entries or not race_date:
            return {}, empty_race

        horse_ids = [e.get("horse_id") for e in entries if e.get("horse_id")]
        if not horse_ids:
            return {}, empty_race

        horse_ids = list(dict.fromkeys(horse_ids))
        min_date = race_date - timedelta(days=365)

        query = text("""
            SELECT
                res.horse_id,
                r.date,
                r.pace_first3f,
                r.pace_last3f,
                r.pace_diff_sec,
                r.lap_slope,
                r.lap_times_200m,
                res.pos_1c,
                r.field_size
            FROM fact_result res
            JOIN fact_race r ON res.race_id = r.race_id
            WHERE res.horse_id = ANY(:horse_ids)
              AND r.date < :race_date
              AND r.date >= :min_date
              AND res.finish_pos IS NOT NULL
              AND (r.pace_first3f IS NOT NULL OR r.lap_times_200m IS NOT NULL)
            ORDER BY r.date DESC
        """)
        rows = self.session.execute(
            query,
            {"horse_ids": horse_ids, "race_date": race_date, "min_date": min_date},
        ).fetchall()

        if not rows:
            return {}, empty_race

        df = pd.DataFrame([dict(r._mapping) for r in rows])
        if df.empty:
            return {}, empty_race

        for c in (
            "pace_first3f",
            "pace_last3f",
            "pace_diff_sec",
            "lap_slope",
            "pos_1c",
            "field_size",
        ):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df["pace_first3f_calc"] = df.get("pace_first3f")
        df["pace_last3f_calc"] = df.get("pace_last3f")
        if "lap_times_200m" in df.columns:
            mask = df["pace_first3f_calc"].isna()
            if mask.any():
                vals = df.loc[mask, "lap_times_200m"].apply(lambda x: sum_first_n(x, 3))
                df.loc[mask, "pace_first3f_calc"] = pd.to_numeric(vals, errors="coerce")
            mask = df["pace_last3f_calc"].isna()
            if mask.any():
                vals = df.loc[mask, "lap_times_200m"].apply(lambda x: sum_last_n(x, 3))
                df.loc[mask, "pace_last3f_calc"] = pd.to_numeric(vals, errors="coerce")

        df["pace_diff_calc"] = df.get("pace_diff_sec")
        mask = (
            df["pace_diff_calc"].isna()
            & df["pace_first3f_calc"].notna()
            & df["pace_last3f_calc"].notna()
        )
        if mask.any():
            df.loc[mask, "pace_diff_calc"] = (
                df.loc[mask, "pace_first3f_calc"] - df.loc[mask, "pace_last3f_calc"]
            )

        df["lap_slope_calc"] = df.get("lap_slope")
        if "lap_times_200m" in df.columns:
            mask = df["lap_slope_calc"].isna()
            if mask.any():
                vals = df.loc[mask, "lap_times_200m"].apply(lambda x: lap_stats(x)[2])
                df.loc[mask, "lap_slope_calc"] = pd.to_numeric(vals, errors="coerce")

        def _q(series: pd.Series, q: float) -> Optional[float]:
            s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                return None
            return float(np.quantile(s, q))

        per_horse: dict[str, dict] = {}
        pos_means: dict[str, Optional[float]] = {}
        for horse_id, g in df.groupby("horse_id"):
            g = g.sort_values("date", ascending=False)
            pace_valid = g[g["pace_first3f_calc"].notna() & g["pace_last3f_calc"].notna()]
            n_pace = int(len(pace_valid))
            per_horse[str(horse_id)] = {
                "horse_past_first3f_p20": _q(pace_valid["pace_first3f_calc"], 0.20),
                "horse_past_last3f_p50": _q(pace_valid["pace_last3f_calc"], 0.50),
                "horse_past_pace_diff_p50": _q(pace_valid["pace_diff_calc"], 0.50),
                "horse_past_lap_slope_p50": _q(g["lap_slope_calc"], 0.50),
                "horse_past_n_races_pace": n_pace,
            }
            if "pos_1c" in g.columns and "field_size" in g.columns:
                pos = pd.to_numeric(g["pos_1c"], errors="coerce")
                fs = pd.to_numeric(g["field_size"], errors="coerce")
                pos_pct = (pos / fs).replace([np.inf, -np.inf], np.nan).dropna()
                if not pos_pct.empty:
                    pos_means[str(horse_id)] = float(pos_pct.head(3).mean())

        first3f_vals: list[float] = []
        last3f_vals: list[float] = []
        for v in per_horse.values():
            raw_first3f = v.get("horse_past_first3f_p20")
            if raw_first3f is not None:
                first3f_vals.append(float(raw_first3f))
            raw_last3f = v.get("horse_past_last3f_p50")
            if raw_last3f is not None:
                last3f_vals.append(float(raw_last3f))

        expected_first3f = _q(pd.Series(first3f_vals), 0.20) if len(first3f_vals) >= 3 else None
        expected_last3f = _q(pd.Series(last3f_vals), 0.50) if len(last3f_vals) >= 3 else None
        expected_pace_diff = None
        if expected_first3f is not None and expected_last3f is not None:
            expected_pace_diff = expected_first3f - expected_last3f

        pace_pressure = None
        pos_vals = [v for v in pos_means.values() if v is not None]
        if len(pos_vals) >= 3:
            pace_pressure = int((np.array(pos_vals) <= 0.33).sum())
        elif len(first3f_vals) >= 3:
            thr = float(np.quantile(first3f_vals, 0.33))
            pace_pressure = int(sum(v <= thr for v in first3f_vals))

        race_expected = {
            "race_expected_first3f": expected_first3f,
            "race_expected_last3f": expected_last3f,
            "race_expected_pace_diff": expected_pace_diff,
            "race_expected_pace_pressure": pace_pressure,
        }
        return per_horse, race_expected

    def _get_race_info(self, race_id: str) -> Optional[dict]:
        """レース情報を取得（C4: 実測ペースは使わない。expectedは別途計算）"""
        query = text("""
            SELECT race_id, date, track_code, race_no,
                   surface, distance, going_turf, going_dirt, field_size
            FROM fact_race
            WHERE race_id = :race_id
        """)
        result = self.session.execute(query, {"race_id": race_id}).fetchone()
        if result:
            return dict(result._mapping)
        return None

    def _get_entries(self, race_id: str) -> list[dict]:
        """出走馬リストを取得"""
        query = text("""
            SELECT horse_id, horse_no, frame_no, jockey_id, trainer_id,
                   weight_carried, horse_weight
            FROM fact_entry
            WHERE race_id = :race_id
            ORDER BY horse_no
        """)
        results = self.session.execute(query, {"race_id": race_id}).fetchall()
        return [dict(r._mapping) for r in results]

    def _build_jt_track_dist_win_rate_maps(
        self,
        *,
        race_info: dict,
        entries: list[dict],
        asof_time: datetime,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        EXP-030: Compute per-(jockey|trainer) win rates conditioned on the current race's
        (track_code, distance_bucket), leak-free w.r.t asof_time.
        """
        from .history_features import compute_entity_track_dist_win_rate_map

        track_code = race_info.get("track_code")
        target_distance = race_info.get("distance")
        if track_code is None or target_distance is None:
            return {}, {}

        jockey_ids = sorted({str(e.get("jockey_id")) for e in entries if e.get("jockey_id")})
        trainer_ids = sorted({str(e.get("trainer_id")) for e in entries if e.get("trainer_id")})
        if not jockey_ids and not trainer_ids:
            return {}, {}

        cutoff = asof_time - timedelta(days=365)

        def _fetch(entity_field: str, ids: list[str]) -> pd.DataFrame:
            if not ids:
                return pd.DataFrame([])
            q = text(f"""
                SELECT
                    e.{entity_field} AS entity_id,
                    (r.date::timestamp + r.start_time) AS race_dt,
                    r.track_code,
                    r.distance,
                    res.finish_pos
                FROM fact_result res
                JOIN fact_race r ON res.race_id = r.race_id
                JOIN fact_entry e ON e.race_id = res.race_id AND e.horse_no = res.horse_no
                WHERE r.start_time IS NOT NULL
                  AND (r.date::timestamp + r.start_time) >= :cutoff
                  AND (r.date::timestamp + r.start_time) < :asof_time
                  AND res.finish_pos IS NOT NULL
                  AND e.{entity_field} = ANY(:ids)
            """)
            rows = self.session.execute(
                q,
                {"cutoff": cutoff, "asof_time": asof_time, "ids": list(ids)},
            ).fetchall()
            return pd.DataFrame([dict(r._mapping) for r in rows])

        jockey_td = compute_entity_track_dist_win_rate_map(
            _fetch("jockey_id", jockey_ids),
            asof_time=asof_time,
            track_code=str(track_code),
            target_distance=target_distance,
            entity_col="entity_id",
        )
        trainer_td = compute_entity_track_dist_win_rate_map(
            _fetch("trainer_id", trainer_ids),
            asof_time=asof_time,
            track_code=str(track_code),
            target_distance=target_distance,
            entity_col="entity_id",
        )
        return jockey_td, trainer_td

    def _build_market_features(self, race_id: str, horse_no: int, asof_time: datetime) -> dict:
        """
        市場特徴量（オッズベース）

        p_mkt = 1/odds / sum(1/odds) で市場確率を計算

        重要:
            - asof_time以前のデータのみを使用してリークを防止
            - 同一スナップショット（t_snap）の行のみで計算して整合性を保つ
        """
        cache_key = (race_id, asof_time)
        all_features = self._market_cache.get(cache_key)
        if all_features is None:
            all_features = self._build_market_features_for_race(race_id, asof_time)
            self._market_cache[cache_key] = all_features

        feat = all_features.get(horse_no)
        if feat is None:
            return self._empty_market_features()
        return feat

    def _build_market_features_for_race(self, race_id: str, asof_time: datetime) -> dict[int, dict]:
        """
        市場特徴量（+ 時系列オッズ変動特徴量）をレース単位で生成する。

        返り値:
          {horse_no: {feature_name: value, ...}}

        方針:
          - t0 = MAX(asof_time) <= buy_time を1つ決める（スナップショット整合性）
          - 直近60分の odds_ts_win を使って「オッズ変化・傾き・分散」を作る
        """
        # まず asof_time 以前の最新スナップショット時刻（t0）を取得
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
            return {}

        t0 = snap_result[0]
        snap_age_min = float((asof_time - t0).total_seconds() / 60.0) if asof_time and t0 else None

        # t0 スナップショット（全馬）
        base_query = text("""
            SELECT DISTINCT ON (horse_no)
              horse_no, odds, popularity, data_kubun, total_sales
            FROM odds_ts_win
            WHERE race_id = :race_id
              AND asof_time = :t0
              AND odds > 0
            ORDER BY horse_no, data_kubun DESC
        """)
        base_rows = self.session.execute(base_query, {"race_id": race_id, "t0": t0}).fetchall()
        if not base_rows:
            return {}

        df0 = pd.DataFrame([dict(r._mapping) for r in base_rows])
        df0["odds"] = df0["odds"].astype(float)

        # odds_rank / is_favorite
        df0_sorted = df0.sort_values("odds").reset_index(drop=True)
        horse_order = df0_sorted["horse_no"].tolist()
        rank_map = {int(hn): int(i + 1) for i, hn in enumerate(horse_order)}
        fav_horse_no = int(horse_order[0]) if horse_order else None

        # ===== 時系列特徴量（直近60分） =====
        t_start = t0 - timedelta(minutes=120)  # 余裕をもって2時間分

        ts_query = text("""
            SELECT DISTINCT ON (horse_no, asof_time)
              horse_no, asof_time, odds, total_sales, data_kubun
            FROM odds_ts_win
            WHERE race_id = :race_id
              AND asof_time <= :t0
              AND asof_time >= :t_start
              AND odds > 0
            ORDER BY horse_no, asof_time, data_kubun DESC
        """)
        ts_rows = self.session.execute(
            ts_query, {"race_id": race_id, "t0": t0, "t_start": t_start}
        ).fetchall()
        df_ts = pd.DataFrame([dict(r._mapping) for r in ts_rows]) if ts_rows else pd.DataFrame()
        if not df_ts.empty:
            df_ts["horse_no"] = df_ts["horse_no"].astype(int)
            df_ts["odds"] = df_ts["odds"].astype(float)
            # total_sales は race 全体の値が入る想定（馬ごとに同値になりがち）
            if "total_sales" in df_ts.columns:
                df_ts["total_sales"] = pd.to_numeric(df_ts["total_sales"], errors="coerce")

            # asof_timeごとの市場確率（p_mkt_t）を作る
            df_ts["inv_odds"] = 1.0 / df_ts["odds"]
            df_ts["inv_sum"] = df_ts.groupby("asof_time")["inv_odds"].transform("sum")
            df_ts["p_mkt_t"] = np.where(
                df_ts["inv_sum"] > 0, df_ts["inv_odds"] / df_ts["inv_sum"], np.nan
            )

        # 参照時刻（t10/t30/t60）はレース内で共通にする（スナップショット混在を避ける）
        times = []
        if not df_ts.empty:
            times = sorted(df_ts["asof_time"].unique().tolist())

        # times は昇順。t0 以前の最大が t0 のはず
        def _pick_time(delta_min: int) -> Optional[datetime]:
            target = t0 - timedelta(minutes=delta_min)
            # target 以下の最大
            for t in reversed(times):
                if t <= target:
                    return t
            return None

        t5 = _pick_time(5)
        t10 = _pick_time(10)
        t30 = _pick_time(30)
        t60 = _pick_time(60)

        def _ratio_change(v0: Optional[float], v1: Optional[float]) -> Optional[float]:
            if v0 is None or v1 is None:
                return None
            if v1 <= 0:
                return None
            return (float(v0) / float(v1)) - 1.0

        def _log(v: Optional[float]) -> Optional[float]:
            if v is None or v <= 0:
                return None
            return float(np.log(float(v)))

        def _get_at(
            df: pd.DataFrame, horse: int, t: Optional[datetime], col: str
        ) -> Optional[float]:
            if t is None or df.empty:
                return None
            x = df[(df["horse_no"] == horse) & (df["asof_time"] == t)]
            if x.empty:
                return None
            val = x[col].iloc[0]
            if pd.isna(val):
                return None
            return float(val)

        out: dict[int, dict] = {}
        for _, row in df0.iterrows():
            hn = int(row["horse_no"])
            odds0 = float(row["odds"])
            odds_rank = rank_map.get(hn)

            # p_mkt（t0, 同一スナップショット）
            inv0 = 1.0 / odds0 if odds0 > 0 else None
            total_inv0 = float((1.0 / df0["odds"]).sum()) if len(df0) > 0 else 0.0
            p_mkt0 = (inv0 / total_inv0) if (inv0 is not None and total_inv0 > 0) else None
            overround_sum_inv = float(total_inv0) if total_inv0 > 0 else None
            takeout_implied = (
                (1.0 - (1.0 / overround_sum_inv))
                if (overround_sum_inv is not None and overround_sum_inv > 0)
                else None
            )
            p_mkt_raw = p_mkt0
            p_mkt_race = p_mkt0

            # 時系列参照（t10/t30/t60）
            odds5 = _get_at(df_ts, hn, t5, "odds")
            odds10 = _get_at(df_ts, hn, t10, "odds")
            odds30 = _get_at(df_ts, hn, t30, "odds")
            odds60 = _get_at(df_ts, hn, t60, "odds")

            p5 = _get_at(df_ts, hn, t5, "p_mkt_t")
            p10 = _get_at(df_ts, hn, t10, "p_mkt_t")
            p30 = _get_at(df_ts, hn, t30, "p_mkt_t")
            p60 = _get_at(df_ts, hn, t60, "p_mkt_t")

            # 直近60分のlog(odds)の傾き・分散（なければNone）
            slope_60 = None
            std_60 = None
            n_pts_60 = 0
            if not df_ts.empty:
                d = df_ts[df_ts["horse_no"] == hn].sort_values("asof_time")
                # t0-60min以降の点だけ
                d60 = d[d["asof_time"] >= (t0 - timedelta(minutes=60))]
                if len(d60) >= 2:
                    n_pts_60 = int(len(d60))
                    x_min = (d60["asof_time"] - t0).dt.total_seconds() / 60.0
                    y = np.log(d60["odds"].astype(float))
                    # y = a*x + b
                    a, b = np.polyfit(x_min.values.astype(float), y.values.astype(float), 1)
                    slope_60 = float(a)
                    std_60 = float(y.std(ddof=0))
                elif len(d) >= 2:
                    # 60分以内が足りない場合は取れる範囲で（保守）
                    n_pts_60 = int(len(d))
                    x_min = (d["asof_time"] - t0).dt.total_seconds() / 60.0
                    y = np.log(d["odds"].astype(float))
                    a, b = np.polyfit(x_min.values.astype(float), y.values.astype(float), 1)
                    slope_60 = float(a)
                    std_60 = float(y.std(ddof=0))

            out[hn] = {
                # 基本（t0）
                "odds": odds0,
                "log_odds": _log(odds0),
                "p_mkt": p_mkt_raw,
                "p_mkt_raw": p_mkt_raw,
                "p_mkt_race": p_mkt_race,
                "overround_sum_inv": overround_sum_inv,
                "takeout_implied": takeout_implied,
                "odds_rank": odds_rank,
                "is_favorite": 1 if (fav_horse_no is not None and hn == fav_horse_no) else 0,
                # スナップショットの古さ（buy_time - t_snap）。スリッページ推定で使う
                "snap_age_min": snap_age_min,
                # 時系列（変動）
                "odds_chg_5m": _ratio_change(odds0, odds5),
                "odds_chg_10m": _ratio_change(odds0, odds10),
                "odds_chg_30m": _ratio_change(odds0, odds30),
                "odds_chg_60m": _ratio_change(odds0, odds60),
                "p_mkt_chg_5m": (p_mkt0 - p5) if (p_mkt0 is not None and p5 is not None) else None,
                "p_mkt_chg_10m": (p_mkt0 - p10)
                if (p_mkt0 is not None and p10 is not None)
                else None,
                "p_mkt_chg_30m": (p_mkt0 - p30)
                if (p_mkt0 is not None and p30 is not None)
                else None,
                "p_mkt_chg_60m": (p_mkt0 - p60)
                if (p_mkt0 is not None and p60 is not None)
                else None,
                "log_odds_slope_60m": slope_60,
                "log_odds_std_60m": std_60,
                "n_pts_60m": n_pts_60,
            }

        return out

    def _empty_market_features(self) -> dict:
        """空の市場特徴量"""
        return {
            "odds": None,
            "log_odds": None,
            "p_mkt": None,
            "p_mkt_raw": None,
            "p_mkt_race": None,
            "overround_sum_inv": None,
            "takeout_implied": None,
            "odds_rank": None,
            "is_favorite": 0,
        }

    def _fallback_market_features(
        self, race_id: str, horse_no: int, asof_time: datetime, snap_df: pd.DataFrame
    ) -> dict:
        """
        フォールバック：該当馬がt_snapにない場合、その馬の直近オッズを使用

        注意: この場合、p_mktの分母がずれる可能性があるためフラグを立てる
        """
        query = text("""
            SELECT odds, popularity
            FROM odds_ts_win
            WHERE race_id = :race_id
              AND horse_no = :horse_no
              AND asof_time <= :asof_time
              AND odds > 0
            ORDER BY asof_time DESC, data_kubun DESC
            LIMIT 1
        """)
        result = self.session.execute(
            query, {"race_id": race_id, "horse_no": horse_no, "asof_time": asof_time}
        ).fetchone()

        if not result:
            return self._empty_market_features()

        odds = float(result[0])

        # p_mktはsnap_dfベースで近似計算（正確ではないが妥当）
        snap_df_copy = snap_df.copy()
        snap_df_copy["inv_odds"] = 1 / snap_df_copy["odds"]
        # 該当馬を追加して計算
        inv_target = 1 / odds
        total_inv = snap_df_copy["inv_odds"].sum() + inv_target
        p_mkt = inv_target / total_inv if total_inv > 0 else None
        overround_sum_inv = float(total_inv) if total_inv > 0 else None
        takeout_implied = (
            (1.0 - (1.0 / overround_sum_inv))
            if (overround_sum_inv is not None and overround_sum_inv > 0)
            else None
        )
        p_mkt_raw = p_mkt
        p_mkt_race = p_mkt

        # オッズ順位は近似
        n_better = (snap_df_copy["odds"] < odds).sum()
        odds_rank = n_better + 1

        return {
            "odds": odds,
            "log_odds": np.log(odds) if odds > 0 else None,
            "p_mkt": p_mkt_raw,
            "p_mkt_raw": p_mkt_raw,
            "p_mkt_race": p_mkt_race,
            "overround_sum_inv": overround_sum_inv,
            "takeout_implied": takeout_implied,
            "odds_rank": odds_rank,
            "is_favorite": 1 if odds_rank == 1 else 0,
        }

    def _build_history_features(
        self,
        *,
        horse_id: Optional[str],
        jockey_id: Optional[str],
        trainer_id: Optional[str],
        asof_time: datetime,
        target_distance: Optional[float],
    ) -> dict:
        """
        過去走特徴量

        リーク防止: race_dt < asof_time のデータのみ使用（同日でも時刻で判定）
        """
        from .history_features import compute_horse_history_features

        out = dict(self._empty_history_features())

        # ---- 馬（過去走） ----
        if horse_id:
            query_h = text("""
                SELECT
                    (r.date::timestamp + r.start_time) AS race_dt,
                    r.distance,
                    r.surface,
                    r.field_size,
                    r.pace_first3f,
                    r.pace_last3f,
                    res.finish_pos,
                    res.odds,
                    res.time_sec,
                    res.last_3f,
                    res.pos_1c,
                    res.pos_2c,
                    res.pos_3c,
                    res.pos_4c
                FROM fact_result res
                JOIN fact_race r ON res.race_id = r.race_id
                WHERE res.horse_id = :horse_id
                  AND r.start_time IS NOT NULL
                  AND (r.date::timestamp + r.start_time) < :asof_time
                ORDER BY (r.date::timestamp + r.start_time) DESC
                LIMIT 50
            """)
            rows = self.session.execute(
                query_h, {"horse_id": horse_id, "asof_time": asof_time}
            ).fetchall()
            if rows:
                df = pd.DataFrame([dict(r._mapping) for r in rows])
                # compute_* 側で未来除外も入っているが二重でもOK
                out.update(
                    compute_horse_history_features(
                        df, asof_time=asof_time, target_distance=target_distance
                    )
                )

        # ---- 騎手/調教師/人馬（過去365日） ----
        cutoff = asof_time - timedelta(days=365)

        def _agg_jt(where_sql: str, params: dict) -> tuple[int, float | None, float | None]:
            q = text(f"""
                SELECT
                    COUNT(*) AS starts,
                    SUM(CASE WHEN res.finish_pos = 1 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN res.finish_pos <= 3 THEN 1 ELSE 0 END) AS places
                FROM fact_result res
                JOIN fact_race r ON res.race_id = r.race_id
                JOIN fact_entry e ON e.race_id = res.race_id AND e.horse_no = res.horse_no
                WHERE r.start_time IS NOT NULL
                  AND (r.date::timestamp + r.start_time) >= :cutoff
                  AND (r.date::timestamp + r.start_time) < :asof_time
                  AND res.finish_pos IS NOT NULL
                  AND {where_sql}
            """)
            row = self.session.execute(
                q, {**params, "cutoff": cutoff, "asof_time": asof_time}
            ).fetchone()
            if not row:
                return 0, None, None
            starts = int(row[0] or 0)
            wins = int(row[1] or 0)
            places = int(row[2] or 0)
            win_rate = (wins / starts) if starts > 0 else None
            place_rate = (places / starts) if starts > 0 else None
            return starts, win_rate, place_rate

        if jockey_id:
            s, w, p = _agg_jt("e.jockey_id = :jockey_id", {"jockey_id": jockey_id})
            out.update(
                {"jockey_starts_365": s, "jockey_win_rate_365": w, "jockey_place_rate_365": p}
            )

        if trainer_id:
            s, w, p = _agg_jt("e.trainer_id = :trainer_id", {"trainer_id": trainer_id})
            out.update(
                {"trainer_starts_365": s, "trainer_win_rate_365": w, "trainer_place_rate_365": p}
            )

        if horse_id and jockey_id:
            s, w, _ = _agg_jt(
                "res.horse_id = :horse_id AND e.jockey_id = :jockey_id",
                {"horse_id": horse_id, "jockey_id": jockey_id},
            )
            out.update({"horse_jockey_starts_365": s, "horse_jockey_win_rate_365": w})

        return out

    def _empty_history_features(self) -> dict:
        return {
            # 既存互換
            "n_races": 0,
            "win_rate": None,
            "place_rate": None,
            "avg_finish_pos": None,
            "avg_last_3f": None,
            "days_since_last": None,
            # Option C1
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
            # C3: perf（time_sec/last_3f）由来
            "horse_speed_z_last": None,
            "horse_speed_z_best3": None,
            "horse_speed_z_mean3": None,
            "horse_speed_z_trend3": None,
            "horse_last3f_z_last": None,
            "horse_last3f_z_best3": None,
            "horse_last3f_z_mean3": None,
            "horse_last3f_z_trend3": None,
            # C4: position / pace
            "horse_pos_4c_pct_last": None,
            "horse_pos_4c_pct_mean3": None,
            "horse_pos_gain_last": None,
            "horse_pos_gain_mean3": None,
            "horse_pace_diff_last": None,
            "horse_pace_diff_mean3": None,
            "horse_past_first3f_p20": None,
            "horse_past_last3f_p50": None,
            "horse_past_pace_diff_p50": None,
            "horse_past_lap_slope_p50": None,
            "horse_past_n_races_pace": 0,
            "jockey_starts_365": 0,
            "jockey_win_rate_365": None,
            "jockey_place_rate_365": None,
            "trainer_starts_365": 0,
            "trainer_win_rate_365": None,
            "trainer_place_rate_365": None,
            "jockey_win_rate_track_dist": None,
            "trainer_win_rate_track_dist": None,
            "horse_jockey_starts_365": 0,
            "horse_jockey_win_rate_365": None,
        }

    def _build_struct_features(self, race_info: dict, entry: dict) -> dict:
        """構造特徴量（レース条件・枠番等 + C4: レースレベルのペース）"""
        field_size = race_info.get("field_size") or 18
        horse_no = entry.get("horse_no") or 1

        features = {
            "field_size": field_size,
            "distance": race_info.get("distance"),
            "is_turf": 1 if str(race_info.get("surface")).strip() == "1" else 0,
            "frame_no": entry.get("frame_no"),
            "horse_no": horse_no,
            "horse_no_pct": horse_no / field_size,
            "weight_carried": entry.get("weight_carried"),
        }

        # C4のレースレベルのペース特徴量は build_for_race で全出走馬から計算して追加される
        # （ここでは構造特徴量のみ）
        return features

    def save_features(self, race_id: str, asof_time: datetime, features_list: list[dict]) -> int:
        """
        特徴量をDBに保存

        重複防止: 同じ race_id + asof_time + feature_version の既存データは
        事前に削除してから再挿入する（再生成時に古いデータが残らない）

        Note: 将来的には Features テーブルに UNIQUE 制約を追加し、
              upsert に移行することを推奨
        """
        # 既存の同一条件のデータを削除（再生成時の重複防止）
        self.session.execute(
            text("""
                DELETE FROM features
                WHERE race_id = :race_id
                  AND feature_version = :version
                  AND asof_time = :asof_time
            """),
            {"race_id": race_id, "version": self.VERSION, "asof_time": asof_time},
        )

        count = 0
        for item in features_list:
            feature = Features(
                race_id=race_id,
                horse_id=item["horse_id"],
                feature_version=self.VERSION,
                asof_time=asof_time,
                payload=_json_safe(item["features"]),
            )
            self.session.add(feature)
            count += 1

        self.session.commit()
        return count


def build_features(
    session: Session,
    race_ids: list[str],
    asof_time: Optional[datetime] = None,
    buy_t_minus_minutes: Optional[int] = None,
    skip_existing: bool = True,
) -> int:
    """
    複数レースの特徴量を生成

    Args:
        session: DBセッション
        race_ids: レースIDリスト
        asof_time: 特徴量計算時点（Noneなら各レースの購入想定時点を使用）
        buy_t_minus_minutes: 発走何分前を購入時点とするか（Noneならconfigから取得）
        skip_existing: 既に同一(race_id, asof_time, feature_version)が存在する場合は
            再計算をスキップする

    Returns:
        生成した特徴量数

    重要:
        asof_time=Noneの場合、各レースの「発走時刻 - buy_t_minus_minutes」を
        asof_timeとして使用します。これによりbacktestと同じ購入時点に揃い、
        リークを防止できます。
    """
    # ★config一元化: buy_t_minus_minutes が None なら config から取得
    if buy_t_minus_minutes is None:
        buy_t_minus_minutes = get_config().backtest.buy_t_minus_minutes

    builder = FeatureBuilder(session)

    total = 0
    if not race_ids:
        return 0

    # -------- asof_time（購入時点）をバルクで求める --------
    # asof_time が明示されている場合は全レース同一時刻
    if asof_time is not None:
        buy_time_map: dict[str, datetime] = {rid: asof_time for rid in race_ids}
    else:
        # DB側で (date + start_time - buy_minutes) を一括算出
        rows = session.execute(
            text(
                """
                SELECT
                    r.race_id,
                    (
                        (r.date::timestamp + r.start_time)
                        - make_interval(mins => :buy_minutes)
                    ) AS buy_time
                FROM fact_race r
                WHERE r.race_id = ANY(:race_ids)
                  AND r.start_time IS NOT NULL
                """
            ),
            {"race_ids": list(race_ids), "buy_minutes": int(buy_t_minus_minutes)},
        ).fetchall()
        buy_time_map = {str(r[0]): r[1] for r in rows if r and r[0] and r[1]}

    # -------- 既存特徴量のバルク判定（再計算の無駄を避ける） --------
    existing_race_ids: set[str] = set()
    if bool(skip_existing):
        if asof_time is not None:
            # asof が固定の場合は単純な絞り込みでOK
            rows = session.execute(
                text(
                    """
                    SELECT DISTINCT f.race_id
                    FROM features f
                    WHERE f.feature_version = :version
                      AND f.asof_time = :asof_time
                      AND f.race_id = ANY(:race_ids)
                    """
                ),
                {"version": builder.VERSION, "asof_time": asof_time, "race_ids": list(race_ids)},
            ).fetchall()
            existing_race_ids = {str(r[0]) for r in rows if r and r[0]}
        else:
            # asof がレースごとに異なる場合は、fact_race から buy_time を算出して JOIN する
            rows = session.execute(
                text(
                    """
                    WITH targets AS (
                        SELECT
                            r.race_id,
                            (
                                (r.date::timestamp + r.start_time)
                                - make_interval(mins => :buy_minutes)
                            ) AS buy_time
                        FROM fact_race r
                        WHERE r.race_id = ANY(:race_ids)
                          AND r.start_time IS NOT NULL
                    )
                    SELECT t.race_id
                    FROM targets t
                    JOIN features f
                      ON f.race_id = t.race_id
                     AND f.asof_time = t.buy_time
                    WHERE f.feature_version = :version
                    GROUP BY t.race_id
                    """
                ),
                {
                    "race_ids": list(race_ids),
                    "buy_minutes": int(buy_t_minus_minutes),
                    "version": builder.VERSION,
                },
            ).fetchall()
            existing_race_ids = {str(r[0]) for r in rows if r and r[0]}

    skipped = 0
    built_races = 0
    for race_id in race_ids:
        if race_id in existing_race_ids:
            skipped += 1
            continue

        asof = buy_time_map.get(race_id)
        if not asof:
            logger.warning(f"Could not determine asof_time for {race_id}, skipping")
            continue

        features = builder.build_for_race(race_id, asof)
        count = builder.save_features(race_id, asof, features)
        total += count
        built_races += 1
        logger.info(f"Built {count} features for {race_id} (asof={asof})")

    if skipped:
        logger.info(
            f"Skipped feature build for {skipped}/{len(race_ids)} races "
            f"(already exists, version={builder.VERSION})"
        )
    if built_races:
        logger.info(
            f"Built features for {built_races}/{len(race_ids)} races (version={builder.VERSION})"
        )

    return total


def _get_buy_time_for_race(
    session: Session, race_id: str, buy_t_minus_minutes: int
) -> Optional[datetime]:
    """
    レースの購入時点を計算

    fact_race.date + fact_race.start_time - buy_t_minus_minutes
    """
    query = text("""
        SELECT date, start_time
        FROM fact_race
        WHERE race_id = :race_id
    """)
    result = session.execute(query, {"race_id": race_id}).fetchone()

    if not result:
        return None

    race_date = result[0]
    start_time = result[1]

    if not start_time:
        # start_timeがない場合はNone（特徴量生成をスキップ）
        return None

    dt = datetime.combine(race_date, start_time)
    buy_time = dt - timedelta(minutes=buy_t_minus_minutes)
    return buy_time
