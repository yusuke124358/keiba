"""
SQLAlchemy ORM モデル定義

すべての時刻はJST前提（naive datetime）
"""
from datetime import datetime, date, time
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, BigInteger, SmallInteger,
    Numeric, Float, Date, Time, DateTime, ForeignKey, Index,
    CheckConstraint, Text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """ベースクラス"""
    pass


# =============================================================================
# Dimension Tables
# =============================================================================

class DimHorse(Base):
    """馬マスタ"""
    __tablename__ = "dim_horse"
    
    horse_id = Column(String(10), primary_key=True)  # 血統登録番号
    name = Column(String(100))
    sex = Column(String(1))  # 牡/牝/セ
    birth_year = Column(SmallInteger)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class DimJockey(Base):
    """騎手マスタ"""
    __tablename__ = "dim_jockey"
    
    jockey_id = Column(String(5), primary_key=True)
    name = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class DimTrainer(Base):
    """調教師マスタ"""
    __tablename__ = "dim_trainer"
    
    trainer_id = Column(String(5), primary_key=True)
    name = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class DimTrack(Base):
    """競馬場マスタ"""
    __tablename__ = "dim_track"
    
    track_code = Column(String(2), primary_key=True)
    name = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)


# =============================================================================
# Fact Tables
# =============================================================================

class FactRace(Base):
    """レースファクト"""
    __tablename__ = "fact_race"
    
    # race_id: YYYYMMDDJJKKNNRR (16桁)
    # YYYY=年, MMDD=月日, JJ=競馬場, KK=回, NN=日目, RR=レース番号
    race_id = Column(String(16), primary_key=True)
    
    date = Column(Date, nullable=False, index=True)
    track_code = Column(String(2), ForeignKey("dim_track.track_code"))
    race_no = Column(SmallInteger, nullable=False)
    start_time = Column(Time)
    
    # コース条件
    surface = Column(String(1))  # 芝/ダ
    distance = Column(SmallInteger)
    track_cd = Column(String(2))  # トラックコード
    
    # 馬場状態
    going_turf = Column(String(1))  # 良/稍/重/不
    going_dirt = Column(String(1))
    weather = Column(String(1))
    
    # 出走頭数
    field_size = Column(SmallInteger)

    # ===== C4: ペース素材（RA由来）=====
    # 200mラップ（最大25個、0.1秒単位 -> float秒で格納）
    lap_times_200m = Column(JSONB)
    # 前後半の区間タイム（0.1秒 -> float秒で格納）
    pace_first3f = Column(Numeric(4, 1))
    pace_first4f = Column(Numeric(4, 1))
    pace_last3f = Column(Numeric(4, 1))
    pace_last4f = Column(Numeric(4, 1))
    pace_diff_sec = Column(Float)
    lap_mean_sec = Column(Float)
    lap_std_sec = Column(Float)
    lap_slope = Column(Float)
    
    # メタ
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # リレーション
    entries = relationship("FactEntry", back_populates="race")
    results = relationship("FactResult", back_populates="race")


class FactEntry(Base):
    """出走馬ファクト"""
    __tablename__ = "fact_entry"
    
    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    horse_id = Column(String(10), ForeignKey("dim_horse.horse_id"))
    
    horse_no = Column(SmallInteger, nullable=False)  # 馬番
    frame_no = Column(SmallInteger)  # 枠番
    
    jockey_id = Column(String(5), ForeignKey("dim_jockey.jockey_id"))
    trainer_id = Column(String(5), ForeignKey("dim_trainer.trainer_id"))
    
    weight_carried = Column(Numeric(4, 1))  # 斤量
    horse_weight = Column(SmallInteger)  # 馬体重
    weight_diff = Column(SmallInteger)  # 馬体重増減
    
    # メタ
    created_at = Column(DateTime, default=datetime.now)
    
    # リレーション
    race = relationship("FactRace", back_populates="entries")
    
    __table_args__ = (
        Index("ix_fact_entry_race_horse", "race_id", "horse_no", unique=True),
    )


class FactResult(Base):
    """レース結果ファクト"""
    __tablename__ = "fact_result"
    
    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    horse_id = Column(String(10), ForeignKey("dim_horse.horse_id"))
    horse_no = Column(SmallInteger, nullable=False)
    
    finish_pos = Column(SmallInteger)  # 着順
    time = Column(String(10))  # 走破タイム（文字列）
    time_sec = Column(Numeric(6, 1))  # 走破タイム（秒）
    margin = Column(String(10))  # 着差
    last_3f = Column(Numeric(4, 1))  # 上がり3F
    
    # 単勝情報（結果時点）
    odds = Column(Numeric(6, 1))
    popularity = Column(SmallInteger)

    # ===== C4: 通過順（SE由来: 1C..4Cの順位）=====
    pos_1c = Column(SmallInteger)
    pos_2c = Column(SmallInteger)
    pos_3c = Column(SmallInteger)
    pos_4c = Column(SmallInteger)
    
    # メタ
    created_at = Column(DateTime, default=datetime.now)
    
    # リレーション
    race = relationship("FactRace", back_populates="results")
    
    __table_args__ = (
        Index("ix_fact_result_race_horse", "race_id", "horse_no", unique=True),
    )


class FactRefundHorse(Base):
    """
    返還対象の馬番（開催中止/出走取消等）

    HRの「返還馬番表（馬番01..28）」から抽出。
    3連系を含む多くの券種で「この馬を含む買い目は返還」という扱いになるため、
    決済時に参照しやすい形で正規化しておく。
    """

    __tablename__ = "fact_refund_horse"

    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    horse_no = Column(SmallInteger, nullable=False)

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_fact_refund_horse_race", "race_id"),
        Index("ix_fact_refund_horse_unique", "race_id", "horse_no", unique=True),
    )


class FactPayoutTrio(Base):
    """3連複 払戻（HR）"""

    __tablename__ = "fact_payout_trio"

    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)

    horse_no_1 = Column(SmallInteger, nullable=False)
    horse_no_2 = Column(SmallInteger, nullable=False)
    horse_no_3 = Column(SmallInteger, nullable=False)

    payout_yen = Column(Integer)  # 100円あたり
    popularity = Column(SmallInteger)

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_fact_payout_trio_race", "race_id"),
        Index(
            "ix_fact_payout_trio_unique",
            "race_id",
            "horse_no_1",
            "horse_no_2",
            "horse_no_3",
            unique=True,
        ),
        CheckConstraint("horse_no_1 < horse_no_2", name="ck_fact_trio_order_12"),
        CheckConstraint("horse_no_2 < horse_no_3", name="ck_fact_trio_order_23"),
    )


class FactPayoutTrifecta(Base):
    """3連単 払戻（HR）"""

    __tablename__ = "fact_payout_trifecta"

    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)

    first_no = Column(SmallInteger, nullable=False)
    second_no = Column(SmallInteger, nullable=False)
    third_no = Column(SmallInteger, nullable=False)

    payout_yen = Column(Integer)  # 100円あたり
    popularity = Column(SmallInteger)

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_fact_payout_trifecta_race", "race_id"),
        Index(
            "ix_fact_payout_trifecta_unique",
            "race_id",
            "first_no",
            "second_no",
            "third_no",
            unique=True,
        ),
    )


# =============================================================================
# Odds Tables (時系列)
# =============================================================================

class OddsTsWin(Base):
    """
    単勝時系列オッズ（5-10分間隔）
    
    asof_time: レコード内の「発表月日時分」から生成（JST）
               ※取得時刻ではなくJV-Data内の時刻を使用
    data_kubun: 1=中間, 2=前日売最終, 3=最終, 4=確定
    """
    __tablename__ = "odds_ts_win"
    
    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    horse_no = Column(SmallInteger, nullable=False)
    
    asof_time = Column(DateTime, nullable=False)
    data_kubun = Column(String(1), nullable=False)
    
    odds = Column(Numeric(6, 1))
    popularity = Column(SmallInteger)
    total_sales = Column(BigInteger)  # 票数合計（誤差あり前提）
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("ix_odds_ts_win_race_asof", "race_id", "asof_time"),
        Index("ix_odds_ts_win_unique", "race_id", "horse_no", "asof_time", 
              "data_kubun", unique=True),
    )


class OddsTsPlace(Base):
    """
    複勝時系列オッズ
    """
    __tablename__ = "odds_ts_place"
    
    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    horse_no = Column(SmallInteger, nullable=False)
    
    asof_time = Column(DateTime, nullable=False)
    data_kubun = Column(String(1), nullable=False)
    
    odds_low = Column(Numeric(6, 1))
    odds_high = Column(Numeric(6, 1))
    popularity = Column(SmallInteger)
    total_sales = Column(BigInteger)
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("ix_odds_ts_place_race_asof", "race_id", "asof_time"),
        Index("ix_odds_ts_place_unique", "race_id", "horse_no", "asof_time",
              "data_kubun", unique=True),
    )


class OddsTsQuinella(Base):
    """
    馬連時系列オッズ（O2から抽出）
    """
    __tablename__ = "odds_ts_quinella"
    
    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    
    # 馬番ペア（小さい方をhorse_no_1に正規化）
    horse_no_1 = Column(SmallInteger, nullable=False)
    horse_no_2 = Column(SmallInteger, nullable=False)
    
    asof_time = Column(DateTime, nullable=False)
    data_kubun = Column(String(1), nullable=False)
    
    odds = Column(Numeric(8, 1))
    popularity = Column(SmallInteger)
    total_sales = Column(BigInteger)
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("ix_odds_ts_quinella_race_asof", "race_id", "asof_time"),
        Index("ix_odds_ts_quinella_unique", "race_id", "horse_no_1", "horse_no_2", 
              "asof_time", "data_kubun", unique=True),
        # 馬番ペアは常に horse_no_1 < horse_no_2 に正規化
        CheckConstraint("horse_no_1 < horse_no_2", name="ck_quinella_order"),
    )


# =============================================================================
# Odds Tables (速報スナップショット: 三連複/三連単)
# =============================================================================

class OddsRtSnapshot(Base):
    """
    速報オッズ（JVRTOpen）を「スナップショット単位」でまとめる親テーブル。

    目的:
      - buy/close の2回収集を安全に保存し、後から時刻条件で抽出しやすくする
      - 同一 asof_time の再取得（fetched_at違い）も区別できる

    注意:
      - asof_time は JV-Data 内の発表時刻（HappyoTime）由来
      - fetched_at は JSONL 生成時刻（_meta.ingested_at）由来
    """

    __tablename__ = "odds_rt_snapshot"

    id = Column(BigInteger, primary_key=True)
    ticket_type = Column(String(20), nullable=False)  # trio / trifecta
    data_spec = Column(String(4), nullable=False)     # 0B35 / 0B36
    record_type = Column(String(2))                   # O5 / O6（想定。揺れた場合も保存）

    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    asof_time = Column(DateTime, nullable=False)
    fetched_at = Column(DateTime, nullable=False)
    data_kubun = Column(String(1), nullable=False)
    sale_flag = Column(String(1))

    total_sales = Column(BigInteger)
    n_rows = Column(Integer)
    note = Column(Text)

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_odds_rt_snapshot_race_asof", "race_id", "asof_time"),
        Index("ix_odds_rt_snapshot_ticket_race_asof", "ticket_type", "race_id", "asof_time"),
        Index(
            "ix_odds_rt_snapshot_unique",
            "ticket_type",
            "race_id",
            "asof_time",
            "data_kubun",
            "fetched_at",
            unique=True,
        ),
    )


class OddsRtTrio(Base):
    """
    速報 3連複オッズ（スナップショット子テーブル）
    """

    __tablename__ = "odds_rt_trio"

    id = Column(BigInteger, primary_key=True)
    snapshot_id = Column(BigInteger, ForeignKey("odds_rt_snapshot.id"), nullable=False)

    horse_no_1 = Column(SmallInteger, nullable=False)
    horse_no_2 = Column(SmallInteger, nullable=False)
    horse_no_3 = Column(SmallInteger, nullable=False)

    odds = Column(Numeric(8, 1))
    popularity = Column(SmallInteger)
    odds_status = Column(String(20))  # ok/refund/no_sale/special_payout/not_registered/unknown

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_odds_rt_trio_snapshot", "snapshot_id"),
        Index(
            "ix_odds_rt_trio_unique",
            "snapshot_id",
            "horse_no_1",
            "horse_no_2",
            "horse_no_3",
            unique=True,
        ),
        CheckConstraint("horse_no_1 < horse_no_2", name="ck_trio_order_12"),
        CheckConstraint("horse_no_2 < horse_no_3", name="ck_trio_order_23"),
    )


class OddsRtTrifecta(Base):
    """
    速報 3連単オッズ（スナップショット子テーブル）
    """

    __tablename__ = "odds_rt_trifecta"

    id = Column(BigInteger, primary_key=True)
    snapshot_id = Column(BigInteger, ForeignKey("odds_rt_snapshot.id"), nullable=False)

    first_no = Column(SmallInteger, nullable=False)
    second_no = Column(SmallInteger, nullable=False)
    third_no = Column(SmallInteger, nullable=False)

    odds = Column(Numeric(9, 1))
    popularity = Column(SmallInteger)
    odds_status = Column(String(20))

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_odds_rt_trifecta_snapshot", "snapshot_id"),
        Index(
            "ix_odds_rt_trifecta_unique",
            "snapshot_id",
            "first_no",
            "second_no",
            "third_no",
            unique=True,
        ),
    )


# =============================================================================
# Features / Predictions
# =============================================================================

class Features(Base):
    """特徴量テーブル"""
    __tablename__ = "features"
    
    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    horse_id = Column(String(10))
    
    feature_version = Column(String(20), nullable=False)
    asof_time = Column(DateTime, nullable=False)  # 必須：リーク防止
    
    payload = Column(JSONB)  # 特徴量データ
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("ix_features_race_version", "race_id", "feature_version"),
    )


class Predictions(Base):
    """予測結果テーブル"""
    __tablename__ = "predictions"
    
    id = Column(BigInteger, primary_key=True)
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    horse_id = Column(String(10))
    
    model_version = Column(String(50), nullable=False)
    asof_time = Column(DateTime, nullable=False)  # 必須：購入時点再現
    
    p_win = Column(Numeric(5, 4))  # 勝利確率
    p_place = Column(Numeric(5, 4))  # 複勝確率
    
    calibration_meta = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("ix_predictions_race_model", "race_id", "model_version"),
    )


# =============================================================================
# Bets / PnL
# =============================================================================

class BetSignal(Base):
    """買い目シグナル"""
    __tablename__ = "bet_signal"
    
    signal_id = Column(String(50), primary_key=True)
    run_id = Column(String(50), nullable=False)  # バッチ実行ID
    
    race_id = Column(String(16), ForeignKey("fact_race.race_id"), nullable=False)
    ticket_type = Column(String(20), nullable=False)  # win, place, quinella, etc.
    
    selection = Column(JSONB)  # 選択（馬番等）
    asof_time = Column(DateTime, nullable=False)
    
    p_hat = Column(Numeric(5, 4))  # 予測確率
    odds_snapshot = Column(Numeric(8, 1))  # 購入時オッズ
    ev = Column(Numeric(6, 4))  # 期待値
    stake_yen = Column(Integer)  # 賭け金
    
    reason = Column(JSONB)  # 買い理由
    
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("ix_bet_signal_race", "race_id"),
        Index("ix_bet_signal_run", "run_id"),
    )


class BetSettlement(Base):
    """買い目決済結果"""
    __tablename__ = "bet_settlement"
    
    id = Column(BigInteger, primary_key=True)
    signal_id = Column(String(50), ForeignKey("bet_signal.signal_id"), nullable=False)
    
    result_status = Column(String(20))  # win, lose, void
    payout_yen = Column(Integer)
    profit_yen = Column(Integer)
    
    settled_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index("ix_bet_settlement_signal", "signal_id", unique=True),
    )


