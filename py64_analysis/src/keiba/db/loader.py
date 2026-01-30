"""
JSONL → PostgreSQL ローダー

py32_fetcherが出力したJSONLファイルをDBに投入する

運用上の注意 (FK制約):
  - odds_rt_snapshot は fact_race.race_id へのFKを持っています
  - そのため、0B35/0B36（三連複/三連単オッズ）をロードする前に、
    必ず対象レースの RA レコード（fact_race）を先に投入してください
  - 順序を誤ると FK 違反でバッチ全体が rollback されます

推奨ロード順序:
  1. RA（レース基本情報）→ fact_race
  2. SE（出馬表）→ fact_entry
  3. HR（払戻）→ fact_result, fact_payout_*, fact_refund_horse
  4. 0B35/0B36（三連系オッズ）→ odds_rt_snapshot, odds_rt_trio, odds_rt_trifecta
"""
import json
import logging
import re
from pathlib import Path
from datetime import datetime, time as dtime
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import insert

from .models import (
    Base, FactRace, FactEntry, FactResult,
    OddsTsWin, OddsTsPlace, OddsTsQuinella,
    OddsRtSnapshot, OddsRtTrio, OddsRtTrifecta,
    FactRefundHorse, FactPayoutTrio, FactPayoutTrifecta,
    DimHorse, DimJockey, DimTrainer, DimTrack
)
from .pace_utils import sum_first_n, sum_last_n, lap_stats, pace_diff
from ..config import get_config

logger = logging.getLogger(__name__)

# race_id妥当性チェック用正規表現（16桁数字）
RACE_ID_PATTERN = re.compile(r"^\d{16}$")


def get_engine():
    """DBエンジンを取得"""
    config = get_config()
    return create_engine(config.database.url)


def get_session() -> Session:
    """セッションを取得"""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def read_jsonl(file_path: Path) -> Iterator[dict]:
    """JSONLファイルを読み込む"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_race_records(session: Session, records: list[dict]) -> int:
    """RAレコードをDBに投入"""
    count = 0
    for record in records:
        if record.get("record_type") != "RA":
            continue
        if "_raw" in record:
            continue  # パース失敗したレコードはスキップ
        
        race_id = record.get("race_id")
        # race_id妥当性チェック（16桁数字）
        if not race_id or not RACE_ID_PATTERN.match(race_id):
            continue
        
        # race_idから日付を抽出
        try:
            race_date = datetime.strptime(race_id[:8], "%Y%m%d").date()
        except ValueError:
            continue
        
        # ★必須項目の補完（race_idから抽出可能）
        # race_id構造: YYYYMMDD + track_code(2桁) + 回次(2桁) + 日次(2桁) + race_no(2桁)
        race_no = record.get("race_no")
        if race_no is None:
            try:
                race_no = int(race_id[14:16])  # race_id の15-16桁目がレース番号
            except (ValueError, IndexError):
                continue  # 補完もできなければスキップ
        
        track_code = record.get("track_code")
        if not track_code:
            try:
                track_code = race_id[8:10]  # race_id の9-10桁目が場コード
            except IndexError:
                pass
        
        # start_time（HHMM形式）をdatetime.timeに変換
        st = record.get("start_time")
        start_time = None
        if st and isinstance(st, str) and len(st) == 4 and st.isdigit():
            try:
                start_time = dtime(int(st[:2]), int(st[2:]))
            except ValueError:
                pass
        
        # ★FK制約対応: DimTrack を事前に upsert
        if track_code:
            stmt_track = insert(DimTrack).values(
                track_code=track_code,
                name=None,  # 名前は後で更新可能
            ).on_conflict_do_nothing()
            session.execute(stmt_track)
        
        lap_times = record.get("lap_times_200m")
        pace_first3f = record.get("pace_first3f")
        if pace_first3f is None:
            pace_first3f = sum_first_n(lap_times, 3)
        pace_first4f = record.get("pace_first4f")
        pace_last3f = record.get("pace_last3f")
        if pace_last3f is None:
            pace_last3f = sum_last_n(lap_times, 3)
        pace_last4f = record.get("pace_last4f")
        pace_diff_sec = pace_diff(pace_first3f, pace_last3f)
        lap_mean_sec, lap_std_sec, lap_slope = lap_stats(lap_times)

        stmt = insert(FactRace).values(
            race_id=race_id,
            date=race_date,
            track_code=track_code,
            race_no=race_no,
            start_time=start_time,
            surface=record.get("track_cd", "")[:1] if record.get("track_cd") else None,
            distance=record.get("distance"),
            track_cd=record.get("track_cd"),
            going_turf=record.get("going_turf"),
            going_dirt=record.get("going_dirt"),
            weather=record.get("weather_cd"),
            field_size=record.get("field_size"),
            # C4: pace（RA由来）
            lap_times_200m=record.get("lap_times_200m"),
            pace_first3f=pace_first3f,
            pace_first4f=pace_first4f,
            pace_last3f=pace_last3f,
            pace_last4f=pace_last4f,
            pace_diff_sec=pace_diff_sec,
            lap_mean_sec=lap_mean_sec,
            lap_std_sec=lap_std_sec,
            lap_slope=lap_slope,
        ).on_conflict_do_update(
            index_elements=["race_id"],
            set_={
                "field_size": record.get("field_size"),
                "start_time": start_time,
                # C4: pace（RA由来）も上書き
                "lap_times_200m": record.get("lap_times_200m"),
                "pace_first3f": pace_first3f,
                "pace_first4f": pace_first4f,
                "pace_last3f": pace_last3f,
                "pace_last4f": pace_last4f,
                "pace_diff_sec": pace_diff_sec,
                "lap_mean_sec": lap_mean_sec,
                "lap_std_sec": lap_std_sec,
                "lap_slope": lap_slope,
                "updated_at": datetime.now(),
            }
        )
        session.execute(stmt)
        count += 1
    
    return count


def load_entry_records(session: Session, records: list[dict]) -> int:
    """SEレコードをDBに投入"""
    count = 0
    for record in records:
        if record.get("record_type") != "SE":
            continue
        if "_raw" in record:
            continue
        
        race_id = record.get("race_id")
        horse_no = record.get("horse_no")
        # race_id妥当性チェック
        if not race_id or not RACE_ID_PATTERN.match(race_id) or not horse_no:
            continue
        
        # 馬マスタ（upsert）
        horse_id = record.get("horse_id")
        if horse_id:
            stmt = insert(DimHorse).values(
                horse_id=horse_id,
                name=record.get("horse_name"),
                sex=record.get("sex_cd"),
            ).on_conflict_do_nothing()
            session.execute(stmt)
        
        # ★FK制約対応: DimJockey を事前に upsert
        jockey_id = record.get("jockey_id")
        if jockey_id:
            stmt_jockey = insert(DimJockey).values(
                jockey_id=jockey_id,
                name=record.get("jockey_name"),  # あれば入れる
            ).on_conflict_do_nothing()
            session.execute(stmt_jockey)
        
        # ★FK制約対応: DimTrainer を事前に upsert
        trainer_id = record.get("trainer_id")
        if trainer_id:
            stmt_trainer = insert(DimTrainer).values(
                trainer_id=trainer_id,
                name=record.get("trainer_name"),  # あれば入れる
            ).on_conflict_do_nothing()
            session.execute(stmt_trainer)
        
        # 出走馬（upsert - 後から確定情報が来る場合があるので更新）
        stmt = insert(FactEntry).values(
            race_id=race_id,
            horse_id=horse_id,
            horse_no=horse_no,
            frame_no=record.get("frame_no"),
            jockey_id=record.get("jockey_id"),
            weight_carried=record.get("weight_carried"),
            horse_weight=record.get("horse_weight"),
        ).on_conflict_do_update(
            index_elements=["race_id", "horse_no"],
            set_={
                "horse_weight": record.get("horse_weight"),
                "weight_carried": record.get("weight_carried"),
            }
        )
        session.execute(stmt)
        
        # 結果もSEに含まれている場合（upsert - 確定情報で更新）
        finish_pos = record.get("finish_pos")
        if finish_pos:
            stmt = insert(FactResult).values(
                race_id=race_id,
                horse_id=horse_id,
                horse_no=horse_no,
                finish_pos=finish_pos,
                time=record.get("time"),
                last_3f=record.get("last_3f"),
                odds=record.get("odds"),
                popularity=record.get("popularity"),
                # C4: 通過順（SE由来）
                pos_1c=record.get("pos_1c"),
                pos_2c=record.get("pos_2c"),
                pos_3c=record.get("pos_3c"),
                pos_4c=record.get("pos_4c"),
            ).on_conflict_do_update(
                index_elements=["race_id", "horse_no"],
                set_={
                    "finish_pos": finish_pos,
                    "time": record.get("time"),
                    "last_3f": record.get("last_3f"),
                    "odds": record.get("odds"),
                    "popularity": record.get("popularity"),
                    "pos_1c": record.get("pos_1c"),
                    "pos_2c": record.get("pos_2c"),
                    "pos_3c": record.get("pos_3c"),
                    "pos_4c": record.get("pos_4c"),
                }
            )
            session.execute(stmt)
        
        count += 1
    
    return count


def load_odds_win_records(session: Session, records: list[dict]) -> int:
    """O1レコード（単勝オッズ）をDBに投入"""
    # ★重要: 1行ずつINSERTすると非常に遅い（時系列オッズは数百万行になりやすい）
    # ここでは executemany（まとめINSERT）で投入する
    count = 0
    win_rows: list[dict] = []
    place_rows: list[dict] = []

    for record in records:
        if record.get("record_type") != "O1":
            continue
        if "_raw" in record:
            continue

        race_id = record.get("race_id")
        asof_time = record.get("asof_time")
        data_kubun = record.get("data_kubun")

        # race_id妥当性チェック
        if not race_id or not RACE_ID_PATTERN.match(race_id) or not asof_time:
            continue

        # ★data_kubun バリデーション（DBは NOT NULL）
        if data_kubun not in ("1", "2", "3", "4"):
            logger.warning(
                f"O1: unexpected data_kubun='{data_kubun}' "
                f"(race_id={race_id}, data_spec={record.get('data_spec')})"
            )
            continue

        # asof_timeがstrの場合はdatetimeに変換
        if isinstance(asof_time, str):
            try:
                asof_time = datetime.fromisoformat(asof_time)
            except ValueError:
                continue

        total_sales_win = record.get("total_sales_win")
        for odds_row in record.get("win_odds", []):
            win_rows.append(
                {
                    "race_id": race_id,
                    "horse_no": odds_row["horse_no"],
                    "asof_time": asof_time,
                    "data_kubun": data_kubun,
                    "odds": odds_row.get("odds"),
                    "popularity": odds_row.get("popularity"),
                    "total_sales": total_sales_win,
                }
            )
            count += 1

        total_sales_place = record.get("total_sales_place")
        for odds_row in record.get("place_odds", []):
            place_rows.append(
                {
                    "race_id": race_id,
                    "horse_no": odds_row["horse_no"],
                    "asof_time": asof_time,
                    "data_kubun": data_kubun,
                    "odds_low": odds_row.get("odds_low"),
                    "odds_high": odds_row.get("odds_high"),
                    "popularity": odds_row.get("popularity"),
                    "total_sales": total_sales_place,
                }
            )
            count += 1

    if win_rows:
        stmt = insert(OddsTsWin).on_conflict_do_nothing()
        session.execute(stmt, win_rows)
    if place_rows:
        stmt = insert(OddsTsPlace).on_conflict_do_nothing()
        session.execute(stmt, place_rows)

    return count


def load_odds_quinella_records(session: Session, records: list[dict]) -> int:
    """O2レコード（馬連オッズ）をDBに投入"""
    count = 0
    rows: list[dict] = []
    for record in records:
        if record.get("record_type") != "O2":
            continue
        if "_raw" in record:
            continue

        race_id = record.get("race_id")
        asof_time = record.get("asof_time")
        data_kubun = record.get("data_kubun")

        if not race_id or not RACE_ID_PATTERN.match(race_id) or not asof_time:
            continue

        if data_kubun not in ("1", "2", "3", "4"):
            logger.warning(
                f"O2: unexpected data_kubun='{data_kubun}' "
                f"(race_id={race_id}, data_spec={record.get('data_spec')})"
            )
            continue

        if isinstance(asof_time, str):
            try:
                asof_time = datetime.fromisoformat(asof_time)
            except ValueError:
                continue

        total_sales = record.get("total_sales")
        for odds_row in record.get("quinella_odds", []):
            rows.append(
                {
                    "race_id": race_id,
                    "horse_no_1": odds_row["horse_no_1"],
                    "horse_no_2": odds_row["horse_no_2"],
                    "asof_time": asof_time,
                    "data_kubun": data_kubun,
                    "odds": odds_row.get("odds"),
                    "popularity": odds_row.get("popularity"),
                    "total_sales": total_sales,
                }
            )
            count += 1

    if rows:
        stmt = insert(OddsTsQuinella).on_conflict_do_nothing()
        session.execute(stmt, rows)

    return count


def _parse_dt(v) -> Optional[datetime]:
    """ISO文字列 or datetime を datetime に揃える（naive, JST前提）。"""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return None
    return None


def load_odds_trio_realtime_records(session: Session, records: list[dict]) -> int:
    """
    O5（3連複速報オッズ）を odds_rt_snapshot / odds_rt_trio に投入する。

    重要:
      - 1レコード = 1スナップショット（asof_time単位）
      - snapshotは upsert で id を確実に取得し、子は bulk insert（冪等）
    """
    count = 0
    for record in records:
        if record.get("record_type") != "O5":
            continue
        if "_raw" in record:
            continue

        race_id = record.get("race_id")
        asof_time = _parse_dt(record.get("asof_time"))
        data_kubun = record.get("data_kubun")
        meta = record.get("_meta") or {}
        fetched_at = _parse_dt(meta.get("ingested_at"))

        if not race_id or not RACE_ID_PATTERN.match(race_id) or not asof_time or not fetched_at:
            continue
        if data_kubun is None:
            continue
        data_kubun = str(data_kubun).strip()[:1]
        if not data_kubun:
            continue

        # snapshot upsert（unique: ticket_type,race_id,asof_time,data_kubun,fetched_at）
        snap_values = {
            "ticket_type": "trio",
            "data_spec": str(meta.get("data_spec") or ""),
            "record_type": str(record.get("record_type") or ""),
            "race_id": race_id,
            "asof_time": asof_time,
            "fetched_at": fetched_at,
            "data_kubun": data_kubun,
            "sale_flag": str(record.get("sale_flag") or "").strip()[:1] or None,
            "total_sales": record.get("total_sales"),
            "n_rows": len(record.get("trio_odds") or []),
        }

        stmt = (
            insert(OddsRtSnapshot)
            .values(**snap_values)
            .on_conflict_do_update(
                index_elements=["ticket_type", "race_id", "asof_time", "data_kubun", "fetched_at"],
                set_={
                    "data_spec": snap_values["data_spec"],
                    "record_type": snap_values["record_type"],
                    "sale_flag": snap_values["sale_flag"],
                    "total_sales": snap_values["total_sales"],
                    "n_rows": snap_values["n_rows"],
                    "note": None,
                },
            )
            .returning(OddsRtSnapshot.id)
        )
        snapshot_id = session.execute(stmt).scalar_one()

        rows: list[dict] = []
        for o in record.get("trio_odds") or []:
            try:
                rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "horse_no_1": int(o["horse_no_1"]),
                        "horse_no_2": int(o["horse_no_2"]),
                        "horse_no_3": int(o["horse_no_3"]),
                        "odds": o.get("odds"),
                        "popularity": o.get("popularity"),
                        "odds_status": o.get("odds_status"),
                    }
                )
            except Exception:
                continue

        if rows:
            stmt2 = insert(OddsRtTrio).on_conflict_do_nothing()
            session.execute(stmt2, rows)
            count += len(rows)

    return count


def load_odds_trifecta_realtime_records(session: Session, records: list[dict]) -> int:
    """
    O6（3連単速報オッズ）を odds_rt_snapshot / odds_rt_trifecta に投入する。
    """
    count = 0
    for record in records:
        if record.get("record_type") != "O6":
            continue
        if "_raw" in record:
            continue

        race_id = record.get("race_id")
        asof_time = _parse_dt(record.get("asof_time"))
        data_kubun = record.get("data_kubun")
        meta = record.get("_meta") or {}
        fetched_at = _parse_dt(meta.get("ingested_at"))

        if not race_id or not RACE_ID_PATTERN.match(race_id) or not asof_time or not fetched_at:
            continue
        if data_kubun is None:
            continue
        data_kubun = str(data_kubun).strip()[:1]
        if not data_kubun:
            continue

        snap_values = {
            "ticket_type": "trifecta",
            "data_spec": str(meta.get("data_spec") or ""),
            "record_type": str(record.get("record_type") or ""),
            "race_id": race_id,
            "asof_time": asof_time,
            "fetched_at": fetched_at,
            "data_kubun": data_kubun,
            "sale_flag": str(record.get("sale_flag") or "").strip()[:1] or None,
            "total_sales": record.get("total_sales"),
            "n_rows": len(record.get("trifecta_odds") or []),
        }

        stmt = (
            insert(OddsRtSnapshot)
            .values(**snap_values)
            .on_conflict_do_update(
                index_elements=["ticket_type", "race_id", "asof_time", "data_kubun", "fetched_at"],
                set_={
                    "data_spec": snap_values["data_spec"],
                    "record_type": snap_values["record_type"],
                    "sale_flag": snap_values["sale_flag"],
                    "total_sales": snap_values["total_sales"],
                    "n_rows": snap_values["n_rows"],
                    "note": None,
                },
            )
            .returning(OddsRtSnapshot.id)
        )
        snapshot_id = session.execute(stmt).scalar_one()

        rows: list[dict] = []
        for o in record.get("trifecta_odds") or []:
            try:
                rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "first_no": int(o["first_no"]),
                        "second_no": int(o["second_no"]),
                        "third_no": int(o["third_no"]),
                        "odds": o.get("odds"),
                        "popularity": o.get("popularity"),
                        "odds_status": o.get("odds_status"),
                    }
                )
            except Exception:
                continue

        if rows:
            stmt2 = insert(OddsRtTrifecta).on_conflict_do_nothing()
            session.execute(stmt2, rows)
            count += len(rows)

    return count


def load_hr_exotics_payout_records(session: Session, records: list[dict]) -> int:
    """
    HR（払戻）から 3連複/3連単の払戻と返還馬番をDBに投入する。
    """
    count = 0

    refund_rows: list[dict] = []
    trio_rows: list[dict] = []
    trifecta_rows: list[dict] = []

    for record in records:
        if record.get("record_type") != "HR":
            continue
        if "_raw" in record:
            continue

        race_id = record.get("race_id")
        if not race_id or not RACE_ID_PATTERN.match(race_id):
            continue

        # 返還馬番（馬番表）
        for h in record.get("refund_horse_nos") or []:
            try:
                refund_rows.append({"race_id": race_id, "horse_no": int(h)})
            except Exception:
                continue

        # 3連複払戻（最大3件）
        for p in record.get("trio_payouts") or []:
            try:
                trio_rows.append(
                    {
                        "race_id": race_id,
                        "horse_no_1": int(p["horse_no_1"]),
                        "horse_no_2": int(p["horse_no_2"]),
                        "horse_no_3": int(p["horse_no_3"]),
                        "payout_yen": p.get("payout_yen"),
                        "popularity": p.get("popularity"),
                    }
                )
            except Exception:
                continue

        # 3連単払戻（最大6件）
        for p in record.get("trifecta_payouts") or []:
            try:
                trifecta_rows.append(
                    {
                        "race_id": race_id,
                        "first_no": int(p["first_no"]),
                        "second_no": int(p["second_no"]),
                        "third_no": int(p["third_no"]),
                        "payout_yen": p.get("payout_yen"),
                        "popularity": p.get("popularity"),
                    }
                )
            except Exception:
                continue

    if refund_rows:
        stmt = insert(FactRefundHorse).on_conflict_do_nothing()
        session.execute(stmt, refund_rows)
        count += len(refund_rows)

    if trio_rows:
        stmt = insert(FactPayoutTrio).on_conflict_do_nothing()
        session.execute(stmt, trio_rows)
        count += len(trio_rows)

    if trifecta_rows:
        stmt = insert(FactPayoutTrifecta).on_conflict_do_nothing()
        session.execute(stmt, trifecta_rows)
        count += len(trifecta_rows)

    return count


def load_jsonl_file(file_path: Path, batch_size: int = 1000) -> dict:
    """
    JSONLファイルをDBに投入（ストリーミング処理）
    
    Args:
        file_path: JSONLファイルパス
        batch_size: コミット単位（メモリ効率のため分割）
    
    Returns:
        {"ra": int, "se": int, "o1": int, "o2": int, "errors": int}
    """
    from .ingestion_log import log_ingestion
    
    session = get_session()
    file_size = file_path.stat().st_size
    
    result = {"ra": 0, "se": 0, "hr": 0, "o1": 0, "o2": 0, "o5": 0, "o6": 0, "errors": 0}
    record_count = 0
    batch = []
    
    # data_specをファイル名から推測
    data_spec = file_path.stem.split("_")[0] if "_" in file_path.stem else "UNKNOWN"
    
    try:
        # ストリーミング処理（メモリ効率改善）
        for record in read_jsonl(file_path):
            record_count += 1
            batch.append(record)
            
            # バッチサイズに達したら処理
            if len(batch) >= batch_size:
                batch_result = _process_batch(session, batch)
                for key in result:
                    result[key] += batch_result.get(key, 0)
                session.commit()
                batch = []
        
        # 残りのバッチを処理
        if batch:
            batch_result = _process_batch(session, batch)
            for key in result:
                result[key] += batch_result.get(key, 0)
            session.commit()
        
        logger.info(f"Load complete: {result} from {file_path}")
        
        # Ingestionログに記録
        log_ingestion(
            session=session,
            file_path=str(file_path),
            data_spec=data_spec,
            file_size=file_size,
            record_count=record_count,
            result=result,
            status="success",
        )
        
        return result
        
    except Exception as e:
        session.rollback()
        logger.error(f"Load failed: {e}")
        
        # エラーをログに記録
        try:
            log_ingestion(
                session=session,
                file_path=str(file_path),
                data_spec=data_spec,
                file_size=file_size,
                record_count=record_count,
                result=result,
                status="error",
                error_message=str(e),
            )
        except Exception:
            pass  # ログ記録失敗は無視
        
        raise
    finally:
        session.close()


def _process_batch(session: Session, records: list[dict]) -> dict:
    """バッチ単位でレコードを処理"""
    result = {"ra": 0, "se": 0, "hr": 0, "o1": 0, "o2": 0, "o5": 0, "o6": 0, "errors": 0}
    
    try:
        result["ra"] = load_race_records(session, records)
        result["se"] = load_entry_records(session, records)
        result["hr"] = load_hr_exotics_payout_records(session, records)
        result["o1"] = load_odds_win_records(session, records)
        result["o2"] = load_odds_quinella_records(session, records)
        # exotics
        result["o5"] = load_odds_trio_realtime_records(session, records)
        result["o6"] = load_odds_trifecta_realtime_records(session, records)
    except Exception as e:
        # ★重要: DB例外が発生するとセッションが"rollback待ち"になる
        #   rollbackしないと次のcommitで落ちる
        logger.warning(f"Batch processing error: {e}")
        session.rollback()
        result["errors"] += 1
    
    return result


def load_all_jsonl_files(input_dir: Path, batch_size: int = 1000) -> dict:
    """
    ディレクトリ内の全JSONLファイルをDBに投入
    
    Args:
        input_dir: 入力ディレクトリ
        batch_size: コミット単位
    
    Returns:
        合計結果
    """
    total = {"ra": 0, "se": 0, "hr": 0, "o1": 0, "o2": 0, "o5": 0, "o6": 0, "errors": 0, "files": 0}
    
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} JSONL files in {input_dir}")
    
    for file_path in jsonl_files:
        try:
            result = load_jsonl_file(file_path, batch_size)
            for key in ["ra", "se", "hr", "o1", "o2", "o5", "o6", "errors"]:
                total[key] += result.get(key, 0)
            total["files"] += 1
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            total["errors"] += 1
    
    logger.info(f"Total load complete: {total}")
    return total
