"""
C4-0バックフィル: 既存のJSONLファイルを再パースしてペース情報をDBに投入

既存のJSONLファイルを拡張パーサーで再パースし、fact_race/fact_resultのペース関連カラムを更新する
"""
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from keiba.db.loader import get_session
from keiba.db.pace_utils import sum_first_n, sum_last_n, lap_stats, pace_diff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_fact_race_pace(session, records: list[dict]) -> int:
    """
    fact_raceのペース関連カラムを更新
    """
    count = 0
    for record in records:
        if record.get("record_type") != "RA":
            continue
        
        race_id = record.get("race_id")
        if not race_id:
            continue
        
        # ペース情報を取得
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
        
        # JSONBに変換（lap_times_200m）
        lap_times_jsonb = None
        if lap_times and isinstance(lap_times, list):
            # Noneを除外
            lap_times_clean = [x for x in lap_times if x is not None]
            if lap_times_clean:
                lap_times_jsonb = json.dumps(lap_times_clean)
        
        # UPDATE（既存レコードのみ）
        if lap_times_jsonb:
            stmt = text("""
                UPDATE fact_race
                SET 
                  lap_times_200m = CAST(:lap_times AS jsonb),
                  pace_first3f = COALESCE(:pace_first3f, pace_first3f),
                  pace_first4f = COALESCE(:pace_first4f, pace_first4f),
                  pace_last3f = COALESCE(:pace_last3f, pace_last3f),
                  pace_last4f = COALESCE(:pace_last4f, pace_last4f),
                  pace_diff_sec = COALESCE(:pace_diff_sec, pace_diff_sec),
                  lap_mean_sec = COALESCE(:lap_mean_sec, lap_mean_sec),
                  lap_std_sec = COALESCE(:lap_std_sec, lap_std_sec),
                  lap_slope = COALESCE(:lap_slope, lap_slope),
                  updated_at = CURRENT_TIMESTAMP
                WHERE race_id = :race_id
            """)
            result = session.execute(
                stmt,
                {
                    "race_id": race_id,
                    "lap_times": lap_times_jsonb,
                    "pace_first3f": pace_first3f,
                    "pace_first4f": pace_first4f,
                    "pace_last3f": pace_last3f,
                    "pace_last4f": pace_last4f,
                    "pace_diff_sec": pace_diff_sec,
                    "lap_mean_sec": lap_mean_sec,
                    "lap_std_sec": lap_std_sec,
                    "lap_slope": lap_slope,
                },
            )
        else:
            stmt = text("""
                UPDATE fact_race
                SET 
                  pace_first3f = COALESCE(:pace_first3f, pace_first3f),
                  pace_first4f = COALESCE(:pace_first4f, pace_first4f),
                  pace_last3f = COALESCE(:pace_last3f, pace_last3f),
                  pace_last4f = COALESCE(:pace_last4f, pace_last4f),
                  pace_diff_sec = COALESCE(:pace_diff_sec, pace_diff_sec),
                  lap_mean_sec = COALESCE(:lap_mean_sec, lap_mean_sec),
                  lap_std_sec = COALESCE(:lap_std_sec, lap_std_sec),
                  lap_slope = COALESCE(:lap_slope, lap_slope),
                  updated_at = CURRENT_TIMESTAMP
                WHERE race_id = :race_id
            """)
            result = session.execute(
                stmt,
                {
                    "race_id": race_id,
                    "pace_first3f": pace_first3f,
                    "pace_first4f": pace_first4f,
                    "pace_last3f": pace_last3f,
                    "pace_last4f": pace_last4f,
                    "pace_diff_sec": pace_diff_sec,
                    "lap_mean_sec": lap_mean_sec,
                    "lap_std_sec": lap_std_sec,
                    "lap_slope": lap_slope,
                },
            )
        
        if result.rowcount > 0:
            count += 1
    
    return count


def update_fact_result_corner(session, records: list[dict]) -> int:
    """
    fact_resultのコーナー順位を更新
    """
    count = 0
    for record in records:
        if record.get("record_type") != "SE":
            continue
        
        race_id = record.get("race_id")
        horse_no = record.get("horse_no")
        if not race_id or not horse_no:
            continue
        
        # コーナー順位を取得
        pos_1c = record.get("pos_1c")
        pos_2c = record.get("pos_2c")
        pos_3c = record.get("pos_3c")
        pos_4c = record.get("pos_4c")
        
        # UPDATE
        stmt = text("""
            UPDATE fact_result
            SET 
              pos_1c = COALESCE(:pos_1c, pos_1c),
              pos_2c = COALESCE(:pos_2c, pos_2c),
              pos_3c = COALESCE(:pos_3c, pos_3c),
              pos_4c = COALESCE(:pos_4c, pos_4c)
            WHERE race_id = :race_id AND horse_no = :horse_no
        """)
        
        result = session.execute(
            stmt,
            {
                "race_id": race_id,
                "horse_no": horse_no,
                "pos_1c": pos_1c,
                "pos_2c": pos_2c,
                "pos_3c": pos_3c,
                "pos_4c": pos_4c,
            },
        )
        
        if result.rowcount > 0:
            count += 1
    
    return count


def backfill_from_jsonl(jsonl_path: Path, batch_size: int = 1000) -> dict:
    """
    JSONLファイルからペース情報を抽出してDBに投入
    """
    session = get_session()
    
    total_ra_updated = 0
    total_se_updated = 0
    batch = []
    
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    batch.append(record)
                    
                    if len(batch) >= batch_size:
                        ra_updated = update_fact_race_pace(session, batch)
                        se_updated = update_fact_result_corner(session, batch)
                        total_ra_updated += ra_updated
                        total_se_updated += se_updated
                        session.commit()
                        batch = []
                        
                        if line_num % (batch_size * 10) == 0:
                            logger.info(f"Processed {line_num:,} lines: RA={total_ra_updated}, SE={total_se_updated}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {line_num}: {e}")
                    continue
        
        # 残りのバッチを処理
        if batch:
            ra_updated = update_fact_race_pace(session, batch)
            se_updated = update_fact_result_corner(session, batch)
            total_ra_updated += ra_updated
            total_se_updated += se_updated
            session.commit()
        
        logger.info(f"Backfill complete: RA={total_ra_updated}, SE={total_se_updated}")
        
        return {
            "ra_updated": total_ra_updated,
            "se_updated": total_se_updated,
        }
    
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill pace fields from JSONL into DB")
    parser.add_argument("jsonl_file", type=Path, help="JSONL file to backfill")
    parser.add_argument("--lock-timeout-seconds", type=float, default=0)
    parser.add_argument("--lock-poll-seconds", type=float, default=0.25)
    parser.add_argument("--no-lock", action="store_true")
    args = parser.parse_args()

    jsonl_path = args.jsonl_file
    if not jsonl_path.exists():
        raise SystemExit(f"Error: File not found: {jsonl_path}")

    def _run() -> None:
        result = backfill_from_jsonl(jsonl_path)
        print(f"\nResult: {result}")

    if args.no_lock:
        print("DB write lock: DISABLED (--no-lock)")
        _run()
        return

    from keiba.utils.locks import db_write_lock, default_db_lock_path

    lock_path = default_db_lock_path()
    print(
        f"Acquiring DB write lock: {lock_path} "
        f"(timeout={args.lock_timeout_seconds}, poll={args.lock_poll_seconds})"
    )
    with db_write_lock(timeout_seconds=args.lock_timeout_seconds, poll_seconds=args.lock_poll_seconds):
        print("Acquired DB write lock")
        try:
            _run()
        finally:
            print("Released DB write lock")


if __name__ == "__main__":
    main()






