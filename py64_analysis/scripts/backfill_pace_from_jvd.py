"""
C4-0バックフィル: JRAVANData内の.jvdファイルからペース情報を抽出してDBに投入

既存の.jvdファイルを拡張パーサーで再パースし、fact_race/fact_resultのペース関連カラムを更新する
"""
import argparse
import json
import logging
import sys
import zlib
from pathlib import Path
from datetime import datetime
from typing import Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from keiba.db.loader import get_session
from keiba.db.pace_utils import (
    sum_first_n,
    sum_last_n,
    lap_stats,
    pace_diff,
    extract_lap_times_200m_from_ra_bytes,
    derive_pace_stats_from_laps,
)
from py32_fetcher.parsers.race_parser import parse_ra_race, parse_se_race_uma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_jvd_payload(jvd_path: Path) -> bytes:
    raw = jvd_path.read_bytes()
    if len(raw) >= 12:
        header = raw[:10]
        if header.strip().isdigit() and raw[10:12] == b"x\x9c":
            try:
                return zlib.decompress(raw[10:])
            except zlib.error as exc:
                logger.warning(f"zlib decompress failed for {jvd_path.name}: {exc}")
    return raw


def _has_ascii_record_types(data: bytes) -> bool:
    return (b"RA" in data) or (b"SE" in data)


def _extract_year_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    if len(stem) < 8:
        return None
    year_str = stem[4:8]
    if not year_str.isdigit():
        return None
    return int(year_str)


def _decode_raw_hex(raw_val) -> Optional[bytes]:
    if raw_val is None:
        return None
    if isinstance(raw_val, bytes):
        return raw_val
    if isinstance(raw_val, str):
        s = raw_val.strip()
        if not s:
            return None
        try:
            return bytes.fromhex(s)
        except ValueError:
            return None
    return None


def _ensure_ra_lap_times(record: dict) -> dict:
    if record.get("record_type") != "RA":
        return record
    lap_times = record.get("lap_times_200m")
    if isinstance(lap_times, list) and any(v is not None for v in lap_times):
        return record
    raw_bytes = _decode_raw_hex(record.get("_raw"))
    if not raw_bytes:
        return record
    laps = extract_lap_times_200m_from_ra_bytes(raw_bytes)
    if not laps:
        return record
    record["lap_times_200m"] = laps
    stats = derive_pace_stats_from_laps(laps)
    for key, value in stats.items():
        if record.get(key) is None:
            record[key] = value
    if record.get("pace_first4f") is None:
        record["pace_first4f"] = sum_first_n(laps, 4)
    if record.get("pace_last4f") is None:
        record["pace_last4f"] = sum_last_n(laps, 4)
    return record


def parse_jvd_file(jvd_path: Path) -> list[dict]:
    """
    .jvdファイルを読み込んでパース
    
    JV-Link形式: 複数のレコードが連結されたバイナリファイル
    - RA: 1272バイト固定長
    - SE: 可変長（先頭2バイトで種別判定、その後レコード長を取得）
    
    Returns:
        list of parsed records
    """
    from py32_fetcher.parsers import parse_record
    
    # ファイル名からrace_dateを推測（簡易）
    filename = jvd_path.stem
    # RASW2026010420260106133206.jvd -> 20260104
    race_date = filename[4:12] if len(filename) >= 12 else None
    
    records = []
    
    try:
        raw_data = _read_jvd_payload(jvd_path)
        if not _has_ascii_record_types(raw_data):
            logger.warning(
                f"No ASCII record types found in {jvd_path.name} after decompress. Skipping."
            )
            return []
        
        # ファイル名からdata_specを推測
        filename = jvd_path.stem
        data_spec = "RACE"  # RA/SEファイルはRACE
        
        pos = 0
        while pos < len(raw_data):
            # レコード種別を確認（先頭2バイト）
            if pos + 2 > len(raw_data):
                break
            
            record_type_bytes = raw_data[pos:pos+2]
            record_type = record_type_bytes.decode("shift_jis", errors="ignore")
            
            if record_type == "RA":
                # RA: 1272バイト固定長
                record_size = 1272
                if pos + record_size > len(raw_data):
                    break
                
                record_bytes = raw_data[pos:pos+record_size]
                parsed = parse_record(record_bytes, data_spec, jvd_path.name)
                if parsed and "_raw" not in parsed:
                    records.append(parsed)
                
                pos += record_size
            
            elif record_type == "SE":
                # SE: 可変長（先頭394バイトで基本情報、その後可変長フィールド）
                # 簡易実装: 最小394バイトを読み、race_idから次のレコード位置を推測
                if pos + 394 > len(raw_data):
                    break
                
                # race_idを取得（12-27バイト目）
                race_id_bytes = raw_data[pos+11:pos+27]
                race_id = race_id_bytes.decode("shift_jis", errors="ignore").strip()
                
                # SEレコードの最小長は394バイト、最大は可変
                # 簡易実装: 次のレコード（RAまたはSE）が見つかるまで読み進める
                # より正確には、SEレコードの構造に従って長さを計算する必要がある
                # ここでは、次の"RA"または"SE"が見つかる位置までを1レコードとする
                next_pos = pos + 394
                found_next = False
                while next_pos + 2 < len(raw_data):
                    next_type = raw_data[next_pos:next_pos+2].decode("shift_jis", errors="ignore")
                    if next_type in ("RA", "SE"):
                        found_next = True
                        break
                    next_pos += 1
                
                if found_next:
                    record_size = next_pos - pos
                else:
                    # 次のレコードが見つからない場合は残り全部
                    record_size = len(raw_data) - pos
                
                record_bytes = raw_data[pos:pos+record_size]
                parsed = parse_record(record_bytes, data_spec, jvd_path.name)
                if parsed and "_raw" not in parsed:
                    records.append(parsed)
                
                pos += record_size
            
            else:
                # 未知のレコード種別: スキップ
                logger.warning(f"Unknown record type '{record_type}' at position {pos} in {jvd_path.name}")
                # 次の"RA"または"SE"を探す
                next_pos = pos + 1
                found = False
                while next_pos + 2 < len(raw_data):
                    next_type = raw_data[next_pos:next_pos+2].decode("shift_jis", errors="ignore")
                    if next_type in ("RA", "SE"):
                        pos = next_pos
                        found = True
                        break
                    next_pos += 1
                if not found:
                    break
    
    except Exception as e:
        logger.error(f"Failed to parse {jvd_path}: {e}", exc_info=True)
    
    return records


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
        # PostgreSQLのJSONB型に直接渡す（SQLAlchemyが自動変換）
        lap_times_jsonb = None
        if lap_times and isinstance(lap_times, list):
            # Noneを除外
            lap_times_clean = [x for x in lap_times if x is not None]
            if lap_times_clean:
                lap_times_jsonb = lap_times_clean  # JSONB型に直接渡す
        
        # UPDATE（既存レコードのみ）
        # JSONBはCASTが必要
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
                    "lap_times": json.dumps(lap_times_jsonb),
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


def update_fact_race_pace_with_stats(session, records: list[dict]) -> tuple[int, int, int]:
    """
    Update fact_race pace fields with counters.
    Returns: (updated, skipped_no_laps, errors)
    """
    updated = 0
    skipped_no_laps = 0
    errors = 0
    for record in records:
        if record.get("record_type") != "RA":
            continue

        race_id = record.get("race_id")
        if not race_id:
            continue

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

        lap_times_jsonb = None
        lap_times_clean = []
        if lap_times and isinstance(lap_times, list):
            lap_times_clean = [x for x in lap_times if x is not None]
            if lap_times_clean:
                lap_times_jsonb = lap_times_clean
        if not lap_times_clean and pace_first3f is None and pace_last3f is None:
            skipped_no_laps += 1
            continue

        try:
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
                        "lap_times": json.dumps(lap_times_jsonb),
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
                updated += 1
        except Exception:
            errors += 1

    return updated, skipped_no_laps, errors


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


def backfill_from_jvd_dir(
    jvd_dir: Path, min_year: Optional[int] = None, max_year: Optional[int] = None
) -> dict:
    """
    JRAVANDataディレクトリ内の.jvdファイルからペース情報を抽出してDBに投入
    """
    session = get_session()
    
    ra_files = sorted(jvd_dir.glob("RA*.jvd"))
    se_files = sorted(jvd_dir.glob("SE*.jvd"))
    if min_year or max_year:
        def _in_range(p: Path) -> bool:
            y = _extract_year_from_name(p)
            if y is None:
                return False
            if min_year and y < min_year:
                return False
            if max_year and y > max_year:
                return False
            return True
        ra_files = [p for p in ra_files if _in_range(p)]
        se_files = [p for p in se_files if _in_range(p)]
    
    logger.info(f"Found {len(ra_files)} RA files and {len(se_files)} SE files")
    
    total_ra_updated = 0
    total_ra_skipped = 0
    total_ra_errors = 0
    total_se_updated = 0
    
    try:
        # RAファイルを処理
        for i, ra_file in enumerate(ra_files):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing RA files: {i+1}/{len(ra_files)}")
            
            records = parse_jvd_file(ra_file)
            if records:
                updated, skipped, errors = update_fact_race_pace_with_stats(session, records)
                total_ra_updated += updated
                total_ra_skipped += skipped
                total_ra_errors += errors
                session.commit()
        
        # SEファイルを処理
        for i, se_file in enumerate(se_files):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing SE files: {i+1}/{len(se_files)}")
            
            records = parse_jvd_file(se_file)
            if records:
                updated = update_fact_result_corner(session, records)
                total_se_updated += updated
                session.commit()
        
        logger.info(
            f"Backfill complete: RA={total_ra_updated} (skipped={total_ra_skipped}, errors={total_ra_errors}), SE={total_se_updated}"
        )
        
        return {
            "ra_files": len(ra_files),
            "se_files": len(se_files),
            "ra_updated": total_ra_updated,
            "ra_skipped_no_laps": total_ra_skipped,
            "ra_errors": total_ra_errors,
            "se_updated": total_se_updated,
        }
    
    finally:
        session.close()


def backfill_from_jsonl_file(
    session,
    jsonl_path: Path,
    batch_size: int = 1000,
) -> tuple[int, int, int]:
    updated = 0
    skipped_no_laps = 0
    errors = 0
    batch: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            if "\"record_type\": \"RA\"" not in line and "\"record_type\": \"SE\"" not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            record = _ensure_ra_lap_times(record)
            batch.append(record)
            if len(batch) >= batch_size:
                u, s, e = update_fact_race_pace_with_stats(session, batch)
                updated += u
                skipped_no_laps += s
                errors += e
                session.commit()
                batch = []
            if line_num % 200000 == 0:
                logger.info(
                    f"Processed {line_num:,} lines in {jsonl_path.name}: updated={updated}, skipped={skipped_no_laps}, errors={errors}"
                )
    if batch:
        u, s, e = update_fact_race_pace_with_stats(session, batch)
        updated += u
        skipped_no_laps += s
        errors += e
        session.commit()
    return updated, skipped_no_laps, errors


def backfill_from_jsonl_dir(jsonl_dir: Path) -> dict:
    session = get_session()
    jsonl_files = sorted(jsonl_dir.glob("RACE_*.jsonl"))
    total_updated = 0
    total_skipped = 0
    total_errors = 0
    try:
        for i, jsonl_file in enumerate(jsonl_files):
            if (i + 1) % 5 == 0:
                logger.info(f"Processing JSONL files: {i+1}/{len(jsonl_files)}")
            updated, skipped, errors = backfill_from_jsonl_file(session, jsonl_file)
            total_updated += updated
            total_skipped += skipped
            total_errors += errors
        return {
            "jsonl_files": len(jsonl_files),
            "ra_updated": total_updated,
            "ra_skipped_no_laps": total_skipped,
            "ra_errors": total_errors,
        }
    finally:
        session.close()


def _compute_pace_nonnull_ratio(session, start_date: str, end_date: str) -> tuple[int, int, float]:
    row = session.execute(
        text(
            """
            SELECT
              COUNT(*) AS total,
              COUNT(CASE WHEN pace_first3f IS NOT NULL THEN 1 END) AS nonnull
            FROM fact_race
            WHERE date >= :start_date AND date <= :end_date
            """
        ),
        {"start_date": start_date, "end_date": end_date},
    ).fetchone()
    total = int(row[0] or 0)
    nonnull = int(row[1] or 0)
    ratio = (nonnull / total) if total > 0 else 0.0
    return total, nonnull, ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jvd-dir",
        type=Path,
        default=PROJECT_ROOT / "JRAVANData" / "data",
        help="Directory containing RA*/SE*.jvd files.",
    )
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    parser.add_argument(
        "--jsonl-dir",
        action="append",
        type=Path,
        default=[],
        help="Directory containing RACE_*.jsonl files (repeatable).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for backfill_pace_nonnull_ratios.csv",
    )
    parser.add_argument("--lock-timeout-seconds", type=float, default=0)
    parser.add_argument("--lock-poll-seconds", type=float, default=0.25)
    parser.add_argument("--no-lock", action="store_true")
    args = parser.parse_args()

    if not args.no_lock:
        if args.lock_timeout_seconds < 0:
            raise SystemExit("--lock-timeout-seconds must be >= 0")
        if args.lock_poll_seconds <= 0:
            raise SystemExit("--lock-poll-seconds must be > 0")

    def _run() -> None:
        total_updated = 0
        total_skipped = 0
        total_errors = 0

        if args.jvd_dir and args.jvd_dir.exists():
            result = backfill_from_jvd_dir(args.jvd_dir, args.min_year, args.max_year)
            total_updated += result.get("ra_updated", 0)
            total_skipped += result.get("ra_skipped_no_laps", 0)
            total_errors += result.get("ra_errors", 0)
            print(f"JVD result: {result}")
        else:
            logger.warning(f"JVD dir not found or skipped: {args.jvd_dir}")

        for jsonl_dir in args.jsonl_dir:
            if not jsonl_dir.exists():
                logger.warning(f"JSONL dir not found: {jsonl_dir}")
                continue
            result = backfill_from_jsonl_dir(jsonl_dir)
            total_updated += result.get("ra_updated", 0)
            total_skipped += result.get("ra_skipped_no_laps", 0)
            total_errors += result.get("ra_errors", 0)
            print(f"JSONL result: {result}")

        session = get_session()
        try:
            total_2024, nonnull_2024, ratio_2024 = _compute_pace_nonnull_ratio(
                session, "2024-01-01", "2024-12-20"
            )
            total_2025, nonnull_2025, ratio_2025 = _compute_pace_nonnull_ratio(
                session, "2025-01-01", "2025-12-20"
            )
        finally:
            session.close()

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / "backfill_pace_nonnull_ratios.csv"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("year,total,nonnull_pace_first3f,ratio_pace_first3f\n")
                f.write(f"2024,{total_2024},{nonnull_2024},{ratio_2024:.6f}\n")
                f.write(f"2025,{total_2025},{nonnull_2025},{ratio_2025:.6f}\n")

        print(
            f"[backfill] updated={total_updated} | skipped_no_laps={total_skipped} | errors={total_errors}"
        )
        print(
            f"[backfill] nonnull_ratio_2024={ratio_2024:.6f} nonnull_ratio_2025={ratio_2025:.6f}"
        )

    if args.no_lock:
        print("DB write lock: DISABLED (--no-lock)")
        _run()
    else:
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
