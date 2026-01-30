"""
featuresテーブルの重複を削除するスクリプト

重複の定義:
  - 同一 (race_id, horse_id, feature_version, asof_time) の組み合わせで複数行存在
  - 同一payloadの場合は、created_atが最も古い1行を残して他を削除
  - 異なるpayloadの場合は警告を出して、created_atが最も古い1行を残す
"""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
src = project_root / "py64_analysis" / "src"
if src.exists():
    sys.path.insert(0, str(src))

from keiba.db.loader import get_session
from sqlalchemy import text


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Deduplicate features table")
    p.add_argument("--dry-run", action="store_true", help="Dry run (show what would be deleted)")
    p.add_argument("--feature-version", type=str, default="1.2.0", help="Feature version to dedupe")
    args = p.parse_args()

    session = get_session()

    # 重複の確認
    check_query = text("""
        WITH dup_counts AS (
            SELECT
                race_id,
                horse_id,
                feature_version,
                asof_time,
                COUNT(*) AS n_rows,
                COUNT(DISTINCT md5(payload::text)) AS n_payload_hash
            FROM features
            WHERE feature_version = :version
            GROUP BY race_id, horse_id, feature_version, asof_time
            HAVING COUNT(*) > 1
        )
        SELECT
            COUNT(*) AS n_dup_groups,
            SUM(n_rows) AS total_dup_rows,
            SUM(n_rows - 1) AS rows_to_delete,
            COUNT(CASE WHEN n_payload_hash > 1 THEN 1 END) AS groups_with_conflict
        FROM dup_counts
    """)
    result = session.execute(check_query, {"version": args.feature_version}).fetchone()
    stats = dict(result._mapping) if result else {}
    
    print(f"Feature version: {args.feature_version}")
    print(f"重複グループ数: {stats.get('n_dup_groups', 0)}")
    print(f"重複行の合計: {stats.get('total_dup_rows', 0)}")
    print(f"削除対象行数: {stats.get('rows_to_delete', 0)}")
    print(f"payload衝突グループ数: {stats.get('groups_with_conflict', 0)}")
    
    if stats.get('groups_with_conflict', 0) > 0:
        print("\nWARNING: payloadが異なる重複が存在します。確認してください。")
        conflict_query = text("""
            WITH dup_counts AS (
                SELECT
                    race_id,
                    horse_id,
                    feature_version,
                    asof_time,
                    COUNT(*) AS n_rows,
                    COUNT(DISTINCT md5(payload::text)) AS n_payload_hash
                FROM features
                WHERE feature_version = :version
                GROUP BY race_id, horse_id, feature_version, asof_time
                HAVING COUNT(*) > 1 AND COUNT(DISTINCT md5(payload::text)) > 1
            )
            SELECT race_id, horse_id, asof_time, n_rows, n_payload_hash
            FROM dup_counts
            ORDER BY race_id, horse_id, asof_time
            LIMIT 20
        """)
        conflicts = session.execute(conflict_query, {"version": args.feature_version}).fetchall()
        print("\n衝突例（最初の20件）:")
        for row in conflicts:
            print(f"  {dict(row._mapping)}")
    
    if args.dry_run:
        print("\n[DRY RUN] 実際の削除は行いません")
        return
    
    # 重複削除（created_atが最も古い1行を残す）
    delete_query = text("""
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY race_id, horse_id, feature_version, asof_time
                    ORDER BY created_at ASC, id ASC
                ) AS rn
            FROM features
            WHERE feature_version = :version
        )
        DELETE FROM features
        WHERE id IN (
            SELECT id FROM ranked WHERE rn > 1
        )
    """)
    
    result = session.execute(delete_query, {"version": args.feature_version})
    deleted_count = result.rowcount
    session.commit()
    
    print(f"\n削除完了: {deleted_count}行を削除しました")


if __name__ == "__main__":
    main()

