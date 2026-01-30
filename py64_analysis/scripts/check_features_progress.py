"""
features再生成の進捗を確認するスクリプト
（DB接続が取れる場合のみ実行可能）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "py64_analysis" / "src"))

from sqlalchemy import text
from keiba.db.loader import get_session


def main():
    try:
        session = get_session()
        try:
            # 現時点のデータ量を確認
            result = session.execute(
                text("""
                    SELECT 
                        COUNT(DISTINCT race_id) as n_races_total
                    FROM fact_race
                    WHERE date >= '2020-01-01' AND date < '2025-01-01'
                """)
            )
            n_races_total = result.fetchone()[0]

            result2 = session.execute(
                text("""
                    SELECT 
                        COUNT(DISTINCT race_id) as n_races_features,
                        COUNT(*) as n_features
                    FROM features
                    WHERE feature_version = '1.4.0'
                      AND SUBSTRING(race_id::text, 1, 8) >= '20200101'
                      AND SUBSTRING(race_id::text, 1, 8) < '20250101'
                """)
            )
            row2 = result2.fetchone()
            n_races_features = row2[0]
            n_features = row2[1]

            result3 = session.execute(
                text("""
                    SELECT 
                        COUNT(CASE WHEN payload ? 'race_expected_pace_pressure' 
                                  AND (payload->>'race_expected_pace_pressure') IS NOT NULL 
                             THEN 1 END) as nn_pressure,
                        COUNT(CASE WHEN payload ? 'race_expected_first3f' 
                                  AND (payload->>'race_expected_first3f') IS NOT NULL 
                             THEN 1 END) as nn_first3f
                    FROM features
                    WHERE feature_version = '1.4.0'
                      AND SUBSTRING(race_id::text, 1, 8) >= '20200101'
                      AND SUBSTRING(race_id::text, 1, 8) < '20250101'
                """)
            )
            row3 = result3.fetchone()
            nn_pressure = row3[0]
            nn_first3f = row3[1]

            pct_races = 100 * n_races_features / n_races_total if n_races_total > 0 else 0
            pct_pressure = 100 * nn_pressure / n_features if n_features > 0 else 0
            pct_first3f = 100 * nn_first3f / n_features if n_features > 0 else 0

            print("=" * 60)
            print("features再生成の進捗状況")
            print("=" * 60)
            print(f"\n対象レース数: {n_races_total:,}")
            print(f"features生成済みレース数: {n_races_features:,} ({pct_races:.1f}%)")
            print(f"総features数: {n_features:,}")
            print(f"\nカバレッジ:")
            print(f"  race_expected_pace_pressure: {nn_pressure:,} ({pct_pressure:.2f}%)")
            print(f"  race_expected_first3f: {nn_first3f:,} ({pct_first3f:.2f}%)")
            print(f"\nDone条件:")
            print(f"  pressure >= 70%: {'✓' if pct_pressure >= 70 else '✗'} ({pct_pressure:.2f}%)")
            print(f"  first3f >= 30%: {'✓' if pct_first3f >= 30 else '✗'} ({pct_first3f:.2f}%)")

            if pct_pressure >= 70 and pct_first3f >= 30:
                print(f"\n✅ カバレッジは基準を満たしています。")
                print(f"   現時点のデータ（{pct_races:.1f}%完了）でC4-2bの再rollingを実行できます。")
            else:
                print(f"\n⚠️ カバレッジが基準未達です。")
            print("=" * 60)

        finally:
            session.close()
    except Exception as e:
        print(f"DB接続エラー: {e}")
        print("\n→ features再生成スクリプトがDB接続を大量に使用している可能性があります。")
        print("→ スクリプトを停止してから再度実行してください。")
        print("\n停止方法:")
        print("  1. タスクマネージャーでpython.exeプロセスを確認")
        print("  2. 'regenerate_features_c4.py'を実行しているプロセスを停止")


if __name__ == "__main__":
    main()






