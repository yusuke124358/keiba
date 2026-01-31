"""
C4-0b完了後のfeatures再生成スクリプト

過去レースのペース素材がDBにバックフィルされた後、
2020-2024年のfeaturesをversion 1.4.0で再生成する。

既存の古いversion（1.0.0, 1.1.0, 1.2.0）のfeaturesは削除してから再生成する。
"""
import sys
from pathlib import Path
from datetime import datetime, date

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "py64_analysis" / "src"))

from sqlalchemy import text
from keiba.db.loader import get_session
from keiba.features.build_features import build_features


def main():
    import argparse

    p = argparse.ArgumentParser(description="Regenerate features for 2020-2024 (v1.4.0)")
    p.add_argument("--lock-timeout-seconds", type=float, default=0)
    p.add_argument("--lock-poll-seconds", type=float, default=0.25)
    p.add_argument("--no-lock", action="store_true")
    args = p.parse_args()

    if not args.no_lock:
        if args.lock_timeout_seconds < 0:
            p.error("--lock-timeout-seconds must be >= 0")
        if args.lock_poll_seconds <= 0:
            p.error("--lock-poll-seconds must be > 0")

    def _run() -> None:
        session = get_session()
        try:

            # 2020-2024年のレースIDを取得
            print("2020-2024年のレースIDを取得中...")
            rows = session.execute(
                text("""
                    SELECT DISTINCT race_id
                    FROM fact_race
                    WHERE date >= '2020-01-01' AND date < '2025-01-01'
                    ORDER BY race_id
                """)
            ).fetchall()
            race_ids = [str(row[0]) for row in rows]
            print(f"対象レース数: {len(race_ids):,}")

            # 古いversionのfeaturesを削除（version 1.4.0以外）
            print("\n古いversionのfeaturesを削除中...")
            result = session.execute(
                text("""
                    DELETE FROM features
                    WHERE race_id = ANY(:race_ids)
                      AND feature_version != '1.4.0'
                """),
                {"race_ids": race_ids}
            )
            deleted_count = result.rowcount
            session.commit()
            print(f"削除件数: {deleted_count:,}")

            # version 1.4.0でfeaturesを再生成（skip_existing=Falseで強制再生成）
            print("\nfeaturesを再生成中（version 1.4.0）...")
            print("（時間がかかります。進捗は表示されません）")

            total_generated = build_features(
                session=session,
                race_ids=race_ids,
                asof_time=None,  # 各レースの購入想定時点を使用
                buy_t_minus_minutes=1,  # buy_minutes=1固定
                skip_existing=False,  # 強制再生成
            )

            print(f"\n再生成完了: {total_generated:,}件のfeaturesを生成しました")

            # カバレッジ確認
            print("\nカバレッジ確認中...")
            result = session.execute(
                text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN payload ? 'race_expected_pace_pressure' 
                                   AND (payload->>'race_expected_pace_pressure') IS NOT NULL 
                              THEN 1 END) as nn_pressure,
                        COUNT(CASE WHEN payload ? 'race_expected_first3f' 
                                   AND (payload->>'race_expected_first3f') IS NOT NULL 
                              THEN 1 END) as nn_first3f
                    FROM features
                    WHERE feature_version = '1.4.0'
                      AND race_id = ANY(:race_ids)
                """),
                {"race_ids": race_ids}
            ).fetchone()

            total = result[0]
            nn_pressure = result[1]
            nn_first3f = result[2]

            print("\nカバレッジ:")
            print(f"  Total features: {total:,}")
            print(f"  race_expected_pace_pressure: {nn_pressure:,} ({100*nn_pressure/total if total > 0 else 0:.2f}%)")
            print(f"  race_expected_first3f: {nn_first3f:,} ({100*nn_first3f/total if total > 0 else 0:.2f}%)")

            # Done条件チェック
            pct_pressure = 100 * nn_pressure / total if total > 0 else 0
            pct_first3f = 100 * nn_first3f / total if total > 0 else 0

            print("\nDone条件:")
            print(f"  race_expected_pace_pressure >= 70%: {'✓' if pct_pressure >= 70 else '✗'} ({pct_pressure:.2f}%)")
            print(f"  race_expected_first3f >= 30%: {'✓' if pct_first3f >= 30 else '✗'} ({pct_first3f:.2f}%)")

            if pct_pressure >= 70 and pct_first3f >= 30:
                print("\n✅ Done条件を満たしています。C4-2b（再rolling）に進めます。")
            else:
                print("\n⚠️ Done条件を満たしていません。追加のバックフィルが必要です。")


        finally:
            session.close()
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






