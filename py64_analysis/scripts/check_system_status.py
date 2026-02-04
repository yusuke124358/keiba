"""
システム全体の進捗状況を確認するスクリプト
- venv/DB/JV-Link/データ取得状況を確認
"""
import os
import sys
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "py64_analysis" / "src"))

from sqlalchemy import text
from keiba.db.loader import get_session


def is_ci() -> bool:
    return os.environ.get("GITHUB_ACTIONS", "").lower() == "true" or os.environ.get(
        "CI", ""
    ).lower() == "true"


def should_skip_db() -> bool:
    return os.environ.get("KEIBA_SKIP_DB_CHECK", "") == "1" or is_ci()


def should_skip_py32() -> bool:
    return os.environ.get("KEIBA_SKIP_PY32_CHECK", "") == "1" or is_ci()


def check_venv():
    """venvの存在確認"""
    py32_venv = PROJECT_ROOT / "py32_fetcher" / ".venv"
    py64_venv = PROJECT_ROOT / "py64_analysis" / ".venv"
    
    return {
        "py32_fetcher": py32_venv.exists(),
        "py64_analysis": py64_venv.exists(),
    }


def check_db():
    """DB接続とデータ量確認"""
    if should_skip_db():
        return {"skipped": True}
    try:
        session = get_session()
        try:
            # レース数（年度別）
            result = session.execute(
                text("""
                    SELECT 
                        EXTRACT(YEAR FROM date)::int as year,
                        COUNT(DISTINCT race_id) as n_races,
                        COUNT(DISTINCT CASE WHEN pace_first3f IS NOT NULL THEN race_id END) as n_races_with_pace
                    FROM fact_race
                    WHERE date >= '2020-01-01' AND date < '2025-01-01'
                    GROUP BY EXTRACT(YEAR FROM date)
                    ORDER BY year
                """)
            )
            races_by_year = {row[0]: {"total": row[1], "with_pace": row[2]} for row in result}
            
            # 総レース数
            result_total = session.execute(
                text("""
                    SELECT COUNT(DISTINCT race_id) as n_races_total
                    FROM fact_race
                    WHERE date >= '2020-01-01' AND date < '2025-01-01'
                """)
            )
            n_races_total = result_total.fetchone()[0]
            
            # 特徴量バージョン別
            result_features = session.execute(
                text("""
                    SELECT 
                        feature_version,
                        COUNT(DISTINCT race_id) as n_races,
                        COUNT(*) as n_features
                    FROM features
                    WHERE SUBSTRING(race_id::text, 1, 8) >= '20200101'
                      AND SUBSTRING(race_id::text, 1, 8) < '20250101'
                    GROUP BY feature_version
                    ORDER BY feature_version DESC
                """)
            )
            features_by_version = {row[0]: {"n_races": row[1], "n_features": row[2]} for row in result_features}
            
            # オッズデータ（0B41）
            result_odds = session.execute(
                text("""
                    SELECT COUNT(DISTINCT race_id) as n_races_with_odds
                    FROM odds_ts_win
                    WHERE race_id LIKE '202%'
                """)
            )
            n_races_with_odds = result_odds.fetchone()[0]
            
            return {
                "connected": True,
                "n_races_total": n_races_total,
                "races_by_year": races_by_year,
                "features_by_version": features_by_version,
                "n_races_with_odds": n_races_with_odds,
            }
        finally:
            session.close()
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
        }


def check_data_files():
    """raw JSONLファイルの確認"""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        return {"exists": False}
    
    jsonl_files = list(raw_dir.glob("*.jsonl"))
    return {
        "exists": True,
        "n_files": len(jsonl_files),
        "total_size_mb": sum(f.stat().st_size for f in jsonl_files) / (1024 * 1024),
    }


def main():
    print("=" * 60)
    print("システム進捗状況確認")
    print("=" * 60)
    
    # venv確認
    print("\n[1] venv環境")
    venv_status = check_venv()
    if should_skip_py32():
        print("  py32_fetcher: SKIP")
    else:
        print(f"  py32_fetcher: {'OK' if venv_status['py32_fetcher'] else 'NG'}")
    print(f"  py64_analysis: {'OK' if venv_status['py64_analysis'] else 'NG'}")
    
    # DB確認
    print("\n[2] PostgreSQL")
    db_status = check_db()
    if db_status.get("skipped"):
        print("  Connection: SKIP (CI)")
    elif db_status.get("connected"):
        print("  接続: OK")
        print(f"  総レース数（2020-2024）: {db_status['n_races_total']:,}")
        print("\n  年度別レース数:")
        for year in sorted(db_status['races_by_year'].keys()):
            info = db_status['races_by_year'][year]
            pace_pct = (info['with_pace'] / info['total'] * 100) if info['total'] > 0 else 0
            print(f"    {year}: {info['total']:,} レース (ペースデータ: {info['with_pace']:,}, {pace_pct:.1f}%)")
        
        print("\n  特徴量バージョン別:")
        for version in sorted(db_status['features_by_version'].keys(), reverse=True):
            info = db_status['features_by_version'][version]
            print(f"    {version}: {info['n_races']:,} レース, {info['n_features']:,} 特徴量")
        
        print(f"\n  オッズデータ（0B41）: {db_status['n_races_with_odds']:,} レース")
    else:
        print("  接続: NG")
        print(f"  エラー: {db_status.get('error', 'Unknown')}")
    
    # データファイル確認
    print("\n[3] raw JSONLファイル")
    file_status = check_data_files()
    if file_status.get("exists"):
        print(f"  ファイル数: {file_status['n_files']}")
        print(f"  総サイズ: {file_status['total_size_mb']:.1f} MB")
    else:
        print("  ディレクトリが存在しません")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
