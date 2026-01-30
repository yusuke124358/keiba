"""
C4-0バックフィル進捗確認スクリプト

2020-2023年のRACEデータ再取得の進捗を確認します
"""
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def check_progress(year: int) -> dict:
    """指定年の進捗を確認"""
    dir_path = PROJECT_ROOT / f"data/raw_backfill_c4_{year}"
    
    if not dir_path.exists():
        return {
            "year": year,
            "status": "not_started",
            "files": 0,
            "total_size": 0,
            "latest_file": None,
            "latest_mtime": None,
        }
    
    jsonl_files = sorted(dir_path.glob("RACE_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not jsonl_files:
        return {
            "year": year,
            "status": "no_files",
            "files": 0,
            "total_size": 0,
            "latest_file": None,
            "latest_mtime": None,
        }
    
    latest = jsonl_files[0]
    mtime = datetime.fromtimestamp(latest.stat().st_mtime)
    total_size = sum(f.stat().st_size for f in jsonl_files)
    elapsed = datetime.now() - mtime
    
    # ファイルが最近更新されているか（5分以内）
    is_active = elapsed.total_seconds() < 300
    
    # レコード数を概算（先頭1000行から推定）
    ra_count = 0
    se_count = 0
    has_pace = False
    
    try:
        with open(latest, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > 1000:
                    break
                try:
                    rec = json.loads(line)
                    rt = rec.get("record_type")
                    race_id = rec.get("race_id", "")
                    
                    if rt == "RA" and race_id.startswith(str(year)):
                        ra_count += 1
                        if rec.get("pace_first3f") is not None:
                            has_pace = True
                    elif rt == "SE" and race_id.startswith(str(year)):
                        se_count += 1
                except:
                    pass
    except:
        pass
    
    return {
        "year": year,
        "status": "active" if is_active else "completed",
        "files": len(jsonl_files),
        "total_size": total_size,
        "total_size_mb": total_size / 1024 / 1024,
        "latest_file": latest.name,
        "latest_mtime": mtime.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_minutes": elapsed.total_seconds() / 60,
        "sample_ra": ra_count,
        "sample_se": se_count,
        "has_pace": has_pace,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("C4-0バックフィル進捗確認")
    print("=" * 60)
    
    for year in [2020, 2021, 2022, 2023]:
        progress = check_progress(year)
        
    print(f"\n{year}年:")
    print(f"  ステータス: {progress['status']}")
    print(f"  ファイル数: {progress['files']}")
    if progress['total_size'] > 0:
        print(f"  総サイズ: {progress['total_size_mb']:.1f} MB")
        
        if progress['latest_file']:
            print(f"  最新ファイル: {progress['latest_file']}")
            print(f"  最終更新: {progress['latest_mtime']}")
            print(f"  経過時間: {progress['elapsed_minutes']:.1f} 分")
            
            if progress['sample_ra'] > 0:
                print(f"  サンプル（先頭1000行）: RA={progress['sample_ra']}, SE={progress['sample_se']}")
                print(f"  ペース情報: {'あり' if progress['has_pace'] else 'なし'}")
        
        if progress['status'] == "active":
            print("  → 処理中")
        elif progress['status'] == "completed":
            print("  → 完了（または停止）")
    
    print("\n" + "=" * 60)

