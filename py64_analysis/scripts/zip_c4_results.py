"""
C4-2のrolling holdout結果をZIP化して単勝フォルダに保存
過去ファイルはoldフォルダに移動
"""
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 出力先
output_dir = PROJECT_ROOT / "output for 5.2pro" / "単勝"
output_dir.mkdir(parents=True, exist_ok=True)

# 既存ZIPファイルをoldフォルダに移動
existing_zips = list(output_dir.glob("*.zip"))
if existing_zips:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    old_dir = output_dir / "old" / timestamp
    old_dir.mkdir(parents=True, exist_ok=True)
    for zip_file in existing_zips:
        shutil.move(str(zip_file), str(old_dir / zip_file.name))
    print(f"Moved {len(existing_zips)} existing ZIP files to {old_dir}")
else:
    print("No existing ZIP files to move")

# C4-2の結果ディレクトリ
c4_dir = PROJECT_ROOT / "data" / "holdout_runs" / "rolling60_s14_valid14_lb365_gap0_univ_buym1_c0c1c3_c4pace_v0_20260107_232128"
baseline_dir = PROJECT_ROOT / "data" / "holdout_runs" / "rolling60_s14_valid14_lb365_gap0_univ_buym1_c0c1c3_20260105_163634"

if not c4_dir.exists():
    raise SystemExit(f"C4 directory not found: {c4_dir}")

# ZIPファイル名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = f"C4_rolling_results_{timestamp}.zip"
zip_path = output_dir / zip_filename

# 含めるファイル
files_to_include = [
    # C4 variant
    ("paired_summary.json", c4_dir / "paired_summary.json"),
    ("paired_compare.csv", c4_dir / "paired_compare.csv"),
    ("ev_lift_summary.json", c4_dir / "ev_lift_summary.json"),
    ("ev_lift.csv", c4_dir / "ev_lift.csv"),
    ("summary.csv", c4_dir / "summary.csv"),
    ("drivers.csv", c4_dir / "drivers.csv"),
    # Baseline（比較用）
    ("baseline_ev_lift_summary.json", baseline_dir / "ev_lift_summary.json"),
    ("baseline_summary.csv", baseline_dir / "summary.csv"),
]

# ZIP作成
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for arcname, filepath in files_to_include:
        if filepath.exists():
            zf.write(filepath, arcname)
            print(f"Added: {arcname}")
        else:
            print(f"Warning: {filepath} not found, skipping")

print(f"\nZIP created: {zip_path}")
print(f"Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")






