"""
baseline rollingの完了を待って、N5分析とpaired比較を自動実行するスクリプト
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
src = project_root / "py64_analysis" / "src"
if src.exists():
    sys.path.insert(0, str(src))


def check_complete(baseline_dir: Path, total_windows: int) -> bool:
    """baseline rollingが完了したか確認"""
    if not baseline_dir.exists():
        return False
    
    summary_csv = baseline_dir / "summary.csv"
    if not summary_csv.exists():
        return False
    
    completed = 0
    for item in sorted(baseline_dir.iterdir()):
        if item.is_dir() and item.name.startswith("w"):
            summary_json = item / "summary.json"
            if summary_json.exists():
                completed += 1
    
    return completed >= total_windows


def main() -> None:
    p = argparse.ArgumentParser(description="Wait for baseline rolling and run analysis")
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--variant-dir", type=Path, required=True)
    p.add_argument("--total-windows", type=int, default=22)
    p.add_argument("--check-interval", type=int, default=60, help="チェック間隔（秒）")
    args = p.parse_args()

    baseline_dir = args.baseline_dir
    variant_dir = args.variant_dir
    total_windows = args.total_windows

    print(f"baseline rollingの完了を待機中: {baseline_dir}")
    print(f"目標窓数: {total_windows}窓")
    print(f"チェック間隔: {args.check_interval}秒\n")

    # 完了を待つ
    while not check_complete(baseline_dir, total_windows):
        completed = sum(
            1
            for item in baseline_dir.iterdir()
            if item.is_dir()
            and item.name.startswith("w")
            and (item / "summary.json").exists()
        )
        print(f"進捗: {completed}/{total_windows}窓完了", end="\r")
        time.sleep(args.check_interval)

    print(f"\n✅ baseline rolling完了: {total_windows}窓")

    # N5分析を実行
    print("\nbaselineのN5分析を実行中...")
    analyze_script = project_root / "py64_analysis" / "scripts" / "analyze_rolling_bets.py"
    subprocess.run(
        [
            sys.executable,
            str(analyze_script),
            "--group-dir",
            str(baseline_dir),
            "--ev-lift",
        ],
        check=True,
    )

    # paired比較を実行
    print("\npaired比較を実行中...")
    compare_script = project_root / "py64_analysis" / "scripts" / "compare_rolling_runs.py"
    subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--baseline-dir",
            str(baseline_dir),
            "--variant-dir",
            str(variant_dir),
        ],
        check=True,
    )

    print("\n✅ すべての分析が完了しました")


if __name__ == "__main__":
    main()






