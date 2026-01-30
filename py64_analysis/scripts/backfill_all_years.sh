#!/bin/bash
# 2020-2023年のRACEデータ再取得とバックフィルを実行

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "2020-2023年のRACEデータ再取得を開始します..."
echo ""

# 各年の再取得をバックグラウンドで実行
for year in 2020 2021 2022 2023; do
    echo "開始: ${year}年"
    py -3.11-32 py32_fetcher/run_fetch.py stored RACE \
        --option 4 \
        --from-time ${year}0101000000 \
        --min-race-date ${year}0101 \
        --max-race-date ${year}1231 \
        --no-state-update \
        --output-dir "data/raw_backfill_c4_${year}" &
done

echo ""
echo "すべてのプロセスを開始しました"
echo "進捗確認: python py64_analysis/scripts/check_backfill_progress.py"






