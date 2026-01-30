# 三連複/三連単データ品質検証

## 概要

`validate_exotics_data_quality.py` は、レビュー指摘の「前向き検証に耐える」チェック項目を実行するスクリプトです。

## チェック項目

1. **当日運用E2E**: buy/closeスナップショットの存在・品質確認
2. **欠落率の把握**: n_rows分布、stale判定で落ちた割合
3. **決済の一致**: HR払戻データとの整合性確認
4. **close vs buy の乖離定量化**: ROI差分の可視化

## 使用方法

```bash
cd py64_analysis
python scripts/validate_exotics_data_quality.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --ticket-type trio \
  --buy-t-minus-minutes 5 \
  --max-staleness-min 10.0 \
  --max-bets-per-race 10 \
  --ev-margin 0.10 \
  --out-dir data/validation_exotics
```

## 出力ファイル

- `coverage_<timestamp>.csv`: スナップショットカバレッジ詳細
- `settlement_<timestamp>.csv`: 決済データ整合性詳細
- `report_<timestamp>.md`: 検証レポート（Markdown形式）

## パラメータ

- `--start-date`: 開始日 (YYYYMMDD)
- `--end-date`: 終了日 (YYYYMMDD)
- `--ticket-type`: 券種 (`trio` または `trifecta`)
- `--buy-t-minus-minutes`: 購入想定時刻（レース開始前N分、デフォルト: 5）
- `--max-staleness-min`: 最大許容stale時間（分、デフォルト: 10.0）
- `--max-bets-per-race`: 1レースあたり最大ベット数（デフォルト: 10）
- `--ev-margin`: EV閾値（デフォルト: 0.10）
- `--out-dir`: 出力ディレクトリ（デフォルト: `data/validation_exotics`）

## 推奨アクション

レポートに自動生成される推奨アクション:

- ⚠️ Buyスナップショット欠落率が高い → forward収集の見直し
- ⚠️ BuyスナップショットのStale率が高い → max_staleness_minの調整検討
- ⚠️ Close vs Buy ROI差分が大きい → オッズ変動耐性の検討
- ⚠️ 払戻データカバレッジが低い → HRローダーの確認

## 前提条件

- `fact_race`, `fact_result` が投入済み
- `odds_rt_snapshot`, `odds_rt_trio`, `odds_rt_trifecta` が投入済み
- `fact_payout_trio` / `fact_payout_trifecta` が投入済み
- `fact_refund_horse` が投入済み

## 注意事項

- ROI比較（チェック4）は実際にバックテストを実行するため、時間がかかります
- 大量のレースを対象にする場合は、日付範囲を分割して実行することを推奨します

