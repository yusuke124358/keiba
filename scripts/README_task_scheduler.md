# タスクスケジューラ設定手順（三連系 forward 収集）

## 概要

2026-01-10（土）〜 2026-01-12（月・祝）の3日開催に向けて、以下の2つのジョブを毎分実行します：

1. **buy収集**: 発走-6分で0B35スナップショットを取得
2. **close収集**: 発走-1分で0B35スナップショットを取得

## 事前準備（1/9に実行）

### 1. RAデータ取得→DB投入

```cmd
cd C:\Users\yyosh\keiba
scripts\prepare_race_data_20260110.cmd
```

**期待される結果**:
- `fact_race` に 2026-01-10, 2026-01-11, 2026-01-12 がそれぞれ `n_races > 0` で入っていること

**確認SQL**:
```sql
SELECT date, COUNT(*) AS n_races, MIN(start_time) AS first_start, MAX(start_time) AS last_start
FROM fact_race
WHERE date BETWEEN DATE '2026-01-10' AND DATE '2026-01-12'
GROUP BY date
ORDER BY date;
```

## タスクスケジューラ設定

### 基本設定（両ジョブ共通）

- **プログラム**: `C:\Windows\System32\cmd.exe`
- **引数**: `/c "C:\Users\yyosh\keiba\scripts\collect_exotics_due_<buy|close>.cmd"`
- **開始場所**: `C:\Users\yyosh\keiba`
- **実行**: ユーザーがログオンしている場合のみ
- **最上位の特権で実行**: オフ（必要に応じてオン）

### buy収集ジョブ

- **名前**: `Exotics Buy Collection (0B35)`
- **トリガー**: 
  - **種類**: 繰り返し
  - **開始**: 2026-01-10 09:40:00
  - **終了**: 2026-01-12 16:30:00
  - **間隔**: 1分間
- **条件**:
  - コンピューターがAC電源で動作している場合のみタスクを開始: **オフ**（バッテリーでも実行）
  - タスクを開始するためにコンピューターを起動: **オフ**

### close収集ジョブ

- **名前**: `Exotics Close Collection (0B35)`
- **トリガー**: 
  - **種類**: 繰り返し
  - **開始**: 2026-01-10 09:40:00
  - **終了**: 2026-01-12 16:30:00
  - **間隔**: 1分間
- **条件**: buy収集と同じ

## 動作確認（1/10朝）

### 1. due race_ids が生成されているか確認

```cmd
cd C:\Users\yyosh\keiba
type data\state\race_ids_due_buy.txt
type data\state\race_ids_due_close.txt
```

**期待される結果**: 09:44頃から `race_ids_due_buy.txt` に race_id が1行ずつ追加される

### 2. スナップショットJSONLが生成されているか確認

```cmd
dir /s /b data\raw\exotics_snapshots\20260110\buy\0B35\*.jsonl
dir /s /b data\raw\exotics_snapshots\20260110\close\0B35\*.jsonl
```

**期待される結果**: 各レースごとに `0B35_*.jsonl` ファイルが生成される

### 3. ログ確認（エラーがないか）

タスクスケジューラの「履歴」タブで、各ジョブの実行結果を確認。

**正常な動作**:
- `due races: 0 (skip)` → 窓外の時刻（正常）
- `[1/2] generate race_id list...` → `[2/2] collect exotics snapshots...` → 完了（正常）

**エラーパターン**:
- `ERROR: No race dates found` → `fact_race` に当日データが無い（RA投入漏れ）
- `pywin32.com_error` → JV-Link COM が起動できない（32bit Python未使用の可能性）

## トラブルシューティング

### 問題: `due races: 0` が続く

**原因**: 
- `fact_race(date=当日)` が0件（非開催日 or RA投入漏れ）
- 時刻が窓外（09:40以前 or 16:30以降）

**確認**:
```cmd
cd C:\Users\yyosh\keiba\py64_analysis
set PYTHONPATH=src
python -c "from datetime import date; from sqlalchemy import text; from keiba.db.loader import get_session; s = get_session(); try: row = s.execute(text('SELECT COUNT(*) FROM fact_race WHERE date = :d'), {'d': date.today()}).fetchone(); print(f'Today races: {row[0]}'); finally: s.close()"
```

### 問題: `pywin32.com_error: class not registered`

**原因**: 32bit Pythonが使われていない

**確認**:
```cmd
py -3.11-32 -c "import sys; print(sys.executable)"
```

**修正**: `.cmd` ファイル内の `python` を `py -3.11-32` に変更（既に修正済み）

### 問題: `ModuleNotFoundError: No module named 'keiba'`

**原因**: `PYTHONPATH` が設定されていない

**確認**: `.cmd` ファイル内で `set PYTHONPATH=%CD%\py64_analysis\src` が実行されているか確認（既に修正済み）

## 次のステップ（収集後）

1. **DB投入**: `py64_analysis/scripts/load_exotics_snapshots_dir.py` でJSONLをDBに投入
2. **データ品質確認**: `py64_analysis/scripts/validate_exotics_data_quality.py` で品質チェック
3. **Stage C検証**: 購入時点オッズでのバックテスト実行

## 参考

- 発走時刻: 最初 09:50、最終 16:25（2場開催想定）
- buy収集タイミング: 発走-6分（例: 09:50発走 → 09:44収集）
- close収集タイミング: 発走-1分（例: 09:50発走 → 09:49収集）
- 窓幅: ±45秒（`window_sec=90`）







