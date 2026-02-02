# 競馬アルゴ（期待値ベース）

JRA-VAN Data Lab を利用した期待値ベースの競馬予測・ベッティングシステム

## 概要

- **目的**: 期待値（EV）プラスの馬券を選別し、資金管理で破産確率を抑えながら運用
- **データソース**: JRA-VAN Data Lab（JV-Link / JV-Data）
- **スコープ**: 買い目生成・金額計算まで（自動投票は対象外）

## アーキテクチャ

```
┌─────────────────┐    ┌─────────────────┐
│  py32_fetcher   │    │  py64_analysis  │
│  (32bit Python) │    │  (64bit Python) │
├─────────────────┤    ├─────────────────┤
│ JV-Link COM     │───>│ DB Loader       │
│ Fetcher         │    │ Features        │
│ Parsers         │    │ Modeling        │
│ State Manager   │    │ Backtest        │
└─────────────────┘    │ Betting Policy  │
         │             └─────────────────┘
         v                     │
   data/raw/*.jsonl            v
                        ┌─────────────────┐
                        │   PostgreSQL    │
                        └─────────────────┘
```

## ディレクトリ構成

```
keiba/
├── py32_fetcher/           # 32bit Python（JV-Link取得専用）
│   ├── jvlink_client.py    # JV-Link COMクライアント
│   ├── dataspec.py         # データ種別ID定義
│   ├── state.py            # ウォーターマーク管理
│   ├── fetcher_stored.py   # 蓄積系データ取得
│   ├── fetcher_realtime.py # 速報系データ取得
│   ├── parsers/            # JV-Dataパーサー
│   └── run_fetch.py        # 取得エントリポイント
│
├── py64_analysis/          # 64bit Python（分析・学習）
│   ├── pyproject.toml
│   ├── alembic/            # DBマイグレーション
│   └── src/keiba/
│       ├── config.py       # 設定管理
│       ├── db/             # モデル・ローダー
│       ├── features/       # 特徴量生成
│       ├── modeling/       # 学習・校正
│       ├── backtest/       # バックテスト
│       └── betting/        # ベッティングポリシー
│
├── config/
│   └── config.yaml         # 設定ファイル
│
├── data/
│   ├── raw/                # JSONL（生データ）
│   ├── processed/          # 加工済みデータ
│   └── state/              # 取得状態ファイル
│
└── reports/                # レポート出力
```

## セットアップ

**重要**: すべてのコマンドは **`keiba/` ディレクトリ直下**で実行してください。
設定ファイル（`config/config.yaml`）やデータディレクトリ（`data/`）の参照が正しく動作します。

### 1. 32bit Python 環境（py32_fetcher）

```bash
# keibaディレクトリに移動
cd keiba

# 32bit Python 3.11をインストール
# https://www.python.org/downloads/ から「Windows installer (32-bit)」を取得

# 仮想環境作成
py -3.11-32 -m venv py32_fetcher/.venv
py32_fetcher\.venv\Scripts\activate
pip install -r py32_fetcher/requirements.txt
```

### 2. 64bit Python 環境（py64_analysis）

```bash
# keibaディレクトリで実行
cd keiba

# 64bit Python（3.12 推奨）
py -3.12 -m venv py64_analysis/.venv
py64_analysis\.venv\Scripts\activate
pip install -e py64_analysis
```

### 3. PostgreSQL

```bash
# Docker Desktop をインストール（Windows推奨）
# - Docker が使えない場合は手元のPostgreSQLでもOK（config/config.yamlのDB URLを合わせる）

# keibaディレクトリでDB起動
cd keiba
docker compose up -d db

# 起動確認（healthy になるまで数秒待つ）
docker ps

# マイグレーション実行
py64_analysis\.venv\Scripts\activate
cd py64_analysis
python -m alembic upgrade head
cd ..
```

### 環境変数（オプション）

設定ファイルやプロジェクトルートを明示的に指定できます：

```bash
# 設定ファイルのパス
set KEIBA_CONFIG_PATH=C:\path\to\keiba\config\config.yaml

# プロジェクトルート
set KEIBA_PROJECT_ROOT=C:\path\to\keiba
```

### 4. JRA-VAN Data Lab

1. [JRA-VAN Data Lab](https://jra-van.jp/dlb/) に加入
2. JV-Linkをインストール
3. ソフトウェアIDを取得

## 使い方

### データ取得（32bit Python）

```bash
# keibaディレクトリから実行
cd keiba
py32_fetcher\.venv\Scripts\activate

# レース情報（蓄積系）
python py32_fetcher/run_fetch.py stored RACE

# 時系列オッズ（蓄積系）
python py32_fetcher/run_fetch.py stored 0B41  # 単勝/複勝/枠連
python py32_fetcher/run_fetch.py stored 0B42  # 馬連

# ★環境によっては 0B41/0B42 を JVOpen(stored) で扱うと -111 になる場合があります。
# その場合は、race_id を指定して JVRTOpen(realtime) でまとめ取りするユーティリティを使用してください：
# 例）期間のrace_id一覧を data/state/race_ids_YYYYMMDD_YYYYMMDD.txt に作ってから実行
python py32_fetcher/bulk_fetch_realtime.py 0B41 --race-ids-file data/state/race_ids_20251129_20251228.txt --chunk-size 50
python py32_fetcher/bulk_fetch_realtime.py 0B41 --race-ids-file data/state/race_ids_20251129_20251228.txt --chunk-size 50 --resume

# 速報オッズ（レース当日）
python py32_fetcher/run_fetch.py realtime 0B31 2024122801010101
```

### DB投入（64bit Python）

```python
from keiba.config import get_data_path
from keiba.db.loader import load_all_jsonl_files

# プロジェクトルート基準でパスを取得
load_all_jsonl_files(get_data_path("data/raw"))
```

### 特徴量生成

```python
# 明示的なimport（__init__.pyは軽量設計）
from keiba.features.build_features import build_features
from keiba.db.loader import get_session

session = get_session()
build_features(session, ["2024122801010101"])
```

### モデル学習

```python
from keiba.modeling.train import train_model
from keiba.db.loader import get_session
from pathlib import Path

session = get_session()
model, metrics = train_model(
    session,
    train_start="2020-01-01",
    train_end="2023-12-31",
    model_path=Path("models/win_model.pkl"),
)
```

### バックテスト

```python
from keiba.backtest.engine import run_backtest
from keiba.backtest.report import generate_report
from keiba.db.loader import get_session
from pathlib import Path

session = get_session()
result = run_backtest(
    session,
    start_date="2024-01-01",
    end_date="2024-12-31",
    model=model,
)
generate_report(result, Path("reports"))
```

## rolling holdout（一括探索）

`py64_analysis/scripts/run_rolling_holdout.py` で、rolling window の holdout をまとめて回し、
最後に `summary.csv` を自動生成できます（比較しながら勝ち筋探索する用途）。

### 注意（重要）

`--test-range-start / --test-range-end` は **「test窓の実日付（test_start〜test_end）」の上限**です。
内部では `test_end = test_start + (test_window_days - 1)` で窓を作るため、
少なくとも次を満たす必要があります：

- `test_range_end - test_range_start + 1 >= test_window_days`

### 一発実行コマンド例（Windows向け ^）

#### 推奨（60日窓・14日ステップ・valid 14日・train 365日）

```bat
python py64_analysis/scripts/run_rolling_holdout.py ^
  --name rolling60_s14_lb365_v14 ^
  --test-range-start 2025-10-01 --test-range-end 2025-12-28 ^
  --test-window-days 60 --step-days 14 ^
  --gap-days 0 ^
  --train-lookback-days 365 ^
  --valid-window-days 14 ^
  --estimate-closing-mult --closing-mult-quantile 0.30
```

#### 速回し（30日窓・7日ステップ・validなし・trainはexpanding）

```bat
python py64_analysis/scripts/run_rolling_holdout.py ^
  --name rolling30_s7_exp_v0 ^
  --test-range-start 2025-11-01 --test-range-end 2025-12-28 ^
  --test-window-days 30 --step-days 7 ^
  --gap-days 0 ^
  --valid-window-days 0 ^
  --estimate-closing-mult --closing-mult-quantile 0.30
```

### 買い目生成

```python
from keiba.betting.policy import generate_bet_signals
from keiba.db.loader import get_session
from pathlib import Path

signals = generate_bet_signals(
    session,
    race_ids=["2024122801010101"],
    predictions_by_race={...},
    bankroll=300000,
    output_path=Path("reports/bets.csv"),
)
```

## 設定

`config/config.yaml` で以下を設定:

- 払戻率（デフォルト・特定日上書き）
- バックテスト設定（購入時点、スリッページ）
- モデル設定（ブレンド比率、校正方法）
- ベッティング設定（期待値マージン、ケリー分数、上限）
- Ticket1: closing_odds_multiplier（買い時点→締切のオッズ悪化/改善を保守的に織り込む倍率）

## データ種別

| dataspec | 用途 | レコード | 取得API |
|----------|------|----------|---------|
| RACE | レース基本情報 | RA/SE/HR | JVOpen |
| 0B31 | 速報オッズ（単複枠） | O1 | JVRTOpen |
| 0B32 | 速報オッズ（馬連） | O2 | JVRTOpen |
| 0B41 | 時系列オッズ（単複枠） | O1 | JVOpen |
| 0B42 | 時系列オッズ（馬連） | O2 | JVOpen |

※上記は仕様上の整理です。実環境で `JVOpen(0B41/0B42)=-111` 等が出る場合は、`bulk_fetch_realtime.py` による `JVRTOpen + race_id` 取得を優先してください。

## 注意事項

- **自動投票は実装していません**: 買い目生成・金額計算までが対象です
- **投票は公式手段で**: 即PAT等を使用し、手動で投票してください
- **成立した投票の取消・変更はできません**: 十分確認してから投票してください
- **JRA-VAN規約を遵守**: データの再配布・外部公開は禁止

## ライセンス

MIT License

## Agent Loop Operations

Local (cron/systemd)
- Linux cron example:
```cron
*/30 * * * * cd /path/to/keiba && bash scripts/agent/loop.sh
```
- Windows Task Scheduler example command:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\agent\loop.ps1
```

GitHub Actions (scheduled)
- Create a scheduled workflow that runs once per schedule and executes:
```bash
bash scripts/agent/loop.sh
```
 - Backlog is in `experiments/backlog.yml`.

Safety and stop conditions
- `risk_level: high` disables auto-merge.
- `approval_policy` and `sandbox_mode` are enforced in `.codex/config.toml`.
- Failures write to `docs/experiments/<id>.md` and the loop exits; rerun on the next schedule.
