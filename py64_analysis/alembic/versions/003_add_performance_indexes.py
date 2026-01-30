"""add performance indexes

Revision ID: 003
Revises: 002
Create Date: 2025-01-01

パフォーマンス向上のためのインデックス追加:
- features テーブル: race_id + feature_version + asof_time での検索
- features テーブル: DISTINCT ON (race_id, horse_id) での最新取得
- fact_result テーブル: horse_id での過去走検索

Note:
    odds_ts_win の ix_odds_ts_win_race_asof は 001_initial_schema で作成済み
    PostgreSQL は btree を逆方向スキャンできるので DESC 専用 index は不要
"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Features テーブルのインデックス
    # 1. race_id + feature_version + asof_time での絞り込み用
    op.create_index(
        'ix_features_race_version_asof',
        'features',
        ['race_id', 'feature_version', 'asof_time'],
        unique=False,
    )
    
    # 2. DISTINCT ON (race_id, horse_id) ORDER BY asof_time DESC 用
    op.create_index(
        'ix_features_race_horse_asof',
        'features',
        ['race_id', 'horse_id', 'feature_version', 'asof_time'],
        unique=False,
    )
    
    # fact_result テーブル: horse_id での過去走検索用
    op.create_index(
        'ix_fact_result_horse_id',
        'fact_result',
        ['horse_id'],
        unique=False,
    )
    
    # ★ odds_ts_win の index は 001_initial_schema で既に作成済みのため不要
    # PostgreSQL は btree index を逆方向スキャンできるので DESC 専用は不要


def downgrade() -> None:
    op.drop_index('ix_fact_result_horse_id', table_name='fact_result')
    op.drop_index('ix_features_race_horse_asof', table_name='features')
    op.drop_index('ix_features_race_version_asof', table_name='features')

