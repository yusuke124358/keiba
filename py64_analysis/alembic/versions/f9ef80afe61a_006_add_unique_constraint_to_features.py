"""006_add_unique_constraint_to_features

Revision ID: f9ef80afe61a
Revises: 005
Create Date: 2026-01-07 02:11:35.768625

目的:
  - featuresテーブルにUNIQUE制約を追加して、重複挿入を防止する
  - (race_id, horse_id, feature_version, asof_time) の組み合わせで一意性を保証
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f9ef80afe61a'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # UNIQUE制約を追加
    # 注意: horse_idはNULL可能なので、NULLは別の値として扱われる
    # 実務上は horse_id が NULL の行は存在しない想定だが、念のため
    op.create_unique_constraint(
        "uq_features_race_horse_version_asof",
        "features",
        ["race_id", "horse_id", "feature_version", "asof_time"],
    )


def downgrade() -> None:
    # 制約が存在する場合のみ削除
    try:
        op.drop_constraint(
            "uq_features_race_horse_version_asof",
            "features",
            type_="unique",
        )
    except Exception:
        # 制約が存在しない場合は無視（既に削除済みなど）
        pass
