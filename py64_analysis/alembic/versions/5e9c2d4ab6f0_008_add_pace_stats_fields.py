"""008 add pace stats fields (C4 pace history)

Revision ID: 5e9c2d4ab6f0
Revises: 3b2e1b7d9b1a
Create Date: 2026-01-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "5e9c2d4ab6f0"
down_revision: Union[str, None] = "3b2e1b7d9b1a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("fact_race", sa.Column("pace_diff_sec", sa.Float(), nullable=True))
    op.add_column("fact_race", sa.Column("lap_mean_sec", sa.Float(), nullable=True))
    op.add_column("fact_race", sa.Column("lap_std_sec", sa.Float(), nullable=True))
    op.add_column("fact_race", sa.Column("lap_slope", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("fact_race", "lap_slope")
    op.drop_column("fact_race", "lap_std_sec")
    op.drop_column("fact_race", "lap_mean_sec")
    op.drop_column("fact_race", "pace_diff_sec")
