"""007 add pace and corner fields (C4)

Revision ID: 3b2e1b7d9b1a
Revises: f9ef80afe61a
Create Date: 2026-01-07
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "3b2e1b7d9b1a"
down_revision: Union[str, None] = "f9ef80afe61a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # fact_race: pace materials (RA)
    op.add_column("fact_race", sa.Column("lap_times_200m", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("fact_race", sa.Column("pace_first3f", sa.Numeric(4, 1), nullable=True))
    op.add_column("fact_race", sa.Column("pace_first4f", sa.Numeric(4, 1), nullable=True))
    op.add_column("fact_race", sa.Column("pace_last3f", sa.Numeric(4, 1), nullable=True))
    op.add_column("fact_race", sa.Column("pace_last4f", sa.Numeric(4, 1), nullable=True))

    # fact_result: corner passing order (SE)
    op.add_column("fact_result", sa.Column("pos_1c", sa.SmallInteger(), nullable=True))
    op.add_column("fact_result", sa.Column("pos_2c", sa.SmallInteger(), nullable=True))
    op.add_column("fact_result", sa.Column("pos_3c", sa.SmallInteger(), nullable=True))
    op.add_column("fact_result", sa.Column("pos_4c", sa.SmallInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column("fact_result", "pos_4c")
    op.drop_column("fact_result", "pos_3c")
    op.drop_column("fact_result", "pos_2c")
    op.drop_column("fact_result", "pos_1c")

    op.drop_column("fact_race", "pace_last4f")
    op.drop_column("fact_race", "pace_last3f")
    op.drop_column("fact_race", "pace_first4f")
    op.drop_column("fact_race", "pace_first3f")
    op.drop_column("fact_race", "lap_times_200m")







