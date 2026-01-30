"""add exotics payouts tables from HR

Revision ID: 005
Revises: 004
Create Date: 2026-01-05

目的:
  - HRレコードから取り込む「3連複/3連単 払戻」と「返還馬番」をDB化し、
    backtest決済（同着/返還）を正しく扱えるようにする。
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fact_refund_horse",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("race_id", sa.String(length=16), nullable=False),
        sa.Column("horse_no", sa.SmallInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["race_id"], ["fact_race.race_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fact_refund_horse_race", "fact_refund_horse", ["race_id"], unique=False)
    op.create_index(
        "ix_fact_refund_horse_unique",
        "fact_refund_horse",
        ["race_id", "horse_no"],
        unique=True,
    )

    op.create_table(
        "fact_payout_trio",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("race_id", sa.String(length=16), nullable=False),
        sa.Column("horse_no_1", sa.SmallInteger(), nullable=False),
        sa.Column("horse_no_2", sa.SmallInteger(), nullable=False),
        sa.Column("horse_no_3", sa.SmallInteger(), nullable=False),
        sa.Column("payout_yen", sa.Integer(), nullable=True),
        sa.Column("popularity", sa.SmallInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["race_id"], ["fact_race.race_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("horse_no_1 < horse_no_2", name="ck_fact_trio_order_12"),
        sa.CheckConstraint("horse_no_2 < horse_no_3", name="ck_fact_trio_order_23"),
    )
    op.create_index("ix_fact_payout_trio_race", "fact_payout_trio", ["race_id"], unique=False)
    op.create_index(
        "ix_fact_payout_trio_unique",
        "fact_payout_trio",
        ["race_id", "horse_no_1", "horse_no_2", "horse_no_3"],
        unique=True,
    )

    op.create_table(
        "fact_payout_trifecta",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("race_id", sa.String(length=16), nullable=False),
        sa.Column("first_no", sa.SmallInteger(), nullable=False),
        sa.Column("second_no", sa.SmallInteger(), nullable=False),
        sa.Column("third_no", sa.SmallInteger(), nullable=False),
        sa.Column("payout_yen", sa.Integer(), nullable=True),
        sa.Column("popularity", sa.SmallInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["race_id"], ["fact_race.race_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fact_payout_trifecta_race", "fact_payout_trifecta", ["race_id"], unique=False)
    op.create_index(
        "ix_fact_payout_trifecta_unique",
        "fact_payout_trifecta",
        ["race_id", "first_no", "second_no", "third_no"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_fact_payout_trifecta_unique", table_name="fact_payout_trifecta")
    op.drop_index("ix_fact_payout_trifecta_race", table_name="fact_payout_trifecta")
    op.drop_table("fact_payout_trifecta")

    op.drop_index("ix_fact_payout_trio_unique", table_name="fact_payout_trio")
    op.drop_index("ix_fact_payout_trio_race", table_name="fact_payout_trio")
    op.drop_table("fact_payout_trio")

    op.drop_index("ix_fact_refund_horse_unique", table_name="fact_refund_horse")
    op.drop_index("ix_fact_refund_horse_race", table_name="fact_refund_horse")
    op.drop_table("fact_refund_horse")


