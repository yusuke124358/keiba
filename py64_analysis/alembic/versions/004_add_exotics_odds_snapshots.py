"""add exotics odds realtime snapshot tables

Revision ID: 004
Revises: 003
Create Date: 2026-01-05

目的:
  - 0B35(三連複) / 0B36(三連単) の速報オッズを「スナップショット単位」で保存する
  - 後から buy/close の抽出や欠落率・鮮度評価ができるようにする
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "odds_rt_snapshot",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("ticket_type", sa.String(length=20), nullable=False),
        sa.Column("data_spec", sa.String(length=4), nullable=False),
        sa.Column("record_type", sa.String(length=2), nullable=True),
        sa.Column("race_id", sa.String(length=16), nullable=False),
        sa.Column("asof_time", sa.DateTime(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(), nullable=False),
        sa.Column("data_kubun", sa.String(length=1), nullable=False),
        sa.Column("sale_flag", sa.String(length=1), nullable=True),
        sa.Column("total_sales", sa.BigInteger(), nullable=True),
        sa.Column("n_rows", sa.Integer(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["race_id"], ["fact_race.race_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_odds_rt_snapshot_race_asof",
        "odds_rt_snapshot",
        ["race_id", "asof_time"],
        unique=False,
    )
    op.create_index(
        "ix_odds_rt_snapshot_ticket_race_asof",
        "odds_rt_snapshot",
        ["ticket_type", "race_id", "asof_time"],
        unique=False,
    )
    op.create_index(
        "ix_odds_rt_snapshot_unique",
        "odds_rt_snapshot",
        ["ticket_type", "race_id", "asof_time", "data_kubun", "fetched_at"],
        unique=True,
    )

    op.create_table(
        "odds_rt_trio",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("snapshot_id", sa.BigInteger(), nullable=False),
        sa.Column("horse_no_1", sa.SmallInteger(), nullable=False),
        sa.Column("horse_no_2", sa.SmallInteger(), nullable=False),
        sa.Column("horse_no_3", sa.SmallInteger(), nullable=False),
        sa.Column("odds", sa.Numeric(precision=8, scale=1), nullable=True),
        sa.Column("popularity", sa.SmallInteger(), nullable=True),
        sa.Column("odds_status", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["snapshot_id"], ["odds_rt_snapshot.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("horse_no_1 < horse_no_2", name="ck_trio_order_12"),
        sa.CheckConstraint("horse_no_2 < horse_no_3", name="ck_trio_order_23"),
    )
    op.create_index(
        "ix_odds_rt_trio_snapshot",
        "odds_rt_trio",
        ["snapshot_id"],
        unique=False,
    )
    op.create_index(
        "ix_odds_rt_trio_unique",
        "odds_rt_trio",
        ["snapshot_id", "horse_no_1", "horse_no_2", "horse_no_3"],
        unique=True,
    )

    op.create_table(
        "odds_rt_trifecta",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("snapshot_id", sa.BigInteger(), nullable=False),
        sa.Column("first_no", sa.SmallInteger(), nullable=False),
        sa.Column("second_no", sa.SmallInteger(), nullable=False),
        sa.Column("third_no", sa.SmallInteger(), nullable=False),
        sa.Column("odds", sa.Numeric(precision=9, scale=1), nullable=True),
        sa.Column("popularity", sa.SmallInteger(), nullable=True),
        sa.Column("odds_status", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["snapshot_id"], ["odds_rt_snapshot.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_odds_rt_trifecta_snapshot",
        "odds_rt_trifecta",
        ["snapshot_id"],
        unique=False,
    )
    op.create_index(
        "ix_odds_rt_trifecta_unique",
        "odds_rt_trifecta",
        ["snapshot_id", "first_no", "second_no", "third_no"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_odds_rt_trifecta_unique", table_name="odds_rt_trifecta")
    op.drop_index("ix_odds_rt_trifecta_snapshot", table_name="odds_rt_trifecta")
    op.drop_table("odds_rt_trifecta")

    op.drop_index("ix_odds_rt_trio_unique", table_name="odds_rt_trio")
    op.drop_index("ix_odds_rt_trio_snapshot", table_name="odds_rt_trio")
    op.drop_table("odds_rt_trio")

    op.drop_index("ix_odds_rt_snapshot_unique", table_name="odds_rt_snapshot")
    op.drop_index("ix_odds_rt_snapshot_ticket_race_asof", table_name="odds_rt_snapshot")
    op.drop_index("ix_odds_rt_snapshot_race_asof", table_name="odds_rt_snapshot")
    op.drop_table("odds_rt_snapshot")


