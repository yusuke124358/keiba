"""Add raw_ingestion_log table

Revision ID: 002
Revises: 001
Create Date: 2024-12-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('raw_ingestion_log',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('data_spec', sa.String(length=20), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('record_count', sa.Integer(), nullable=True),
        sa.Column('ra_count', sa.Integer(), nullable=True),
        sa.Column('se_count', sa.Integer(), nullable=True),
        sa.Column('o1_count', sa.Integer(), nullable=True),
        sa.Column('o2_count', sa.Integer(), nullable=True),
        sa.Column('error_count', sa.Integer(), nullable=True),
        sa.Column('ingested_at', sa.DateTime(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('file_path')
    )


def downgrade() -> None:
    op.drop_table('raw_ingestion_log')



