"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-12-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Dimension Tables
    op.create_table('dim_track',
        sa.Column('track_code', sa.String(length=2), nullable=False),
        sa.Column('name', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('track_code')
    )
    
    op.create_table('dim_horse',
        sa.Column('horse_id', sa.String(length=10), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=True),
        sa.Column('sex', sa.String(length=1), nullable=True),
        sa.Column('birth_year', sa.SmallInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('horse_id')
    )
    
    op.create_table('dim_jockey',
        sa.Column('jockey_id', sa.String(length=5), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('jockey_id')
    )
    
    op.create_table('dim_trainer',
        sa.Column('trainer_id', sa.String(length=5), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('trainer_id')
    )
    
    # Fact Tables
    op.create_table('fact_race',
        sa.Column('race_id', sa.String(length=16), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('track_code', sa.String(length=2), nullable=True),
        sa.Column('race_no', sa.SmallInteger(), nullable=False),
        sa.Column('start_time', sa.Time(), nullable=True),
        sa.Column('surface', sa.String(length=1), nullable=True),
        sa.Column('distance', sa.SmallInteger(), nullable=True),
        sa.Column('track_cd', sa.String(length=2), nullable=True),
        sa.Column('going_turf', sa.String(length=1), nullable=True),
        sa.Column('going_dirt', sa.String(length=1), nullable=True),
        sa.Column('weather', sa.String(length=1), nullable=True),
        sa.Column('field_size', sa.SmallInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['track_code'], ['dim_track.track_code'], ),
        sa.PrimaryKeyConstraint('race_id')
    )
    op.create_index('ix_fact_race_date', 'fact_race', ['date'], unique=False)
    
    op.create_table('fact_entry',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('race_id', sa.String(length=16), nullable=False),
        sa.Column('horse_id', sa.String(length=10), nullable=True),
        sa.Column('horse_no', sa.SmallInteger(), nullable=False),
        sa.Column('frame_no', sa.SmallInteger(), nullable=True),
        sa.Column('jockey_id', sa.String(length=5), nullable=True),
        sa.Column('trainer_id', sa.String(length=5), nullable=True),
        sa.Column('weight_carried', sa.Numeric(precision=4, scale=1), nullable=True),
        sa.Column('horse_weight', sa.SmallInteger(), nullable=True),
        sa.Column('weight_diff', sa.SmallInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['horse_id'], ['dim_horse.horse_id'], ),
        sa.ForeignKeyConstraint(['jockey_id'], ['dim_jockey.jockey_id'], ),
        sa.ForeignKeyConstraint(['race_id'], ['fact_race.race_id'], ),
        sa.ForeignKeyConstraint(['trainer_id'], ['dim_trainer.trainer_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_fact_entry_race_horse', 'fact_entry', ['race_id', 'horse_no'], unique=True)
    
    op.create_table('fact_result',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('race_id', sa.String(length=16), nullable=False),
        sa.Column('horse_id', sa.String(length=10), nullable=True),
        sa.Column('horse_no', sa.SmallInteger(), nullable=False),
        sa.Column('finish_pos', sa.SmallInteger(), nullable=True),
        sa.Column('time', sa.String(length=10), nullable=True),
        sa.Column('time_sec', sa.Numeric(precision=6, scale=1), nullable=True),
        sa.Column('margin', sa.String(length=10), nullable=True),
        sa.Column('last_3f', sa.Numeric(precision=4, scale=1), nullable=True),
        sa.Column('odds', sa.Numeric(precision=6, scale=1), nullable=True),
        sa.Column('popularity', sa.SmallInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['horse_id'], ['dim_horse.horse_id'], ),
        sa.ForeignKeyConstraint(['race_id'], ['fact_race.race_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_fact_result_race_horse', 'fact_result', ['race_id', 'horse_no'], unique=True)
    
    # Odds Tables
    op.create_table('odds_ts_win',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('race_id', sa.String(length=16), nullable=False),
        sa.Column('horse_no', sa.SmallInteger(), nullable=False),
        sa.Column('asof_time', sa.DateTime(), nullable=False),
        sa.Column('data_kubun', sa.String(length=1), nullable=False),
        sa.Column('odds', sa.Numeric(precision=6, scale=1), nullable=True),
        sa.Column('popularity', sa.SmallInteger(), nullable=True),
        sa.Column('total_sales', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['race_id'], ['fact_race.race_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_odds_ts_win_race_asof', 'odds_ts_win', ['race_id', 'asof_time'], unique=False)
    op.create_index('ix_odds_ts_win_unique', 'odds_ts_win', ['race_id', 'horse_no', 'asof_time', 'data_kubun'], unique=True)
    
    op.create_table('odds_ts_place',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('race_id', sa.String(length=16), nullable=False),
        sa.Column('horse_no', sa.SmallInteger(), nullable=False),
        sa.Column('asof_time', sa.DateTime(), nullable=False),
        sa.Column('data_kubun', sa.String(length=1), nullable=False),
        sa.Column('odds_low', sa.Numeric(precision=6, scale=1), nullable=True),
        sa.Column('odds_high', sa.Numeric(precision=6, scale=1), nullable=True),
        sa.Column('popularity', sa.SmallInteger(), nullable=True),
        sa.Column('total_sales', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['race_id'], ['fact_race.race_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_odds_ts_place_race_asof', 'odds_ts_place', ['race_id', 'asof_time'], unique=False)
    op.create_index('ix_odds_ts_place_unique', 'odds_ts_place', ['race_id', 'horse_no', 'asof_time', 'data_kubun'], unique=True)
    
    op.create_table('odds_ts_quinella',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('race_id', sa.String(length=16), nullable=False),
        sa.Column('horse_no_1', sa.SmallInteger(), nullable=False),
        sa.Column('horse_no_2', sa.SmallInteger(), nullable=False),
        sa.Column('asof_time', sa.DateTime(), nullable=False),
        sa.Column('data_kubun', sa.String(length=1), nullable=False),
        sa.Column('odds', sa.Numeric(precision=8, scale=1), nullable=True),
        sa.Column('popularity', sa.SmallInteger(), nullable=True),
        sa.Column('total_sales', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['race_id'], ['fact_race.race_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint('horse_no_1 < horse_no_2', name='ck_quinella_order')
    )
    op.create_index('ix_odds_ts_quinella_race_asof', 'odds_ts_quinella', ['race_id', 'asof_time'], unique=False)
    op.create_index('ix_odds_ts_quinella_unique', 'odds_ts_quinella', ['race_id', 'horse_no_1', 'horse_no_2', 'asof_time', 'data_kubun'], unique=True)
    
    # Features / Predictions
    op.create_table('features',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('race_id', sa.String(length=16), nullable=False),
        sa.Column('horse_id', sa.String(length=10), nullable=True),
        sa.Column('feature_version', sa.String(length=20), nullable=False),
        sa.Column('asof_time', sa.DateTime(), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['race_id'], ['fact_race.race_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_features_race_version', 'features', ['race_id', 'feature_version'], unique=False)
    
    op.create_table('predictions',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('race_id', sa.String(length=16), nullable=False),
        sa.Column('horse_id', sa.String(length=10), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('asof_time', sa.DateTime(), nullable=False),
        sa.Column('p_win', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('p_place', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('calibration_meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['race_id'], ['fact_race.race_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_predictions_race_model', 'predictions', ['race_id', 'model_version'], unique=False)
    
    # Bets
    op.create_table('bet_signal',
        sa.Column('signal_id', sa.String(length=50), nullable=False),
        sa.Column('run_id', sa.String(length=50), nullable=False),
        sa.Column('race_id', sa.String(length=16), nullable=False),
        sa.Column('ticket_type', sa.String(length=20), nullable=False),
        sa.Column('selection', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('asof_time', sa.DateTime(), nullable=False),
        sa.Column('p_hat', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('odds_snapshot', sa.Numeric(precision=8, scale=1), nullable=True),
        sa.Column('ev', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('stake_yen', sa.Integer(), nullable=True),
        sa.Column('reason', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['race_id'], ['fact_race.race_id'], ),
        sa.PrimaryKeyConstraint('signal_id')
    )
    op.create_index('ix_bet_signal_race', 'bet_signal', ['race_id'], unique=False)
    op.create_index('ix_bet_signal_run', 'bet_signal', ['run_id'], unique=False)
    
    op.create_table('bet_settlement',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('signal_id', sa.String(length=50), nullable=False),
        sa.Column('result_status', sa.String(length=20), nullable=True),
        sa.Column('payout_yen', sa.Integer(), nullable=True),
        sa.Column('profit_yen', sa.Integer(), nullable=True),
        sa.Column('settled_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['signal_id'], ['bet_signal.signal_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_bet_settlement_signal', 'bet_settlement', ['signal_id'], unique=True)


def downgrade() -> None:
    op.drop_table('bet_settlement')
    op.drop_table('bet_signal')
    op.drop_table('predictions')
    op.drop_table('features')
    op.drop_table('odds_ts_quinella')
    op.drop_table('odds_ts_place')
    op.drop_table('odds_ts_win')
    op.drop_table('fact_result')
    op.drop_table('fact_entry')
    op.drop_table('fact_race')
    op.drop_table('dim_trainer')
    op.drop_table('dim_jockey')
    op.drop_table('dim_horse')
    op.drop_table('dim_track')



