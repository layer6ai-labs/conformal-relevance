"""add_split_to_run_samples

Revision ID: c5f575c7ea10
Revises: 20260310_add_run_samples
Create Date: 2026-03-12 23:39:43.200881

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c5f575c7ea10'
down_revision: Union[str, None] = '20260310_add_run_samples'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add split column (existing rows default to 'test')
    op.add_column('run_samples', sa.Column('split', sa.String(length=8), nullable=False, server_default='test'))
    # Update unique constraint to include split (allows same run_id+sample_index across cal/test)
    op.drop_constraint('run_samples_unique_sample', 'run_samples', type_='unique')
    op.create_unique_constraint('run_samples_unique_sample', 'run_samples', ['run_id', 'split', 'sample_index'])


def downgrade() -> None:
    op.drop_constraint('run_samples_unique_sample', 'run_samples', type_='unique')
    op.create_unique_constraint('run_samples_unique_sample', 'run_samples', ['run_id', 'sample_index'])
    op.drop_column('run_samples', 'split')
