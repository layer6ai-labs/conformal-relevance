"""Rename summary to output_text in dataset_samples table.

Revision ID: 002
Revises: 001
Create Date: 2025-02-01

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename columns in dataset_samples table
    op.alter_column(
        "dataset_samples",
        "summary",
        new_column_name="output_text",
    )
    op.alter_column(
        "dataset_samples",
        "summary_units",
        new_column_name="output_units",
    )


def downgrade() -> None:
    # Revert column names
    op.alter_column(
        "dataset_samples",
        "output_text",
        new_column_name="summary",
    )
    op.alter_column(
        "dataset_samples",
        "output_units",
        new_column_name="summary_units",
    )
