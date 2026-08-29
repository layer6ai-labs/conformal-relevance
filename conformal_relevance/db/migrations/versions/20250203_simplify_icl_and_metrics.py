"""Simplify schema: remove ICL configurations table, add run_icl_examples junction table,
remove computed metrics from runs table.

Revision ID: 003
Revises: 002
Create Date: 2025-02-03

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create run_icl_examples junction table
    op.create_table(
        "run_icl_examples",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("sample_id", sa.Integer(), nullable=False),
        sa.Column("example_order", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f("fk_run_icl_examples_run_id_runs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["sample_id"],
            ["dataset_samples.id"],
            name=op.f("fk_run_icl_examples_sample_id_dataset_samples"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run_icl_examples")),
        sa.UniqueConstraint(
            "run_id", "sample_id", name="uq_run_icl_examples_run_sample"
        ),
    )
    op.create_index(
        "ix_run_icl_examples_run",
        "run_icl_examples",
        ["run_id"],
        unique=False,
    )

    # Remove columns from runs table
    op.drop_constraint("fk_runs_icl_config_id_icl_configurations", "runs", type_="foreignkey")
    op.drop_column("runs", "icl_config_id")
    op.drop_column("runs", "mean_average_precision")
    op.drop_column("runs", "n_samples_scored")
    op.drop_column("runs", "n_samples_failed")

    # Drop icl_configurations table
    op.drop_table("icl_configurations")


def downgrade() -> None:
    # Recreate icl_configurations table
    op.create_table(
        "icl_configurations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("n_examples", sa.Integer(), nullable=True),
        sa.Column("strategy", sa.String(length=64), nullable=True),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("random_seed", sa.Integer(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name=op.f("fk_icl_configurations_dataset_id_datasets"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icl_configurations")),
        sa.UniqueConstraint(
            "dataset_id", "name", name="uq_icl_configurations_dataset_name"
        ),
    )
    op.create_index(
        "ix_icl_dataset_name",
        "icl_configurations",
        ["dataset_id", "name"],
        unique=False,
    )

    # Add columns back to runs table
    op.add_column("runs", sa.Column("icl_config_id", sa.Integer(), nullable=True))
    op.add_column(
        "runs", sa.Column("mean_average_precision", sa.Float(), nullable=True)
    )
    op.add_column("runs", sa.Column("n_samples_scored", sa.Integer(), nullable=True))
    op.add_column("runs", sa.Column("n_samples_failed", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_runs_icl_config_id_icl_configurations",
        "runs",
        "icl_configurations",
        ["icl_config_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Drop run_icl_examples table
    op.drop_index("ix_run_icl_examples_run", table_name="run_icl_examples")
    op.drop_table("run_icl_examples")
