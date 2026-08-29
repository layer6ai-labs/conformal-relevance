"""Add ICL strategy tracking tables and columns.

Revision ID: 20250204_add_icl_strategies
Revises: 20250204_add_profiling
Create Date: 2025-02-04 00:00:01.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision = "20250204_add_icl_strategies"
down_revision = "20250204_add_profiling"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add ICL strategy columns to runs table
    op.add_column(
        "runs",
        sa.Column("icl_strategy_name", sa.String(64), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("icl_strategy_params", JSONB, nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("icl_permutation_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("icl_diversity_metrics", JSONB, nullable=True),
    )

    # Create bandit_arm_states table for persisting bandit learning state
    op.create_table(
        "bandit_arm_states",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sample_id", sa.Integer(), sa.ForeignKey("dataset_samples.id", ondelete="CASCADE"), nullable=False),
        sa.Column("algorithm", sa.String(32), nullable=False),
        sa.Column("n_pulls", sa.Integer(), default=0, nullable=False),
        sa.Column("total_reward", sa.Float(), default=0.0, nullable=False),
        sa.Column("successes", sa.Float(), default=1.0, nullable=False),
        sa.Column("failures", sa.Float(), default=1.0, nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("dataset_id", "sample_id", "algorithm", name="uq_bandit_arm_dataset_sample_algo"),
    )
    op.create_index("ix_bandit_arm_states_dataset", "bandit_arm_states", ["dataset_id"])

    # Create icl_strategy_results table for aggregated strategy comparison
    op.create_table(
        "icl_strategy_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("llm_model_id", sa.Integer(), sa.ForeignKey("llm_models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("strategy_params", JSONB, nullable=True),
        sa.Column("n_runs", sa.Integer(), default=0, nullable=False),
        sa.Column("mean_ap", sa.Float(), nullable=True),
        sa.Column("std_ap", sa.Float(), nullable=True),
        sa.Column("min_ap", sa.Float(), nullable=True),
        sa.Column("max_ap", sa.Float(), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.UniqueConstraint("dataset_id", "llm_model_id", "strategy_name", name="uq_strategy_results_dataset_model_strategy"),
    )
    op.create_index("ix_icl_strategy_results_dataset", "icl_strategy_results", ["dataset_id"])
    op.create_index("ix_icl_strategy_results_strategy", "icl_strategy_results", ["strategy_name"])


def downgrade() -> None:
    # Drop tables
    op.drop_table("icl_strategy_results")
    op.drop_table("bandit_arm_states")

    # Remove columns from runs
    op.drop_column("runs", "icl_diversity_metrics")
    op.drop_column("runs", "icl_permutation_id")
    op.drop_column("runs", "icl_strategy_params")
    op.drop_column("runs", "icl_strategy_name")
