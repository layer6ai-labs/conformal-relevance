"""Add profiling columns to runs and sample_scores.

Revision ID: 20250204_add_profiling
Revises: 20250203_simplify_icl_and_metrics
Create Date: 2025-02-04 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20250204_add_profiling"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add profiling columns to runs table
    op.add_column(
        "runs",
        sa.Column("total_duration_ms", sa.Integer(), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("total_llm_time_ms", sa.Integer(), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("avg_sample_time_ms", sa.Float(), nullable=True),
    )

    # Add profiling columns to sample_scores table
    op.add_column(
        "sample_scores",
        sa.Column("scoring_time_ms", sa.Integer(), nullable=True),
    )
    op.add_column(
        "sample_scores",
        sa.Column("llm_latency_ms", sa.Integer(), nullable=True),
    )
    op.add_column(
        "sample_scores",
        sa.Column("targeted_retry_count", sa.Integer(), default=0, nullable=True),
    )


def downgrade() -> None:
    # Remove profiling columns from sample_scores
    op.drop_column("sample_scores", "targeted_retry_count")
    op.drop_column("sample_scores", "llm_latency_ms")
    op.drop_column("sample_scores", "scoring_time_ms")

    # Remove profiling columns from runs
    op.drop_column("runs", "avg_sample_time_ms")
    op.drop_column("runs", "total_llm_time_ms")
    op.drop_column("runs", "total_duration_ms")
