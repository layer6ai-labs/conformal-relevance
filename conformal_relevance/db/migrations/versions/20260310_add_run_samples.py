"""Add run_samples table for per-sample score storage.

Revision ID: 20260310_add_run_samples
Revises: 20250204_add_materialized_views
Create Date: 2026-03-10 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260310_add_run_samples"
down_revision = "20250204_add_materialized_views"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_samples",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("sample_index", sa.Integer(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=True),
        sa.Column("ap", sa.Double(), nullable=True),
        sa.Column("scores", postgresql.JSONB(), nullable=True),
        sa.Column("labels", postgresql.JSONB(), nullable=False),
        sa.Column("n_sentences", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="scored"),
        sa.Column("split", sa.String(8), nullable=False, server_default="test"),
        sa.Column("sub_scores", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.CheckConstraint("status IN ('scored', 'failed')", name="run_samples_status_check"),
        sa.UniqueConstraint("run_id", "split", "sample_index", name="run_samples_unique_sample"),
    )
    op.create_index("ix_run_samples_run_id", "run_samples", ["run_id"])
    op.create_index("ix_run_samples_intent", "run_samples", ["run_id", "intent"])


def downgrade() -> None:
    op.drop_index("ix_run_samples_intent", table_name="run_samples")
    op.drop_index("ix_run_samples_run_id", table_name="run_samples")
    op.drop_table("run_samples")
