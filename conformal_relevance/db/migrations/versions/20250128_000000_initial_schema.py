"""Initial schema for LLM run results database.

Revision ID: 001
Revises:
Create Date: 2025-01-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create datasets table
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=True),
        sa.Column("has_intent", sa.Boolean(), nullable=True, default=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_datasets")),
        sa.UniqueConstraint("name", name=op.f("uq_datasets_name")),
    )
    op.create_index(op.f("ix_datasets_name"), "datasets", ["name"], unique=False)

    # Create dataset_samples table
    op.create_table(
        "dataset_samples",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("sample_hash", sa.String(length=64), nullable=True),
        sa.Column("split", sa.String(length=32), nullable=False),
        sa.Column("intent", sa.Text(), nullable=True),
        sa.Column("input_text", sa.Text(), nullable=True),
        sa.Column("input_units", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("input_unit_labels", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_units", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name=op.f("fk_dataset_samples_dataset_id_datasets"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_samples")),
        sa.UniqueConstraint("sample_hash", name=op.f("uq_dataset_samples_sample_hash")),
    )
    op.create_index(
        op.f("ix_samples_dataset_split"),
        "dataset_samples",
        ["dataset_id", "split"],
        unique=False,
    )
    op.create_index(
        op.f("ix_samples_dataset_intent"),
        "dataset_samples",
        ["dataset_id", "intent"],
        unique=False,
    )
    op.create_index(op.f("ix_samples_hash"), "dataset_samples", ["sample_hash"], unique=False)

    # Create llm_models table
    op.create_table(
        "llm_models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_models")),
        sa.UniqueConstraint("name", name=op.f("uq_llm_models_name")),
    )
    op.create_index(op.f("ix_llm_models_name"), "llm_models", ["name"], unique=False)
    op.create_index(op.f("ix_llm_models_provider"), "llm_models", ["provider"], unique=False)

    # Create icl_configurations table
    op.create_table(
        "icl_configurations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("n_examples", sa.Integer(), nullable=True, default=3),
        sa.Column("strategy", sa.String(length=64), nullable=True, default="random"),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("random_seed", sa.Integer(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name=op.f("fk_icl_configurations_dataset_id_datasets"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icl_configurations")),
        sa.UniqueConstraint(
            "dataset_id", "name", name=op.f("uq_icl_configurations_dataset_name")
        ),
    )
    op.create_index(
        op.f("ix_icl_dataset_name"),
        "icl_configurations",
        ["dataset_id", "name"],
        unique=False,
    )

    # Create runs table
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=True),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("llm_model_id", sa.Integer(), nullable=False),
        sa.Column("icl_config_id", sa.Integer(), nullable=True),
        sa.Column("split", sa.String(length=32), nullable=True, default="test"),
        sa.Column("max_concurrency", sa.Integer(), nullable=True),
        sa.Column("max_retries", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True, default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("mean_average_precision", sa.Float(), nullable=True),
        sa.Column("n_samples_scored", sa.Integer(), nullable=True),
        sa.Column("n_samples_failed", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name=op.f("fk_runs_dataset_id_datasets"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["llm_model_id"],
            ["llm_models.id"],
            name=op.f("fk_runs_llm_model_id_llm_models"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["icl_config_id"],
            ["icl_configurations.id"],
            name=op.f("fk_runs_icl_config_id_icl_configurations"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
    )
    op.create_index(
        op.f("ix_runs_dataset_model"), "runs", ["dataset_id", "llm_model_id"], unique=False
    )
    op.create_index(
        op.f("ix_runs_status"), "runs", ["status", "created_at"], unique=False
    )

    # Create sample_scores table
    op.create_table(
        "sample_scores",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("sample_id", sa.Integer(), nullable=False),
        sa.Column("scores", postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column("average_precision", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True, default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=True, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f("fk_sample_scores_run_id_runs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["sample_id"],
            ["dataset_samples.id"],
            name=op.f("fk_sample_scores_sample_id_dataset_samples"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sample_scores")),
        sa.UniqueConstraint("run_id", "sample_id", name=op.f("uq_sample_scores_run_sample")),
    )
    op.create_index(op.f("ix_scores_run"), "sample_scores", ["run_id"], unique=False)
    op.create_index(op.f("ix_scores_sample"), "sample_scores", ["sample_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_scores_sample"), table_name="sample_scores")
    op.drop_index(op.f("ix_scores_run"), table_name="sample_scores")
    op.drop_table("sample_scores")

    op.drop_index(op.f("ix_runs_status"), table_name="runs")
    op.drop_index(op.f("ix_runs_dataset_model"), table_name="runs")
    op.drop_table("runs")

    op.drop_index(op.f("ix_icl_dataset_name"), table_name="icl_configurations")
    op.drop_table("icl_configurations")

    op.drop_index(op.f("ix_llm_models_provider"), table_name="llm_models")
    op.drop_index(op.f("ix_llm_models_name"), table_name="llm_models")
    op.drop_table("llm_models")

    op.drop_index(op.f("ix_samples_hash"), table_name="dataset_samples")
    op.drop_index(op.f("ix_samples_dataset_intent"), table_name="dataset_samples")
    op.drop_index(op.f("ix_samples_dataset_split"), table_name="dataset_samples")
    op.drop_table("dataset_samples")

    op.drop_index(op.f("ix_datasets_name"), table_name="datasets")
    op.drop_table("datasets")
