"""Add materialized view for dashboard queries.

Revision ID: 20250204_add_materialized_views
Revises: 20250204_add_icl_strategies
Create Date: 2025-02-04 00:00:02.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "20250204_add_materialized_views"
down_revision = "20250204_add_icl_strategies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create materialized view for run summaries (optimized for dashboard)
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_run_summary AS
        SELECT
            r.id AS run_id,
            r.name AS run_name,
            r.status,
            r.created_at,
            r.started_at,
            r.completed_at,
            r.total_duration_ms,
            r.total_llm_time_ms,
            r.avg_sample_time_ms,
            r.icl_strategy_name,
            r.icl_strategy_params,
            d.id AS dataset_id,
            d.name AS dataset_name,
            d.has_intent,
            m.id AS model_id,
            m.name AS model_name,
            m.provider AS model_provider,
            COUNT(ss.id) AS n_samples,
            COUNT(ss.id) FILTER (WHERE ss.status = 'success') AS n_success,
            COUNT(ss.id) FILTER (WHERE ss.status = 'failed') AS n_failed,
            AVG(ss.average_precision) AS mean_ap,
            STDDEV(ss.average_precision) AS std_ap,
            MIN(ss.average_precision) AS min_ap,
            MAX(ss.average_precision) AS max_ap,
            AVG(ss.scoring_time_ms) AS avg_scoring_time_ms,
            SUM(ss.retry_count) AS total_retries,
            SUM(ss.targeted_retry_count) AS total_targeted_retries
        FROM runs r
        JOIN datasets d ON r.dataset_id = d.id
        JOIN llm_models m ON r.llm_model_id = m.id
        LEFT JOIN sample_scores ss ON r.id = ss.run_id
        GROUP BY r.id, d.id, m.id
    """)

    # Create index on materialized view
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_mv_run_summary_run_id
        ON mv_run_summary (run_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_mv_run_summary_dataset
        ON mv_run_summary (dataset_id, created_at DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_mv_run_summary_model
        ON mv_run_summary (model_id, created_at DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_mv_run_summary_strategy
        ON mv_run_summary (icl_strategy_name, created_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS mv_run_summary CASCADE")
