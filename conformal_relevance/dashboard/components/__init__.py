"""Reusable dashboard components."""

from conformal_relevance.dashboard.components.charts import (
    create_ap_histogram,
    create_model_comparison_bar,
    create_strategy_heatmap,
    create_intent_breakdown_bar,
)
from conformal_relevance.dashboard.components.tables import (
    create_runs_table,
    create_scores_table,
    create_comparison_table,
    create_performance_matrix_table,
    create_strategy_icl_table,
)

__all__ = [
    # Charts
    "create_ap_histogram",
    "create_model_comparison_bar",
    "create_strategy_heatmap",
    "create_intent_breakdown_bar",
    # Tables
    "create_runs_table",
    "create_scores_table",
    "create_comparison_table",
    "create_performance_matrix_table",
    "create_strategy_icl_table",
]
