"""Evaluation framework for relevancy scoring experiments.

This module provides metrics computation, statistical analysis,
and report generation for scoring experiment results.
"""

from conformal_relevance.evaluation.metrics import (
    IntentMetrics,
    RunReport,
    compute_run_metrics,
    generate_run_report,
)
from conformal_relevance.evaluation.statistics import (
    BootstrapResult,
    BootstrapMAPResult,
    StatisticalTestResult,
    bootstrap_map_ci,
    compare_models_bootstrap,
    compare_models_paired_ttest,
    compare_strategies,
    compute_confidence_interval,
)
from conformal_relevance.evaluation.paper_data import (
    extract_paper_data,
    compute_ap_distribution_stats,
)
from conformal_relevance.evaluation.queries import (
    get_run_summary,
    get_runs_comparison,
    get_strategy_comparison,
    get_performance_matrix,
    query_run_metrics,
)
from conformal_relevance.evaluation.error_analysis import (
    ErrorCategory,
    ErrorSeverity,
    ErrorInfo,
    ErrorPattern,
    ErrorAnalysisResult,
    categorize_error,
    analyze_errors,
    find_error_patterns,
    analyze_errors_by_intent,
    analyze_errors_by_input_length,
    get_error_summary_for_run,
    get_error_summary_for_dataset,
    format_error_report,
)

__all__ = [
    # Metrics
    "IntentMetrics",
    "RunReport",
    "compute_run_metrics",
    "generate_run_report",
    # Statistics
    "StatisticalTestResult",
    "BootstrapResult",
    "BootstrapMAPResult",
    "bootstrap_map_ci",
    "compare_models_paired_ttest",
    "compare_models_bootstrap",
    "compare_strategies",
    "compute_confidence_interval",
    # Paper data extraction
    "extract_paper_data",
    "compute_ap_distribution_stats",
    # Queries
    "get_run_summary",
    "get_runs_comparison",
    "get_strategy_comparison",
    "get_performance_matrix",
    "query_run_metrics",
    # Error Analysis
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorInfo",
    "ErrorPattern",
    "ErrorAnalysisResult",
    "categorize_error",
    "analyze_errors",
    "find_error_patterns",
    "analyze_errors_by_intent",
    "analyze_errors_by_input_length",
    "get_error_summary_for_run",
    "get_error_summary_for_dataset",
    "format_error_report",
]
