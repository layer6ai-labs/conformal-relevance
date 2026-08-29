"""Dashboard layout modules."""

from conformal_relevance.dashboard.layouts.overview import create_overview_layout
from conformal_relevance.dashboard.layouts.run_detail import create_run_detail_layout
from conformal_relevance.dashboard.layouts.comparison import create_comparison_layout
from conformal_relevance.dashboard.layouts.error_analysis import create_error_analysis_layout

__all__ = [
    "create_overview_layout",
    "create_run_detail_layout",
    "create_comparison_layout",
    "create_error_analysis_layout",
]
