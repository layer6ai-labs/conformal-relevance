"""Dashboard callback modules."""

from conformal_relevance.dashboard.callbacks.navigation import register_navigation_callbacks
from conformal_relevance.dashboard.callbacks.overview import register_overview_callbacks
from conformal_relevance.dashboard.callbacks.run_detail import register_run_detail_callbacks
from conformal_relevance.dashboard.callbacks.comparison import register_comparison_callbacks
from conformal_relevance.dashboard.callbacks.error_analysis import register_error_analysis_callbacks


def register_callbacks(app):
    """Register all dashboard callbacks.

    Args:
        app: Dash application instance.
    """
    register_navigation_callbacks(app)
    register_overview_callbacks(app)
    register_run_detail_callbacks(app)
    register_comparison_callbacks(app)
    register_error_analysis_callbacks(app)
