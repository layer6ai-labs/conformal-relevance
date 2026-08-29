"""Plotly Dash dashboard for experiment evaluation.

This module provides an interactive dashboard for visualizing and comparing
relevancy scoring experiment results.
"""

from conformal_relevance.dashboard.app import create_app, run_dashboard

__all__ = [
    "create_app",
    "run_dashboard",
]
