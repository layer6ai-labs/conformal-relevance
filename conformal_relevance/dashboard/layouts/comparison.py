"""Comparison page layout for the dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_comparison_layout() -> html.Div:
    """Create the comparison page layout.

    Returns:
        Dash HTML component.
    """
    return html.Div([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Strategy Comparison", className="mb-4"),
            ]),
        ]),

        # Filters
        dbc.Row([
            dbc.Col([
                dbc.Label("Dataset"),
                dcc.Dropdown(
                    id="compare-filter-dataset",
                    placeholder="Select Dataset",
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Baseline Strategy"),
                dcc.Dropdown(
                    id="compare-baseline-strategy",
                    placeholder="Select Baseline",
                    value="icl0",
                ),
            ], width=3),
        ], className="mb-4"),

        # Performance comparison table (strategies x ICL numbers)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Comparison (Best Run per Configuration)"),
                    dbc.CardBody([
                        html.Div(id="table-performance-comparison"),
                    ]),
                ]),
            ]),
        ], className="mb-4"),

        # Statistical comparison table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Statistical Comparison vs Baseline"),
                    dbc.CardBody([
                        html.Div(id="table-statistical-comparison"),
                    ]),
                ]),
            ]),
        ], className="mb-4"),

        # Cross-model comparison table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Cross-Model Comparison"),
                    dbc.CardBody([
                        html.Div(id="table-cross-model-comparison"),
                    ]),
                ]),
            ]),
        ], className="mb-4"),
    ])
