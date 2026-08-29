"""Error analysis page layout for the dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_error_analysis_layout() -> html.Div:
    """Create the error analysis page layout.

    Returns:
        Dash HTML component.
    """
    return html.Div([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Error Analysis", className="mb-4"),
                html.P(
                    "Analyze and categorize errors from scoring experiments to identify "
                    "patterns and improve reliability.",
                    className="text-muted",
                ),
            ]),
        ]),

        # Filters Row
        dbc.Row([
            dbc.Col([
                dbc.Label("Dataset"),
                dcc.Dropdown(
                    id="error-filter-dataset",
                    placeholder="Select Dataset",
                    clearable=True,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Model"),
                dcc.Dropdown(
                    id="error-filter-model",
                    placeholder="All Models",
                    clearable=True,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Run"),
                dcc.Dropdown(
                    id="error-filter-run",
                    placeholder="All Runs",
                    clearable=True,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Error Category"),
                dcc.Dropdown(
                    id="error-filter-category",
                    placeholder="All Categories",
                    options=[
                        {"label": "API Error", "value": "api_error"},
                        {"label": "Rate Limit", "value": "rate_limit"},
                        {"label": "Timeout", "value": "timeout"},
                        {"label": "Parsing Error", "value": "parsing_error"},
                        {"label": "Validation Error", "value": "validation_error"},
                        {"label": "Context Length", "value": "context_length"},
                        {"label": "Content Filter", "value": "content_filter"},
                        {"label": "Authentication", "value": "authentication"},
                        {"label": "Network Error", "value": "network_error"},
                        {"label": "Unknown", "value": "unknown"},
                    ],
                    clearable=True,
                ),
            ], width=3),
        ], className="mb-4"),

        # KPI Cards Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Samples", className="card-title text-muted"),
                        html.H2(id="error-kpi-total-samples", children="0"),
                    ])
                ], className="text-center"),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Errors", className="card-title text-muted"),
                        html.H2(id="error-kpi-total-errors", children="0", className="text-danger"),
                    ])
                ], className="text-center"),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Error Rate", className="card-title text-muted"),
                        html.H2(id="error-kpi-error-rate", children="0%"),
                    ])
                ], className="text-center"),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Critical Errors", className="card-title text-muted"),
                        html.H2(id="error-kpi-critical", children="0", className="text-danger"),
                    ])
                ], className="text-center"),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Top Error Type", className="card-title text-muted"),
                        html.H5(id="error-kpi-top-type", children="-", className="text-warning"),
                    ])
                ], className="text-center"),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Avg Retries", className="card-title text-muted"),
                        html.H2(id="error-kpi-avg-retries", children="0"),
                    ])
                ], className="text-center"),
            ], width=2),
        ], className="mb-4"),

        # Error Gauge and Category Breakdown Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Error Rate"),
                    dbc.CardBody([
                        dcc.Graph(id="chart-error-rate-gauge"),
                    ]),
                ]),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Errors by Category"),
                    dbc.CardBody([
                        dcc.Graph(id="chart-error-category-pie"),
                    ]),
                ]),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Errors by Severity"),
                    dbc.CardBody([
                        dcc.Graph(id="chart-error-severity-bar"),
                    ]),
                ]),
            ], width=4),
        ], className="mb-4"),

        # Error Types and Timeline Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Top Error Types"),
                    dbc.CardBody([
                        dcc.Graph(id="chart-error-type-bar"),
                    ]),
                ]),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Error Timeline"),
                    dbc.CardBody([
                        dcc.Graph(id="chart-error-timeline"),
                    ]),
                ]),
            ], width=6),
        ], className="mb-4"),

        # Error Patterns Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Error Patterns & Recommended Actions", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        html.Div(id="table-error-patterns"),
                    ]),
                ]),
            ]),
        ], className="mb-4"),

        # Errors by Intent (if applicable)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Errors by Intent"),
                    dbc.CardBody([
                        dcc.Graph(id="chart-error-by-intent"),
                    ]),
                ]),
            ]),
        ], className="mb-4"),

        # Detailed Errors Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("Error Details", className="mb-0 d-inline"),
                            dbc.Button(
                                "Export CSV",
                                id="btn-export-errors",
                                color="secondary",
                                size="sm",
                                className="float-end",
                            ),
                        ]),
                    ]),
                    dbc.CardBody([
                        html.Div(id="table-error-details"),
                    ]),
                ]),
            ]),
        ]),

        # Hidden store for error data
        dcc.Store(id="error-data-store"),

        # Download component for CSV export
        dcc.Download(id="download-errors-csv"),
    ])
