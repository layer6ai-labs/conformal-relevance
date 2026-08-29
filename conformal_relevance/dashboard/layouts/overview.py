"""Overview page layout for the dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_overview_layout() -> html.Div:
    """Create the overview page layout.

    Returns:
        Dash HTML component.
    """
    return html.Div([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Experiment Overview", className="mb-4"),
            ]),
        ]),

        # KPI Cards Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Runs", className="card-title text-muted"),
                        html.H2(id="kpi-total-runs", children="0"),
                    ])
                ], className="text-center"),
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Completed", className="card-title text-muted"),
                        html.H2(id="kpi-completed-runs", children="0", className="text-success"),
                    ])
                ], className="text-center"),
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Avg Mean AP", className="card-title text-muted"),
                        html.H2(id="kpi-avg-ap", children="0.000"),
                    ])
                ], className="text-center"),
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Active Runs", className="card-title text-muted"),
                        html.H2(id="kpi-active-runs", children="0", className="text-warning"),
                    ])
                ], className="text-center"),
            ], width=3),
        ], className="mb-4"),

        # Filters Row
        dbc.Row([
            dbc.Col([
                dbc.Label("Dataset"),
                dcc.Dropdown(
                    id="filter-dataset",
                    placeholder="All Datasets",
                    clearable=True,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Model"),
                dcc.Dropdown(
                    id="filter-model",
                    placeholder="All Models",
                    clearable=True,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("ICL Strategy"),
                dcc.Dropdown(
                    id="filter-strategy",
                    placeholder="All Strategies",
                    clearable=True,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Date Range"),
                dcc.DatePickerRange(
                    id="filter-date-range",
                    start_date_placeholder_text="Start",
                    end_date_placeholder_text="End",
                ),
            ], width=3),
        ], className="mb-3"),

        # Tag filter row
        dbc.Row([
            dbc.Col([
                dbc.Label("Tags"),
                dcc.Dropdown(
                    id="filter-tags",
                    placeholder="Filter by tags...",
                    clearable=True,
                    multi=True,
                ),
            ], width=4),
        ], className="mb-4"),

        # Performance Matrix
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Matrix (Latest Runs)"),
                    dbc.CardBody([
                        html.Div(id="table-performance-matrix"),
                    ]),
                ]),
            ]),
        ], className="mb-4"),

        # Recent Runs Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Runs"),
                    dbc.CardBody([
                        html.Div(id="table-recent-runs"),
                    ]),
                ]),
            ]),
        ]),
    ])
