"""Run detail page layout for the dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_run_detail_layout() -> html.Div:
    """Create the run detail page layout.

    Returns:
        Dash HTML component.
    """
    return html.Div([
        # Confirm dialog for delete run
        dcc.ConfirmDialog(
            id="confirm-delete-run",
            message="Are you sure you want to delete this run? This cannot be undone.",
        ),

        # Confirm dialog for clean database (two-step confirmation)
        dcc.ConfirmDialog(
            id="confirm-clean-db-1",
            message=(
                "WARNING: This will permanently delete ALL runs and their "
                "associated scores from the database.\n\n"
                "This action CANNOT be undone.\n\n"
                "Are you sure you want to continue?"
            ),
        ),
        dcc.ConfirmDialog(
            id="confirm-clean-db-2",
            message=(
                "FINAL CONFIRMATION: You are about to delete ALL experiment "
                "data from the database.\n\n"
                "Click OK to permanently erase everything."
            ),
        ),

        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Run Details", className="mb-4"),
            ], width=12),
        ]),

        # Filters + run selector row
        dbc.Row([
            dbc.Col([
                dbc.Label("Filter by Dataset"),
                dcc.Dropdown(
                    id="run-filter-dataset",
                    placeholder="All datasets",
                    clearable=True,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Filter by Strategy"),
                dcc.Dropdown(
                    id="run-filter-strategy",
                    placeholder="All strategies",
                    clearable=True,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Select Run"),
                html.Div([
                    dcc.Dropdown(
                        id="run-selector",
                        placeholder="Select a run...",
                        style={"flex": "1"},
                    ),
                    dbc.Button(
                        "Delete Run",
                        id="btn-delete-run",
                        color="danger",
                        size="sm",
                        className="ms-2",
                        style={"whiteSpace": "nowrap"},
                    ),
                    dbc.Button(
                        "Clean Database",
                        id="btn-clean-db",
                        color="danger",
                        outline=True,
                        size="sm",
                        className="ms-2",
                        style={"whiteSpace": "nowrap"},
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
            ], width=6),
        ], className="mb-4"),

        # Hidden divs for callback output sinks
        html.Div(id="delete-run-result", style={"display": "none"}),
        html.Div(id="clean-db-result", style={"display": "none"}),

        # Toast notification for clean database result
        dbc.Toast(
            id="clean-db-toast",
            header="Database Cleaned",
            is_open=False,
            dismissable=True,
            duration=5000,
            icon="success",
            style={"position": "fixed", "top": 20, "right": 20, "width": 350, "zIndex": 9999},
        ),

        # Run metadata
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Run Information", className="card-title"),
                        html.Div(id="run-info"),
                    ]),
                ]),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Performance Summary", className="card-title"),
                        html.Div(id="run-performance-summary"),
                    ]),
                ]),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Profiling", className="card-title"),
                        html.Div(id="run-profiling"),
                    ]),
                ]),
            ], width=4),
        ], className="mb-4"),

        # Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("AP Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id="chart-ap-histogram"),
                    ]),
                ]),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("AP by Intent"),
                    dbc.CardBody([
                        dcc.Graph(id="chart-ap-by-intent"),
                    ]),
                ]),
            ], width=6),
        ], className="mb-4"),

        # Sample details
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Sample Scores"),
                    dbc.CardBody([
                        html.Div(id="table-sample-scores"),
                    ]),
                ]),
            ]),
        ], className="mb-4"),

        # Error analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Failed Samples Analysis"),
                    dbc.CardBody([
                        html.Div(id="failed-samples-analysis"),
                    ]),
                ]),
            ]),
        ]),

        # Prompt content viewer
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Prompt Content"),
                    dbc.CardBody([
                        html.Div(id="prompt-content-viewer"),
                    ]),
                ]),
            ]),
        ], className="mt-4"),
    ])
