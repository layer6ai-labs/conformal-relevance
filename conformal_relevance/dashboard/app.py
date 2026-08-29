"""Main Dash application for the evaluation dashboard."""

try:
    from dash import Dash, html, dcc
    import dash_bootstrap_components as dbc
except ImportError:
    raise ImportError(
        "Dashboard dependencies not installed. Install with: "
        "pip install conformal-relevancy[dashboard]"
    )

from conformal_relevance.dashboard.callbacks import register_callbacks


def create_app(
    database_url: str | None = None,
    debug: bool = False,
) -> Dash:
    """Create the Dash application.

    Args:
        database_url: Database URL. If None, uses SUPABASE_DATABASE_URL env var.
        debug: Enable debug mode.

    Returns:
        Configured Dash application.
    """
    # Initialize app with Bootstrap theme
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="Conformal Relevancy Dashboard",
    )

    # Store database URL in app config
    if database_url:
        app.server.config["DATABASE_URL"] = database_url

    # Main layout with navigation
    app.layout = html.Div(
        id="page-root",
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="color-theme", data="light"),
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("Overview", href="/")),
                    dbc.NavItem(dbc.NavLink("Run Details", href="/run")),
                    dbc.NavItem(dbc.NavLink("Comparison", href="/comparison")),
                    dbc.NavItem(dbc.NavLink("Error Analysis", href="/errors")),
                ],
                brand="Conformal Relevancy Dashboard",
                brand_href="/",
                color="primary",
                dark=True,
                className="mb-4",
            ),
            dbc.Container(
                id="page-content",
                fluid=True,
                className="px-4",
            ),
            dcc.Interval(
                id="refresh-interval",
                interval=30000,  # 30 seconds
                n_intervals=0,
            ),
        ],
    )

    # Register callbacks
    register_callbacks(app)

    # OS-level dark mode detection + Bootstrap data-bs-theme attribute
    from dash import Output, Input
    app.clientside_callback(
        """
        function(_) {
            var mq = window.matchMedia('(prefers-color-scheme: dark)');
            function apply(theme) {
                document.getElementById('page-root').setAttribute('data-bs-theme', theme);
                return theme;
            }
            mq.addEventListener('change', function(e) {
                var theme = e.matches ? 'dark' : 'light';
                apply(theme);
                window.dash_clientside.set_props('color-theme', {data: theme});
            });
            return apply(mq.matches ? 'dark' : 'light');
        }
        """,
        Output("color-theme", "data"),
        Input("color-theme", "id"),  # fires once on page mount
    )

    return app


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    database_url: str | None = None,
) -> None:
    """Run the dashboard server.

    Args:
        host: Host address.
        port: Port number.
        debug: Enable debug mode.
        database_url: Database URL.
    """
    app = create_app(database_url=database_url, debug=debug)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)
