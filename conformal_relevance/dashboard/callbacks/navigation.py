"""Navigation callbacks for the dashboard."""

from dash import Input, Output, html

from conformal_relevance.dashboard.layouts import (
    create_overview_layout,
    create_run_detail_layout,
    create_comparison_layout,
    create_error_analysis_layout,
)


def register_navigation_callbacks(app):
    """Register navigation callbacks.

    Args:
        app: Dash application instance.
    """

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def display_page(pathname):
        """Route to appropriate page based on URL."""
        if pathname == "/" or pathname == "/overview":
            return create_overview_layout()
        elif pathname == "/run":
            return create_run_detail_layout()
        elif pathname == "/comparison":
            return create_comparison_layout()
        elif pathname == "/errors":
            return create_error_analysis_layout()
        else:
            return html.Div([
                html.H1("404 - Page Not Found"),
                html.P(f"The page '{pathname}' was not found."),
            ])
