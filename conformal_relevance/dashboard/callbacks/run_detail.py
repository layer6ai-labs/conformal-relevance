"""Run detail page callbacks."""

from functools import wraps

from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate


def _db_callback(fallback):
    """Decorator for DB callbacks: wraps body with session management and error handling.

    The decorated function receives the DB session as its first argument.
    On any exception, returns the static fallback value.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                from conformal_relevance.db import get_session
                with get_session() as session:
                    return fn(session, *args, **kwargs)
            except Exception:
                return fallback
        return wrapper
    return decorator


def register_run_detail_callbacks(app):
    """Register run detail page callbacks.

    Args:
        app: Dash application instance.
    """

    @app.callback(
        [
            Output("run-filter-dataset", "options"),
            Output("run-filter-strategy", "options"),
        ],
        Input("refresh-interval", "n_intervals"),
    )
    @_db_callback(fallback=([], []))
    def update_filter_options(session, n_intervals):
        """Populate dataset and strategy filter dropdowns from all runs."""
        from conformal_relevance.evaluation.queries import get_runs_comparison

        runs = get_runs_comparison(session, limit=1000)
        datasets = sorted({r["dataset_name"] for r in runs if r.get("dataset_name")})
        strategies = sorted({r["icl_strategy_name"] for r in runs if r.get("icl_strategy_name")})
        return (
            [{"label": d, "value": d} for d in datasets],
            [{"label": s, "value": s} for s in strategies],
        )

    def _build_run_options(runs, filter_dataset=None, filter_strategy=None):
        """Filter runs and format dropdown option labels."""
        if filter_dataset:
            runs = [r for r in runs if r.get("dataset_name") == filter_dataset]
        if filter_strategy:
            runs = [r for r in runs if r.get("icl_strategy_name") == filter_strategy]
        return [
            {
                "label": f"#{r['run_id']} - {r.get('dataset_name', 'N/A')} - {r.get('icl_strategy_name') or r.get('run_name') or 'N/A'} - icl{r.get('n_icl', 0)}",
                "value": r["run_id"],
            }
            for r in runs
        ]

    @app.callback(
        Output("run-selector", "options"),
        [
            Input("refresh-interval", "n_intervals"),
            Input("run-filter-dataset", "value"),
            Input("run-filter-strategy", "value"),
        ],
    )
    @_db_callback(fallback=[])
    def update_run_selector(session, n_intervals, filter_dataset, filter_strategy):
        """Update run selector dropdown options, respecting active filters."""
        from conformal_relevance.evaluation.queries import get_runs_comparison

        runs = get_runs_comparison(session, limit=1000)
        return _build_run_options(runs, filter_dataset, filter_strategy)

    _run_info_error = html.P("Database error", className="text-danger")

    @app.callback(
        [
            Output("run-info", "children"),
            Output("run-performance-summary", "children"),
            Output("run-profiling", "children"),
        ],
        Input("run-selector", "value"),
    )
    @_db_callback(fallback=(_run_info_error, _run_info_error, _run_info_error))
    def update_run_info(session, run_id):
        """Update run information panels."""
        if not run_id:
            placeholder = html.P("Select a run to view details", className="text-muted")
            return placeholder, placeholder, placeholder

        from conformal_relevance.evaluation.queries import get_run_summary

        summary = get_run_summary(session, run_id)
        if not summary:
            error = html.P("Run not found", className="text-danger")
            return error, error, error

        mean_ap = summary.get('mean_ap')
        duration_ms = summary.get('total_duration_ms')
        run_info = html.Div([
            html.P([html.Strong("Run ID: "), f"#{summary['run_id']}"]),
            html.P([html.Strong("Name: "), summary.get('run_name') or "-"]),
            html.P([html.Strong("Dataset: "), summary.get('dataset_name', 'N/A')]),
            html.P([html.Strong("Model: "), summary.get('model_name', 'N/A')]),
            html.P([html.Strong("Status: "), summary.get('status', 'N/A')]),
            html.P([html.Strong("Strategy: "), summary.get('icl_strategy_name') or "-"]),
        ])
        perf_summary = html.Div([
            html.P([html.Strong("Mean AP: "), f"{mean_ap:.3f}" if mean_ap else "N/A"]),
            html.P([html.Strong("Samples: "), str(summary.get('n_samples', 0))]),
            html.P([html.Strong("Success: "), str(summary.get('n_success', 0))]),
            html.P([html.Strong("Failed: "), str(summary.get('n_failed', 0))]),
        ])
        profiling = html.Div([
            html.P([html.Strong("Duration: "), f"{duration_ms / 1000:.1f}s" if duration_ms else "N/A"]),
            html.P([html.Strong("LLM Time: "),
                    f"{summary.get('total_llm_time_ms', 0) / 1000:.1f}s"
                    if summary.get('total_llm_time_ms') else "N/A"]),
            html.P([html.Strong("Avg Sample: "),
                    f"{summary.get('avg_sample_time_ms', 0):.0f}ms"
                    if summary.get('avg_sample_time_ms') else "N/A"]),
        ])
        return run_info, perf_summary, profiling

    @app.callback(
        Output("chart-ap-histogram", "figure"),
        [Input("run-selector", "value"), Input("color-theme", "data")],
    )
    @_db_callback(fallback={})
    def update_ap_histogram(session, run_id, theme):
        """Update AP histogram chart."""
        from conformal_relevance.dashboard.components.charts import create_ap_histogram
        from conformal_relevance.db import get_run

        if not run_id:
            return create_ap_histogram([], theme=theme or "light")

        run = get_run(session, run_id)
        if not run or not run.ap_percentiles:
            return create_ap_histogram([], theme=theme or "light")
        pctl = run.ap_percentiles
        ap_values = [
            pctl.get("min", 0), pctl.get("p5", 0), pctl.get("p25", 0),
            pctl.get("p50", 0), pctl.get("p75", 0), pctl.get("p95", 0),
            pctl.get("max", 1),
        ]
        return create_ap_histogram(ap_values, theme=theme or "light")

    @app.callback(
        Output("chart-ap-by-intent", "figure"),
        [Input("run-selector", "value"), Input("color-theme", "data")],
    )
    @_db_callback(fallback={})
    def update_ap_by_intent(session, run_id, theme):
        """Update AP by intent chart."""
        from conformal_relevance.dashboard.components.charts import create_intent_breakdown_bar
        from conformal_relevance.db import get_run

        if not run_id:
            return create_intent_breakdown_bar([], theme=theme or "light")

        run = get_run(session, run_id)
        if not run or not run.intent_metrics:
            return create_intent_breakdown_bar([], theme=theme or "light")
        intent_data = [
            {"intent": intent, "mean_ap": metrics.get("mean_ap"), "n_samples": metrics.get("n", 0)}
            for intent, metrics in run.intent_metrics.items()
        ]
        return create_intent_breakdown_bar(intent_data, theme=theme or "light")

    @app.callback(
        Output("table-sample-scores", "children"),
        Input("run-selector", "value"),
    )
    @_db_callback(fallback=html.P("Error loading samples", className="text-danger"))
    def update_sample_scores_table(session, run_id):
        """Update sample scores table with per-sample data."""
        if not run_id:
            return html.P("Select a run to view scores", className="text-muted")

        from conformal_relevance.db import get_run_samples, RunSample
        from sqlalchemy import select, func
        import dash_bootstrap_components as dbc

        count = session.execute(
            select(func.count(RunSample.id)).where(RunSample.run_id == run_id)
        ).scalar()
        if not count:
            return html.P("No sample-level data for this run.", className="text-muted")

        samples = get_run_samples(session, run_id, split="test")
        MAX_DISPLAY = 200
        display_samples = samples[:MAX_DISPLAY]
        rows = [
            html.Tr([
                html.Td(s.sample_index),
                html.Td(s.intent or "\u2014"),
                html.Td(f"{s.ap:.4f}" if s.ap is not None else "\u2014"),
                html.Td(s.n_sentences),
                html.Td(s.status),
            ])
            for s in display_samples
        ]
        table = dbc.Table(
            [html.Thead(html.Tr([
                html.Th("Index"), html.Th("Intent"), html.Th("AP"),
                html.Th("Sentences"), html.Th("Status"),
            ]))] + [html.Tbody(rows)],
            bordered=True, hover=True, size="sm",
        )
        children = [table]
        if len(samples) > MAX_DISPLAY:
            children.append(html.P(
                f"Showing {MAX_DISPLAY} of {len(samples)} samples.",
                className="text-muted mt-2",
            ))
        return html.Div(children)

    @app.callback(
        Output("failed-samples-analysis", "children"),
        Input("run-selector", "value"),
    )
    @_db_callback(fallback=html.P("Error loading failed samples", className="text-danger"))
    def update_failed_samples(session, run_id):
        """Update failed samples analysis."""
        if not run_id:
            return html.P("Select a run to view failed samples", className="text-muted")

        from conformal_relevance.db import get_run

        run = get_run(session, run_id)
        if not run:
            return html.P("Run not found", className="text-danger")

        n_failed = run.n_samples_failed or 0
        if n_failed == 0:
            return html.P("No failed samples", className="text-success")

        summary = run.error_summary or {}
        samples = summary.get("samples", [])
        children = [html.P(f"Total failed: {n_failed}")]
        if samples:
            children.append(html.Ul([
                html.Li([
                    f"Sample #{s.get('index', '?')}: ",
                    html.Code(s.get("message", "Unknown error")[:100]),
                ])
                for s in samples[:10]
            ]))
        else:
            children.append(html.P("No error details available"))
        if n_failed > 10:
            children.append(html.P(f"...and {n_failed - 10} more"))
        return html.Div(children)

    @app.callback(
        Output("confirm-delete-run", "displayed"),
        Input("btn-delete-run", "n_clicks"),
        State("run-selector", "value"),
        prevent_initial_call=True,
    )
    def open_delete_confirmation(n_clicks, run_id):
        """Show confirmation dialog when Delete Run is clicked."""
        if not n_clicks or not run_id:
            raise PreventUpdate
        return True

    @app.callback(
        [
            Output("run-selector", "value"),
            Output("run-selector", "options", allow_duplicate=True),
            Output("delete-run-result", "children"),
        ],
        Input("confirm-delete-run", "submit_n_clicks"),
        [
            State("run-selector", "value"),
            State("run-filter-dataset", "value"),
            State("run-filter-strategy", "value"),
        ],
        prevent_initial_call=True,
    )
    def delete_run_and_reset(submit_n_clicks, run_id, filter_dataset, filter_strategy):
        """Delete the selected run and reset the selector."""
        if not submit_n_clicks or not run_id:
            raise PreventUpdate

        try:
            from conformal_relevance.db import get_session, delete_run
            from conformal_relevance.evaluation.queries import get_runs_comparison

            with get_session() as session:
                deleted = delete_run(session, int(run_id))
                if not deleted:
                    raise PreventUpdate

            with get_session() as session:
                runs = get_runs_comparison(session, limit=1000)

            return None, _build_run_options(runs, filter_dataset, filter_strategy), ""

        except PreventUpdate:
            raise
        except Exception:
            raise PreventUpdate

    # ------------------------------------------------------------------
    # Clean Database: two-step confirmation flow
    # ------------------------------------------------------------------

    @app.callback(
        Output("confirm-clean-db-1", "displayed"),
        Input("btn-clean-db", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_clean_db_confirmation_1(n_clicks):
        """Show the first confirmation dialog when Clean Database is clicked."""
        if not n_clicks:
            raise PreventUpdate
        return True

    @app.callback(
        Output("confirm-clean-db-2", "displayed"),
        Input("confirm-clean-db-1", "submit_n_clicks"),
        prevent_initial_call=True,
    )
    def open_clean_db_confirmation_2(submit_n_clicks):
        """Show the second (final) confirmation dialog."""
        if not submit_n_clicks:
            raise PreventUpdate
        return True

    @app.callback(
        [
            Output("run-selector", "value", allow_duplicate=True),
            Output("run-selector", "options", allow_duplicate=True),
            Output("run-filter-dataset", "options", allow_duplicate=True),
            Output("run-filter-strategy", "options", allow_duplicate=True),
            Output("clean-db-result", "children"),
            Output("clean-db-toast", "children"),
            Output("clean-db-toast", "is_open"),
        ],
        Input("confirm-clean-db-2", "submit_n_clicks"),
        prevent_initial_call=True,
    )
    @_db_callback(fallback=(no_update, no_update, no_update, no_update, "", "Database error", True))
    def clean_database(session, submit_n_clicks):
        """Delete all runs from the database after double confirmation."""
        if not submit_n_clicks:
            raise PreventUpdate

        from conformal_relevance.db import delete_all_runs

        count = delete_all_runs(session)
        return None, [], [], [], "", f"Successfully deleted {count} run(s).", True

    # ------------------------------------------------------------------
    # Prompt Content Viewer
    # ------------------------------------------------------------------

    @app.callback(
        Output("prompt-content-viewer", "children"),
        Input("run-selector", "value"),
    )
    @_db_callback(fallback=html.P("Error loading prompt content", className="text-danger"))
    def update_prompt_content(session, run_id):
        """Show the prompt template and ICL examples used for the run."""
        if not run_id:
            return html.P("Select a run to view prompt content", className="text-muted")

        from conformal_relevance.db import get_run

        run = get_run(session, run_id)
        if not run or not run.prompt_content:
            return html.P("No prompt content stored for this run", className="text-muted")

        content = run.prompt_content
        children = []
        template = content.get("template", "")
        if template:
            children.append(html.H6("Prompt Template"))
            children.append(html.Pre(
                template,
                style={
                    "maxHeight": "300px", "overflow": "auto",
                    "backgroundColor": "var(--bs-tertiary-bg)", "padding": "10px",
                    "borderRadius": "4px", "fontSize": "12px",
                },
            ))
        icl = content.get("icl_examples", {})
        if icl:
            children.append(html.H6("ICL Examples", className="mt-3"))
            for key, value in icl.items():
                if value:
                    children.append(html.Strong(f"{key}:"))
                    children.append(html.Pre(
                        str(value),
                        style={
                            "maxHeight": "500px", "overflow": "auto",
                            "backgroundColor": "var(--bs-tertiary-bg)", "padding": "10px",
                            "borderRadius": "4px", "fontSize": "11px",
                        },
                    ))
                else:
                    children.append(html.P(f"{key}: (none)", className="text-muted"))
        return html.Div(children) if children else html.P("Empty prompt content", className="text-muted")
