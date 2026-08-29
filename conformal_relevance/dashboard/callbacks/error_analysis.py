"""Error analysis page callbacks."""

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate


def register_error_analysis_callbacks(app):
    """Register error analysis page callbacks.

    Args:
        app: Dash application instance.
    """

    @app.callback(
        [
            Output("error-filter-dataset", "options"),
            Output("error-filter-model", "options"),
            Output("error-filter-run", "options"),
        ],
        Input("refresh-interval", "n_intervals"),
    )
    def update_error_filter_options(n_intervals):
        """Update error filter dropdown options."""
        try:
            from conformal_relevance.db import get_session, list_datasets, list_llm_models
            from sqlalchemy import select
            from conformal_relevance.db.models import Run

            with get_session() as session:
                # Dataset options
                datasets = list_datasets(session)
                dataset_options = [
                    {"label": d.name, "value": d.id}
                    for d in datasets
                ]

                # Model options
                models = list_llm_models(session)
                model_options = [
                    {"label": f"{m.name} ({m.provider})", "value": m.id}
                    for m in models
                ]

                # Run options (with failed samples)
                runs = session.execute(
                    select(Run)
                    .where(Run.status.in_(["completed", "failed"]))
                    .order_by(Run.created_at.desc())
                    .limit(100)
                ).scalars().all()
                run_options = [
                    {"label": f"#{r.id} - {r.name or 'Unnamed'}", "value": r.id}
                    for r in runs
                ]

                return dataset_options, model_options, run_options

        except Exception:
            return [], [], []

    @app.callback(
        Output("error-data-store", "data"),
        [
            Input("error-filter-dataset", "value"),
            Input("error-filter-model", "value"),
            Input("error-filter-run", "value"),
            Input("error-filter-category", "value"),
        ],
    )
    def load_error_data(dataset_id, model_id, run_id, category_filter):
        """Load and analyze error data based on filters."""
        try:
            from conformal_relevance.db import get_session
            from conformal_relevance.evaluation.error_analysis import (
                categorize_error,
                analyze_errors,
                ErrorInfo,
            )
            from sqlalchemy import select
            from conformal_relevance.db.models import Run

            with get_session() as session:
                # Build query for runs with errors
                stmt = select(Run).where(Run.status.in_(["completed", "failed"]))

                if dataset_id:
                    stmt = stmt.where(Run.dataset_id == dataset_id)
                if model_id:
                    stmt = stmt.where(Run.llm_model_id == model_id)
                if run_id:
                    stmt = stmt.where(Run.id == run_id)

                runs = session.execute(stmt).scalars().all()

                # Collect errors from error_summary JSONB
                errors = []
                total_samples = 0
                for run in runs:
                    total_samples += (run.n_samples_scored or 0) + (run.n_samples_failed or 0)
                    summary = run.error_summary or {}
                    for sample_entry in summary.get("samples", []):
                        error_info = categorize_error(
                            error_message=sample_entry.get("message", "Unknown error"),
                            sample_id=sample_entry.get("index"),
                            run_id=run.id,
                            retry_count=sample_entry.get("retries", 0),
                        )
                        # Apply category filter
                        if category_filter and error_info.category.value != category_filter:
                            continue
                        errors.append(error_info.to_dict())

                # Analyze errors
                analysis = analyze_errors(
                    [ErrorInfo(**e) for e in errors],
                    total_samples=total_samples,
                )

                return {
                    "errors": errors,
                    "total_samples": analysis.total_samples,
                    "total_errors": analysis.total_errors,
                    "error_rate": analysis.error_rate,
                    "by_category": {k.value if hasattr(k, 'value') else k: v for k, v in analysis.by_category.items()},
                    "by_severity": {k.value if hasattr(k, 'value') else k: v for k, v in analysis.by_severity.items()},
                    "by_type": analysis.by_type,
                    "patterns": [
                        {
                            "pattern": p.pattern,
                            "category": p.category.value if hasattr(p.category, 'value') else p.category,
                            "count": p.count,
                            "suggested_action": p.suggested_action,
                        }
                        for p in analysis.patterns
                    ],
                    "timeline": [
                        (t[0].isoformat() if hasattr(t[0], 'isoformat') else str(t[0]), t[1])
                        for t in (analysis.timeline or [])
                    ],
                }

        except Exception as e:
            return {
                "errors": [],
                "total_samples": 0,
                "total_errors": 0,
                "error_rate": 0,
                "by_category": {},
                "by_severity": {},
                "by_type": {},
                "patterns": [],
                "timeline": [],
                "error": str(e),
            }

    @app.callback(
        [
            Output("error-kpi-total-samples", "children"),
            Output("error-kpi-total-errors", "children"),
            Output("error-kpi-error-rate", "children"),
            Output("error-kpi-critical", "children"),
            Output("error-kpi-top-type", "children"),
            Output("error-kpi-avg-retries", "children"),
        ],
        Input("error-data-store", "data"),
    )
    def update_error_kpis(data):
        """Update error KPI cards."""
        if not data:
            return "0", "0", "0%", "0", "-", "0"

        total_samples = data.get("total_samples", 0)
        total_errors = data.get("total_errors", 0)
        error_rate = data.get("error_rate", 0)
        critical_count = data.get("by_severity", {}).get("critical", 0)

        # Find top error type
        by_type = data.get("by_type", {})
        top_type = "-"
        if by_type:
            top_type = max(by_type.items(), key=lambda x: x[1])[0]

        # Calculate average retries
        errors = data.get("errors", [])
        avg_retries = 0
        if errors:
            total_retries = sum(e.get("retry_count", 0) for e in errors)
            avg_retries = total_retries / len(errors)

        return (
            str(total_samples),
            str(total_errors),
            f"{error_rate * 100:.1f}%",
            str(critical_count),
            top_type[:20],
            f"{avg_retries:.1f}",
        )

    @app.callback(
        Output("chart-error-rate-gauge", "figure"),
        [Input("error-data-store", "data"), Input("color-theme", "data")],
    )
    def update_error_rate_gauge(data, theme):
        """Update error rate gauge chart."""
        from conformal_relevance.dashboard.components.charts import create_error_rate_gauge

        if not data:
            return create_error_rate_gauge(0, theme=theme or "light")

        error_rate = data.get("error_rate", 0)
        return create_error_rate_gauge(error_rate, theme=theme or "light")

    @app.callback(
        Output("chart-error-category-pie", "figure"),
        [Input("error-data-store", "data"), Input("color-theme", "data")],
    )
    def update_error_category_pie(data, theme):
        """Update error category pie chart."""
        from conformal_relevance.dashboard.components.charts import create_error_category_pie

        if not data:
            return create_error_category_pie({}, theme=theme or "light")

        by_category = data.get("by_category", {})
        return create_error_category_pie(by_category, theme=theme or "light")

    @app.callback(
        Output("chart-error-severity-bar", "figure"),
        [Input("error-data-store", "data"), Input("color-theme", "data")],
    )
    def update_error_severity_bar(data, theme):
        """Update error severity bar chart."""
        from conformal_relevance.dashboard.components.charts import create_error_severity_bar

        if not data:
            return create_error_severity_bar({}, theme=theme or "light")

        by_severity = data.get("by_severity", {})
        return create_error_severity_bar(by_severity, theme=theme or "light")

    @app.callback(
        Output("chart-error-type-bar", "figure"),
        [Input("error-data-store", "data"), Input("color-theme", "data")],
    )
    def update_error_type_bar(data, theme):
        """Update error type bar chart."""
        from conformal_relevance.dashboard.components.charts import create_error_type_bar

        if not data:
            return create_error_type_bar({}, theme=theme or "light")

        by_type = data.get("by_type", {})
        return create_error_type_bar(by_type, theme=theme or "light")

    @app.callback(
        Output("chart-error-timeline", "figure"),
        [Input("error-data-store", "data"), Input("color-theme", "data")],
    )
    def update_error_timeline(data, theme):
        """Update error timeline chart."""
        from conformal_relevance.dashboard.components.charts import create_error_timeline
        from datetime import datetime

        if not data:
            return create_error_timeline([], theme=theme or "light")

        timeline = data.get("timeline", [])
        # Convert ISO strings back to datetime
        parsed_timeline = []
        for t in timeline:
            try:
                dt = datetime.fromisoformat(t[0])
                parsed_timeline.append((dt, t[1]))
            except (ValueError, TypeError):
                pass

        return create_error_timeline(parsed_timeline, theme=theme or "light")

    @app.callback(
        Output("table-error-patterns", "children"),
        Input("error-data-store", "data"),
    )
    def update_error_patterns_table(data):
        """Update error patterns table."""
        from conformal_relevance.dashboard.components.tables import create_error_patterns_table

        if not data:
            return html.P("No data available", className="text-muted")

        patterns = data.get("patterns", [])
        return create_error_patterns_table(patterns)

    @app.callback(
        Output("chart-error-by-intent", "figure"),
        [Input("error-data-store", "data"), Input("color-theme", "data")],
    )
    def update_error_by_intent(data, theme):
        """Update errors by intent chart."""
        from conformal_relevance.dashboard.components.charts import create_error_by_intent_bar

        if not data:
            return create_error_by_intent_bar({}, theme=theme or "light")

        errors = data.get("errors", [])

        # Group errors by intent and category
        intent_errors = {}
        for e in errors:
            intent = e.get("intent") or "No Intent"
            category = e.get("category", "unknown")

            if intent not in intent_errors:
                intent_errors[intent] = {}
            if category not in intent_errors[intent]:
                intent_errors[intent][category] = 0
            intent_errors[intent][category] += 1

        return create_error_by_intent_bar(intent_errors, theme=theme or "light")

    @app.callback(
        Output("table-error-details", "children"),
        Input("error-data-store", "data"),
    )
    def update_error_details_table(data):
        """Update error details table."""
        from conformal_relevance.dashboard.components.tables import create_error_details_table

        if not data:
            return html.P("No data available", className="text-muted")

        errors = data.get("errors", [])
        return create_error_details_table(errors)

    @app.callback(
        Output("download-errors-csv", "data"),
        Input("btn-export-errors", "n_clicks"),
        State("error-data-store", "data"),
        prevent_initial_call=True,
    )
    def export_errors_csv(n_clicks, data):
        """Export error data as CSV."""
        if not n_clicks or not data:
            raise PreventUpdate

        errors = data.get("errors", [])
        if not errors:
            raise PreventUpdate

        import io
        import csv

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "sample_id", "category", "error_type",
                "severity", "message", "retry_count", "intent",
            ],
        )
        writer.writeheader()
        for error in errors:
            writer.writerow({
                "sample_id": error.get("sample_id", ""),
                "category": error.get("category", ""),
                "error_type": error.get("error_type", ""),
                "severity": error.get("severity", ""),
                "message": error.get("message", "")[:200],
                "retry_count": error.get("retry_count", 0),
                "intent": error.get("intent", ""),
            })

        return dict(content=output.getvalue(), filename="errors_export.csv")
