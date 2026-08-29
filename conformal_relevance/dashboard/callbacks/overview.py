"""Overview page callbacks."""

from dash import Input, Output, html


def _apply_date_filter(query, start_date, end_date):
    """Apply date range filter to a SQLAlchemy query on Run.created_at.

    Args:
        query: SQLAlchemy select query.
        start_date: ISO format start date string or None.
        end_date: ISO format end date string or None.

    Returns:
        Filtered query.
    """
    from conformal_relevance.db.models import Run

    if start_date:
        from datetime import datetime
        query = query.where(Run.created_at >= datetime.fromisoformat(start_date))
    if end_date:
        from datetime import datetime, timedelta
        end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
        query = query.where(Run.created_at < end_dt)
    return query


def _filter_runs_by_date(runs, start_date, end_date):
    """Filter a list of run dicts by date range (Python-side filtering).

    Used for results from get_runs_comparison which doesn't accept date parameters.

    Args:
        runs: List of run dicts with 'created_at' key.
        start_date: ISO format start date string or None.
        end_date: ISO format end date string or None.

    Returns:
        Filtered list of run dicts.
    """
    if not start_date and not end_date:
        return runs

    from datetime import datetime, timedelta

    filtered = []
    for r in runs:
        created = r.get("created_at")
        if created is None:
            continue
        # Handle both datetime objects and strings
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created)
            except (ValueError, TypeError):
                continue
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
            if created < start_dt:
                continue
        if end_date:
            end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
            if created >= end_dt:
                continue
        filtered.append(r)
    return filtered


def register_overview_callbacks(app):
    """Register overview page callbacks.

    Args:
        app: Dash application instance.
    """

    @app.callback(
        [
            Output("kpi-total-runs", "children"),
            Output("kpi-completed-runs", "children"),
            Output("kpi-avg-ap", "children"),
            Output("kpi-active-runs", "children"),
        ],
        [
            Input("refresh-interval", "n_intervals"),
            Input("filter-dataset", "value"),
            Input("filter-model", "value"),
            Input("filter-date-range", "start_date"),
            Input("filter-date-range", "end_date"),
        ],
    )
    def update_kpis(n_intervals, dataset_filter, model_filter, start_date, end_date):
        """Update KPI cards."""
        try:
            from conformal_relevance.db import get_session
            from sqlalchemy import func, select
            from conformal_relevance.db.models import Run

            with get_session() as session:
                # Build base query
                query = select(func.count(Run.id))

                if dataset_filter:
                    query = query.where(Run.dataset_id == dataset_filter)
                if model_filter:
                    query = query.where(Run.llm_model_id == model_filter)

                query = _apply_date_filter(query, start_date, end_date)

                total_runs = session.execute(query).scalar() or 0

                # Completed runs
                completed_query = select(func.count(Run.id)).where(Run.status == "completed")
                if dataset_filter:
                    completed_query = completed_query.where(Run.dataset_id == dataset_filter)
                if model_filter:
                    completed_query = completed_query.where(Run.llm_model_id == model_filter)
                completed_query = _apply_date_filter(completed_query, start_date, end_date)
                completed_runs = session.execute(completed_query).scalar() or 0

                # Active runs
                active_query = select(func.count(Run.id)).where(Run.status == "running")
                if dataset_filter:
                    active_query = active_query.where(Run.dataset_id == dataset_filter)
                if model_filter:
                    active_query = active_query.where(Run.llm_model_id == model_filter)
                active_query = _apply_date_filter(active_query, start_date, end_date)
                active_runs = session.execute(active_query).scalar() or 0

                # Average AP from run-level mean_ap column
                avg_ap_query = (
                    select(func.avg(Run.mean_ap))
                    .where(Run.status == "completed")
                )
                if dataset_filter:
                    avg_ap_query = avg_ap_query.where(Run.dataset_id == dataset_filter)
                if model_filter:
                    avg_ap_query = avg_ap_query.where(Run.llm_model_id == model_filter)
                avg_ap_query = _apply_date_filter(avg_ap_query, start_date, end_date)
                avg_ap = session.execute(avg_ap_query).scalar()
                avg_ap_str = f"{avg_ap:.3f}" if avg_ap else "N/A"

                return str(total_runs), str(completed_runs), avg_ap_str, str(active_runs)

        except Exception:
            # Return placeholders on error
            return "0", "0", "N/A", "0"

    @app.callback(
        [
            Output("filter-dataset", "options"),
            Output("filter-model", "options"),
            Output("filter-strategy", "options"),
            Output("filter-tags", "options"),
            Output("filter-model", "value"),
        ],
        Input("refresh-interval", "n_intervals"),
    )
    def update_filter_options(n_intervals):
        """Update filter dropdown options including tags."""
        try:
            from conformal_relevance.db import get_session, list_datasets, list_llm_models
            from sqlalchemy import select, func
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

                # Strategy options (normalize: strip '_stratified' suffix)
                from conformal_relevance.dashboard.callbacks.comparison import _normalized_strategy_name
                strategies = session.execute(
                    select(_normalized_strategy_name().label("strategy"))
                    .where(Run.icl_strategy_name.isnot(None))
                    .distinct()
                ).scalars().all()
                strategy_options = [
                    {"label": s, "value": s}
                    for s in sorted(set(strategies)) if s
                ]

                # Tag options: unnest the tags arrays to get all distinct tags
                try:
                    tag_rows = session.execute(
                        select(func.unnest(Run.tags).label("tag"))
                        .where(Run.tags.isnot(None))
                        .distinct()
                    ).scalars().all()
                    tag_options = [{"label": t, "value": t} for t in sorted(tag_rows) if t]
                except Exception:
                    tag_options = []

                default_model_id = next(
                    (m.id for m in models if m.name == "gemini-2.5-flash-lite"), None
                )
                return dataset_options, model_options, strategy_options, tag_options, default_model_id

        except Exception:
            return [], [], [], [], None

    @app.callback(
        Output("table-recent-runs", "children"),
        [
            Input("refresh-interval", "n_intervals"),
            Input("filter-dataset", "value"),
            Input("filter-model", "value"),
            Input("filter-strategy", "value"),
            Input("filter-date-range", "start_date"),
            Input("filter-date-range", "end_date"),
            Input("filter-tags", "value"),
        ],
    )
    def update_recent_runs_table(
        n_intervals, dataset_filter, model_filter, strategy_filter,
        start_date, end_date, tag_filter,
    ):
        """Update recent runs table."""
        from conformal_relevance.dashboard.components.tables import create_runs_table

        try:
            from conformal_relevance.db import get_session
            from conformal_relevance.evaluation.queries import get_runs_comparison

            with get_session() as session:
                # Expand normalized strategy filter to match both base and _stratified variants
                strategy_names = None
                if strategy_filter:
                    strategy_names = [strategy_filter, f"{strategy_filter}_stratified"]

                runs = get_runs_comparison(
                    session,
                    dataset_id=dataset_filter,
                    model_ids=[model_filter] if model_filter else None,
                    strategy_names=strategy_names,
                    limit=50,
                )

                # Apply date filtering
                runs = _filter_runs_by_date(runs, start_date, end_date)

                # Apply tag filtering
                if tag_filter:
                    tag_set = set(tag_filter)
                    runs = [
                        r for r in runs
                        if r.get("tags") and tag_set.intersection(r["tags"])
                    ]

                return create_runs_table(runs)

        except Exception as e:
            return html.P(f"Error loading runs: {str(e)}", className="text-danger")

    @app.callback(
        Output("table-performance-matrix", "children"),
        [
            Input("refresh-interval", "n_intervals"),
            Input("filter-model", "value"),
        ],
    )
    def update_performance_matrix(n_intervals, model_filter):
        """Update performance matrix table."""
        from conformal_relevance.dashboard.components.tables import create_performance_matrix_table

        try:
            from conformal_relevance.db import get_session
            from conformal_relevance.evaluation.queries import get_performance_matrix
            from conformal_relevance.db.models import LLMModel

            with get_session() as session:
                model_name = None
                if model_filter:
                    model = session.get(LLMModel, model_filter)
                    if model:
                        model_name = model.name

                matrix_data = get_performance_matrix(session, model_name=model_name)
                return create_performance_matrix_table(matrix_data)

        except Exception as e:
            return html.P(f"Error loading matrix: {str(e)}", className="text-danger")
