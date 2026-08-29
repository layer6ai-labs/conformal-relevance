"""Comparison page callbacks."""

from dash import Input, Output, html

PRIMARY_MODEL_NAME = "gemini-2.5-flash-lite"


def _normalized_strategy_name():
    """Return SQLAlchemy expression: icl_strategy_name with '_stratified' suffix stripped."""
    from sqlalchemy import func
    from conformal_relevance.db.models import Run
    return func.regexp_replace(Run.icl_strategy_name, '_stratified$', '')


def _latest_run_subquery(filters, partition_extra=None, extra_cols=None, cte_name="numbered_runs"):
    """Build a numbered CTE row_numbering runs by recency within strategy groups.

    Parameters
    ----------
    filters : list
        SQLAlchemy filter expressions (passed to .where()).
    partition_extra : list, optional
        Additional partition columns beyond the normalized strategy name.
    extra_cols : list, optional
        Additional columns to include in the CTE select beyond [strategy_name, mean_ap, rn].
    cte_name : str
        Name for the resulting CTE.

    Returns
    -------
    CTE with columns: strategy_name, mean_ap, rn, + extra_cols.
    """
    from sqlalchemy import func, select
    from conformal_relevance.db.models import Run
    norm = _normalized_strategy_name()
    partition = [norm] + (partition_extra or [])
    row_num = func.row_number().over(
        partition_by=partition,
        order_by=Run.created_at.desc(),
    ).label("rn")
    base_cols = [norm.label("strategy_name"), Run.mean_ap, row_num]
    return select(*(base_cols + (extra_cols or []))).where(*filters).cte(cte_name)


def register_comparison_callbacks(app):
    """Register comparison page callbacks.

    Args:
        app: Dash application instance.
    """

    @app.callback(
        [
            Output("compare-filter-dataset", "options"),
            Output("compare-baseline-strategy", "options"),
        ],
        Input("refresh-interval", "n_intervals"),
    )
    def update_comparison_filters(n_intervals):
        """Update comparison filter dropdown options."""
        try:
            from conformal_relevance.db import get_session, list_datasets
            from sqlalchemy import select
            from conformal_relevance.db.models import Run

            with get_session() as session:
                # Dataset options
                datasets = list_datasets(session)
                dataset_options = [
                    {"label": d.name, "value": d.id}
                    for d in datasets
                ]

                # Strategy options for baseline (normalize: strip '_stratified')
                strategies = session.execute(
                    select(_normalized_strategy_name().label("strategy"))
                    .where(Run.icl_strategy_name.isnot(None))
                    .distinct()
                ).scalars().all()
                strategy_options = [
                    {"label": s, "value": s}
                    for s in sorted(set(strategies)) if s
                ]

                return dataset_options, strategy_options

        except Exception:
            return [], []

    @app.callback(
        Output("table-performance-comparison", "children"),
        Input("compare-filter-dataset", "value"),
    )
    def update_performance_comparison(dataset_id):
        """Update the strategy x ICL count performance table.

        Queries all completed runs for the dataset scoped to the primary model,
        groups by (strategy_name, n_icl_examples), and picks the latest run's
        mean_ap per group (consistent with the overview tab).
        ICL0 baseline is shown as a separate benchmark row.
        """
        from conformal_relevance.dashboard.components.tables import create_strategy_icl_table

        if not dataset_id:
            return html.P("Select a dataset to see performance comparison", className="text-muted")

        try:
            from sqlalchemy import func, literal_column, select
            from conformal_relevance.db import get_session
            from conformal_relevance.db.models import Run, LLMModel

            with get_session() as session:
                # Look up primary model ID
                primary = session.execute(
                    select(LLMModel).where(LLMModel.name == PRIMARY_MODEL_NAME)
                ).scalar_one_or_none()
                primary_model_id = primary.id if primary else None

                # Base filter
                filters = [
                    Run.status == "completed",
                    Run.dataset_id == dataset_id,
                    Run.mean_ap.isnot(None),
                ]
                if primary_model_id:
                    filters.append(Run.llm_model_id == primary_model_id)

                n_icl_text = func.coalesce(
                    Run.icl_strategy_params.op("->>")(literal_column("'n_icl_examples'")),
                    "0",
                )

                # Latest run per (strategy, n_icl) group
                numbered = _latest_run_subquery(
                    filters=[*filters, Run.icl_strategy_name.isnot(None)],
                    partition_extra=[n_icl_text],
                    extra_cols=[n_icl_text.label("n_icl_text"), Run.icl_strategy_name],
                )

                # Get ICL0 baseline: latest run among icl0 runs
                baseline_ap = None
                baseline_q = (
                    select(numbered.c.mean_ap)
                    .where(numbered.c.icl_strategy_name == "icl0")
                    .where(numbered.c.rn == 1)
                )
                result = session.execute(baseline_q).scalar()
                if result is not None:
                    baseline_ap = float(result)

                # Get latest run per (strategy, n_icl) for non-icl0
                latest_q = (
                    select(
                        numbered.c.strategy_name,
                        numbered.c.n_icl_text,
                        numbered.c.mean_ap.label("best_mean_ap"),
                    )
                    .where(numbered.c.icl_strategy_name != "icl0")
                    .where(numbered.c.rn == 1)
                    .order_by(numbered.c.strategy_name, numbered.c.n_icl_text)
                )

                rows = session.execute(latest_q).all()

                def _safe_int(val: str) -> int:
                    """Parse n_icl from text, handling dict-like strings."""
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return 0

                data = [
                    {
                        "strategy_name": r.strategy_name,
                        "n_icl": _safe_int(r.n_icl_text),
                        "best_mean_ap": float(r.best_mean_ap) if r.best_mean_ap is not None else None,
                    }
                    for r in rows
                ]

                return create_strategy_icl_table(data, baseline_ap=baseline_ap)

        except Exception as e:
            return html.P(f"Error: {str(e)}", className="text-danger")

    @app.callback(
        Output("table-statistical-comparison", "children"),
        [
            Input("compare-filter-dataset", "value"),
            Input("compare-baseline-strategy", "value"),
        ],
    )
    def update_statistical_comparison(dataset_id, baseline):
        """Update statistical comparison table scoped to the primary model.

        For each strategy, picks the latest run's mean_ap under
        this dataset (consistent with the overview tab).
        """
        from conformal_relevance.dashboard.components.tables import create_comparison_table

        if not dataset_id or not baseline:
            return html.P("Select dataset and baseline to see comparison", className="text-muted")

        try:
            from sqlalchemy import func, select
            from conformal_relevance.db import get_session
            from conformal_relevance.db.models import Run, LLMModel

            with get_session() as session:
                # Look up primary model ID
                primary = session.execute(
                    select(LLMModel).where(LLMModel.name == PRIMARY_MODEL_NAME)
                ).scalar_one_or_none()
                primary_model_id = primary.id if primary else None

                # Base filter
                filters = [
                    Run.status == "completed",
                    Run.dataset_id == dataset_id,
                    Run.mean_ap.isnot(None),
                    Run.icl_strategy_name.isnot(None),
                ]
                if primary_model_id:
                    filters.append(Run.llm_model_id == primary_model_id)

                # Latest run per strategy
                numbered = _latest_run_subquery(filters)

                # Count runs per strategy (separate query for n_runs)
                count_q = (
                    select(
                        _normalized_strategy_name().label("strategy_name"),
                        func.count(Run.id).label("n_runs"),
                    )
                    .where(*filters)
                    .group_by(_normalized_strategy_name())
                )
                counts = {
                    r.strategy_name: r.n_runs
                    for r in session.execute(count_q).all()
                }

                # Get latest run per strategy
                latest_q = (
                    select(
                        numbered.c.strategy_name,
                        numbered.c.mean_ap.label("best_ap"),
                    )
                    .where(numbered.c.rn == 1)
                    .order_by(numbered.c.mean_ap.desc())
                )

                rows = session.execute(latest_q).all()

                if not rows:
                    return html.P("No strategy data available", className="text-muted")

                # Build strategy results dict using latest run's AP
                strategy_results = {}
                for r in rows:
                    strategy_results[r.strategy_name] = {
                        "mean_ap": float(r.best_ap),
                        "n_runs": counts.get(r.strategy_name, 0),
                    }

                # Compute comparison vs baseline
                baseline_ap = strategy_results.get(baseline, {}).get("mean_ap", 0)
                for name, data in strategy_results.items():
                    if name != baseline:
                        diff = data["mean_ap"] - baseline_ap
                        data["comparison"] = {
                            "mean_diff": diff,
                            "p_value": None,
                            "effect_size": None,
                            "is_significant": abs(diff) > 0.02,
                        }

                return create_comparison_table(strategy_results, baseline)

        except Exception as e:
            return html.P(f"Error: {str(e)}", className="text-danger")

    @app.callback(
        Output("table-cross-model-comparison", "children"),
        Input("compare-filter-dataset", "value"),
    )
    def update_cross_model_comparison(dataset_id):
        """Update cross-model comparison table.

        Shows latest MAP per (strategy_label, model_name) for strategies
        that have been run on at least one non-primary model.
        """
        if not dataset_id:
            return html.P("Select a dataset to see cross-model comparison", className="text-muted")

        try:
            from conformal_relevance.db import get_session
            from conformal_relevance.evaluation.queries import get_cross_model_comparison
            from conformal_relevance.dashboard.components.tables import create_cross_model_table

            with get_session() as session:
                result = get_cross_model_comparison(session, dataset_id, PRIMARY_MODEL_NAME)

            if not result["rows"]:
                return html.P("No alternative model runs for this dataset.", className="text-muted")

            return create_cross_model_table(result["models"], result["rows"])

        except Exception as e:
            return html.P(f"Error: {str(e)}", className="text-danger")
