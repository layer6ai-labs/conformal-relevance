"""Database query utilities for evaluation framework.

All queries read directly from the `runs` table (3-table schema:
datasets, llm_models, runs). No per-sample joins.
"""

from typing import Any, Sequence

from sqlalchemy import case, func, literal_column, select
from sqlalchemy.orm import Session

from conformal_relevance.db.models import Run, Dataset, LLMModel


def get_run_summary(session: Session, run_id: int) -> dict[str, Any] | None:
    """Get summary data for a run directly from run columns.

    Args:
        session: Database session.
        run_id: Run ID.

    Returns:
        Dictionary with run summary or None if not found.
    """
    run = session.get(Run, run_id)
    if not run:
        return None

    return {
        "run_id": run.id,
        "run_name": run.name,
        "status": run.status,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "dataset_id": run.dataset_id,
        "dataset_name": run.dataset.name if run.dataset else None,
        "model_id": run.llm_model_id,
        "model_name": run.llm_model.name if run.llm_model else None,
        "model_provider": run.llm_model.provider if run.llm_model else None,
        "n_samples": run.n_samples_scored,
        "n_success": run.n_samples_scored,
        "n_failed": run.n_samples_failed,
        "mean_ap": float(run.mean_ap) if run.mean_ap is not None else None,
        "std_ap": float(run.std_ap) if run.std_ap is not None else None,
        "error_rate": float(run.error_rate) if run.error_rate is not None else None,
        "ap_percentiles": run.ap_percentiles,
        "intent_metrics": run.intent_metrics,
        "prompt_content": run.prompt_content,
        "error_summary": run.error_summary,
        "tags": run.tags,
        "total_duration_ms": run.total_duration_ms,
        "total_llm_time_ms": run.total_llm_time_ms,
        "avg_sample_time_ms": run.avg_sample_time_ms,
        "icl_strategy_name": run.icl_strategy_name,
        "icl_strategy_params": run.icl_strategy_params,
    }


def query_run_metrics(
    session: Session,
    run_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Query metrics for multiple runs.

    Args:
        session: Database session.
        run_ids: List of run IDs.

    Returns:
        List of dictionaries with run metrics.
    """
    results = []
    for run_id in run_ids:
        summary = get_run_summary(session, run_id)
        if summary:
            results.append(summary)
    return results


def get_runs_comparison(
    session: Session,
    dataset_id: int | None = None,
    model_ids: Sequence[int] | None = None,
    strategy_names: Sequence[str] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get runs for comparison with optional filters.

    Reads n_samples and mean_ap directly from Run columns.
    No outer join to SampleScore, no group_by needed.

    Args:
        session: Database session.
        dataset_id: Optional dataset filter.
        model_ids: Optional model ID filter.
        strategy_names: Optional strategy name filter.
        limit: Maximum runs to return.

    Returns:
        List of run summary dictionaries.
    """
    query = (
        select(
            Run.id.label("run_id"),
            Run.name.label("run_name"),
            Run.status,
            Run.created_at,
            Run.dataset_id,
            Dataset.name.label("dataset_name"),
            Run.llm_model_id,
            LLMModel.name.label("model_name"),
            Run.icl_strategy_name,
            Run.icl_strategy_params.op("->>")(literal_column("'n_icl_examples'")).label("n_icl_str"),
            Run.total_duration_ms,
            Run.n_samples_scored.label("n_samples"),
            Run.mean_ap,
            Run.std_ap,
            Run.tags,
        )
        .join(Dataset, Run.dataset_id == Dataset.id)
        .join(LLMModel, Run.llm_model_id == LLMModel.id)
        .where(Run.status == "completed")
        .order_by(Run.created_at.desc())
        .limit(limit)
    )

    if dataset_id:
        query = query.where(Run.dataset_id == dataset_id)
    if model_ids:
        query = query.where(Run.llm_model_id.in_(model_ids))
    if strategy_names:
        query = query.where(Run.icl_strategy_name.in_(strategy_names))

    results = session.execute(query).all()

    return [
        {
            "run_id": r.run_id,
            "run_name": r.run_name,
            "status": r.status,
            "created_at": r.created_at,
            "dataset_id": r.dataset_id,
            "dataset_name": r.dataset_name,
            "model_id": r.llm_model_id,
            "model_name": r.model_name,
            "icl_strategy_name": r.icl_strategy_name,
            "n_icl": int(r.n_icl_str) if r.n_icl_str is not None else 0,
            "total_duration_ms": r.total_duration_ms,
            "n_samples": r.n_samples,
            "mean_ap": float(r.mean_ap) if r.mean_ap is not None else None,
            "std_ap": float(r.std_ap) if r.std_ap is not None else None,
            "tags": r.tags,
        }
        for r in results
    ]


def get_strategy_comparison(
    session: Session,
    dataset_id: int,
    model_id: int | None = None,
) -> list[dict[str, Any]]:
    """Get aggregated metrics by ICL strategy for a dataset.

    Groups runs by icl_strategy_name and computes avg of Run.mean_ap
    across runs. No SampleScore join.

    Args:
        session: Database session.
        dataset_id: Dataset ID.
        model_id: Optional model filter.

    Returns:
        List of strategy comparison results.
    """
    query = (
        select(
            Run.icl_strategy_name,
            func.count(Run.id).label("n_runs"),
            func.sum(Run.n_samples_scored).label("n_samples"),
            func.avg(Run.mean_ap).label("mean_ap"),
            func.min(Run.mean_ap).label("min_ap"),
            func.max(Run.mean_ap).label("max_ap"),
        )
        .where(Run.dataset_id == dataset_id)
        .where(Run.status == "completed")
        .where(Run.icl_strategy_name.isnot(None))
        .group_by(Run.icl_strategy_name)
        .order_by(func.avg(Run.mean_ap).desc())
    )

    if model_id:
        query = query.where(Run.llm_model_id == model_id)

    results = session.execute(query).all()

    return [
        {
            "strategy_name": r.icl_strategy_name,
            "n_runs": r.n_runs,
            "n_samples": r.n_samples,
            "mean_ap": float(r.mean_ap) if r.mean_ap is not None else None,
            "min_ap": float(r.min_ap) if r.min_ap is not None else None,
            "max_ap": float(r.max_ap) if r.max_ap is not None else None,
        }
        for r in results
    ]


def get_performance_matrix(
    session: Session,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Get the latest completed run's mean AP for each (dataset, strategy) pair.

    Uses a window function to select the most recent and second-most-recent
    run per (dataset, strategy_label) group. Constructs strategy_label as:
      - For icl_strategy_name='icl0': 'icl0'
      - Otherwise: 'icl' + COALESCE(icl_strategy_params->>'n_icl_examples', '0')
        + '_' + icl_strategy_name

    Optionally filters by model_name. Excludes datasets with zero completed runs.

    Args:
        session: Database session.
        model_name: Optional model name filter (e.g. 'gemini-2.5-flash-lite').

    Returns:
        List of dicts with: dataset_name, strategy_label, mean_ap, std_ap,
        prev_mean_ap, delta, run_id, created_at, model_name, n_samples.
    """
    # Build strategy_label expression:
    # Normalize strategy names by stripping '_stratified' suffix so that
    # historical stratified runs collapse into the same row as base-name runs.
    # icl0 stays as 'icl0'; otherwise concat 'icl' + n_icl_examples + '_' + normalized_name
    n_icl_expr = func.coalesce(
        Run.icl_strategy_params.op("->>")(literal_column("'n_icl_examples'")),
        "0",
    )
    normalized_strategy = func.regexp_replace(
        Run.icl_strategy_name,
        '_stratified$',
        '',
    )
    strategy_label_expr = case(
        (
            Run.icl_strategy_name.like("icl0%"),
            normalized_strategy,
        ),
        else_=func.concat(
            literal_column("'icl'"),
            n_icl_expr,
            literal_column("'_'"),
            normalized_strategy,
        ),
    ).label("strategy_label")

    # Window function: partition by (dataset_id, model_name, normalized strategy, n_icl)
    # so that 'random' and 'random_stratified' collapse into one partition,
    # and cross-model runs don't pollute each other's latest-run selection.
    # The most recent run wins within each partition.
    row_num = func.row_number().over(
        partition_by=[
            Run.dataset_id,
            LLMModel.name,
            normalized_strategy,
            Run.icl_strategy_params.op("->>")(literal_column("'n_icl_examples'")),
        ],
        order_by=Run.created_at.desc(),
    ).label("rn")

    # Base query: exclude failed/null-AP runs so the latest *valid* run wins each partition
    base_filter = [Run.status == "completed", Run.mean_ap.isnot(None)]
    if model_name:
        base_filter.append(LLMModel.name == model_name)

    # CTE with row numbers
    numbered = (
        select(
            Run.id.label("run_id"),
            Run.dataset_id,
            Dataset.name.label("dataset_name"),
            LLMModel.name.label("model_name"),
            Run.mean_ap,
            Run.std_ap,
            Run.n_samples_scored.label("n_samples"),
            Run.created_at,
            strategy_label_expr,
            row_num,
        )
        .join(Dataset, Run.dataset_id == Dataset.id)
        .join(LLMModel, Run.llm_model_id == LLMModel.id)
        .where(*base_filter)
        .cte("numbered_runs")
    )

    # Select latest (rn=1) with a self-join for second-latest (rn=2)
    latest = numbered.alias("latest")
    prev = numbered.alias("prev")

    query = (
        select(
            latest.c.dataset_name,
            latest.c.strategy_label,
            latest.c.mean_ap,
            latest.c.std_ap,
            prev.c.mean_ap.label("prev_mean_ap"),
            latest.c.run_id,
            latest.c.created_at,
            latest.c.model_name,
            latest.c.n_samples,
        )
        .select_from(latest)
        .outerjoin(
            prev,
            (latest.c.dataset_id == prev.c.dataset_id)
            & (latest.c.strategy_label == prev.c.strategy_label)
            & (prev.c.rn == 2),
        )
        .where(latest.c.rn == 1)
        .order_by(latest.c.dataset_name, latest.c.strategy_label)
    )

    results = session.execute(query).all()

    return [
        {
            "dataset_name": r.dataset_name,
            "strategy_label": r.strategy_label,
            "mean_ap": float(r.mean_ap) if r.mean_ap is not None else None,
            "std_ap": float(r.std_ap) if r.std_ap is not None else None,
            "prev_mean_ap": float(r.prev_mean_ap) if r.prev_mean_ap is not None else None,
            "delta": (
                round(float(r.mean_ap) - float(r.prev_mean_ap), 4)
                if r.mean_ap is not None and r.prev_mean_ap is not None
                else None
            ),
            "run_id": r.run_id,
            "created_at": r.created_at,
            "model_name": r.model_name,
            "n_samples": r.n_samples,
        }
        for r in results
    ]


def get_cross_model_comparison(
    session: Session,
    dataset_id: int,
    primary_model_name: str = "gemini-2.5-flash-lite",
) -> dict:
    """Get latest MAP per (strategy_label, model_name) for cross-model comparison.

    Returns only strategy configurations run on at least one non-primary model.
    Primary model column always included when data exists.

    Args:
        session: Database session.
        dataset_id: Dataset ID to filter by.
        primary_model_name: Primary model name (always leftmost column).

    Returns:
        {
            "models": ["gemini-2.5-flash-lite", "qwen3:8b", ...],
            "rows": [{"strategy_label": "icl2_moe4", "maps": {"gemini-...": 0.428, "qwen3:8b": 0.391}}, ...]
        }
    """
    n_icl_expr = func.coalesce(
        Run.icl_strategy_params.op("->>")(literal_column("'n_icl_examples'")),
        "0",
    )
    normalized_strategy = func.regexp_replace(
        Run.icl_strategy_name,
        '_stratified$',
        '',
    )
    strategy_label_expr = case(
        (
            Run.icl_strategy_name.like("icl0%"),
            normalized_strategy,
        ),
        else_=func.concat(
            literal_column("'icl'"),
            n_icl_expr,
            literal_column("'_'"),
            normalized_strategy,
        ),
    ).label("strategy_label")

    # Window function: latest run per (strategy_label, model_name)
    row_num = func.row_number().over(
        partition_by=[
            LLMModel.name,
            normalized_strategy,
            Run.icl_strategy_params.op("->>")(literal_column("'n_icl_examples'")),
        ],
        order_by=Run.created_at.desc(),
    ).label("rn")

    numbered = (
        select(
            LLMModel.name.label("model_name"),
            Run.mean_ap,
            strategy_label_expr,
            row_num,
        )
        .join(Dataset, Run.dataset_id == Dataset.id)
        .join(LLMModel, Run.llm_model_id == LLMModel.id)
        .where(
            Run.dataset_id == dataset_id,
            Run.status == "completed",
            Run.mean_ap.isnot(None),
        )
        .cte("numbered_cross_model")
    )

    query = (
        select(
            numbered.c.strategy_label,
            numbered.c.model_name,
            numbered.c.mean_ap,
        )
        .where(numbered.c.rn == 1)
        .order_by(numbered.c.strategy_label, numbered.c.model_name)
    )

    results = session.execute(query).all()

    # Group by strategy_label → {model_name: mean_ap}
    strategy_maps: dict[str, dict[str, float]] = {}
    all_models: set[str] = set()
    for r in results:
        all_models.add(r.model_name)
        if r.strategy_label not in strategy_maps:
            strategy_maps[r.strategy_label] = {}
        strategy_maps[r.strategy_label][r.model_name] = float(r.mean_ap)

    # Keep only strategies with at least one non-primary model run
    non_primary_strategies = {
        label
        for label, maps in strategy_maps.items()
        if any(m != primary_model_name for m in maps)
    }

    if not non_primary_strategies:
        return {"models": [], "rows": []}

    # Order models: primary first, then alphabetical
    other_models = sorted(m for m in all_models if m != primary_model_name)
    models = ([primary_model_name] if primary_model_name in all_models else []) + other_models

    rows = [
        {
            "strategy_label": label,
            "maps": strategy_maps[label],
        }
        for label in sorted(non_primary_strategies)
    ]

    return {"models": models, "rows": rows}
