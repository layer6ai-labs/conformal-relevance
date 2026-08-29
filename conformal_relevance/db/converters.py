"""Converters for serializing SQLAlchemy models.

Under the 3-table schema (datasets, llm_models, runs), there are no more
dataset_samples or sample_scores tables. The only converter needed is
run_to_dict for JSON export of Run objects.
"""

from typing import Any

from conformal_relevance.db.models import Run


def run_to_dict(run: Run) -> dict[str, Any]:
    """Serialize a Run to a dictionary for JSON export.

    Includes all run-level columns: aggregate metrics (mean_ap, std_ap),
    sample counts, JSONB fields (ap_percentiles, intent_metrics,
    prompt_content, error_summary), ICL strategy info, profiling, and tags.

    Args:
        run: Run instance.

    Returns:
        Dictionary representation of the run.
    """
    return {
        "id": run.id,
        "name": run.name,
        "dataset_id": run.dataset_id,
        "dataset_name": run.dataset.name if run.dataset else None,
        "llm_model_id": run.llm_model_id,
        "llm_model_name": run.llm_model.name if run.llm_model else None,
        "llm_model_provider": run.llm_model.provider if run.llm_model else None,
        # ICL strategy
        "icl_strategy_name": run.icl_strategy_name,
        "icl_strategy_params": run.icl_strategy_params,
        "icl_permutation_id": run.icl_permutation_id,
        "icl_diversity_metrics": run.icl_diversity_metrics,
        # Run state
        "split": run.split,
        "status": run.status,
        "max_concurrency": run.max_concurrency,
        "max_retries": run.max_retries,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "error_message": run.error_message,
        # Aggregate metrics (populated at run completion)
        "mean_ap": float(run.mean_ap) if run.mean_ap is not None else None,
        "std_ap": float(run.std_ap) if run.std_ap is not None else None,
        "n_samples_scored": run.n_samples_scored,
        "n_samples_failed": run.n_samples_failed,
        "n_retries_total": run.n_retries_total,
        "error_rate": float(run.error_rate) if run.error_rate is not None else None,
        # JSONB fields
        "ap_percentiles": run.ap_percentiles,
        "intent_metrics": run.intent_metrics,
        "prompt_content": run.prompt_content,
        "error_summary": run.error_summary,
        # Profiling
        "total_duration_ms": run.total_duration_ms,
        "total_llm_time_ms": run.total_llm_time_ms,
        "avg_sample_time_ms": run.avg_sample_time_ms,
        # Tags
        "tags": run.tags,
        # Metadata
        "metadata": run.metadata_,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
    }
