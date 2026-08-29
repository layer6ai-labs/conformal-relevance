"""CRUD operations and query functions for the database.

Simplified schema: only `datasets`, `llm_models`, and `runs` tables.
Aggregate metrics are stored directly on the Run row.
"""

from typing import Any, Sequence

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

import polars as pl

from conformal_relevance.db.models import (
    Dataset,
    LLMModel,
    Run,
    RunSample,
)


# =============================================================================
# Dataset Operations
# =============================================================================


def get_or_create_dataset(
    session: Session,
    name: str,
    display_name: str | None = None,
    has_intent: bool = False,
    description: str | None = None,
    metadata: dict | None = None,
) -> Dataset:
    """Get existing dataset or create a new one.

    Args:
        session: Database session.
        name: Unique dataset name (e.g., 'subsume', 'puma').
        display_name: Human-readable name.
        has_intent: Whether dataset has intent/query column.
        description: Dataset description.
        metadata: Additional metadata.

    Returns:
        Dataset instance.
    """
    dataset = session.execute(select(Dataset).where(Dataset.name == name)).scalar_one_or_none()

    if dataset is None:
        dataset = Dataset(
            name=name,
            display_name=display_name,
            has_intent=has_intent,
            description=description,
            metadata_=metadata,
        )
        session.add(dataset)
        session.flush()

    return dataset


def get_dataset_by_name(session: Session, name: str) -> Dataset | None:
    """Get dataset by name.

    Args:
        session: Database session.
        name: Dataset name.

    Returns:
        Dataset instance or None if not found.
    """
    return session.execute(select(Dataset).where(Dataset.name == name)).scalar_one_or_none()


def list_datasets(session: Session) -> Sequence[Dataset]:
    """List all datasets.

    Args:
        session: Database session.

    Returns:
        List of Dataset instances.
    """
    return session.execute(select(Dataset).order_by(Dataset.name)).scalars().all()


# =============================================================================
# LLM Model Operations
# =============================================================================


def get_or_create_llm_model(
    session: Session,
    name: str,
    provider: str,
    config: dict | None = None,
    description: str | None = None,
) -> LLMModel:
    """Get existing LLM model or create a new one.

    Args:
        session: Database session.
        name: Model name (e.g., 'gpt-4o-mini', 'gemini-2.5-flash-lite').
        provider: Provider name. If not provided, inferred from model name.
        config: Model configuration (temperature, max_tokens, etc.).
        description: Model description.

    Returns:
        LLMModel instance.
    """
    model = session.execute(select(LLMModel).where(LLMModel.name == name)).scalar_one_or_none()

    if model is None:
        model = LLMModel(
            name=name,
            provider=provider,
            config=config,
            description=description,
        )
        session.add(model)
        session.flush()

    return model


def get_llm_model_by_name(session: Session, name: str) -> LLMModel | None:
    """Get LLM model by name.

    Args:
        session: Database session.
        name: Model name.

    Returns:
        LLMModel instance or None if not found.
    """
    return session.execute(select(LLMModel).where(LLMModel.name == name)).scalar_one_or_none()


def list_llm_models(session: Session, provider: str | None = None) -> Sequence[LLMModel]:
    """List LLM models, optionally filtered by provider.

    Args:
        session: Database session.
        provider: Optional provider filter.

    Returns:
        List of LLMModel instances.
    """
    query = select(LLMModel)
    if provider:
        query = query.where(LLMModel.provider == provider)
    return session.execute(query.order_by(LLMModel.name)).scalars().all()


# =============================================================================
# Run Operations
# =============================================================================


def create_run(
    session: Session,
    dataset_id: int,
    llm_model_id: int,
    name: str | None = None,
    split: str = "test",
    max_concurrency: int | None = None,
    max_retries: int | None = None,
    metadata: dict | None = None,
) -> Run:
    """Create a new scoring run.

    Args:
        session: Database session.
        dataset_id: Dataset ID.
        llm_model_id: LLM model ID.
        name: Optional run name.
        split: Data split ('calibration' or 'test').
        max_concurrency: Max concurrent requests.
        max_retries: Max retries per sample.
        metadata: Additional metadata.

    Returns:
        Run instance.
    """
    run = Run(
        dataset_id=dataset_id,
        llm_model_id=llm_model_id,
        name=name,
        split=split,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        metadata_=metadata,
    )
    session.add(run)
    session.flush()
    return run


def start_run(session: Session, run_id: int) -> Run:
    """Mark a run as started.

    Args:
        session: Database session.
        run_id: Run ID.

    Returns:
        Updated Run instance.
    """
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    run.start()
    session.flush()
    return run


# Allowed keyword arguments for complete_run that map directly to Run columns.
_COMPLETE_RUN_FIELDS = frozenset({
    "mean_ap",
    "std_ap",
    "n_samples_scored",
    "n_samples_failed",
    "n_retries_total",
    "error_rate",
    "ap_percentiles",
    "intent_metrics",
    "prompt_content",
    "error_summary",
    "tags",
})


def complete_run(session: Session, run_id: int, **kwargs: Any) -> Run:
    """Mark a run as completed and set aggregate metrics.

    Any keyword argument whose name matches an aggregate column on the Run
    model will be set before the run is marked complete.

    Accepted keyword args:
        mean_ap: Mean Average Precision across scored samples.
        std_ap: Standard deviation of AP across scored samples.
        n_samples_scored: Number of samples successfully scored.
        n_samples_failed: Number of samples that failed scoring.
        n_retries_total: Total retry count across all samples.
        error_rate: Fraction of samples that failed (0.0-1.0).
        ap_percentiles: JSONB dict of AP percentiles (e.g. {"25": 0.6, "50": 0.75, "75": 0.9}).
        intent_metrics: JSONB dict of per-intent metrics.
        prompt_content: Full prompt text used for the run.
        error_summary: JSONB dict summarising error categories and counts.
        tags: JSONB list of tags for filtering/grouping runs.

    Args:
        session: Database session.
        run_id: Run ID.
        **kwargs: Aggregate metric values to store on the Run.

    Returns:
        Updated Run instance.

    Raises:
        ValueError: If run_id is not found or an unrecognised kwarg is passed.
    """
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found")

    unknown = set(kwargs) - _COMPLETE_RUN_FIELDS
    if unknown:
        raise ValueError(f"Unrecognised complete_run kwargs: {unknown}")

    for key, value in kwargs.items():
        setattr(run, key, value)

    run.complete()
    session.flush()
    return run


def fail_run(
    session: Session, run_id: int, error_message: str | None = None
) -> Run:
    """Mark a run as failed.

    Args:
        session: Database session.
        run_id: Run ID.
        error_message: Error message.

    Returns:
        Updated Run instance.
    """
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    run.fail(error_message=error_message)
    session.flush()
    return run


def get_run(session: Session, run_id: int) -> Run | None:
    """Get a run by ID.

    Args:
        session: Database session.
        run_id: Run ID.

    Returns:
        Run instance or None if not found.
    """
    return session.get(Run, run_id)


def get_runs_by_dataset(
    session: Session,
    dataset_id: int,
    status: str | None = None,
    llm_model_id: int | None = None,
) -> Sequence[Run]:
    """Get runs for a dataset with optional filters.

    Args:
        session: Database session.
        dataset_id: Dataset ID.
        status: Optional status filter.
        llm_model_id: Optional model filter.

    Returns:
        List of Run instances.
    """
    query = select(Run).where(Run.dataset_id == dataset_id)

    if status:
        query = query.where(Run.status == status)
    if llm_model_id:
        query = query.where(Run.llm_model_id == llm_model_id)

    return session.execute(query.order_by(Run.created_at.desc())).scalars().all()


def delete_run(session: Session, run_id: int) -> bool:
    """Delete a single run.

    Args:
        session: Database session.
        run_id: Run ID to delete.

    Returns:
        True if the run was found and deleted, False if not found.
    """
    # Use a direct SQL DELETE to avoid ORM cascade loading all run_samples into memory
    # (N+1 deletes). The FK has ondelete="CASCADE" so the DB handles run_samples cleanup.
    result = session.execute(delete(Run).where(Run.id == run_id))
    session.flush()
    return result.rowcount > 0


def delete_all_runs(session: Session, include_datasets: bool = False) -> int:
    """Delete all runs, optionally with datasets and LLM models.

    Args:
        session: Database session.
        include_datasets: If True, also delete all datasets and LLM models.
            Use this for a full database reset.

    Returns:
        Number of runs deleted.
    """
    count = session.execute(select(func.count(Run.id))).scalar() or 0

    session.execute(delete(Run))
    if include_datasets:
        session.execute(delete(LLMModel))
        session.execute(delete(Dataset))
    session.flush()

    # Reset auto-increment sequences so IDs start from 1 again.
    # PostgreSQL sequences are independent of table data and survive DELETEs.
    tables_to_reset = ["runs"]
    if include_datasets:
        tables_to_reset += ["llm_models", "datasets"]
    for table in tables_to_reset:
        session.execute(text(f"ALTER SEQUENCE {table}_id_seq RESTART WITH 1"))

    return count


# =============================================================================
# Sample Operations
# =============================================================================


def store_run_samples(
    session: Session,
    run_id: int,
    df_scored: pl.DataFrame,
    scores_col: str,
    ap_col: str,
    labels_col: str = "input_unit_labels",
    intent_col: str = "intent",
    include_intent: bool = True,
    sub_scores: dict[str, list[list[float]]] | None = None,
    split: str = "test",
    index_col: str | None = None,
) -> int:
    """Bulk insert per-sample results for a completed run.

    Parameters
    ----------
    include_intent : bool
        Pass False for HotpotQA (unique question per sample, not useful for grouping).
    sub_scores : dict[str, list[list[float]]] | None
        MoE runs: {strategy_name: [scores_sample_0, ...]}. None for single-strategy.
    split : str
        Data split this batch belongs to: 'test' (default) or 'cal'.
    index_col : str | None
        Column in df_scored to use as sample_index. If None, uses the loop counter.
        Pass '_original_cal_index' for calibration samples that were filtered.

    Returns
    -------
    int : Number of rows inserted.
    """
    rows = []
    for idx, row in enumerate(df_scored.iter_rows(named=True)):

        scores = row.get(scores_col)
        labels = row.get(labels_col, [])
        intent = row.get(intent_col) if include_intent else None
        status = "scored" if scores is not None else "failed"
        sample_index = row[index_col] if index_col else idx

        # Extract per-strategy scores for this sample.
        # NOTE: strategies whose scores are None for this sample are silently
        # omitted from the dict (partial failure). If *all* strategies failed,
        # the dict is empty and gets stored as NULL (line below). This means
        # the DB record does not distinguish "non-MoE run" from "MoE run where
        # every sub-strategy failed" — both are NULL. The `status` field
        # ("scored" vs "failed") on the parent row disambiguates.
        sample_sub_scores = None
        if sub_scores is not None:
            sample_sub_scores = {
                strat: strat_scores[idx]
                for strat, strat_scores in sub_scores.items()
                if strat_scores[idx] is not None
            }

        rows.append(RunSample(
            run_id=run_id,
            sample_index=sample_index,
            intent=intent,
            ap=row.get(ap_col),
            scores=scores,
            labels=labels,
            n_sentences=len(labels),
            status=status,
            split=split,
            sub_scores=sample_sub_scores if sample_sub_scores else None,
        ))

    session.add_all(rows)
    session.flush()
    return len(rows)


def get_run_samples(
    session: Session,
    run_id: int,
    split: str | None = None,
    status: str | None = None,
    intent: str | None = None,
) -> list[RunSample]:
    """Fetch per-sample results for a run, optionally filtered."""
    stmt = select(RunSample).where(RunSample.run_id == run_id)
    if split:
        stmt = stmt.where(RunSample.split == split)
    if status:
        stmt = stmt.where(RunSample.status == status)
    if intent:
        stmt = stmt.where(RunSample.intent == intent)
    return session.execute(stmt.order_by(RunSample.sample_index)).scalars().all()


def get_cross_run_sample_aps(
    session: Session,
    run_ids: list[int],
    split: str | None = None,
) -> dict[int, dict[int, float | None]]:
    """Return {run_id: {sample_index: ap}} for cross-run comparison."""
    stmt = (
        select(RunSample.run_id, RunSample.sample_index, RunSample.ap)
        .where(RunSample.run_id.in_(run_ids))
    )
    if split:
        stmt = stmt.where(RunSample.split == split)
    result: dict[int, dict[int, float | None]] = {rid: {} for rid in run_ids}
    for run_id, sample_index, ap in session.execute(stmt).all():
        result[run_id][sample_index] = ap
    return result
