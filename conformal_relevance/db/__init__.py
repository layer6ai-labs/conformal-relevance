"""Database module for storing LLM run results.

This module provides a SQLAlchemy ORM layer for storing and querying
relevancy scoring experiment results in a PostgreSQL database (Supabase).

Simplified 3-table schema: datasets, llm_models, runs.
Aggregate metrics (mean_ap, std_ap, intent_metrics, etc.) are stored
directly on the Run row.

Example usage:
    from conformal_relevance.db import (
        get_session,
        get_or_create_dataset,
        get_or_create_llm_model,
        create_run,
        start_run,
        complete_run,
    )

    with get_session() as session:
        dataset = get_or_create_dataset(session, "subsume", has_intent=True)
        model = get_or_create_llm_model(session, "gemini-2.5-flash-lite")
        run = create_run(session, dataset.id, model.id, name="exp_001")
        start_run(session, run.id)
        # ... perform scoring ...
        complete_run(session, run.id, mean_ap=0.85, std_ap=0.12, n_samples_scored=50)
"""

# Session management
from conformal_relevance.db.session import (
    get_database_url,
    get_engine,
    get_session,
    get_session_factory,
    init_database,
    reset_connection,
)

# ORM Models
from conformal_relevance.db.models import (
    Dataset,
    LLMModel,
    Run,
    RunSample,
)

# CRUD Operations
from conformal_relevance.db.operations import (
    # Dataset operations
    get_or_create_dataset,
    get_dataset_by_name,
    list_datasets,
    # LLM Model operations
    get_or_create_llm_model,
    get_llm_model_by_name,
    list_llm_models,
    # Run operations
    create_run,
    start_run,
    complete_run,
    fail_run,
    get_run,
    get_runs_by_dataset,
    delete_run,
    delete_all_runs,
    # Sample operations
    store_run_samples,
    get_run_samples,
    get_cross_run_sample_aps,
)

# Converters
from conformal_relevance.db.converters import (
    run_to_dict,
)

__all__ = [
    # Session management
    "get_database_url",
    "get_engine",
    "get_session",
    "get_session_factory",
    "init_database",
    "reset_connection",
    # Models
    "Dataset",
    "LLMModel",
    "Run",
    "RunSample",
    # Dataset operations
    "get_or_create_dataset",
    "get_dataset_by_name",
    "list_datasets",
    # LLM Model operations
    "get_or_create_llm_model",
    "get_llm_model_by_name",
    "list_llm_models",
    # Run operations
    "create_run",
    "start_run",
    "complete_run",
    "fail_run",
    "get_run",
    "get_runs_by_dataset",
    "delete_run",
    "delete_all_runs",
    # Sample operations
    "store_run_samples",
    "get_run_samples",
    "get_cross_run_sample_aps",
    # Converters
    "run_to_dict",
]
