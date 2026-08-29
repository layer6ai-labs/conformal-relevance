"""Experiment run storage orchestration.

Single entry point for persisting experiment results to the database.
Handles ICL-free, local strategy, global strategy, and MoE runs.

Usage:
    from conformal_relevance.db.run_storage import store_experiment_run, RunParams

    run_id = store_experiment_run(RunParams(
        settings=settings,
        dataset_name="subsume",
        has_intent=True,
        provider="google",
        model_name="gemini-2.5-flash-lite",
        scoring_result=scoring_result,
        df_scored=df_scored,
        output_col="scores_icl3_random",
        max_concurrency=10,
        run_name=None,
        strategy_result=icl_result.strategy_result,
        icl_examples=icl_result.icl_examples,
    ))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class RunParams:
    """All parameters needed to persist a single experiment run."""

    settings: object
    dataset_name: str
    has_intent: bool
    provider: str
    model_name: str
    scoring_result: object
    df_scored: pl.DataFrame
    output_col: str
    max_concurrency: int
    run_name: str | None
    # Mode flags and optional fields (defaults for backward compatibility)
    is_icl_free: bool = False
    is_moe: bool = False
    strategy_result: object = None
    icl_examples: str | dict | None = None
    display_strategy_name: str | None = None
    n_icl: int | None = None
    tags: list[str] | None = None
    cal_scored: pl.DataFrame | None = None
    strategy_name_suffix: str | None = None


@dataclass
class _RunInfo:
    """Strategy-specific metadata built by mode helpers."""

    strategy_config: dict
    icl_strategy_name: str
    icl_strategy_params: dict | None
    icl_strategy_metrics: dict | None
    sub_scores: dict | None
    prompt_content: dict


@dataclass
class _RunAggregates:
    mean_ap: float | None
    std_ap: float | None
    ap_percentiles: dict | None
    n_samples_scored: int
    n_samples_failed: int
    error_rate: float
    n_retries_total: int | None
    intent_metrics: dict | None
    error_summary: dict | None


def _compute_ap_stats(ap_values: list[float]) -> tuple[float | None, float | None, dict | None]:
    """Compute mean, std, and percentiles from a list of AP values."""
    mean_ap = float(np.mean(ap_values)) if ap_values else None
    std_ap = float(np.std(ap_values, ddof=1)) if len(ap_values) > 1 else None
    ap_percentiles = None
    if ap_values:
        ap_arr = np.array(ap_values)
        ap_percentiles = {
            "p5": float(np.percentile(ap_arr, 5)),
            "p25": float(np.percentile(ap_arr, 25)),
            "p50": float(np.percentile(ap_arr, 50)),
            "p75": float(np.percentile(ap_arr, 75)),
            "p95": float(np.percentile(ap_arr, 95)),
            "min": float(np.min(ap_arr)),
            "max": float(np.max(ap_arr)),
        }
    return mean_ap, std_ap, ap_percentiles


def _compute_error_metrics(n_scored: int, n_failed: int) -> tuple[float, int | None]:
    """Compute error rate from scored/failed counts. Returns (error_rate, total_samples)."""
    total = n_scored + n_failed
    return (n_failed / total if total > 0 else 0.0), total


def _compute_per_intent_metrics(
    df_scored: pl.DataFrame, ap_col: str, dataset_name: str
) -> dict | None:
    """Compute per-intent MAP metrics for multi-intent datasets."""
    from conformal_relevance.icl.registry import DATASET_REGISTRY

    icl_config = DATASET_REGISTRY.get(dataset_name, {})
    if not bool(icl_config.get("multi_intent", False)):
        return None
    if "intent" not in df_scored.columns or ap_col not in df_scored.columns:
        return None
    intent_groups = (
        df_scored
        .filter(pl.col(ap_col).is_not_null())
        .group_by("intent")
        .agg([
            pl.col(ap_col).mean().alias("mean_ap"),
            pl.col(ap_col).count().alias("n"),
        ])
    )
    return {
        row["intent"]: {
            "mean_ap": float(row["mean_ap"]) if row["mean_ap"] is not None else None,
            "n": int(row["n"]),
        }
        for row in intent_groups.iter_rows(named=True)
    }


def _build_error_summary(
    df_scored: pl.DataFrame, output_col: str, n_failed: int, error_rate: float
) -> dict | None:
    """Build error summary dict if any samples failed."""
    if n_failed == 0:
        return None
    failed_df = df_scored.filter(df_scored[output_col].is_null())
    error_samples = [
        {"index": idx, "message": "Scoring failed (null scores)", "retries": 0}
        for idx in range(min(failed_df.shape[0], 20))
    ]
    return {
        "total_errors": n_failed,
        "error_rate": error_rate,
        "by_category": {"scoring_failure": n_failed},
        "samples": error_samples,
    }


def _build_run_aggregates(
    df_scored: pl.DataFrame,
    output_col: str,
    scoring_result,
    dataset_name: str,
) -> _RunAggregates:
    """Compute aggregate metrics from a scored DataFrame and scoring result."""
    ap_col = f"{output_col}_ap"
    ap_values = df_scored[ap_col].drop_nulls().to_list() if ap_col in df_scored.columns else []
    mean_ap, std_ap, ap_percentiles = _compute_ap_stats(ap_values)

    n_samples_scored = scoring_result.n_scored
    n_samples_failed = scoring_result.n_failed
    error_rate, _ = _compute_error_metrics(n_samples_scored, n_samples_failed)

    n_retries_total = scoring_result.profile.n_retries if scoring_result.profile else None
    intent_metrics = _compute_per_intent_metrics(df_scored, ap_col, dataset_name)
    error_summary = _build_error_summary(df_scored, output_col, n_samples_failed, error_rate)

    return _RunAggregates(
        mean_ap=mean_ap,
        std_ap=std_ap,
        ap_percentiles=ap_percentiles,
        n_samples_scored=n_samples_scored,
        n_samples_failed=n_samples_failed,
        error_rate=error_rate,
        n_retries_total=n_retries_total,
        intent_metrics=intent_metrics,
        error_summary=error_summary,
    )


def _extract_strategy_metrics(strategy_result) -> dict | None:
    """Extract strategy-specific metrics for the icl_diversity_metrics DB column."""
    from conformal_relevance.icl.strategies.base import StratifiedICLSelectionResult

    meta = strategy_result.metadata
    metrics = {}

    if isinstance(strategy_result, StratifiedICLSelectionResult):
        for intent_val, intent_result in strategy_result.by_intent.items():
            intent_meta = intent_result.metadata
            if intent_meta.get("log_det_score") is not None:
                metrics.setdefault("per_intent_log_det", {})[intent_val] = intent_meta["log_det_score"]
            if intent_meta.get("distances") is not None:
                metrics.setdefault("per_intent_distances", {})[intent_val] = intent_meta["distances"]
    else:
        if meta.get("log_det_score") is not None:
            metrics["log_det_score"] = meta["log_det_score"]
        if meta.get("distances") is not None:
            metrics["distances"] = meta["distances"]

    return metrics if metrics else None


def _store_moe_run(scoring_result, n_icl: int | None, strategy_name_suffix: str | None = None) -> _RunInfo:
    """Build strategy info for a MoE run."""
    from conformal_relevance.prompts.templates import RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT

    meta = scoring_result.metadata or {}
    sub_strategies = meta.get("sub_strategies", [])
    moe_k_actual = len(sub_strategies)
    suffix = f"_{strategy_name_suffix}" if strategy_name_suffix else ""
    strategy_config = {
        "strategy_name": f"moe{moe_k_actual}{suffix}",
        "n_icl_examples": n_icl,
        "fusion_method": meta.get("fusion_method", "score_average"),
        "sub_strategies": sub_strategies,
        "weights": meta.get("weights", []),
        "icl_overlap": meta.get("icl_overlap", {}),
    }
    return _RunInfo(
        strategy_config=strategy_config,
        icl_strategy_name=f"moe{moe_k_actual}{suffix}",
        icl_strategy_params=strategy_config,
        icl_strategy_metrics=None,
        sub_scores=meta.get("sub_strategy_scores"),
        prompt_content={
            "template": RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT,
            "icl_examples": {"default": "MoE fusion (see sub_strategies in metadata)"},
        },
    )


def _store_icl_free_run(dataset_name: str, strategy_name_suffix: str | None = None) -> _RunInfo:
    """Build strategy info for an ICL-free run."""
    from conformal_relevance.prompts.icl_free import ICL_FREE_PROMPT_REGISTRY

    template_text = ICL_FREE_PROMPT_REGISTRY.get(dataset_name, (None,))[0] or ""
    suffix = f"_{strategy_name_suffix}" if strategy_name_suffix else ""
    name = f"icl0{suffix}"
    return _RunInfo(
        strategy_config={"strategy_name": name, "n_icl": 0},
        icl_strategy_name=name,
        icl_strategy_params=None,
        icl_strategy_metrics=None,
        sub_scores=None,
        prompt_content={"template": template_text, "icl_examples": {"default": None}},
    )


def _store_local_run(display_strategy_name: str | None, n_icl: int | None = None) -> _RunInfo:
    """Build strategy info for a local (per-sample) ICL strategy run."""
    from conformal_relevance.prompts.templates import RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT

    name = display_strategy_name or "unknown"
    strategy_config: dict = {"strategy_name": name}
    if n_icl is not None:
        strategy_config["n_icl_examples"] = n_icl
    return _RunInfo(
        strategy_config=strategy_config,
        icl_strategy_name=display_strategy_name or "local",
        icl_strategy_params=strategy_config,
        icl_strategy_metrics=None,
        sub_scores=None,
        prompt_content={
            "template": RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT,
            "icl_examples": {"default": None},
        },
    )


def _store_global_run(strategy_result, icl_examples, display_strategy_name: str | None) -> _RunInfo:
    """Build strategy info for a global ICL strategy run."""
    from conformal_relevance.icl.strategies.base import StratifiedICLSelectionResult
    from conformal_relevance.prompts.templates import RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT

    strategy_config = strategy_result.metadata.copy()
    strategy_config["strategy_name"] = strategy_result.strategy_name

    if isinstance(strategy_result, StratifiedICLSelectionResult):
        per_intent = strategy_result.n_examples_per_intent
        strategy_config["n_examples_per_intent"] = per_intent
        strategy_config["n_icl_examples"] = next(iter(per_intent.values()), 0)
        per_intent_indices: dict = {}
        all_indices: list = []
        for intent_val, intent_result in strategy_result.by_intent.items():
            per_intent_indices[intent_val] = intent_result.sample_indices
            all_indices.extend(intent_result.sample_indices)
        strategy_config["sample_indices_by_intent"] = per_intent_indices
        strategy_config["sample_indices"] = all_indices
    else:
        strategy_config["n_icl_examples"] = strategy_result.n_examples
        strategy_config["sample_indices"] = strategy_result.sample_indices

    prompt_content = {
        "template": RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT,
        "icl_examples": icl_examples if isinstance(icl_examples, dict) else {"default": icl_examples},
    }
    return _RunInfo(
        strategy_config=strategy_config,
        icl_strategy_name=display_strategy_name or strategy_result.strategy_name,
        icl_strategy_params=strategy_config,
        icl_strategy_metrics=_extract_strategy_metrics(strategy_result),
        sub_scores=None,
        prompt_content=prompt_content,
    )


def store_experiment_run(params: RunParams) -> int | None:
    """Persist a single experiment run to the database.

    Dispatches to the appropriate mode helper (MoE, ICL-free, local, global)
    to build strategy metadata, then performs common DB operations.

    Returns the run ID, or None if storage was skipped or failed.
    """
    db_url = params.settings.database_url
    if not db_url:
        print("Warning: No database URL configured. Skipping DB storage.")
        return None

    try:
        from conformal_relevance.db import (
            create_run,
            get_engine,
            get_or_create_dataset,
            get_or_create_llm_model,
            get_session,
            init_database,
            store_run_samples,
        )
    except ImportError:
        print("Warning: Database dependencies not installed. Skipping DB storage.")
        return None

    if params.is_moe:
        info = _store_moe_run(params.scoring_result, params.n_icl, params.strategy_name_suffix)
    elif params.is_icl_free:
        info = _store_icl_free_run(params.dataset_name, params.strategy_name_suffix)
    elif params.strategy_result is None:
        info = _store_local_run(params.display_strategy_name, n_icl=params.n_icl)
    else:
        info = _store_global_run(params.strategy_result, params.icl_examples, params.display_strategy_name)

    try:
        engine = get_engine(database_url=db_url)
        init_database(engine=engine)

        with get_session(engine=engine) as session:
            dataset = get_or_create_dataset(session, params.dataset_name, has_intent=params.has_intent)
            llm_model = get_or_create_llm_model(session, params.model_name, provider=params.provider)

            run = create_run(
                session,
                dataset.id,
                llm_model.id,
                name=params.run_name,
                split="test",
                max_concurrency=params.max_concurrency,
                metadata=info.strategy_config,
            )
            run.start()
            session.flush()

            run.set_icl_strategy(
                strategy_name=info.icl_strategy_name,
                strategy_params=info.icl_strategy_params,
                strategy_metrics=info.icl_strategy_metrics,
            )

            aggregates = _build_run_aggregates(
                params.df_scored, params.output_col, params.scoring_result, params.dataset_name
            )
            profile = params.scoring_result.profile
            if profile:
                run.set_profiling(
                    total_duration_ms=int(profile.total_duration_ms),
                    total_llm_time_ms=int(profile.total_llm_time_ms),
                    avg_sample_time_ms=profile.avg_sample_time_ms,
                )

            run.mean_ap = aggregates.mean_ap
            run.std_ap = aggregates.std_ap
            run.n_samples_scored = aggregates.n_samples_scored
            run.n_samples_failed = aggregates.n_samples_failed
            run.n_retries_total = aggregates.n_retries_total
            run.error_rate = aggregates.error_rate
            run.ap_percentiles = aggregates.ap_percentiles
            run.intent_metrics = aggregates.intent_metrics
            run.error_summary = aggregates.error_summary
            run.prompt_content = info.prompt_content
            if params.tags:
                run.tags = params.tags

            run.complete()

            include_intent = (params.dataset_name != "hotpotqa")
            n_samples = params.df_scored.shape[0]
            print(f"  Storing {n_samples} sample records to DB...", end="", flush=True)
            n_inserted = store_run_samples(
                session=session,
                run_id=run.id,
                df_scored=params.df_scored,
                scores_col=params.output_col,
                ap_col=f"{params.output_col}_ap",
                include_intent=include_intent,
                sub_scores=info.sub_scores,
                split="test",
            )
            print(f" done ({n_inserted} rows)")

            if params.cal_scored is not None:
                n_cal = params.cal_scored.shape[0]
                print(f"  Storing {n_cal} calibration sample records to DB...", end="", flush=True)
                n_cal_inserted = store_run_samples(
                    session=session,
                    run_id=run.id,
                    df_scored=params.cal_scored,
                    scores_col=params.output_col,
                    ap_col=f"{params.output_col}_ap",
                    include_intent=include_intent,
                    split="cal",
                )
                print(f" done ({n_cal_inserted} rows)")

            session.commit()
            print(f"  Stored to DB (run_id={run.id})")
            return run.id

    except Exception as e:
        logger.exception("Failed to store run to database")
        print(f"Warning: DB storage failed: {e}")
        return None
