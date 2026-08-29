"""MoE score fusion: orchestrate K scoring passes and fuse results.

MoE is an output-side scoring mode, not an ICL selection strategy.
It runs K independent score_dataframe() calls -- each with a different
pre-built ICL block -- and element-wise averages the resulting score
vectors per sample.

ICL selection (pool → ICLResult) is the caller's responsibility.
Pass the formatted ICL blocks via `icl_by_strategy`.
"""

from __future__ import annotations

import logging

import polars as pl
from langchain_core.language_models import BaseChatModel

from conformal_relevance.types import (
    DEFAULT_N_ICL,
    normalize_strategy_name,
)
from conformal_relevance.icl.strategies.registry import StrategyRegistry
from conformal_relevance.scoring.pipeline import (
    DEFAULT_MAX_CONCURRENCY,
    ScoringResult,
    score_dataframe,
)
from conformal_relevance.scoring.profiling import RunProfile

logger = logging.getLogger(__name__)

# --- Constants ---

# Ranked by empirical fusion contribution (MoE ablation, 2026-03-19).
# Order derived from C(6,2..6) exhaustive combination ablation on cal subsets.
# Optimal universal combos (n_icl=3): k=2 anchor_dpp+bm25, k=3 +pattern_dpp,
# k=4 +random (adopted default), k=5 +knn, k=6 +clarity.
MOE_STRATEGY_RANKING: list[str] = [
    "anchor_dpp",
    "bm25",
    "pattern_dpp",
    "random",
    "knn",
    "clarity",
]
DEFAULT_MOE_K: int = 4
DEFAULT_MOE_STRATEGIES: list[str] = MOE_STRATEGY_RANKING[:DEFAULT_MOE_K]
_MIN_MOE_K: int = 2
_MAX_MOE_K: int = len(MOE_STRATEGY_RANKING)
_MOE_SUB_PREFIX: str = "_moe_sub_"


# --- Public functions ---


def resolve_moe_strategies(
    strategies: list[str] | None = None,
    k: int | None = None,
) -> list[str]:
    """Resolve the list of sub-strategies for MoE fusion.

    Resolution priority:
    1. Explicit strategies list -> use exactly these (validated, len must be 2–6)
    2. k -> top-k from MOE_STRATEGY_RANKING (k must be 2–6)
    3. Neither -> DEFAULT_MOE_STRATEGIES
    """
    if strategies is not None:
        if not (_MIN_MOE_K <= len(strategies) <= _MAX_MOE_K):
            msg = f"MoE requires 2–{_MAX_MOE_K} strategies (got {len(strategies)})"
            raise ValueError(msg)
        from conformal_relevance.icl.strategies.registry import is_local_strategy
        available_global = StrategyRegistry.list_strategies()
        for name in strategies:
            if name not in available_global and not is_local_strategy(name):
                msg = f"Unknown strategy: '{name}'. Available: {available_global + ['bm25', 'knn']}"
                raise ValueError(msg)
        return list(strategies)

    if k is not None:
        if not (_MIN_MOE_K <= k <= _MAX_MOE_K):
            msg = f"MoE k must be 2–{_MAX_MOE_K} (got {k})"
            raise ValueError(msg)
        return MOE_STRATEGY_RANKING[:k]

    return list(DEFAULT_MOE_STRATEGIES)


def fuse_scores(
    score_vectors: list[list[float] | None],
    weights: list[float] | None = None,
) -> list[float] | None:
    """Element-wise weighted average of K score vectors.

    Skips None inputs and re-normalizes weights over valid inputs.
    Returns None if all inputs are None.
    """
    n_vectors = len(score_vectors)

    # Default weights: equal
    effective_weights = (
        weights if weights is not None else [1.0 / n_vectors] * n_vectors
    )

    if len(effective_weights) != n_vectors:
        msg = f"len(weights)={len(effective_weights)} != len(score_vectors)={n_vectors}"
        raise ValueError(msg)

    # Filter valid inputs
    valid = [
        (sv, w)
        for sv, w in zip(score_vectors, effective_weights)
        if sv is not None
    ]
    if not valid:
        return None

    # Validate vector lengths match
    lengths = [len(sv) for sv, _ in valid]
    if len(set(lengths)) > 1:
        msg = f"Score vectors have mismatched lengths: {lengths}"
        raise ValueError(msg)

    n_sentences = lengths[0]

    # Re-normalize weights over valid inputs
    valid_weight_sum = sum(w for _, w in valid)
    normed = [(sv, w / valid_weight_sum) for sv, w in valid]

    # Element-wise weighted average
    fused = [0.0] * n_sentences
    for sv, w in normed:
        for j in range(n_sentences):
            fused[j] += w * sv[j]

    return fused


def score_dataframe_moe(
    df: pl.DataFrame,
    llm: BaseChatModel,
    icl_by_strategy: dict[str, str | dict[str, str] | list[str]],
    dataset: str,
    n_icl: int = DEFAULT_N_ICL,
    weights: list[float] | None = None,
    output_col: str | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    show_progress: bool = True,
    verbose: bool = True,
    dataset_task: str | None = None,
) -> ScoringResult:
    """Run K scoring passes with different pre-built ICL blocks and fuse results.

    Callers are responsible for building ICL examples via `select_icl_examples()`
    and passing the resulting formatted blocks in `icl_by_strategy`.

    Args:
        df: Input dataframe to score.
        llm: LangChain LLM instance.
        icl_by_strategy: Maps strategy name → pre-formatted ICL block
            (str for uni-intent, dict[str, str] for multi-intent).
        dataset: Dataset name for has_intent / dataset_task resolution.
        n_icl: Number of ICL examples used (for output column naming only).
        weights: Optional per-strategy weights for fusion. Equal weights if None.
        output_col: Override for the fused output column name.
        max_concurrency: Maximum concurrent LLM requests per sub-pass.
        show_progress: Whether to show progress bars.
        verbose: Whether to print summary info.
        dataset_task: Uniform task description override. Registry default used if None.
    """
    resolved_strategies = list(icl_by_strategy.keys())
    effective_weights = (
        weights
        if weights is not None
        else [1.0 / len(resolved_strategies)] * len(resolved_strategies)
    )
    if weights is not None and len(weights) != len(resolved_strategies):
        msg = (
            f"len(weights)={len(weights)} != "
            f"len(strategies)={len(resolved_strategies)}"
        )
        raise ValueError(msg)

    # Output column name
    moe_output_col = output_col or f"scores_icl{n_icl}_moe"

    # Run K scoring passes sequentially
    running_df = df
    sub_scoring_results: dict[str, ScoringResult] = {}
    sub_strategy_col_names: dict[str, str] = {}

    for strategy_name, icl_examples in icl_by_strategy.items():
        sub_output_col = f"scores_icl{n_icl}_{normalize_strategy_name(strategy_name)}"

        logger.info("MoE: scoring with sub-strategy '%s'", strategy_name)

        sub_result = score_dataframe(
            df=running_df,
            llm=llm,
            icl_examples=icl_examples,
            output_col=sub_output_col,
            dataset=dataset,
            dataset_task=dataset_task,
            max_concurrency=max_concurrency,
            show_progress=show_progress,
            verbose=verbose,
        )

        sub_strategy_col_names[strategy_name] = sub_output_col
        running_df = sub_result.df.rename({sub_output_col: f"{_MOE_SUB_PREFIX}{strategy_name}"})
        sub_scoring_results[strategy_name] = sub_result

    # Fuse scores per row
    sub_cols = [f"{_MOE_SUB_PREFIX}{name}" for name in resolved_strategies]
    n_rows = running_df.shape[0]
    fused_scores: list[list[float] | None] = []

    col_lists = {col: running_df[col].to_list() for col in sub_cols}
    for row_idx in range(n_rows):
        row_vectors = [col_lists[col][row_idx] for col in sub_cols]
        fused = fuse_scores(row_vectors, weights=effective_weights)
        fused_scores.append(fused)

    # Extract sub-strategy score vectors before dropping intermediate columns
    sub_strategy_scores: dict[str, list[list[float] | None]] = {}
    for strategy_name, col in zip(resolved_strategies, sub_cols):
        sub_strategy_scores[strategy_name] = running_df[col].to_list()

    # Attach fused column, drop intermediates
    result_df = running_df.with_columns(
        pl.Series(name=moe_output_col, values=fused_scores)
    ).drop(sub_cols)

    # Aggregate profiling across all sub-strategy calls
    n_fused_scored = sum(1 for s in fused_scores if s is not None)
    n_fused_failed = sum(1 for s in fused_scores if s is None)
    n_fused_total = n_fused_scored + n_fused_failed

    sub_profiles = [sr.profile for sr in sub_scoring_results.values() if sr.profile is not None]
    if sub_profiles:
        total_duration_ms = sum(p.total_duration_ms for p in sub_profiles)
        total_llm_time_ms = sum(p.total_llm_time_ms for p in sub_profiles)
        n_llm_calls = sum(p.n_llm_calls for p in sub_profiles)
        moe_profile = RunProfile(
            total_duration_ms=total_duration_ms,
            total_llm_time_ms=total_llm_time_ms,
            n_samples=n_fused_total,
            n_successful=n_fused_scored,
            n_failed=n_fused_failed,
            n_llm_calls=n_llm_calls,
            n_retries=sum(p.n_retries for p in sub_profiles),
            n_targeted_retries=sum(p.n_targeted_retries for p in sub_profiles),
            avg_sample_time_ms=total_duration_ms / n_fused_total if n_fused_total > 0 else 0.0,
            avg_llm_latency_ms=total_llm_time_ms / n_llm_calls if n_llm_calls > 0 else 0.0,
            samples=[],  # per-sample breakdown lives in each sub-strategy's profile
        )
    else:
        moe_profile = None

    return ScoringResult(
        df=result_df,
        n_scored=n_fused_scored,
        n_failed=n_fused_failed,
        profile=moe_profile,
        metadata={
            "fusion_method": "score_average",
            "sub_strategies": resolved_strategies,
            "weights": effective_weights,
            "sub_strategy_scores": sub_strategy_scores,
            "sub_scoring_results": sub_scoring_results,
            "sub_strategy_col_names": sub_strategy_col_names,
            "per_strategy_n_scored": {
                name: sr.n_scored for name, sr in sub_scoring_results.items()
            },
            "per_strategy_n_failed": {
                name: sr.n_failed for name, sr in sub_scoring_results.items()
            },
        },
    )
