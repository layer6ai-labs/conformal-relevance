"""Batch scoring pipeline for processing dataframes."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import polars as pl
from langchain_core.language_models import BaseChatModel

from conformal_relevance.prompts.templates import ICL_HAS_INTENT
from conformal_relevance.scoring._async_engine import (
    run_async,
    score_all_async,
)
from conformal_relevance.scoring.cache import enable_llm_cache
from conformal_relevance.scoring.profiling import ProfileCollector, RunProfile
from conformal_relevance.types import DEFAULT_CACHE_PATH, DEFAULT_INTENT

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_MAX_RETRIES = 2


@dataclass
class ScoringResult:
    """Result from a scoring run with optional profiling and metrics."""

    df: pl.DataFrame
    run_id: int | None = None
    profile: RunProfile | None = None
    mean_ap: float | None = None
    n_scored: int = 0
    n_failed: int = 0
    metadata: dict = field(default_factory=dict)


def _score_dataframe_base(
    df: pl.DataFrame,
    llm: BaseChatModel,
    output_col: str,
    intents: list[str],
    sentences_list: list[Sequence[str]],
    icl_examples_list: list,
    default_intent: str,
    max_retries: int,
    max_concurrency: int,
    show_progress: bool,
    verbose: bool,
    cache_path: str | Path | None,
    track_profiling: bool,
    max_missing_rate: float,
    prompt_builder=None,
    missing_prompt_builder=None,
    has_intent: bool = True,
    dataset_task: str = "",
) -> ScoringResult:
    """Shared scoring execution: profiling init, cache setup, async run, result assembly."""
    profile_collector = ProfileCollector() if track_profiling else None
    if profile_collector:
        profile_collector.start_run()

    if cache_path is not None:
        enable_llm_cache(cache_path)

    scores_list = run_async(
        score_all_async(
            llm=llm,
            intents=intents,
            sentences_list=sentences_list,
            icl_examples_list=icl_examples_list,
            default_intent=default_intent,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            show_progress=show_progress,
            profile_collector=profile_collector,
            prompt_builder=prompt_builder,
            missing_prompt_builder=missing_prompt_builder,
            has_intent=has_intent,
            dataset_task=dataset_task,
            max_missing_rate=max_missing_rate,
        )
    )

    df_result = df.with_columns(pl.Series(name=output_col, values=scores_list))
    n_success = sum(1 for s in scores_list if s is not None)
    n_failed = sum(1 for s in scores_list if s is None)

    if verbose:
        failed_indices = [i for i, s in enumerate(scores_list) if s is None]
        print(f"\nScoring complete:")
        print(f"  Successful: {n_success}/{df.shape[0]}")
        if n_failed > 0:
            print(
                f"  Failed: {n_failed} (rows: {failed_indices[:10]}"
                f"{'...' if n_failed > 10 else ''})"
            )

    profile = profile_collector.finalize() if profile_collector else None
    return ScoringResult(df=df_result, profile=profile, n_scored=n_success, n_failed=n_failed)


def score_dataframe(
    df: pl.DataFrame,
    llm: BaseChatModel,
    output_col: str = "relevancy_scores",
    intent_col: str = "intent",
    sentences_col: str = "input_units",
    icl_examples: str | dict[str, str] | list[str] | None = None,
    default_intent: str = DEFAULT_INTENT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    show_progress: bool = True,
    verbose: bool = True,
    cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    track_profiling: bool = True,
    dataset: str | None = None,
    dataset_task: str | None = None,
    # Partial-score recovery
    max_missing_rate: float = 0.25,
) -> ScoringResult:
    """Score a dataframe of samples for relevancy.

    Receives pre-formatted ICL examples via `icl_examples`. To build ICL examples,
    call `select_icl_examples()` first and pass `icl_result.icl_examples`.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with columns for intent and sentences.
    llm : BaseChatModel
        LangChain LLM instance.
    output_col : str
        Name of the output column to store scores.
    intent_col : str
        Name of the intent column.
    sentences_col : str
        Name of the sentences list column.
    icl_examples : str | dict[str, str] | list[str] | None
        Pre-formatted ICL examples (str for uni-intent, dict for multi-intent,
        list[str] for per-sample local ICL).
    default_intent : str
        Intent string to use for ICL examples lacking 'intent' key.
    max_retries : int
        Number of retry attempts per sample on failure.
    max_concurrency : int
        Maximum number of concurrent LLM requests.
    show_progress : bool
        Whether to show progress bar.
    verbose : bool
        Whether to print summary information.
    cache_path : str | Path | None
        Path to SQLite cache database.
    track_profiling : bool
        Whether to collect profiling data.
    dataset : str | None
        Dataset name used to resolve `has_intent` (prompt template variant) and
        registry `dataset_task` default. Does not trigger ICL selection.
    dataset_task : str | None
        Uniform task description for all samples (e.g., "PHI detection", "Question answering").
        Appears as "Task: {description}" in every inference prompt. Separate from the intent
        field (per-sample category in multi-intent datasets). A registry-configured default is
        used when this is None and the dataset has a dataset_task entry. Explicit kwarg takes
        priority over registry default. Pass empty string to suppress registry default.
    max_missing_rate : float
        Maximum allowed fraction of samples with missing/failed scores (default: 0.25).
        If the fraction exceeds this threshold, ValueError is raised.

    Returns
    -------
    ScoringResult
        Result object containing scored DataFrame and optional profiling data.
    """
    has_intent = ICL_HAS_INTENT.get(dataset, True) if dataset is not None else True

    # Resolve uniform dataset task. Priority: explicit kwarg > registry default > "".
    # None means "use registry default"; "" means "explicitly suppress task".
    if dataset_task is not None:
        effective_dataset_task: str = dataset_task
    elif dataset is not None:
        from conformal_relevance.icl.registry import DATASET_REGISTRY
        effective_dataset_task = DATASET_REGISTRY.get(dataset, {}).get("dataset_task", "")
    else:
        effective_dataset_task = ""

    if verbose:
        if cache_path is not None:
            print(f"LLM cache enabled: {cache_path}")
        print(f"Scoring {df.shape[0]} samples for relevancy")
        print(f"  Output column: {output_col}")
        print(f"  Max concurrency: {max_concurrency}")
        icl_type = (
            "None"
            if icl_examples is None
            else "intent-keyed dict"
            if isinstance(icl_examples, dict)
            else "per-sample list"
            if isinstance(icl_examples, list)
            else "shared string/examples"
        )
        print(f"  ICL examples: {icl_type}")

    if sentences_col not in df.columns:
        raise ValueError(f"Missing required column: {sentences_col}")

    has_intent_col = intent_col in df.columns
    if not has_intent_col:
        if not has_intent:
            # No-intent datasets (PhysioNet, ECTSum) never render intent in the prompt,
            # so override default_intent to "" to keep internal state honest.
            default_intent = ""
        if verbose:
            if has_intent:
                print(f"  No '{intent_col}' column found - using default_intent: '{default_intent}'")
            else:
                print(f"  No '{intent_col}' column found (not needed — has_intent=False)")

    if isinstance(icl_examples, list) and len(icl_examples) != df.shape[0]:
        raise ValueError(
            f"icl_examples list length ({len(icl_examples)}) != "
            f"df row count ({df.shape[0]})"
        )

    # Extract raw data from dataframe
    intents: list[str] = []
    sentences_list: list[Sequence[str]] = []
    icl_examples_list: list[str | None] = []

    for idx, row in enumerate(df.iter_rows(named=True)):
        intent = row[intent_col] if has_intent_col else default_intent
        sentences = row[sentences_col]

        # Resolve per-sample list, per-intent dict, or shared string
        if isinstance(icl_examples, list):
            icl_for_row = icl_examples[idx]
        elif isinstance(icl_examples, dict):
            icl_for_row = icl_examples.get(intent)
        else:
            icl_for_row = icl_examples

        intents.append(intent)
        sentences_list.append(sentences)
        icl_examples_list.append(icl_for_row)

    if verbose:
        print(f"  Prepared {len(intents)} samples for scoring")

    return _score_dataframe_base(
        df=df, llm=llm, output_col=output_col,
        intents=intents, sentences_list=sentences_list, icl_examples_list=icl_examples_list,
        default_intent=default_intent,
        max_retries=max_retries, max_concurrency=max_concurrency,
        show_progress=show_progress, verbose=verbose, cache_path=cache_path,
        track_profiling=track_profiling, max_missing_rate=max_missing_rate,
        has_intent=has_intent, dataset_task=effective_dataset_task,
    )


def score_dataframe_no_icl(
    df: pl.DataFrame,
    llm: BaseChatModel,
    dataset: str,
    output_col: str = "scores_icl0",
    intent_col: str = "intent",
    sentences_col: str = "input_units",
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    show_progress: bool = True,
    verbose: bool = True,
    cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    track_profiling: bool = True,
    max_missing_rate: float = 0.25,
) -> ScoringResult:
    """Score sentences using dataset-specific ICL-free prompts.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with sentences to score.
    llm : BaseChatModel
        LangChain LLM instance.
    dataset : str
        Dataset name. Must be one of the keys in ICL_FREE_PROMPT_REGISTRY.
    output_col : str
        Name of the output column for scores.
    intent_col : str
        Column containing intent (used for datasets with intent).
    sentences_col : str
        Column containing sentence lists.
    max_retries : int
        Number of retry attempts per sample on failure.
    max_concurrency : int
        Maximum number of concurrent LLM requests.
    show_progress : bool
        Whether to show progress bar.
    verbose : bool
        Whether to print summary information.
    cache_path : str | Path | None
        Path to SQLite cache database.
    track_profiling : bool
        Whether to collect profiling data.
    max_missing_rate : float
        Maximum allowed fraction of samples with missing/failed scores (default: 0.25).
        If the fraction exceeds this threshold, ValueError is raised.

    Returns
    -------
    ScoringResult
        Result object containing scored DataFrame.
    """
    from conformal_relevance.prompts.icl_free import ICL_FREE_PROMPT_REGISTRY
    from conformal_relevance.prompts.formatting import format_sentences_numbered

    if dataset not in ICL_FREE_PROMPT_REGISTRY:
        available = list(ICL_FREE_PROMPT_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {dataset!r}. Available: {available}")

    main_prompt, retry_prompt, has_intent = ICL_FREE_PROMPT_REGISTRY[dataset]

    def prompt_builder(intent: str, sentences: Sequence[str]) -> str:
        sentences_text = format_sentences_numbered(sentences)
        return main_prompt.format_map(
            defaultdict(str, intent=intent, sentences=sentences_text)
        )

    def missing_prompt_builder(intent: str, sentences: Sequence[str], missing: set[str]) -> str:
        missing_idx_ints = sorted(int(i) for i in missing)
        missing_lines = [f"{i}. {sentences[i]}" for i in missing_idx_ints]
        missing_text = "\n".join(missing_lines)
        return retry_prompt.format_map(
            defaultdict(str, intent=intent, missing_sentences=missing_text)
        )

    sentences_list = df[sentences_col].to_list()

    if has_intent:
        if intent_col not in df.columns:
            raise ValueError(
                f"Dataset '{dataset}' requires intent but column '{intent_col}' "
                f"not found in dataframe. Columns: {df.columns}"
            )
        intents_list = df[intent_col].to_list()
    else:
        intents_list = [""] * len(sentences_list)

    if verbose:
        print(f"Scoring {len(sentences_list)} samples for {dataset} (ICL-free)")
        print(f"  Has intent: {has_intent}")
        print(f"  Max concurrency: {max_concurrency}")

    icl_examples_list = [None] * len(sentences_list)

    return _score_dataframe_base(
        df=df, llm=llm, output_col=output_col,
        intents=intents_list, sentences_list=sentences_list, icl_examples_list=icl_examples_list,
        default_intent="",
        max_retries=max_retries, max_concurrency=max_concurrency,
        show_progress=show_progress, verbose=verbose, cache_path=cache_path,
        track_profiling=track_profiling, max_missing_rate=max_missing_rate,
        prompt_builder=prompt_builder, missing_prompt_builder=missing_prompt_builder,
        has_intent=has_intent,
    )


def compute_average_precision(
    df: pl.DataFrame,
    scores_col: str,
    labels_col: str = "input_unit_labels",
    output_col: str | None = None,
) -> pl.DataFrame:
    """Compute average precision for each row."""
    from sklearn.metrics import average_precision_score

    if output_col is None:
        output_col = f"{scores_col}_ap"

    def safe_ap(scores: list | None, labels: list) -> float | None:
        if scores is None:
            return None
        if not any(labels) or all(labels):
            return None
        try:
            return average_precision_score(labels, scores)
        except (ValueError, TypeError):
            return None

    ap_values = []
    for row in df.iter_rows(named=True):
        scores = row[scores_col]
        labels = row[labels_col]
        ap_values.append(safe_ap(scores, labels))

    return df.with_columns(pl.Series(name=output_col, values=ap_values))


def summarize_results(
    df: pl.DataFrame,
    score_cols: Sequence[str] | None = None,
    labels_col: str = "input_unit_labels",
) -> dict[str, float]:
    """Compute mean average precision for multiple score columns."""
    if score_cols is None:
        score_cols = [c for c in df.columns if c.startswith("scores_")]
        if not score_cols:
            print("No score columns found (looking for 'scores_*' pattern)")
            return {}

    results = {}
    print("Mean Average Precision:")

    for col in score_cols:
        if col not in df.columns:
            print(f"  {col}: [column not found]")
            continue

        df_with_ap = compute_average_precision(df, col, labels_col)
        ap_col = f"{col}_ap"

        ap_values = df_with_ap[ap_col].drop_nulls().to_list()
        if ap_values:
            mean_ap = sum(ap_values) / len(ap_values)
            results[col] = mean_ap
            print(f"  {col}: {mean_ap:.3f}")
        else:
            results[col] = float("nan")
            print(f"  {col}: NaN (no valid scores)")

    return results


def score_and_evaluate(
    df: pl.DataFrame,
    llm: BaseChatModel,
    output_col: str = "relevancy_scores",
    icl_examples: str | dict[str, str] | list[str] | None = None,
    labels_col: str = "input_unit_labels",
    **kwargs,
) -> tuple[pl.DataFrame, float]:
    """Score a dataframe and compute mean average precision."""
    result = score_dataframe(
        df, llm, output_col=output_col, icl_examples=icl_examples, **kwargs
    )

    df_with_ap = compute_average_precision(result.df, output_col, labels_col)

    ap_col = f"{output_col}_ap"
    ap_values = df_with_ap[ap_col].drop_nulls().to_list()
    mean_ap = sum(ap_values) / len(ap_values) if ap_values else float("nan")

    print(f"\nMean Average Precision for {output_col}: {mean_ap:.3f}")

    return df_with_ap, mean_ap
