"""ICL example generation for relevancy scoring.

This module provides functions to select and format ICL examples on-the-fly
from pool data using pluggable strategies, with optional windowed
sampling for long documents.
"""

import math
import random
from dataclasses import dataclass
from typing import Sequence

import polars as pl

from conformal_relevance.icl.strategies.base import (
    DEFAULT_SEED,
    ICLSelectionContext,
    ICLSelectionResult,
    ICLSelectionStrategy,
    StratifiedICLSelectionResult,
)
from conformal_relevance.icl.strategies.local import (
    LocalICLContext,
    LocalICLStrategy,
    _build_retrieval_text,
    _get_input_text,
)
from conformal_relevance.prompts.formatting import format_relevancy_icl_examples
from conformal_relevance.types import DEFAULT_N_ICL, normalize_strategy_name


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ICLResult:
    """Result of ICL example selection and formatting.

    icl_examples is a contrastive ICL block — the "examples" portion of the prompt,
    not a complete prompt. Per-sample prompt assembly happens later in
    build_relevancy_prompt() at scoring time.
    """

    icl_examples: str | dict[str, str]  # str (uni-intent) or dict (multi-intent)
    strategy_result: ICLSelectionResult | StratifiedICLSelectionResult | None
    output_col: str  # e.g. "scores_icl3_anchor_dpp"


@dataclass
class LocalICLResult:
    """Result of local (per-sample) ICL selection and formatting."""

    icl_examples: list[str]               # One formatted ICL block per inference sample
    output_col: str                        # e.g. "scores_icl3_bm25"
    per_sample_indices: list[list[int]]    # Pool indices selected per sample (for analysis)


# ---------------------------------------------------------------------------
# Windowing helpers
# ---------------------------------------------------------------------------


def _create_sampled_example(
    sentences: Sequence[str],
    labels: Sequence[int],
    context_window: int = 2,
    sampling_min_negatives: int = 3,
    n_remote_neg_ratio: float = 0.5,
    random_seed: int | None = None,
) -> tuple[list[str], list[int]]:
    """Create a sampled ICL example with asymmetric context and remote negatives.

    Steps:
    1. Select all positive sentences
    2. For each positive, independently sample left ~ U(0, context_window) and
       right ~ U(0, context_window) to build an asymmetric neighborhood
    3. Merge overlapping neighborhoods (union of selected indices)
    4. Enforce minimum negatives floor on core — if core has fewer than
       sampling_min_negatives negatives, sample additional from available
    5. Add remote negatives for distributional coverage:
       ceil(n_remote_neg_ratio * n_pos) non-core sentences are sampled
    6. Sort all selected indices by original position
    7. Return (sentences, labels) renumbered from 0

    Returns ([], []) if the example has no positive sentences.
    """
    n = len(sentences)
    rng = random.Random(random_seed)

    n_pos = sum(1 for lbl in labels if lbl)
    if n_pos == 0:
        return [], []

    # Step 1-2: Build core mask — positives + asymmetric neighborhoods
    core_mask = [False] * n
    for i, label in enumerate(labels):
        if label:
            core_mask[i] = True  # Always include the positive itself
            left = rng.randint(0, context_window)
            right = rng.randint(0, context_window)
            lo = max(0, i - left)
            hi = min(n, i + right + 1)
            for j in range(lo, hi):
                core_mask[j] = True

    # Step 3: Partition indices (merging is implicit via the mask)
    core_indices = [i for i in range(n) if core_mask[i]]
    available_indices = [i for i in range(n) if not core_mask[i]]

    # Step 4: Enforce minimum negatives floor on core
    n_core_neg = len(core_indices) - n_pos
    if n_core_neg < sampling_min_negatives and available_indices:
        floor_to_sample = min(sampling_min_negatives - n_core_neg, len(available_indices))
        floor_sampled = rng.sample(available_indices, floor_to_sample)
        floor_sampled_set = set(floor_sampled)
        remaining_available = [i for i in available_indices if i not in floor_sampled_set]
    else:
        floor_sampled = []
        remaining_available = available_indices

    # Step 5: Add remote negatives for distributional coverage
    # ceil(ratio * n_pos) ensures easy negatives scale with positive density
    n_remote = min(math.ceil(n_remote_neg_ratio * n_pos), len(remaining_available))
    remote_sampled = rng.sample(remaining_available, n_remote) if n_remote > 0 else []

    all_indices = sorted(core_indices + floor_sampled + remote_sampled)

    # Step 6-7: Extract renumbered
    return [sentences[i] for i in all_indices], [labels[i] for i in all_indices]


def _should_window_dataset(
    pool_df: pl.DataFrame,
    sentences_col: str = "input_units",
    labels_col: str = "input_unit_labels",
    min_avg_sentences: int = 30,
    max_avg_positive_rate: float = 0.15,
) -> bool:
    """Return True if EITHER: avg sentence count > 30 OR avg positive rate < 15%.

    Computes dataset-level statistics over the full pool set so that
    all rows are treated uniformly -- no mixed windowed/non-windowed pool sets.
    """
    avg_sents = pool_df.select(
        pl.col(sentences_col).list.len().mean()
    ).item()
    avg_pos_rate = pool_df.select(
        (pl.col(labels_col).list.sum() / pl.col(labels_col).list.len()).mean()
    ).item()
    return avg_sents > min_avg_sentences or avg_pos_rate < max_avg_positive_rate


def _window_pool_df(
    pool_df: pl.DataFrame,
    context_window: int,
    sampling_min_negatives: int = 3,
    n_remote_neg_ratio: float = 0.5,
    seed: int = DEFAULT_SEED,
    sentences_col: str = "input_units",
    labels_col: str = "input_unit_labels",
) -> pl.DataFrame:
    """Apply windowed sampling to each row of a pool DataFrame.

    All rows are windowed unconditionally -- the dataset-level guard
    (_should_window_dataset) lives in the callers. Rows with no positives
    after windowing are dropped.

    Returns a new DataFrame with the same schema as the input.
    """
    windowed_rows = []
    for i, row in enumerate(pool_df.iter_rows(named=True)):
        sentences = row[sentences_col]
        labels = row[labels_col]

        w_sents, w_labels = _create_sampled_example(
            sentences=sentences,
            labels=labels,
            context_window=context_window,
            sampling_min_negatives=sampling_min_negatives,
            n_remote_neg_ratio=n_remote_neg_ratio,
            random_seed=seed + i,
        )

        if not w_sents:
            continue  # Drop rows with no positives

        new_row = dict(row)
        new_row[sentences_col] = w_sents
        new_row[labels_col] = w_labels
        windowed_rows.append(new_row)

    return pl.DataFrame(windowed_rows, schema=pool_df.schema)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_icl_examples(
    pool_df: pl.DataFrame,
    strategy: ICLSelectionStrategy,
    dataset: str,
    n_icl: int = DEFAULT_N_ICL,
    seed: int = DEFAULT_SEED,
    dataset_task: str | None = None,
    context_window: int | None = None,
    n_remote_neg_ratio: float = 0.5,
    sampling_min_negatives: int = 3,
    intent_col: str = "intent",
    sentences_col: str = "input_units",
    verbose: bool = True,
) -> ICLResult:
    """Select and format ICL examples from pool.

    Called once per strategy per dataset. Delegates to
    select_and_format_icl_examples_by_intent() for multi-intent datasets,
    select_and_format_icl_examples() otherwise.

    Args:
        pool_df: Pool dataframe (used only for ICL selection, never scored).
        strategy: ICL selection strategy.
        dataset: Dataset name for registry lookup.
        n_icl: Number of ICL examples to select.
        seed: Random seed for reproducibility.
        dataset_task: Uniform task description (priority: explicit kwarg > registry default > "").
        context_window: If set, apply windowed sampling before strategy selection.
        n_remote_neg_ratio: Ratio of remote negatives per positive in windowed examples.
        sampling_min_negatives: Min negatives guaranteed per windowed example.
        intent_col: Name of the intent column in pool_df.
        sentences_col: Name of the sentences column in pool_df.
        verbose: Whether to print summary information.

    Returns:
        ICLResult with formatted ICL block, strategy result, and output column name.
    """
    from conformal_relevance.icl.registry import DATASET_REGISTRY, get_dataset_config
    from conformal_relevance.prompts.templates import ICL_HAS_INTENT

    # Resolve has_intent (for formatting)
    has_intent = ICL_HAS_INTENT.get(dataset, True)

    # Resolve dataset_task (priority: explicit kwarg > registry default > "")
    if dataset_task is not None:
        effective_task: str = dataset_task
    else:
        effective_task = DATASET_REGISTRY.get(dataset, {}).get("dataset_task", "")

    # Resolve multi_intent
    config = get_dataset_config(dataset)
    multi_intent = bool(config["multi_intent"])

    # Derive output column name
    base_name = normalize_strategy_name(strategy.name)
    output_col = f"scores_icl{n_icl}_{base_name}"

    if multi_intent and intent_col in pool_df.columns:
        icl_examples, strategy_result = select_and_format_icl_examples_by_intent(
            pool_df=pool_df,
            strategy=strategy,
            n_examples=n_icl,
            intent_col=intent_col,
            sentences_col=sentences_col,
            seed=seed,
            task=effective_task,
            context_window=context_window,
            n_remote_neg_ratio=n_remote_neg_ratio,
            sampling_min_negatives=sampling_min_negatives,
        )
        if verbose:
            print(f"Generated ICL examples for {len(icl_examples)} intents using {strategy.name}")
    else:
        icl_examples, strategy_result = select_and_format_icl_examples(
            pool_df=pool_df,
            strategy=strategy,
            n_examples=n_icl,
            sentences_col=sentences_col,
            seed=seed,
            has_intent=has_intent,
            task=effective_task,
            context_window=context_window,
            n_remote_neg_ratio=n_remote_neg_ratio,
            sampling_min_negatives=sampling_min_negatives,
        )
        if verbose:
            print(f"Generated global ICL examples using {strategy.name}")

    return ICLResult(
        icl_examples=icl_examples,
        strategy_result=strategy_result,
        output_col=output_col,
    )


def select_and_format_icl_examples(
    pool_df: pl.DataFrame,
    strategy: ICLSelectionStrategy,
    n_examples: int,
    intent: str | None = None,
    seed: int = DEFAULT_SEED,
    intent_col: str = "intent",
    sentences_col: str = "input_units",
    labels_col: str = "input_unit_labels",
    min_negatives: int = 2,
    has_intent: bool = True,
    task: str = "",
    # Windowing parameters
    context_window: int | None = None,
    sampling_min_negatives: int = 3,
    n_remote_neg_ratio: float = 0.5,
) -> tuple[str, ICLSelectionResult]:
    """Select ICL examples using strategy and format them as contrastive prompt text.

    Args:
        pool_df: Pool dataframe.
        strategy: ICL selection strategy.
        n_examples: Number of examples to select.
        intent: Target intent for filtering (filters pool_df to this intent).
        seed: Random seed for reproducibility.
        intent_col: Name of the intent column in pool_df.
        sentences_col: Name of the sentences/units column in pool_df.
        labels_col: Name of the labels column in pool_df.
        min_negatives: Min negatives for contrastive formatting.
        has_intent: Whether examples include Intent line in formatting.
        context_window: If set, apply windowed sampling to pool data
            before strategy selection. None means no windowing.
        sampling_min_negatives: Min negatives guaranteed per windowed example.
        n_remote_neg_ratio: Ratio of remote negatives per positive in windowed
            ICL examples (default: 0.5).

    Returns:
        Tuple of (formatted ICL examples string, strategy selection result).
    """
    # Apply windowing before intent filtering (with dataset-level guard)
    if context_window is not None and _should_window_dataset(pool_df, sentences_col, labels_col):
        pool_df = _window_pool_df(
            pool_df,
            context_window=context_window,
            sampling_min_negatives=sampling_min_negatives,
            n_remote_neg_ratio=n_remote_neg_ratio,
            seed=seed,
            sentences_col=sentences_col,
            labels_col=labels_col,
        )

    # If intent specified, filter pool data to that intent
    if intent is not None and intent_col in pool_df.columns:
        intent_df = pool_df.filter(pl.col(intent_col) == intent)
        # Fall back to full dataset if not enough samples for this intent
        if intent_df.shape[0] >= n_examples:
            pool_df = intent_df

    context = ICLSelectionContext(
        pool_df=pool_df,
        target_intent=intent,
        n_examples=n_examples,
        seed=seed,
        intent_col=intent_col,
        sentences_col=sentences_col,
        labels_col=labels_col,
    )
    result = strategy.select(context, n_examples=n_examples)
    formatted = format_relevancy_icl_examples(
        result.examples,
        min_negatives=min_negatives,
        random_seed=seed,
        has_intent=has_intent,
        task=task,
    )
    return formatted, result


def format_stratified_icl_result(
    result: StratifiedICLSelectionResult,
    min_negatives: int = 2,
    seed: int = DEFAULT_SEED,
    has_intent: bool = True,
    task: str = "",
) -> dict[str, str]:
    """Format a StratifiedICLSelectionResult into per-intent prompt strings.

    Args:
        result: Stratified selection result from a stratified strategy.
        min_negatives: Min negatives for contrastive formatting.
        seed: Random seed for reproducibility in formatting.
        has_intent: Whether examples include Intent line in formatting.
            Stratified results are always for multi-intent datasets, so
            has_intent=True is expected; parameter kept for API consistency.
        task: Uniform task description (e.g., "PHI detection"). Empty string = no Task line.

    Returns:
        Dict mapping each intent value to its formatted ICL examples string.
    """
    return {
        intent_val: format_relevancy_icl_examples(
            intent_result.examples,
            min_negatives=min_negatives,
            random_seed=seed,
            has_intent=has_intent,
            task=task,
        )
        for intent_val, intent_result in result.by_intent.items()
    }


def select_and_format_icl_examples_by_intent(
    pool_df: pl.DataFrame,
    strategy: ICLSelectionStrategy,
    n_examples: int,
    intent_col: str = "intent",
    sentences_col: str = "input_units",
    labels_col: str = "input_unit_labels",
    seed: int = DEFAULT_SEED,
    min_negatives: int = 2,
    has_intent: bool = True,
    task: str = "",
    # Windowing parameters
    context_window: int | None = None,
    sampling_min_negatives: int = 3,
    n_remote_neg_ratio: float = 0.5,
) -> tuple[dict[str, str], ICLSelectionResult | StratifiedICLSelectionResult]:
    """Generate separate ICL example strings for each intent in pool_df.

    For multi-intent datasets (e.g. PUMA, SubSumE, Evidence Inference),
    this produces one formatted ICL string per intent, with examples drawn
    only from pool rows matching that intent.

    If the strategy is a stratified strategy (returns StratifiedICLSelectionResult),
    it delegates directly to the strategy. Otherwise, it iterates over unique
    intents and calls select_and_format_icl_examples for each.

    Args:
        pool_df: Pool dataframe with an intent column.
        strategy: ICL selection strategy.
        n_examples: Number of examples to select per intent.
        intent_col: Name of the intent column in pool_df.
        sentences_col: Name of the sentences/units column in pool_df.
        labels_col: Name of the labels column in pool_df.
        seed: Random seed for reproducibility.
        min_negatives: Min negatives for contrastive formatting.
        has_intent: Whether examples include Intent line in formatting.
        context_window: If set, apply windowed sampling before strategy selection.
        sampling_min_negatives: Min negatives guaranteed per windowed example.
        n_remote_neg_ratio: Ratio of remote negatives per positive in windowed
            ICL examples (default: 0.5).

    Returns:
        Tuple of (dict mapping intent to formatted ICL string, strategy result).
    """
    # Apply windowing once at the top level (with dataset-level guard)
    # Guard sees the full pool set before intent filtering.
    if context_window is not None and _should_window_dataset(pool_df, sentences_col, labels_col):
        pool_df = _window_pool_df(
            pool_df,
            context_window=context_window,
            sampling_min_negatives=sampling_min_negatives,
            n_remote_neg_ratio=n_remote_neg_ratio,
            seed=seed,
            sentences_col=sentences_col,
            labels_col=labels_col,
        )

    # Build context and try strategy.select() -- if it returns a
    # StratifiedICLSelectionResult, format that directly.
    context = ICLSelectionContext(
        pool_df=pool_df,
        n_examples=n_examples,
        seed=seed,
        intent_col=intent_col,
        sentences_col=sentences_col,
        labels_col=labels_col,
    )
    result = strategy.select(context, n_examples=n_examples)

    if isinstance(result, StratifiedICLSelectionResult):
        formatted = format_stratified_icl_result(
            result,
            min_negatives=min_negatives,
            seed=seed,
            has_intent=has_intent,
            task=task,
        )
        return formatted, result

    # Fallback for non-stratified strategies: iterate over intents manually.
    # Pass context_window=None to inner calls -- windowing already applied above.
    unique_intents = pool_df[intent_col].unique().to_list()
    formatted = {}
    for intent_val in unique_intents:
        text, _ = select_and_format_icl_examples(
            pool_df=pool_df,
            strategy=strategy,
            n_examples=n_examples,
            intent=intent_val,
            seed=seed,
            intent_col=intent_col,
            sentences_col=sentences_col,
            labels_col=labels_col,
            min_negatives=min_negatives,
            has_intent=has_intent,
            task=task,
            context_window=None,  # Already windowed above
        )
        formatted[intent_val] = text
    return formatted, result


# ---------------------------------------------------------------------------
# Local (per-sample) ICL API
# ---------------------------------------------------------------------------


def _apply_windowing_to_row(
    sentences: list,
    labels: list,
    context_window: int,
    seed: int,
    pool_row_idx: int,
    sampling_min_negatives: int,
    n_remote_neg_ratio: float,
) -> tuple[list, list] | None:
    """Apply windowed sampling to a pool row. Returns (sentences, labels) or None if no positives."""
    w_sents, w_labels = _create_sampled_example(
        sentences=sentences,
        labels=labels,
        context_window=context_window,
        sampling_min_negatives=sampling_min_negatives,
        n_remote_neg_ratio=n_remote_neg_ratio,
        random_seed=seed + pool_row_idx,
    )
    return (w_sents, w_labels) if w_sents else None


def _format_icl_block_for_sample(
    pool_df: pl.DataFrame,
    pool_row_indices: list[int],
    should_window: bool,
    context_window: int | None,
    seed: int,
    sampling_min_negatives: int,
    n_remote_neg_ratio: float,
    intent_col: str,
    sentences_col: str,
    labels_col: str,
    has_intent: bool,
    effective_task: str,
) -> str:
    """Format a contrastive ICL block for one inference sample."""
    formatted_examples = []
    for inner_idx, row in enumerate(pool_df[pool_row_indices].iter_rows(named=True)):
        sentences = list(row[sentences_col])
        labels = list(row[labels_col])
        if should_window:
            windowed = _apply_windowing_to_row(
                sentences=sentences,
                labels=labels,
                context_window=context_window,
                seed=seed,
                pool_row_idx=pool_row_indices[inner_idx],
                sampling_min_negatives=sampling_min_negatives,
                n_remote_neg_ratio=n_remote_neg_ratio,
            )
            if windowed is None:
                continue
            sentences, labels = windowed
        formatted_examples.append({
            "intent": row.get(intent_col, ""),
            "input_units": sentences,
            "input_unit_labels": labels,
        })
    return format_relevancy_icl_examples(
        formatted_examples,
        min_negatives=2,
        random_seed=seed,
        has_intent=has_intent,
        task=effective_task,
    )


def select_local_icl_examples(
    pool_df: pl.DataFrame,
    inference_df: pl.DataFrame,
    strategy: LocalICLStrategy,
    dataset: str,
    n_icl: int = DEFAULT_N_ICL,
    seed: int = DEFAULT_SEED,
    dataset_task: str | None = None,
    context_window: int | None = None,
    n_remote_neg_ratio: float = 0.5,
    sampling_min_negatives: int = 3,
    intent_col: str = "intent",
    sentences_col: str = "input_units",
    labels_col: str = "input_unit_labels",
    input_col: str = "input",
    verbose: bool = True,
) -> LocalICLResult:
    """Select and format per-sample ICL examples using a local strategy.

    Unlike ``select_icl_examples`` (which produces one fixed ICL block for all
    samples), this function produces one contrastive ICL block per inference row,
    conditioned on that row's content.

    Args:
        pool_df: Pool dataframe (used only for ICL selection, never scored).
        inference_df: Inference samples (calibration or test split).
        strategy: Local ICL strategy (already instantiated; ``prepare()`` is
            called once here to build the retrieval index).
        dataset: Dataset name for registry lookup (has_intent, dataset_task).
        n_icl: Number of ICL examples to select per inference sample.
        seed: Random seed for deterministic windowing.
        dataset_task: Explicit task description. Registry default used if None.
        context_window: If set, apply windowed sampling to selected pool rows
            after retrieval (same logic as global ICL windowing).
        n_remote_neg_ratio: Remote negative ratio for windowed examples.
        sampling_min_negatives: Min negatives per windowed example.
        intent_col: Name of the intent column.
        sentences_col: Name of the sentences/units column.
        labels_col: Name of the labels column.
        input_col: Name of the raw input text column.
        verbose: Whether to print summary information.

    Returns:
        LocalICLResult with per-sample ICL blocks, output column name,
        and per-sample pool indices for analysis.
    """
    from conformal_relevance.icl.registry import DATASET_REGISTRY, get_dataset_config
    from conformal_relevance.prompts.templates import ICL_HAS_INTENT

    # Resolve has_intent, dataset_task, multi_intent from registry
    has_intent = ICL_HAS_INTENT.get(dataset, True)

    if dataset_task is not None:
        effective_task: str = dataset_task
    else:
        effective_task = DATASET_REGISTRY.get(dataset, {}).get("dataset_task", "")

    config = get_dataset_config(dataset)
    multi_intent = bool(config["multi_intent"])

    # Derive output column name
    output_col = f"scores_icl{n_icl}_{normalize_strategy_name(strategy.name)}"

    # Step 2: Windowing guard (once) — evaluate on full pool to get dataset-level stats
    should_window = (
        context_window is not None
        and _should_window_dataset(pool_df, sentences_col, labels_col)
    )

    # Step 3: Build LocalICLContext and prepare strategy (builds index over raw pool)
    context = LocalICLContext(
        pool_df=pool_df,
        n_examples=n_icl,
        seed=seed,
        has_intent=has_intent and multi_intent,
        dataset_task=effective_task,
        intent_col=intent_col,
        sentences_col=sentences_col,
        labels_col=labels_col,
        input_col=input_col,
    )
    strategy.prepare(context)

    # Step 4: Build query texts for all inference rows
    has_intent_col = intent_col in inference_df.columns
    query_texts: list[str] = []
    query_intents: list[str] = []

    for row in inference_df.iter_rows(named=True):
        intent_val = row.get(intent_col, "") if has_intent_col else ""
        query_text = _build_retrieval_text(
            input_text=_get_input_text(row, input_col, sentences_col),
            intent=intent_val,
            dataset_task=effective_task,
            has_intent=has_intent,
        )
        query_texts.append(query_text)
        query_intents.append(intent_val)

    # Step 5: Batch selection
    per_sample_indices = strategy.select_for_batch(query_texts, query_intents)

    # Step 6: Format per-sample ICL blocks
    icl_blocks = [
        _format_icl_block_for_sample(
            pool_df=pool_df,
            pool_row_indices=indices,
            should_window=should_window,
            context_window=context_window,
            seed=seed,
            sampling_min_negatives=sampling_min_negatives,
            n_remote_neg_ratio=n_remote_neg_ratio,
            intent_col=intent_col,
            sentences_col=sentences_col,
            labels_col=labels_col,
            has_intent=has_intent,
            effective_task=effective_task,
        )
        for indices in per_sample_indices
    ]

    if verbose:
        n_inf = inference_df.shape[0]
        print(f"Generated per-sample local ICL blocks: {n_inf} samples using {strategy.name}")

    return LocalICLResult(
        icl_examples=icl_blocks,
        output_col=output_col,
        per_sample_indices=per_sample_indices,
    )
