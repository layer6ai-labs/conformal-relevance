"""ICL (In-Context Learning) module for relevancy scoring.

Provides strategy-based on-the-fly ICL example generation.
"""

from conformal_relevance.icl.generation import (
    ICLResult,
    LocalICLResult,
    format_stratified_icl_result,
    select_icl_examples,
    select_local_icl_examples,
    select_and_format_icl_examples,
    select_and_format_icl_examples_by_intent,
)
from conformal_relevance.icl.strategies import (
    DEFAULT_N_ICL,
    DEFAULT_SEED,
    ICLSelectionStrategy,
    ICLSelectionContext,
    ICLSelectionResult,
    StratifiedICLSelectionResult,
    StrategyRegistry,
    get_local_strategy,
    is_local_strategy,
    LocalICLContext,
    LocalICLStrategy,
)
from conformal_relevance.icl.registry import DATASET_REGISTRY, get_dataset_config

__all__ = [
    "ICLResult",
    "LocalICLResult",
    "select_icl_examples",
    "select_local_icl_examples",
    "format_stratified_icl_result",
    "select_and_format_icl_examples",
    "select_and_format_icl_examples_by_intent",
    "DEFAULT_N_ICL",
    "DEFAULT_SEED",
    "ICLSelectionStrategy",
    "ICLSelectionContext",
    "ICLSelectionResult",
    "StratifiedICLSelectionResult",
    "StrategyRegistry",
    "get_local_strategy",
    "is_local_strategy",
    "LocalICLContext",
    "LocalICLStrategy",
    "DATASET_REGISTRY",
    "get_dataset_config",
]
