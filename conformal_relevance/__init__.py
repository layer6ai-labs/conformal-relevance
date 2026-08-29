"""Conformal Relevancy Library.

A library for intent-based relevancy scoring using LLMs with in-context learning.
"""

# Scoring
from conformal_relevance.scoring import (
    ScoringResult,
    score_dataframe,
    score_dataframe_no_icl,
    score_and_evaluate,
    compute_average_precision,
    summarize_results,
    relevancy_dict_scorer,
    enable_llm_cache,
)

# Prompts
from conformal_relevance.prompts import (
    RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT,
    ICL_FREE_PROMPT_REGISTRY,
    build_relevancy_prompt,
    format_relevancy_icl_example,
    format_relevancy_icl_examples,
    format_sentences_numbered,
    labels_to_score_dict,
)
from conformal_relevance.prompts.templates import ICL_HAS_INTENT

# ICL (strategy-based on-the-fly)
from conformal_relevance.icl import (
    ICLResult,
    LocalICLResult,
    select_icl_examples,
    select_local_icl_examples,
    select_and_format_icl_examples,
    select_and_format_icl_examples_by_intent,
    DEFAULT_N_ICL,
    DEFAULT_SEED,
    StrategyRegistry,
    get_local_strategy,
    is_local_strategy,
)

# Types
from conformal_relevance.types import RelevancyICLExample, RelevancyScorer
from conformal_relevance.models import DictScoringResponse

# Utilities
from conformal_relevance.utils import split_sentences

__version__ = "0.1.0"

__all__ = [
    # Main scorer
    "relevancy_dict_scorer",
    # Batch scoring pipeline
    "score_dataframe",
    "score_dataframe_no_icl",
    "score_and_evaluate",
    "compute_average_precision",
    "summarize_results",
    "ScoringResult",
    # Caching utilities
    "enable_llm_cache",
    # ICL (strategy-based on-the-fly)
    "ICLResult",
    "LocalICLResult",
    "select_icl_examples",
    "select_local_icl_examples",
    "select_and_format_icl_examples",
    "select_and_format_icl_examples_by_intent",
    "DEFAULT_N_ICL",
    "DEFAULT_SEED",
    "StrategyRegistry",
    "get_local_strategy",
    "is_local_strategy",
    # Prompt helpers and constants
    "RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT",
    "ICL_FREE_PROMPT_REGISTRY",
    "ICL_HAS_INTENT",
    "build_relevancy_prompt",
    "format_relevancy_icl_example",
    "format_relevancy_icl_examples",
    "format_sentences_numbered",
    "labels_to_score_dict",
    # Types
    "RelevancyICLExample",
    "RelevancyScorer",
    "DictScoringResponse",
    # Utility functions
    "split_sentences",
]
