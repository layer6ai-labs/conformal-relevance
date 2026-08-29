"""Prompt templates and formatting utilities for relevancy scoring."""

from conformal_relevance.prompts.templates import (
    RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT,
    RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT_NO_INTENT,
    RELEVANCY_MISSING_SCORES_PROMPT,
    RELEVANCY_MISSING_SCORES_PROMPT_NO_INTENT,
)
from conformal_relevance.prompts.formatting import (
    build_relevancy_prompt,
    format_relevancy_icl_example,
    format_relevancy_icl_examples,
    format_score_dict,
    format_sentences_numbered,
    labels_to_score_dict,
)
from conformal_relevance.prompts.icl_free import ICL_FREE_PROMPT_REGISTRY

__all__ = [
    # Templates
    "RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT",
    "RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT_NO_INTENT",
    "RELEVANCY_MISSING_SCORES_PROMPT",
    "RELEVANCY_MISSING_SCORES_PROMPT_NO_INTENT",
    # Formatting functions
    "build_relevancy_prompt",
    "format_relevancy_icl_example",
    "format_relevancy_icl_examples",
    "format_score_dict",
    "format_sentences_numbered",
    "labels_to_score_dict",
    # ICL-free registry
    "ICL_FREE_PROMPT_REGISTRY",
]
