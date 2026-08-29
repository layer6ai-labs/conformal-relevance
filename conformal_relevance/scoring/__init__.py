"""Scoring module for relevancy-based sentence scoring."""

from conformal_relevance.scoring.cache import enable_llm_cache
from conformal_relevance.scoring.pipeline import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    ScoringResult,
    compute_average_precision,
    score_and_evaluate,
    score_dataframe,
    score_dataframe_no_icl,
    summarize_results,
)
from conformal_relevance.scoring.profiling import (
    ProfileCollector,
    RunProfile,
    SampleProfile,
)
from conformal_relevance.scoring.fusion import score_dataframe_moe
from conformal_relevance.scoring.scorers import relevancy_dict_scorer

__all__ = [
    "score_dataframe",
    "score_dataframe_moe",
    "score_dataframe_no_icl",
    "score_and_evaluate",
    "compute_average_precision",
    "summarize_results",
    "ScoringResult",
    "relevancy_dict_scorer",
    "enable_llm_cache",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_RETRIES",
    "ProfileCollector",
    "RunProfile",
    "SampleProfile",
]
