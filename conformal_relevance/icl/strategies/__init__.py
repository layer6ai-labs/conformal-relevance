"""ICL selection strategies for robustness testing."""

from conformal_relevance.icl.strategies.base import (
    DEFAULT_N_ICL,
    DEFAULT_SEED,
    BaseStrategy,
    ICLSelectionContext,
    ICLSelectionResult,
    ICLSelectionStrategy,
    StratifiedICLSelectionResult,
    extract_examples_from_df,
    make_stratified,
)
from conformal_relevance.icl.strategies.random import RandomSamplingStrategy
from conformal_relevance.icl.strategies.most_positive import MostPositiveStrategy
from conformal_relevance.icl.strategies.diversity import (
    AnchoredDiversitySelectionStrategy,
    DiversitySelectionStrategy,
    EmbeddingStrategy,
    RepresentativeSelectionStrategy,
)
from conformal_relevance.icl.strategies.sentence_selection import (
    BoundarySelectionStrategy,
    ClaritySelectionStrategy,
    FisherSelectionStrategy,
    PatternDPPSelectionStrategy,
)
from conformal_relevance.icl.strategies.registry import (
    StrategyRegistry,
    # Auto-generated stratified variants (backward-compatible names)
    StratifiedMostPositiveStrategy,
    StratifiedRandomSamplingStrategy,
    StratifiedDiversitySelectionStrategy,
    StratifiedRepresentativeSelectionStrategy,
    StratifiedAnchoredDiversitySelectionStrategy,
    StratifiedBoundarySelectionStrategy,
    StratifiedPatternDPPSelectionStrategy,
    StratifiedClaritySelectionStrategy,
    StratifiedFisherSelectionStrategy,
    # Local strategy registry
    LOCAL_STRATEGY_REGISTRY,
    get_local_strategy,
    is_local_strategy,
)
from conformal_relevance.icl.strategies.local import (
    BM25LocalStrategy,
    LocalBaseStrategy,
    LocalICLContext,
    LocalICLStrategy,
    KNNLocalStrategy,
)

__all__ = [
    # Base types & utilities
    "DEFAULT_N_ICL",
    "DEFAULT_SEED",
    "BaseStrategy",
    "EmbeddingStrategy",
    "ICLSelectionStrategy",
    "ICLSelectionContext",
    "ICLSelectionResult",
    "StratifiedICLSelectionResult",
    "extract_examples_from_df",
    "make_stratified",
    # Global strategies
    "MostPositiveStrategy",
    "RandomSamplingStrategy",
    "DiversitySelectionStrategy",
    "RepresentativeSelectionStrategy",
    "AnchoredDiversitySelectionStrategy",
    "BoundarySelectionStrategy",
    "PatternDPPSelectionStrategy",
    "ClaritySelectionStrategy",
    "FisherSelectionStrategy",
    # Auto-generated stratified variants
    "StratifiedMostPositiveStrategy",
    "StratifiedRandomSamplingStrategy",
    "StratifiedDiversitySelectionStrategy",
    "StratifiedRepresentativeSelectionStrategy",
    "StratifiedAnchoredDiversitySelectionStrategy",
    "StratifiedBoundarySelectionStrategy",
    "StratifiedPatternDPPSelectionStrategy",
    "StratifiedClaritySelectionStrategy",
    "StratifiedFisherSelectionStrategy",
    # Registry
    "StrategyRegistry",
    # Local strategies
    "LOCAL_STRATEGY_REGISTRY",
    "get_local_strategy",
    "is_local_strategy",
    "BM25LocalStrategy",
    "KNNLocalStrategy",
    "LocalBaseStrategy",
    "LocalICLContext",
    "LocalICLStrategy",
]
