"""Strategy registry for ICL selection strategies."""

from typing import Any, Type

from conformal_relevance.icl.strategies.base import (
    DEFAULT_SEED,
    ICLSelectionStrategy,
    make_stratified,
)
from conformal_relevance.icl.strategies.local import (
    BM25LocalStrategy,
    LocalICLStrategy,
    KNNLocalStrategy,
)
from conformal_relevance.icl.strategies.random import RandomSamplingStrategy
from conformal_relevance.icl.strategies.diversity import (
    AnchoredDiversitySelectionStrategy,
    DiversitySelectionStrategy,
    RepresentativeSelectionStrategy,
)
from conformal_relevance.icl.strategies.sentence_selection import (
    BoundarySelectionStrategy,
    ClaritySelectionStrategy,
    FisherSelectionStrategy,
    PatternDPPSelectionStrategy,
)
from conformal_relevance.icl.strategies.most_positive import MostPositiveStrategy

# Auto-generate stratified variants
StratifiedRandomSamplingStrategy = make_stratified(RandomSamplingStrategy, "random_stratified")
StratifiedDiversitySelectionStrategy = make_stratified(DiversitySelectionStrategy, "dpp_stratified")
StratifiedRepresentativeSelectionStrategy = make_stratified(RepresentativeSelectionStrategy, "rep_stratified")
StratifiedAnchoredDiversitySelectionStrategy = make_stratified(AnchoredDiversitySelectionStrategy, "anchor_dpp_stratified")
StratifiedBoundarySelectionStrategy = make_stratified(BoundarySelectionStrategy, "boundary_stratified")
StratifiedPatternDPPSelectionStrategy = make_stratified(PatternDPPSelectionStrategy, "pattern_dpp_stratified")
StratifiedClaritySelectionStrategy = make_stratified(ClaritySelectionStrategy, "clarity_stratified")
StratifiedFisherSelectionStrategy = make_stratified(FisherSelectionStrategy, "fisher_stratified")
StratifiedMostPositiveStrategy = make_stratified(MostPositiveStrategy, "most_positive_stratified")


class StrategyRegistry:
    """Registry for ICL selection strategies."""

    _strategies: dict[str, Type] = {
        "random": RandomSamplingStrategy,
        "random_stratified": StratifiedRandomSamplingStrategy,
        "dpp": DiversitySelectionStrategy,
        "dpp_stratified": StratifiedDiversitySelectionStrategy,
        "rep": RepresentativeSelectionStrategy,
        "rep_stratified": StratifiedRepresentativeSelectionStrategy,
        "anchor_dpp": AnchoredDiversitySelectionStrategy,
        "anchor_dpp_stratified": StratifiedAnchoredDiversitySelectionStrategy,
        "boundary": BoundarySelectionStrategy,
        "boundary_stratified": StratifiedBoundarySelectionStrategy,
        "pattern_dpp": PatternDPPSelectionStrategy,
        "pattern_dpp_stratified": StratifiedPatternDPPSelectionStrategy,
        "clarity": ClaritySelectionStrategy,
        "clarity_stratified": StratifiedClaritySelectionStrategy,
        "fisher": FisherSelectionStrategy,
        "fisher_stratified": StratifiedFisherSelectionStrategy,
        "most_positive": MostPositiveStrategy,
        "most_positive_stratified": StratifiedMostPositiveStrategy,
    }

    @classmethod
    def get(cls, name: str, **kwargs) -> ICLSelectionStrategy:
        """Get a strategy by name.

        Use resolve_strategy_name() first to auto-route base names
        to stratified variants for multi-intent datasets.
        """
        if name not in cls._strategies:
            available = cls.list_strategies(include_stratified=True)
            raise ValueError(f"Unknown strategy: {name}. Available: {available}")

        kwargs.setdefault("seed", DEFAULT_SEED)

        strategy_class = cls._strategies[name]
        return strategy_class(**kwargs)

    @classmethod
    def resolve_strategy_name(cls, name: str, multi_intent: bool = False) -> str:
        """Resolve a user-facing strategy name to its registry key.

        For multi-intent datasets, auto-routes base names to their
        stratified variants (e.g., 'random' -> 'random_stratified').
        """
        if multi_intent and not name.endswith("_stratified"):
            stratified = f"{name}_stratified"
            if stratified in cls._strategies:
                return stratified
        return name

    @classmethod
    def list_strategies(cls, include_stratified: bool = False) -> list[str]:
        """List available strategy names.

        Args:
            include_stratified: If True, include *_stratified variants.
                Defaults to False (user-facing base names only).
        """
        if include_stratified:
            return list(cls._strategies.keys())
        return [name for name in cls._strategies if not name.endswith("_stratified")]

    @classmethod
    def register(cls, name: str, strategy_class: Type) -> None:
        cls._strategies[name] = strategy_class

    @classmethod
    def get_strategy_info(cls, name: str) -> dict[str, Any]:
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        strategy_class = cls._strategies[name]
        info = {"name": name, "class": strategy_class.__name__, "module": strategy_class.__module__}
        if strategy_class.__doc__:
            info["description"] = strategy_class.__doc__.split("\n")[0]
        return info


# ---------------------------------------------------------------------------
# Local strategy registry
# ---------------------------------------------------------------------------


LOCAL_STRATEGY_REGISTRY: dict[str, type] = {
    "bm25": BM25LocalStrategy,
    "knn": KNNLocalStrategy,
}


def get_local_strategy(name: str, **kwargs) -> LocalICLStrategy:
    """Get a local ICL strategy instance by name."""
    if name not in LOCAL_STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown local strategy: {name!r}. "
            f"Available: {list(LOCAL_STRATEGY_REGISTRY.keys())}"
        )
    return LOCAL_STRATEGY_REGISTRY[name](**kwargs)


def is_local_strategy(name: str) -> bool:
    """Check if a strategy name is a local (per-sample) strategy."""
    return name in LOCAL_STRATEGY_REGISTRY
