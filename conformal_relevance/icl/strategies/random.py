"""Random sampling strategy for ICL example selection."""

import random

from conformal_relevance.icl.strategies.base import (
    BaseStrategy,
    ICLSelectionContext,
    ICLSelectionResult,
    extract_examples_from_df,
)


class RandomSamplingStrategy(BaseStrategy):
    """Uniform random sampling strategy for ICL example selection."""

    _name = "random"

    def select(
        self,
        context: ICLSelectionContext,
        n_examples: int,
        performance_history: dict[int, list[float]] | None = None,
    ) -> ICLSelectionResult:
        df = context.pool_df
        n_samples = df.shape[0]

        if n_examples > n_samples:
            raise ValueError(
                f"Requested {n_examples} examples but only {n_samples} available"
            )

        rng = random.Random(self._seed)
        indices = rng.sample(range(n_samples), n_examples)

        examples = extract_examples_from_df(
            df, indices, context.intent_col, context.sentences_col, context.labels_col,
        )

        return ICLSelectionResult(
            sample_indices=indices,
            examples=examples,
            strategy_name=self.name,
            metadata={"seed": self._seed},
        )
