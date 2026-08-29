"""Most-positive labels strategy for ICL example selection."""

import random

from conformal_relevance.icl.strategies.base import (
    BaseStrategy,
    ICLSelectionContext,
    ICLSelectionResult,
    extract_examples_from_df,
)


class MostPositiveStrategy(BaseStrategy):
    """Select ICL examples with the highest positive label count.

    Embedding-free. Selects the k calibration examples whose ``input_unit_labels``
    list contains the most ``True`` values. Ties are broken by a seeded shuffle
    so results are reproducible but not biased toward low row indices.

    Targets datasets where the positive class is semantically heterogeneous and
    sparse (e.g., PhysioNet PHI detection) — examples with many positives provide
    stronger task-inference signal for the LLM than sparse-positive examples.

    See: PlanDocs/physionet_random_dominance_investigation.md § Proposal 1
    """

    _name = "most_positive"

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

        pos_counts = [
            sum(row[context.labels_col])
            for row in df.iter_rows(named=True)
        ]

        # Seeded shuffle for tie-breaking, then stable descending sort
        rng = random.Random(self._seed)
        indexed = list(enumerate(pos_counts))
        rng.shuffle(indexed)
        indexed.sort(key=lambda x: x[1], reverse=True)

        indices = [idx for idx, _ in indexed[:n_examples]]
        selected_counts = [pos_counts[i] for i in indices]

        examples = extract_examples_from_df(
            df, indices, context.intent_col, context.sentences_col, context.labels_col,
        )

        return ICLSelectionResult(
            sample_indices=indices,
            examples=examples,
            strategy_name=self.name,
            metadata={"positive_counts": selected_counts, "seed": self._seed},
        )
