"""Base types, protocol, and shared infrastructure for ICL selection strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import polars as pl

from conformal_relevance.types import (
    DEFAULT_INTENT,
    DEFAULT_N_ICL,
    DEFAULT_SEED,
    INPUT_UNITS_COLUMN,
    INPUT_UNIT_LABELS_COLUMN,
    INTENT_COLUMN,
)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ICLSelectionContext:
    """Context for ICL example selection."""

    pool_df: pl.DataFrame
    target_intent: str | None = None
    n_examples: int = DEFAULT_N_ICL
    seed: int = DEFAULT_SEED
    intent_col: str = INTENT_COLUMN
    sentences_col: str = INPUT_UNITS_COLUMN
    labels_col: str = INPUT_UNIT_LABELS_COLUMN


@dataclass
class ICLSelectionResult:
    """Result of ICL example selection."""

    sample_indices: list[int]
    examples: list[dict]
    strategy_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return len(self.examples)


@dataclass
class StratifiedICLSelectionResult:
    """Result of per-intent ICL example selection for multi-intent datasets.

    Each key in ``by_intent`` maps an intent value to an ``ICLSelectionResult``
    containing the examples selected specifically for that intent.
    """

    by_intent: dict[str, ICLSelectionResult]
    strategy_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def intents(self) -> list[str]:
        return list(self.by_intent.keys())

    @property
    def n_intents(self) -> int:
        return len(self.by_intent)

    @property
    def n_examples_per_intent(self) -> dict[str, int]:
        return {k: v.n_examples for k, v in self.by_intent.items()}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ICLSelectionStrategy(Protocol):
    """Protocol for ICL selection strategies."""

    @property
    def name(self) -> str: ...

    @property
    def requires_performance_history(self) -> bool: ...

    def select(
        self,
        context: ICLSelectionContext,
        n_examples: int,
        performance_history: dict[int, list[float]] | None = None,
    ) -> ICLSelectionResult: ...

    def get_config(self) -> dict: ...


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def extract_examples_from_df(
    df: pl.DataFrame,
    indices: Sequence[int],
    intent_col: str = INTENT_COLUMN,
    sentences_col: str = INPUT_UNITS_COLUMN,
    labels_col: str = INPUT_UNIT_LABELS_COLUMN,
) -> list[dict]:
    """Extract ICL example dictionaries from dataframe rows."""
    examples = []
    for row in df[indices].iter_rows(named=True):
        examples.append({
            "intent": row.get(intent_col, DEFAULT_INTENT),
            "input_units": row[sentences_col],
            "input_unit_labels": row[labels_col],
        })
    return examples


def _group_by_intent(
    df: pl.DataFrame,
    intent_col: str,
) -> dict[str, list[int]]:
    """Return ``{intent_value: [row_indices]}`` mapping."""
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(df.iter_rows(named=True)):
        groups.setdefault(row[intent_col], []).append(i)
    return groups


# ---------------------------------------------------------------------------
# Base strategy class
# ---------------------------------------------------------------------------


class BaseStrategy:
    """Shared base for all ICL selection strategies.

    Provides default implementations of ``name``, ``requires_performance_history``,
    and ``get_config()`` so that concrete subclasses only need to define
    ``_name`` and ``select()``.
    """

    _name: str

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_performance_history(self) -> bool:
        return False

    def get_config(self) -> dict:
        return {"name": self.name, "seed": self._seed}


# ---------------------------------------------------------------------------
# Generic stratified wrapper factory
# ---------------------------------------------------------------------------


def make_stratified(
    base_class: type,
    stratified_name: str,
) -> type:
    """Auto-generate a stratified variant of any global strategy class.

    The returned class:
    1. Accepts the same constructor args as *base_class*
    2. Validates intent_col is present in the pool dataframe
    3. Groups rows by intent via ``_group_by_intent()``
    4. For each intent group, builds a sub-context and delegates to the
       base class's ``select()`` method
    5. Maps local (sub-dataframe) indices back to global (original df) indices
    6. Returns a ``StratifiedICLSelectionResult``
    """

    class _StratifiedStrategy:
        __doc__ = f"Per-intent stratified variant of {base_class.__name__}."

        def __init__(self, **kwargs) -> None:
            self._delegate = base_class(**kwargs)
            self._stratified_name = stratified_name

        @property
        def name(self) -> str:
            return self._stratified_name

        @property
        def requires_performance_history(self) -> bool:
            return self._delegate.requires_performance_history

        def get_config(self) -> dict:
            config = self._delegate.get_config()
            config["name"] = self._stratified_name
            return config

        def select(
            self,
            context: ICLSelectionContext,
            n_examples: int,
            performance_history: dict[int, list[float]] | None = None,
        ) -> StratifiedICLSelectionResult:
            df = context.pool_df
            intent_col = context.intent_col

            if intent_col not in df.columns:
                raise ValueError(
                    f"Stratified strategy '{self._stratified_name}' requires an "
                    f"intent column '{intent_col}' in the pool data."
                )

            groups = _group_by_intent(df, intent_col)
            by_intent: dict[str, ICLSelectionResult] = {}

            for intent_val, row_indices in groups.items():
                subdf = df[row_indices]
                sub_context = ICLSelectionContext(
                    pool_df=subdf,
                    target_intent=intent_val,
                    n_examples=n_examples,
                    seed=context.seed,
                    intent_col=intent_col,
                    sentences_col=context.sentences_col,
                    labels_col=context.labels_col,
                )

                k = min(n_examples, len(row_indices))
                sub_result = self._delegate.select(
                    sub_context, n_examples=k, performance_history=performance_history,
                )

                # Map sub-dataframe-local indices back to original df indices
                global_indices = [row_indices[i] for i in sub_result.sample_indices]
                examples = extract_examples_from_df(
                    df, global_indices, intent_col,
                    context.sentences_col, context.labels_col,
                )

                by_intent[intent_val] = ICLSelectionResult(
                    sample_indices=global_indices,
                    examples=examples,
                    strategy_name=self._stratified_name,
                    metadata=sub_result.metadata,
                )

            return StratifiedICLSelectionResult(
                by_intent=by_intent,
                strategy_name=self._stratified_name,
                metadata={"n_intents": len(by_intent)},
            )

    _StratifiedStrategy.__name__ = f"Stratified{base_class.__name__}"
    _StratifiedStrategy.__qualname__ = f"Stratified{base_class.__name__}"
    return _StratifiedStrategy
