"""Local (per-sample) ICL selection strategies.

Unlike global strategies that select one fixed ICL block for all samples,
local strategies select ICL examples conditioned on each inference sample
individually using retrieval methods.

Strategies:
    BM25LocalStrategy   – lexical retrieval via BM25 (rank-bm25)
    KNNLocalStrategy    – embedding kNN via sentence-transformers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


@dataclass
class LocalICLContext:
    """Context for local (per-sample) ICL selection."""

    pool_df: pl.DataFrame
    n_examples: int
    seed: int
    has_intent: bool
    dataset_task: str
    intent_col: str = "intent"
    sentences_col: str = "input_units"
    labels_col: str = "input_unit_labels"
    input_col: str = "input"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LocalICLStrategy(Protocol):
    """Protocol for local (per-sample) ICL selection strategies."""

    @property
    def name(self) -> str: ...

    def prepare(self, context: LocalICLContext) -> None: ...

    def select_for_sample(self, sample_text: str, intent: str = "") -> list[int]: ...

    def select_for_batch(
        self, sample_texts: list[str], intents: list[str] | None = None
    ) -> list[list[int]]: ...


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class LocalBaseStrategy:
    """Shared base for local ICL strategies (mirrors BaseStrategy pattern).

    Provides default ``name``, ``prepare`` (no-op), and ``select_for_batch``
    (loops over ``select_for_sample``). Concrete subclasses must set ``_name``
    and implement ``select_for_sample``. Override ``prepare`` and
    ``select_for_batch`` as needed.
    """

    _name: str

    @property
    def name(self) -> str:
        return self._name

    def prepare(self, context: LocalICLContext) -> None:
        pass  # Subclasses override to build index

    def _prepare_common(self, context: LocalICLContext) -> None:
        """Unpack shared context fields and build intent-to-indices mapping."""
        self._n_examples = context.n_examples
        self._pool_df = context.pool_df
        self._has_intent = context.has_intent
        self._dataset_task = context.dataset_task
        self._intent_col = context.intent_col
        self._input_col = context.input_col

        pool_df = context.pool_df
        has_intent_col = context.intent_col in pool_df.columns
        if context.has_intent and has_intent_col:
            self._intent_to_indices = _build_intent_groups(pool_df, context.intent_col)
        else:
            self._intent_to_indices = {"": list(range(pool_df.shape[0]))}

    def _resolve_intent(self, intent: str, retrieval_dict: dict) -> tuple:
        """Return (retrieval_value, pool_indices) for an intent, falling back to full pool."""
        if intent in retrieval_dict:
            return retrieval_dict[intent], self._intent_to_indices[intent]
        val = retrieval_dict.get("", next(iter(retrieval_dict.values())))
        pool_indices = self._intent_to_indices.get("", next(iter(self._intent_to_indices.values())))
        return val, pool_indices

    def select_for_sample(self, sample_text: str, intent: str = "") -> list[int]:
        raise NotImplementedError

    def select_for_batch(
        self, sample_texts: list[str], intents: list[str] | None = None
    ) -> list[list[int]]:
        """Default: loop over select_for_sample(). KNNLocalStrategy overrides for batch encoding."""
        intents = intents or [""] * len(sample_texts)
        return [self.select_for_sample(t, i) for t, i in zip(sample_texts, intents)]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _get_input_text(row: dict, input_col: str, sentences_col: str = "input_units") -> str:
    """Return the input text for a pool row, falling back to joining input_units.

    HotpotQA omits the ``input`` column — the document is only stored as
    ``input_units`` (sentence list). Joining them reconstructs the full text.
    """
    if input_col in row:
        return row[input_col]
    units = row.get(sentences_col, [])
    return " ".join(units) if units else ""


def _build_intent_groups(pool_df: pl.DataFrame, intent_col: str) -> dict[str, list[int]]:
    """Map intent value → list of pool row indices (shared by BM25 and KNN prepare())."""
    intent_to_indices: dict[str, list[int]] = {}
    for i, row in enumerate(pool_df.iter_rows(named=True)):
        intent_val = row.get(intent_col, "")
        intent_to_indices.setdefault(intent_val, []).append(i)
    return intent_to_indices


def _build_retrieval_text(
    input_text: str, intent: str, dataset_task: str, has_intent: bool
) -> str:
    """Build prompt-mirror text for retrieval (query or pool target).

    Format:
        Task: {dataset_task}       <- omit line entirely when dataset_task is ""
        Intent: {intent}           <- omit line entirely when has_intent is False
        {input_text}

    Used by both ``prepare()`` (pool text construction) and the bridge function
    (query text construction). Keeping both in sync ensures the retrieval space
    matches the scoring context.
    """
    parts = []
    if dataset_task:
        parts.append(f"Task: {dataset_task}")
    if has_intent and intent:
        parts.append(f"Intent: {intent}")
    parts.append(input_text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# BM25 strategy
# ---------------------------------------------------------------------------


class BM25LocalStrategy(LocalBaseStrategy):
    """Select ICL examples by BM25 lexical relevance to each inference sample."""

    _name = "bm25"

    def prepare(self, context: LocalICLContext) -> None:
        from rank_bm25 import BM25Okapi

        self._prepare_common(context)
        pool_df = self._pool_df

        self._intent_to_bm25: dict[str, BM25Okapi] = {}
        for intent_val, indices in self._intent_to_indices.items():
            texts = [
                _build_retrieval_text(
                    _get_input_text(row, context.input_col, context.sentences_col),
                    row.get(context.intent_col, ""), context.dataset_task, context.has_intent,
                ).lower().split()
                for row in pool_df[indices].iter_rows(named=True)
            ]
            self._intent_to_bm25[intent_val] = BM25Okapi(texts)

    def select_for_sample(self, sample_text: str, intent: str = "") -> list[int]:
        bm25, pool_indices = self._resolve_intent(intent, self._intent_to_bm25)
        query_tokens = sample_text.lower().split()
        scores = bm25.get_scores(query_tokens)

        k = min(self._n_examples, len(pool_indices))
        # Get top-k by score (descending)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [pool_indices[i] for i in ranked[:k]]


# ---------------------------------------------------------------------------
# KNN strategy (embedding kNN)
# ---------------------------------------------------------------------------


class KNNLocalStrategy(LocalBaseStrategy):
    """Select ICL examples by embedding-space kNN to each inference sample."""

    _name = "knn"

    def prepare(self, context: LocalICLContext) -> None:
        from conformal_relevance.icl.strategies.diversity import _get_sentence_transformer

        self._prepare_common(context)
        self._model = _get_sentence_transformer()
        pool_df = self._pool_df

        self._intent_to_matrix: dict[str, np.ndarray] = {}
        for intent_val, indices in self._intent_to_indices.items():
            texts = [
                _build_retrieval_text(
                    _get_input_text(row, context.input_col, context.sentences_col),
                    row.get(context.intent_col, ""), context.dataset_task, context.has_intent,
                )
                for row in pool_df[indices].iter_rows(named=True)
            ]
            embs = self._model.encode(texts, normalize_embeddings=True)
            self._intent_to_matrix[intent_val] = embs

    def select_for_sample(self, sample_text: str, intent: str = "") -> list[int]:
        query_emb = self._model.encode([sample_text], normalize_embeddings=True)[0]
        pool_matrix, pool_indices = self._resolve_intent(intent, self._intent_to_matrix)
        similarities = pool_matrix @ query_emb
        k = min(self._n_examples, len(pool_indices))
        top_k_local = np.argpartition(similarities, -k)[-k:]
        top_k_local = top_k_local[np.argsort(similarities[top_k_local])[::-1]]
        return [pool_indices[i] for i in top_k_local]

    def select_for_batch(
        self, sample_texts: list[str], intents: list[str] | None = None
    ) -> list[list[int]]:
        """Batch-encode all inference samples in one forward pass."""
        intents = intents or [""] * len(sample_texts)

        # Batch encode all query texts
        query_matrix = self._model.encode(sample_texts, normalize_embeddings=True)  # (N, d)

        results: list[list[int]] = []
        for i, (query_emb, intent) in enumerate(zip(query_matrix, intents)):
            pool_matrix, pool_indices = self._resolve_intent(intent, self._intent_to_matrix)
            similarities = pool_matrix @ query_emb
            k = min(self._n_examples, len(pool_indices))
            top_k_local = np.argpartition(similarities, -k)[-k:]
            top_k_local = top_k_local[np.argsort(similarities[top_k_local])[::-1]]
            results.append([pool_indices[j] for j in top_k_local])

        return results
