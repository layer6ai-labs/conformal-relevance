"""Sentence-level ICL selection strategies based on label structure.

Provides four families of strategies that operate on per-sentence embeddings
and binary relevancy labels within each calibration example:

    Boundary   – selects examples where positive/negative sentences are hardest
                 to distinguish (high cross-class cosine similarity)
    PatternDPP – diversifies the *type* of relevancy distinction demonstrated
                 (DPP over relevancy-direction vectors)
    Clarity    – selects examples with the most coherent within-class structure
                 (high intra-class cosine similarity)
    Fisher     – selects examples with the highest signal-to-noise ratio
                 (Fisher discriminant: between-class distance² / within-class scatter)

Stratified (per-intent) variants are auto-generated via ``make_stratified()``
in ``base.py`` and registered in ``registry.py``.

Strategies:
    BoundarySelectionStrategy   – top-k by boundary difficulty
    PatternDPPSelectionStrategy – DPP over relevancy directions
    ClaritySelectionStrategy    – top-k by contrastive clarity
    FisherSelectionStrategy     – top-k by Fisher discriminant ratio
"""

from __future__ import annotations

import numpy as np
import polars as pl

from conformal_relevance.icl.strategies.base import (
    DEFAULT_SEED,
    ICLSelectionContext,
    ICLSelectionResult,
    extract_examples_from_df,
)
from conformal_relevance.icl.strategies.diversity import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingStrategy,
    _get_sentence_transformer,
    _gram_cosine_psd,
    _logdet_submatrix,
    _select_k_diverse_dpp,
)

# ---------------------------------------------------------------------------
# Sentence-level embedding computation
# ---------------------------------------------------------------------------

_DIRECTION_NORM_EPSILON = 1e-8


def _compute_sentence_embeddings(
    df: pl.DataFrame,
    context: ICLSelectionContext,
    embedding_model: str,
) -> list[np.ndarray]:
    """Compute sentence-transformer embeddings for individual sentences in each row.

    Returns a list of length ``df.shape[0]``, where element ``i`` is an ndarray
    of shape ``(n_sentences_i, embedding_dim)`` — the embeddings of the individual
    sentences in calibration example ``i``.

    Uses the same SentenceTransformer model as ``_compute_embeddings()`` to
    ensure embedding-space alignment.
    """
    # Flatten all sentences across all rows for batch encoding
    all_sentences: list[str] = []
    row_lengths: list[int] = []
    for row in df.iter_rows(named=True):
        sentences = row[context.sentences_col]
        all_sentences.extend(sentences)
        row_lengths.append(len(sentences))

    model = _get_sentence_transformer(embedding_model)
    all_embeddings: np.ndarray = model.encode(
        all_sentences,
        batch_size=64,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Split back into per-example arrays
    result: list[np.ndarray] = []
    offset = 0
    for length in row_lengths:
        result.append(all_embeddings[offset : offset + length])
        offset += length
    return result


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def _boundary_difficulty(
    sentence_embeddings: np.ndarray,  # (n_sentences, d)
    labels: list[int],
) -> float:
    """Mean cosine similarity between positive and negative sentence embeddings.

    High score = hard boundary (positives and negatives are semantically close).
    Low score = easy boundary (positives and negatives are obviously different).
    """
    pos_mask = np.array(labels, dtype=bool)
    neg_mask = ~pos_mask

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return 0.0  # Degenerate case: all same label

    pos_emb = sentence_embeddings[pos_mask]
    neg_emb = sentence_embeddings[neg_mask]

    # L2 normalize
    pos_emb = pos_emb / np.maximum(np.linalg.norm(pos_emb, axis=1, keepdims=True), 1e-12)
    neg_emb = neg_emb / np.maximum(np.linalg.norm(neg_emb, axis=1, keepdims=True), 1e-12)

    # Cross-similarity matrix: (n_pos, n_neg)
    sim_matrix = pos_emb @ neg_emb.T
    return float(sim_matrix.mean())


def _relevancy_direction(
    sentence_embeddings: np.ndarray,  # (n_sentences, d)
    labels: list[int],
) -> np.ndarray:
    """Compute the relevancy direction vector for one calibration example.

    Returns the vector (centroid_positive - centroid_negative), representing
    the semantic axis that separates relevant from not-relevant sentences.

    Returns a zero vector if either class is empty (degenerate case).
    """
    pos_mask = np.array(labels, dtype=bool)
    neg_mask = ~pos_mask

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return np.zeros(sentence_embeddings.shape[1])

    c_pos = sentence_embeddings[pos_mask].mean(axis=0)
    c_neg = sentence_embeddings[neg_mask].mean(axis=0)
    return c_pos - c_neg


def _contrastive_clarity(
    sentence_embeddings: np.ndarray,  # (n_sentences, d)
    labels: list[int],
) -> float:
    """Within-class cohesion of positive and negative sentence groups.

    High clarity = each class is internally coherent (tight clusters),
                   producing a clear contrastive signal for the LLM.
    Low clarity  = sentences within each class are scattered/incoherent,
                   diluting the teaching signal.
    """
    pos_mask = np.array(labels, dtype=bool)
    neg_mask = ~pos_mask

    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        return 0.0  # Need at least 2 in each class for pairwise similarity

    pos_emb = sentence_embeddings[pos_mask]
    neg_emb = sentence_embeddings[neg_mask]

    # L2 normalize
    pos_emb = pos_emb / np.maximum(np.linalg.norm(pos_emb, axis=1, keepdims=True), 1e-12)
    neg_emb = neg_emb / np.maximum(np.linalg.norm(neg_emb, axis=1, keepdims=True), 1e-12)

    # Mean pairwise cosine similarity within each class
    pos_sim = pos_emb @ pos_emb.T
    neg_sim = neg_emb @ neg_emb.T

    # Exclude self-similarities (diagonal = 1.0) from the mean
    n_pos, n_neg = pos_emb.shape[0], neg_emb.shape[0]
    pos_coherence = (pos_sim.sum() - n_pos) / (n_pos * (n_pos - 1))
    neg_coherence = (neg_sim.sum() - n_neg) / (n_neg * (n_neg - 1))

    return float((pos_coherence + neg_coherence) / 2)


def _fisher_discriminant(
    sentence_embeddings: np.ndarray,  # (n_sentences, d)
    labels: list[int],
) -> float:
    """Fisher discriminant ratio: between-class distance² / within-class scatter.

    Measures the signal-to-noise ratio of the relevancy pattern in one calibration
    example. High score = positive and negative sentences are well-separated
    (large centroid distance) AND internally coherent (low within-class variance).

    Returns 0.0 for degenerate cases (all same label).
    """
    pos_mask = np.array(labels, dtype=bool)
    neg_mask = ~pos_mask

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return 0.0

    pos_emb = sentence_embeddings[pos_mask]
    neg_emb = sentence_embeddings[neg_mask]

    # L2 normalize (consistent with boundary and clarity)
    pos_emb = pos_emb / np.maximum(np.linalg.norm(pos_emb, axis=1, keepdims=True), 1e-12)
    neg_emb = neg_emb / np.maximum(np.linalg.norm(neg_emb, axis=1, keepdims=True), 1e-12)

    c_pos = pos_emb.mean(axis=0)
    c_neg = neg_emb.mean(axis=0)

    # Between-class: squared Euclidean distance between centroids
    between = np.sum((c_pos - c_neg) ** 2)

    # Within-class: mean squared distance from centroid (0 if class has < 2 members)
    var_pos = np.mean(np.sum((pos_emb - c_pos) ** 2, axis=1)) if pos_emb.shape[0] >= 2 else 0.0
    var_neg = np.mean(np.sum((neg_emb - c_neg) ** 2, axis=1)) if neg_emb.shape[0] >= 2 else 0.0

    return float(between / (var_pos + var_neg + 1e-8))


# ---------------------------------------------------------------------------
# Sentence-embedding strategy base
# ---------------------------------------------------------------------------


class _SentenceEmbeddingBase(EmbeddingStrategy):
    """Base for sentence-level strategies with cached sentence embeddings."""

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        lambda_val: float | None = None,
    ) -> None:
        super().__init__(seed=seed, embedding_model=embedding_model, lambda_val=lambda_val)
        # Cache maps id(df) → (df_ref, embeddings). Storing a strong reference to
        # the dataframe prevents Python from garbage-collecting it and reusing its
        # memory address for a different object, which would produce a false cache hit.
        self._sentence_emb_cache: dict[int, tuple[pl.DataFrame, list[np.ndarray]]] = {}

    def _get_sentence_embeddings(
        self, df: pl.DataFrame, context: ICLSelectionContext
    ) -> list[np.ndarray]:
        cache_id = id(df)
        if cache_id in self._sentence_emb_cache:
            cached_df, cached_embs = self._sentence_emb_cache[cache_id]
            if cached_df is df:
                return cached_embs
        embs = _compute_sentence_embeddings(df, context, self._embedding_model)
        self._sentence_emb_cache[cache_id] = (df, embs)
        return embs


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------


class BoundarySelectionStrategy(_SentenceEmbeddingBase):
    """Sentence-level boundary difficulty selection over the full calibration set.

    Selects calibration examples where positive and negative sentences are
    hardest to distinguish (highest cross-class cosine similarity), forcing
    the LLM to learn fine-grained relevancy discrimination.
    """

    _name = "boundary"

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        super().__init__(seed=seed, embedding_model=embedding_model)

    def select(
        self,
        context: ICLSelectionContext,
        n_examples: int,
        performance_history: dict[int, list[float]] | None = None,
    ) -> ICLSelectionResult:
        df = context.pool_df
        all_sentence_embs = self._get_sentence_embeddings(df, context)

        # Score each calibration example by boundary difficulty
        scores: list[float] = []
        for i, row in enumerate(df.iter_rows(named=True)):
            labels = row[context.labels_col]
            scores.append(_boundary_difficulty(all_sentence_embs[i], labels))

        # Select top-k by difficulty (highest cross-class similarity)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected = ranked[:n_examples]

        examples = extract_examples_from_df(
            df, selected, context.intent_col, context.sentences_col, context.labels_col,
        )

        return ICLSelectionResult(
            sample_indices=selected,
            examples=examples,
            strategy_name=self.name,
            metadata={"boundary_scores": {i: scores[i] for i in selected}},
        )


class PatternDPPSelectionStrategy(_SentenceEmbeddingBase):
    """DPP diversity over relevancy-direction vectors.

    Computes a relevancy direction (centroid_positive - centroid_negative) for
    each calibration example and selects a set that maximises direction
    diversity via greedy log-det DPP.  This ensures the LLM sees maximally
    diverse *types* of relevancy distinctions rather than diverse documents.
    """

    _name = "pattern_dpp"

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        lambda_val: float = 1e-6,
    ) -> None:
        super().__init__(seed=seed, embedding_model=embedding_model, lambda_val=lambda_val)

    def select(
        self,
        context: ICLSelectionContext,
        n_examples: int,
        performance_history: dict[int, list[float]] | None = None,
    ) -> ICLSelectionResult:
        if n_examples < 1:
            raise ValueError(f"n_examples must be >= 1 (got {n_examples})")

        df = context.pool_df
        all_sentence_embs = self._get_sentence_embeddings(df, context)

        # Compute direction vectors, filtering out degenerate examples
        directions: list[np.ndarray] = []
        valid_indices: list[int] = []
        for i, row in enumerate(df.iter_rows(named=True)):
            labels = row[context.labels_col]
            r_i = _relevancy_direction(all_sentence_embs[i], labels)
            if np.linalg.norm(r_i) > _DIRECTION_NORM_EPSILON:
                directions.append(r_i)
                valid_indices.append(i)

        if len(valid_indices) < n_examples:
            raise ValueError(
                f"Only {len(valid_indices)} calibration examples have non-degenerate "
                f"relevancy directions, but {n_examples} were requested."
            )

        if n_examples == 1:
            # k=1 degenerate case: pick the example with the strongest relevancy signal
            # (largest ||centroid_pos - centroid_neg||).
            direction_norms = np.array([np.linalg.norm(d) for d in directions])
            best_local = int(np.argmax(direction_norms))
            selected = [valid_indices[best_local]]

            examples = extract_examples_from_df(
                df, selected, context.intent_col, context.sentences_col, context.labels_col,
            )

            return ICLSelectionResult(
                sample_indices=selected,
                examples=examples,
                strategy_name=self.name,
                metadata={
                    "fallback": "max_direction_norm",
                    "direction_norm": float(direction_norms[best_local]),
                    "n_degenerate_excluded": df.shape[0] - len(valid_indices),
                },
            )

        R = np.stack(directions)  # (n_valid, d)

        # DPP over direction vectors — _select_k_diverse_dpp L2-normalizes
        # internally via _gram_cosine_psd, so direction magnitude is stripped
        k = min(n_examples, len(valid_indices))
        local_selected = _select_k_diverse_dpp(R, k, self._lambda_val)

        # Map DPP-local indices back to dataframe indices
        selected = [valid_indices[j] for j in local_selected]

        log_det = _logdet_submatrix(
            _gram_cosine_psd(R, ridge_eps=self._lambda_val), local_selected
        )
        examples = extract_examples_from_df(
            df, selected, context.intent_col, context.sentences_col, context.labels_col,
        )

        return ICLSelectionResult(
            sample_indices=selected,
            examples=examples,
            strategy_name=self.name,
            metadata={
                "log_det_score": float(log_det),
                "n_degenerate_excluded": df.shape[0] - len(valid_indices),
            },
        )


class ClaritySelectionStrategy(_SentenceEmbeddingBase):
    """Contrastive clarity selection over the full calibration set.

    Selects calibration examples where each class (relevant / not-relevant) is
    internally coherent, producing the clearest contrastive teaching signal
    for the LLM.
    """

    _name = "clarity"

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        super().__init__(seed=seed, embedding_model=embedding_model)

    def select(
        self,
        context: ICLSelectionContext,
        n_examples: int,
        performance_history: dict[int, list[float]] | None = None,
    ) -> ICLSelectionResult:
        df = context.pool_df
        all_sentence_embs = self._get_sentence_embeddings(df, context)

        # Score each calibration example by contrastive clarity
        scores: list[float] = []
        for i, row in enumerate(df.iter_rows(named=True)):
            labels = row[context.labels_col]
            scores.append(_contrastive_clarity(all_sentence_embs[i], labels))

        # Select top-k by clarity (highest within-class coherence)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected = ranked[:n_examples]

        examples = extract_examples_from_df(
            df, selected, context.intent_col, context.sentences_col, context.labels_col,
        )

        return ICLSelectionResult(
            sample_indices=selected,
            examples=examples,
            strategy_name=self.name,
            metadata={"clarity_scores": {i: scores[i] for i in selected}},
        )


class FisherSelectionStrategy(_SentenceEmbeddingBase):
    """Fisher discriminant selection over the full calibration set.

    Selects calibration examples with the highest signal-to-noise ratio:
    strong separation between positive and negative sentence centroids
    relative to within-class scatter. Combines the discriminative aspects
    of boundary (separation) and clarity (coherence) into a single score.
    """

    _name = "fisher"

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        super().__init__(seed=seed, embedding_model=embedding_model)

    def select(
        self,
        context: ICLSelectionContext,
        n_examples: int,
        performance_history: dict[int, list[float]] | None = None,
    ) -> ICLSelectionResult:
        df = context.pool_df
        all_sentence_embs = self._get_sentence_embeddings(df, context)

        # Score each calibration example by Fisher discriminant ratio
        scores: list[float] = []
        for i, row in enumerate(df.iter_rows(named=True)):
            labels = row[context.labels_col]
            scores.append(_fisher_discriminant(all_sentence_embs[i], labels))

        # Select top-k by Fisher score (highest signal-to-noise first)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected = ranked[:n_examples]

        examples = extract_examples_from_df(
            df, selected, context.intent_col, context.sentences_col, context.labels_col,
        )

        return ICLSelectionResult(
            sample_indices=selected,
            examples=examples,
            strategy_name=self.name,
            metadata={"fisher_scores": {i: scores[i] for i in selected}},
        )


