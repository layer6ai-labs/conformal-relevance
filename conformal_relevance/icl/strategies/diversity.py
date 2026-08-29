"""Embedding-based selection strategies for ICL examples.

Provides DPP (Determinantal Point Process) diversity, centroid-based
representative, and anchored-diversity selection strategies.

Strategies:
    DiversitySelectionStrategy          – greedy k-DPP over full calibration set
    RepresentativeSelectionStrategy     – centroid selection over full set
    AnchoredDiversitySelectionStrategy  – centroid anchor + conditional DPP

Stratified (per-intent) variants are auto-generated via ``make_stratified()``
in ``base.py`` and registered in ``registry.py``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from conformal_relevance.icl.strategies.base import (
    DEFAULT_SEED,
    BaseStrategy,
    ICLSelectionContext,
    ICLSelectionResult,
    extract_examples_from_df,
)
from conformal_relevance.prompts.formatting import format_relevancy_icl_example
from conformal_relevance.types import DEFAULT_INTENT

# ---------------------------------------------------------------------------
# Lazy import machinery for sentence-transformers
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SENTENCE_TRANSFORMERS_AVAILABLE: bool | None = None  # None = not yet checked


def _check_sentence_transformers() -> bool:
    """Check if sentence-transformers is available (lazy check)."""
    global SENTENCE_TRANSFORMERS_AVAILABLE
    if SENTENCE_TRANSFORMERS_AVAILABLE is None:
        try:
            import sentence_transformers  # noqa: F401

            SENTENCE_TRANSFORMERS_AVAILABLE = True
        except ImportError:
            SENTENCE_TRANSFORMERS_AVAILABLE = False
    return SENTENCE_TRANSFORMERS_AVAILABLE


_SENTENCE_TRANSFORMER_CACHE: dict[str, object] = {}


def _get_sentence_transformer(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Lazy import and instantiate SentenceTransformer (cached per model name)."""
    if model_name not in _SENTENCE_TRANSFORMER_CACHE:
        from sentence_transformers import SentenceTransformer

        _SENTENCE_TRANSFORMER_CACHE[model_name] = SentenceTransformer(model_name)
    return _SENTENCE_TRANSFORMER_CACHE[model_name]


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------


def _compute_embeddings(
    df: pl.DataFrame,
    context: ICLSelectionContext,
    embedding_model: str,
) -> np.ndarray:
    """Compute sentence-transformer embeddings for each calibration row.

    For every row the text is constructed via
    ``format_relevancy_icl_example(intent, sentences, labels, random_seed=0)``
    so that the embedding space is aligned with what the LLM actually sees at
    scoring time.  A fixed ``random_seed=0`` is used for negative
    sub-sampling to keep embeddings deterministic for the same calibration
    data.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape ``(n_rows, embedding_dim)``.
    """
    has_intent = context.intent_col in df.columns

    texts: list[str] = []
    for row in df.iter_rows(named=True):
        intent = row.get(context.intent_col, DEFAULT_INTENT) if has_intent else DEFAULT_INTENT
        sentences = row[context.sentences_col]
        labels = row[context.labels_col]
        text = format_relevancy_icl_example(
            intent=intent,
            sentences=sentences,
            labels=labels,
            random_seed=0,
        )
        texts.append(text)

    model = _get_sentence_transformer(embedding_model)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings


# ---------------------------------------------------------------------------
# DPP math helpers
# ---------------------------------------------------------------------------


def _gram_cosine_psd(
    embeddings: np.ndarray,
    ridge_eps: float = 1e-6,
) -> np.ndarray:
    """Build a PSD Gram matrix: ``L = E_norm @ E_norm.T + ridge * I``.

    Parameters
    ----------
    embeddings : np.ndarray
        2-D array of shape ``(n, d)``.
    ridge_eps : float
        Ridge regularization added to the diagonal.

    Returns
    -------
    np.ndarray
        PSD Gram matrix of shape ``(n, n)``.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array of shape (n, d).")
    n = embeddings.shape[0]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # clamp to avoid division by zero
    E = embeddings / norms

    L = E @ E.T + ridge_eps * np.eye(n)
    return L


def _logdet_submatrix(L: np.ndarray, indices: list[int]) -> float:
    """Log-determinant of the principal submatrix ``L[indices, indices]``.

    Returns ``0.0`` for an empty index list and ``-inf`` when the submatrix
    is not positive-definite.
    """
    if len(indices) == 0:
        return 0.0
    M = L[np.ix_(indices, indices)]
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return -np.inf
    return float(logdet)


# ---------------------------------------------------------------------------
# Selection algorithms
# ---------------------------------------------------------------------------


def _select_k_diverse_dpp(
    embeddings: np.ndarray,
    k: int,
    lambda_val: float = 1e-6,
    initial_indices: list[int] | None = None,
) -> list[int]:
    """Greedy k-DPP MAP approximation via log-det maximization.

    Parameters
    ----------
    embeddings : np.ndarray
        ``(n, d)`` embedding matrix.
    k : int
        Total number of items to select (including any initial indices).
    lambda_val : float
        Ridge regularization.
    initial_indices : list[int] | None
        Optional pre-selected indices to include in the result.  The
        remaining ``k - len(initial_indices)`` items are greedily
        selected to maximize the log-det conditioned on them.
        Pass ``None`` or ``[]`` for standard (unconditional) DPP.

    Returns
    -------
    list[int]
        Selected indices in selection order (initial indices first,
        then greedy picks).
    """
    n = embeddings.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, {n}], got {k}")

    L = _gram_cosine_psd(embeddings, ridge_eps=lambda_val)

    selected: list[int] = list(initial_indices) if initial_indices else []
    remaining = set(range(n)) - set(selected)
    current_logdet = _logdet_submatrix(L, selected) if selected else 0.0

    if len(selected) >= k:
        return selected[:k]

    for _ in range(k - len(selected)):
        best_i: int | None = None
        best_gain = -np.inf
        for i in remaining:
            trial = selected + [i]
            gain = _logdet_submatrix(L, trial) - current_logdet
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is not None:
            selected.append(best_i)
            remaining.discard(best_i)
            current_logdet = _logdet_submatrix(L, selected)

    return selected


def _select_k_representative(
    embeddings: np.ndarray,
    k: int,
) -> tuple[list[int], np.ndarray]:
    """Select *k* items closest to the embedding centroid.

    Parameters
    ----------
    embeddings : np.ndarray
        ``(n, d)`` embedding matrix.
    k : int
        Number of items to select.

    Returns
    -------
    tuple[list[int], np.ndarray]
        ``(indices, distances)`` where *indices* are sorted by ascending
        distance to the centroid and *distances* is the corresponding array
        of Euclidean distances (length *k*).
    """
    n = embeddings.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, {n}], got {k}")

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    E = embeddings / norms

    centroid = E.mean(axis=0)
    distances = np.linalg.norm(E - centroid, axis=1)

    order = np.argsort(distances)[:k]
    return order.tolist(), distances[order]


# ---------------------------------------------------------------------------
# Embedding strategy base class
# ---------------------------------------------------------------------------


class EmbeddingStrategy(BaseStrategy):
    """Base for embedding-based strategies.

    Handles the sentence-transformers import check, ``embedding_model`` and
    ``lambda_val`` parameters, and provides a cached ``_get_embeddings()``
    helper for full-example embeddings.
    """

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        lambda_val: float | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if not _check_sentence_transformers():
            raise ImportError(
                "sentence-transformers is required for this strategy. "
                "Install with: uv add sentence-transformers"
            )
        self._embedding_model = embedding_model
        self._lambda_val = lambda_val
        # Cache maps id(df) → (df_ref, embeddings). Storing a strong reference to
        # the dataframe prevents Python from garbage-collecting it and reusing its
        # memory address for a different object, which would produce a false cache hit.
        self._embeddings_cache: dict[int, tuple[pl.DataFrame, np.ndarray]] = {}

    def _get_embeddings(self, df: pl.DataFrame, context: ICLSelectionContext) -> np.ndarray:
        cache_id = id(df)
        if cache_id in self._embeddings_cache:
            cached_df, cached_embs = self._embeddings_cache[cache_id]
            if cached_df is df:
                return cached_embs
        embs = _compute_embeddings(df, context, self._embedding_model)
        self._embeddings_cache[cache_id] = (df, embs)
        return embs

    def get_config(self) -> dict:
        config = super().get_config()
        config["embedding_model"] = self._embedding_model
        if self._lambda_val is not None:
            config["lambda_val"] = self._lambda_val
        return config


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------


class DiversitySelectionStrategy(EmbeddingStrategy):
    """Embedding-based DPP diversity selection over the full calibration set.

    Uses greedy log-det maximization over a cosine Gram matrix to select
    maximally spread-out ICL examples.
    """

    _name = "dpp"

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
        if n_examples < 2:
            raise ValueError(
                f"DPP diversity selection requires at least 2 examples (got {n_examples}). "
                f"Diversity is meaningless with a single sample. Use 'random' or 'rep' for n_icl=1."
            )
        df = context.pool_df
        embeddings = self._get_embeddings(df, context)

        selected = _select_k_diverse_dpp(embeddings, n_examples, self._lambda_val)
        log_det = _logdet_submatrix(
            _gram_cosine_psd(embeddings, ridge_eps=self._lambda_val),
            selected,
        )

        examples = extract_examples_from_df(
            df, selected, context.intent_col, context.sentences_col, context.labels_col,
        )

        return ICLSelectionResult(
            sample_indices=selected,
            examples=examples,
            strategy_name=self.name,
            metadata={"log_det_score": float(log_det)},
        )


class RepresentativeSelectionStrategy(EmbeddingStrategy):
    """Embedding-based centroid selection over the full calibration set.

    Selects the N examples whose embeddings are closest to the mean
    embedding — the "most typical" examples.
    """

    _name = "rep"

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
        embeddings = self._get_embeddings(df, context)

        indices, distances = _select_k_representative(embeddings, n_examples)

        examples = extract_examples_from_df(
            df, indices, context.intent_col, context.sentences_col, context.labels_col,
        )

        return ICLSelectionResult(
            sample_indices=indices,
            examples=examples,
            strategy_name=self.name,
            metadata={"distances": distances.tolist()},
        )


class AnchoredDiversitySelectionStrategy(EmbeddingStrategy):
    """Anchored DPP diversity: centroid anchor + conditional DPP expansion.

    Selects one anchor example closest to the embedding centroid (most
    representative), then greedily expands to k total examples by maximizing
    DPP log-det diversity conditioned on the anchor.  For k=1, degenerates
    to centroid-nearest (same as ``rep``).
    """

    _name = "anchor_dpp"

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
        df = context.pool_df
        embeddings = self._get_embeddings(df, context)

        # Phase 1: Anchor — single closest to centroid
        anchor_indices, anchor_distances = _select_k_representative(embeddings, 1)
        anchor_idx = anchor_indices[0]
        anchor_distance = float(anchor_distances[0])

        # Phase 2: Conditional DPP expansion (includes anchor)
        selected = _select_k_diverse_dpp(
            embeddings, n_examples, self._lambda_val,
            initial_indices=[anchor_idx],
        )

        log_det = _logdet_submatrix(
            _gram_cosine_psd(embeddings, ridge_eps=self._lambda_val),
            selected,
        )

        examples = extract_examples_from_df(
            df, selected, context.intent_col, context.sentences_col, context.labels_col,
        )

        return ICLSelectionResult(
            sample_indices=selected,
            examples=examples,
            strategy_name=self.name,
            metadata={
                "anchor_idx": anchor_idx,
                "anchor_distance": anchor_distance,
                "log_det_score": float(log_det),
            },
        )


