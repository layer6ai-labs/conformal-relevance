from typing import Iterable
import numpy as np
import polars as pl
import math

# Scale of uniform noise added to break ties.
# Chosen to be negligible relative to score precision (scores are 0-1)
# but sufficient to break exact ties between identical scores.
TIE_BREAK_NOISE_SCALE = 1e-6

# Default seed for reproducibility. Calibration-side tie-breaking and
# inference-side tie-breaking (apply_threshold) derive from the same seed so
# that, at fixed (α, scorer), varying β produces a monotonic retention curve
# rather than ±1e-6 jitter around exact-tie score values (e.g., HotpotQA where
# 16% of scores are exactly 0.1).
DEFAULT_TIE_BREAK_SEED = 42


def _add_tie_break_noise(
    scores: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add small uniform noise to scores to break ties.

    If `rng` is None, uses a deterministic seed (`DEFAULT_TIE_BREAK_SEED`)
    so results are reproducible across calls.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_TIE_BREAK_SEED)
    noise = rng.uniform(-TIE_BREAK_NOISE_SCALE, TIE_BREAK_NOISE_SCALE, size=scores.shape)
    return scores + noise


def apply_threshold(
    scores: np.ndarray | Iterable[float],
    threshold: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a boolean mask of which scores are retained under `threshold`.

    Applies the same tie-breaking noise scheme used at calibration time, so
    test-time comparisons are symmetric with the threshold returned by
    `compute_threshold` (both sides see ±TIE_BREAK_NOISE_SCALE jitter). Without
    this, raw scores that cluster on exact values (e.g., 0.1) can flip in/out
    across β values purely due to cal-side noise, producing non-monotonic
    retention curves.
    """
    scores_arr = np.asarray(scores, dtype=float)
    noisy_scores = _add_tie_break_noise(scores_arr, rng)
    return noisy_scores >= threshold


def get_conformal_score(
        confidence_scores: np.ndarray,
        labels: np.ndarray,
        beta: float,
        rng: np.random.Generator | None = None,
    ) -> float:
    """
    Compute the conformal importance score given a set of confidence scores and binary importance labels.

    Returns the largest threshold such that at least a fraction beta of important claims
    have confidence scores >= that threshold (i.e., would be retained).

    Small uniform noise is added to break ties between identical scores.

    Edge cases:
    - beta = 0: no important claims need to be retained, so the threshold is +inf.
    - No important claims: the retention constraint is vacuously satisfied, so the threshold is +inf.
    - beta = 1: the threshold equals the lowest confidence score among important claims.
    """

    # Edge cases:
    # - beta = 0: +inf (no retention required)
    if beta == 0:
        return np.inf

    confidence_scores = _add_tie_break_noise(confidence_scores, rng)

    sort_idx = np.argsort(-confidence_scores)  # sort in descending order
    sorted_confidence_scores = confidence_scores[sort_idx]
    labels = labels[sort_idx]
    total_important = np.sum(labels)

    # Edge cases:
    # - No important claims: +inf (constraint vacuously satisfied)

    if total_important == 0:
        return np.inf

    important_fractions = np.cumsum(labels) / total_important
    indices = np.asarray(important_fractions >= beta).nonzero()[0]
    return float(sorted_confidence_scores[indices[0]] if indices.size > 0 else np.inf)


def get_conformal_score_column(
    dataset: pl.DataFrame, 
    score_column: str,
    label_column: str,
    beta: float
) -> pl.Series:
    """
    Apply get_conformal_score to each row of the dataset and return a new column with the conformal scores.
    """
    return dataset.select(
        pl.struct([score_column, label_column]).map_elements(
            lambda row: get_conformal_score(
                np.array(row[score_column]), 
                np.array(row[label_column]), 
                beta
            ),
            return_dtype=pl.Float64)
            )


def compute_threshold(
        dataset: pl.DataFrame,
        score_column: str,
        label_column: str,
        alpha: float, 
        beta: float
    ) -> float:
    """
    Compute the conformal threshold for a given dataset, score column, and label column.
    """
    conformal_scores = get_conformal_score_column(dataset, score_column, label_column, beta).to_numpy().flatten()

    # The quantile is floor(alpha*(n+1))/n, and we map this to the index by
    # dropping the division by n and subtracting one (for zero-index).
    quantile_target_index = int(math.floor((len(conformal_scores) + 1) * alpha))
    quantile_target_index = max(1, min(quantile_target_index, len(conformal_scores))) - 1

    # np.partition finds the kth smallest element without fully sorting the array.
    threshold = np.partition(conformal_scores, quantile_target_index)[quantile_target_index]

    return float(threshold)



def get_accepted_subclaims(
        claims: Iterable[str],
        scores: Iterable[float],
        threshold: float,
        rng: np.random.Generator | None = None,
) -> list[str]:
    scores_arr = _add_tie_break_noise(np.asarray(scores, dtype=float), rng)
    return [claim for claim, score in zip(claims, scores_arr) if score >= threshold]


