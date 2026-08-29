"""Type definitions for conformal relevancy."""

from typing import Callable, Sequence, TypedDict


class RelevancyICLExample(TypedDict):
    """In-context learning example for relevancy-based scoring.

    Includes intent to demonstrate how relevancy is evaluated
    with respect to a specific query/perspective.
    """

    intent: str
    input_units: Sequence[str]
    input_unit_labels: Sequence[bool]
    output_text: str


# Relevancy scorer type: (intent, sentences) -> scores
# Note: No separate 'text' parameter since sentences ARE the document content
RelevancyScorer = Callable[[str, Sequence[str]], Sequence[float]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default random seed for reproducibility across all modules
DEFAULT_SEED = 3

# Default number of ICL examples
DEFAULT_N_ICL = 2

# Default intent string for datasets without per-sample intent
DEFAULT_INTENT = "General relevancy"

# Standard column names for conformal relevancy dataframes
INTENT_COLUMN = "intent"
INPUT_UNITS_COLUMN = "input_units"
INPUT_UNIT_LABELS_COLUMN = "input_unit_labels"
OUTPUT_TEXT_COLUMN = "output_text"
OUTPUT_UNITS_COLUMN = "output_units"

# Score column prefix for ICL-based scoring results
# Full column name format: f"{SCORES_PREFIX}{n_icl}_{strategy}"
SCORES_PREFIX = "scores_icl"

# Default LangChain cache database path
DEFAULT_CACHE_PATH = ".langchain_cache.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_strategy_name(name: str) -> str:
    """Strip the ``_stratified`` suffix from a strategy name.

    Used when deriving output column names so that stratified variants
    map to the same column as their base strategy.
    """
    return name.removesuffix("_stratified")
