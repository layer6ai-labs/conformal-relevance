"""Pydantic models for structured LLM outputs."""

import logging

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class DictScoringResponse(BaseModel):
    """LLM response for sentence scoring with explicit key-value mapping.

    Uses a dictionary with sentence indices as string keys (e.g., "0", "1", "2")
    to ensure explicit mapping between sentences and their scores, preventing
    order mix-ups in long documents.
    """

    scores: dict[str, float] = Field(
        description="Dictionary mapping sentence indices (as strings '0', '1', etc.) "
        "to relevancy scores. Each score should be between 0.0 and 1.0."
    )

    @field_validator("scores", mode="before")
    @classmethod
    def normalize_scores(cls, v: dict) -> dict[str, float]:
        """Normalize dict values to [0, 1] range.

        Handles:
        - 0-10 scale normalization (if any score > 1)
        - Clipping to valid range
        - Converting keys to strings
        """
        if not v:
            return v

        # Ensure keys are strings and values are floats
        scores = {str(k): float(val) for k, val in v.items()}

        # Normalize if scores appear to be on 0-10 scale
        if any(s > 1.0 for s in scores.values()):
            logger.debug("Normalizing scores from 0-10 scale to 0-1")
            scores = {k: s / 10.0 for k, s in scores.items()}

        # Clip to valid range
        scores = {k: max(0.0, min(1.0, s)) for k, s in scores.items()}

        return scores
