"""Relevancy scorer for intent-based sentence scoring."""

import json
import logging
import time
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from conformal_relevance.prompts import build_relevancy_prompt
from conformal_relevance.prompts.templates import RELEVANCY_MISSING_SCORES_PROMPT
from conformal_relevance.scoring._response_parser import (
    normalize_scores_dict,
    parse_json_response,
)
from conformal_relevance.scoring.pipeline import DEFAULT_MAX_RETRIES
from conformal_relevance.types import DEFAULT_INTENT, RelevancyScorer

logger = logging.getLogger(__name__)


def relevancy_dict_scorer(
    llm: BaseChatModel,
    icl_examples: str | Sequence[dict] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = 1.0,
    default_intent: str = DEFAULT_INTENT,
) -> RelevancyScorer:
    """Create a relevancy scorer using dict output with targeted retry for missing scores.

    Args:
        llm: Language model to use for scoring.
        icl_examples: In-context learning examples (pre-formatted string or list of dicts).
        max_retries: Maximum number of retry attempts for failed API calls.
        retry_base_delay: Base delay in seconds between retries (exponential backoff).
        default_intent: Default intent string to use for ICL examples lacking 'intent' key.

    Returns:
        A RelevancyScorer function that scores sentences given an intent.
    """

    def _get_missing_keys(scores: dict[str, float], n: int) -> set[str]:
        expected = {str(i) for i in range(n)}
        return expected - set(scores.keys())

    def _call_initial(intent: str, sentences: Sequence[str]) -> dict[str, float]:
        prompt = build_relevancy_prompt(
            intent=intent,
            sentences=sentences,
            icl_examples=icl_examples,
            default_intent=default_intent,
        )

        last_exception = None
        response = None
        for attempt in range(max_retries):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                scores = parse_json_response(response.content)
                return normalize_scores_dict(scores)
            except json.JSONDecodeError as e:
                last_exception = e
                content = response.content if response is not None else "N/A"
                logger.warning(
                    f"Relevancy scoring attempt {attempt + 1} failed: {e}. "
                    f"Response length: {len(content)} chars, "
                    f"ends with: ...{content[-80:] if len(content) > 80 else content}"
                )
                if attempt < max_retries - 1:
                    delay = retry_base_delay * (2**attempt)
                    logger.warning(f"Retrying in {delay}s...")
                    time.sleep(delay)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = retry_base_delay * (2**attempt)
                    logger.warning(
                        f"Relevancy scoring attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"Relevancy scoring failed after {max_retries} attempts"
        ) from last_exception

    def _call_for_missing(
        missing_keys: set[str],
        intent: str,
        sentences: Sequence[str],
    ) -> dict[str, float]:
        missing_sentences_text = "\n".join(
            f"{key}. {sentences[int(key)]}" for key in sorted(missing_keys, key=int)
        )
        prompt = RELEVANCY_MISSING_SCORES_PROMPT.format(
            icl_examples="",
            task_line="",
            intent=intent,
            missing_sentences=missing_sentences_text,
        )

        logger.info(
            f"Requesting {len(missing_keys)} missing relevancy scores via targeted retry"
        )

        last_exception = None
        response = None
        for attempt in range(max_retries):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                scores = parse_json_response(response.content)
                return normalize_scores_dict(scores)
            except json.JSONDecodeError as e:
                last_exception = e
                content = response.content if response is not None else "N/A"
                logger.warning(
                    f"Targeted retry attempt {attempt + 1} failed: {e}. "
                    f"Response ends with: ...{content[-80:] if len(content) > 80 else content}"
                )
                if attempt < max_retries - 1:
                    delay = retry_base_delay * (2**attempt)
                    time.sleep(delay)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = retry_base_delay * (2**attempt)
                    logger.warning(
                        f"Targeted retry attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"Targeted retry failed after {max_retries} attempts"
        ) from last_exception

    def score(intent: str, sentences: Sequence[str]) -> Sequence[float]:
        n = len(sentences)
        logger.info(f"Scoring {n} sentences for relevancy with LLM")

        all_scores = _call_initial(intent, sentences)
        logger.debug(f"Initial call returned {len(all_scores)} scores")

        missing = _get_missing_keys(all_scores, n)
        retry_count = 0

        while missing and retry_count < max_retries:
            logger.warning(
                f"Missing {len(missing)} scores after attempt {retry_count + 1}: "
                f"{sorted(missing, key=int)[:10]}{'...' if len(missing) > 10 else ''}"
            )
            try:
                retry_scores = _call_for_missing(missing, intent, sentences)
                all_scores.update(retry_scores)
            except RuntimeError as e:
                logger.error(f"Targeted retry failed: {e}")
            missing = _get_missing_keys(all_scores, n)
            retry_count += 1

        if missing:
            raise ValueError(
                f"Failed to get relevancy scores for {len(missing)} sentences after "
                f"{max_retries} targeted retries. "
                f"Missing: {sorted(missing, key=int)[:20]}"
                f"{'...' if len(missing) > 20 else ''}"
            )

        logger.debug(f"Relevancy scoring complete for {n} sentences")
        return [all_scores[str(i)] for i in range(n)]

    return score
