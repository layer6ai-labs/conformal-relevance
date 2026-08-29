"""Prompt formatting utilities for relevancy scoring."""

import json
import random
from typing import Sequence

from conformal_relevance.prompts.templates import (
    RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT,
    RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT_NO_INTENT,
)
from conformal_relevance.types import DEFAULT_INTENT


def labels_to_score_dict(
    labels: Sequence[bool] | Sequence[int],
    high_score: float = 1.0,
    low_score: float = 0.0,
) -> dict[str, float]:
    """Convert a list of binary labels to a 0-indexed score dictionary."""
    return {
        str(i): high_score if label else low_score
        for i, label in enumerate(labels)
    }


def format_sentences_numbered(sentences: Sequence[str]) -> str:
    """Format sentences as a numbered list (0-indexed)."""
    return "\n".join(f"{i}. {sent}" for i, sent in enumerate(sentences))


def format_score_dict(scores: dict[str, float]) -> str:
    """Format a score dictionary as a JSON-like string."""
    sorted_scores = {k: scores[k] for k in sorted(scores.keys(), key=int)}
    return json.dumps(sorted_scores)


def format_relevancy_icl_example(
    intent: str,
    sentences: Sequence[str],
    labels: Sequence[bool] | Sequence[int],
    min_negatives: int = 2,
    random_seed: int | None = None,
    has_intent: bool = True,
    task: str = "",
) -> str:
    """Format a single ICL example with contrastive (positive + negative) groups.

    Shows ALL positive sentences as quoted text, and a SUB-SAMPLED set of
    negative sentences for contrast. Negatives are sub-sampled at a 1:1 ratio
    with positives (floored by min_negatives, capped by availability).

    Args:
        intent: The intent/query perspective.
        sentences: List of sentences (input_units).
        labels: Binary labels for each sentence (input_unit_labels).
        min_negatives: Minimum number of negative examples. Default is 2.
        random_seed: Random seed for reproducible negative sampling.
        has_intent: Whether to include Intent line in the example header.
        task: Uniform task description (e.g., "PHI detection"). Empty string = no Task line.

    Returns:
        Formatted ICL example string with contrastive groups.
    """
    sentences_text = format_sentences_numbered(sentences)

    # Separate positive and negative sentences
    positive_sentences = [s for s, l in zip(sentences, labels) if l]
    negative_sentences = [s for s, l in zip(sentences, labels) if not l]

    # Sub-sample negatives at 1:1 ratio with positives (floored by min_negatives)
    num_positives = len(positive_sentences)
    num_negatives = min(max(num_positives, min_negatives), len(negative_sentences))
    rng = random.Random(random_seed)
    sampled_negatives = rng.sample(negative_sentences, num_negatives) if num_negatives > 0 else []

    # Build the header section (task and/or intent if present)
    header_lines = []
    if task:
        header_lines.append(f"Task: {task}")
    if has_intent:
        header_lines.append(f"Intent: {intent}")

    if header_lines:
        result = "\n".join(header_lines) + "\n"
        result += f"Sentences to evaluate:\n{sentences_text}"
    else:
        result = f"Sentences to evaluate:\n{sentences_text}"

    if positive_sentences:
        pos_text = ", ".join(f'"{s}"' for s in positive_sentences)
        if has_intent:
            result += f'\nSentences selected for "{intent}": {pos_text}'
        else:
            result += f'\nSentences selected: {pos_text}'

    if sampled_negatives:
        neg_text = ", ".join(f'"{s}"' for s in sampled_negatives)
        if has_intent:
            result += f'\nSentences NOT selected for "{intent}" (examples): {neg_text}'
        else:
            result += f'\nSentences NOT selected (examples): {neg_text}'

    result += "\n---"
    return result


def format_relevancy_icl_examples(
    examples: Sequence[dict],
    default_intent: str = DEFAULT_INTENT,
    min_negatives: int = 2,
    random_seed: int | None = None,
    has_intent: bool = True,
    task: str = "",
) -> str:
    """Format multiple ICL examples for prompt insertion using contrastive format.

    Each example dict should have keys:
    - 'intent': str (optional, uses default_intent if missing)
    - 'input_units' or 'sentences': Sequence[str]
    - 'input_unit_labels' or 'labels': Sequence[bool]

    Args:
        examples: List of example dictionaries.
        default_intent: Intent string to use when example lacks 'intent' key.
        min_negatives: Minimum number of negative examples per ICL example.
        random_seed: Random seed for reproducible negative sampling.
        has_intent: Whether examples include Intent line.
        task: Uniform task description (e.g., "PHI detection"). Empty string = no Task line.

    Returns:
        Formatted string for prompt insertion.
    """
    if not examples:
        return ""

    parts = []
    for i, ex in enumerate(examples):
        intent = ex.get('intent') or default_intent
        sentences = ex.get('input_units') or ex.get('sentences', [])
        labels = ex.get('input_unit_labels') or ex.get('labels', [])

        # Use per-example seed for reproducibility
        example_seed = random_seed + i if random_seed is not None else None

        example_text = format_relevancy_icl_example(
            intent=intent,
            sentences=sentences,
            labels=labels,
            min_negatives=min_negatives,
            random_seed=example_seed,
            has_intent=has_intent,
            task=task,
        )
        parts.append(f"Example {i}:\n{example_text}")

    examples_text = "\n\n".join(parts)
    return examples_text + "\n\n"


def build_relevancy_prompt(
    intent: str,
    sentences: Sequence[str],
    icl_examples: str | Sequence[dict] | None = None,
    default_intent: str = DEFAULT_INTENT,
    min_negatives: int = 2,
    random_seed: int | None = None,
    has_intent: bool = True,
    task: str = "",
) -> str:
    """Build a complete relevancy scoring prompt.

    Args:
        intent: The intent/query perspective.
        sentences: List of sentences to score.
        icl_examples: Either a pre-formatted ICL string or list of example dicts.
            Each dict should have 'intent', 'input_units', 'input_unit_labels'.
        default_intent: Intent string to use for ICL examples lacking 'intent' key.
        min_negatives: Min negatives for contrastive ICL formatting.
        random_seed: Random seed for negative sampling in ICL examples.
        has_intent: Whether to include Intent line in the prompt.
        task: Uniform task description (e.g., "PHI detection"). Empty string = no Task line.

    Returns:
        Complete formatted prompt ready for LLM invocation.
    """
    sentences_text = format_sentences_numbered(sentences)

    if icl_examples is None:
        icl_text = ""
    elif isinstance(icl_examples, str):
        icl_text = icl_examples
    else:
        icl_text = format_relevancy_icl_examples(
            icl_examples,
            default_intent=default_intent,
            min_negatives=min_negatives,
            random_seed=random_seed,
            has_intent=has_intent,
            task=task,
        )

    # Build task_line conditionally — empty string when no task
    task_line = f"Task: {task}\n" if task else ""

    if has_intent:
        return RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT.format(
            icl_examples=icl_text,
            task_line=task_line,
            intent=intent,
            sentences=sentences_text,
        )
    else:
        return RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT_NO_INTENT.format(
            icl_examples=icl_text,
            task_line=task_line,
            sentences=sentences_text,
        )
