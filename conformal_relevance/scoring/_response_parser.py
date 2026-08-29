"""Response parsing utilities for LLM outputs."""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)
_SCORE_PAIR_RE = re.compile(r'"(\d+)":\s*(\d+(?:\.\d+)?)')
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking_tags(content: str) -> str:
    """Strip <think>...</think> blocks emitted by models with thinking mode.

    Models like Qwen3 emit ``<think>...</think>`` reasoning traces before the
    actual JSON response, even when ``format="json"`` is set.  Stripping these
    blocks before parsing prevents both direct-parse failures and false matches
    from the regex repair step (which could pick up stray numbers from the
    thinking narrative).
    """
    return _THINK_BLOCK_RE.sub("", content).strip()


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks.

    Handles responses where the LLM writes explanatory text before or after
    a JSON code block.  Extraction order:
      1. If content is bare JSON, parse directly.
      2. Extract the **last** ```json ... ``` fenced block (covers both
         responses that start with a code block and those with preamble).
      3. Fall back to repair_truncated_json regex scan.
    """
    content = _strip_thinking_tags(content)

    # 1. Try direct parse (handles pure-JSON responses)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Extract last fenced JSON code block anywhere in the response
    code_block_match = _CODE_BLOCK_RE.search(content)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        try:
            return json.loads(block_content)
        except json.JSONDecodeError:
            # Code block exists but JSON inside is truncated — repair it
            repaired = repair_truncated_json(block_content)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

    # 3. Fall back to regex repair on the full text
    logger.debug(
        f"JSON parse failed, attempting repair. "
        f"Content ends with: ...{content[-100:]}"
    )
    repaired = repair_truncated_json(content)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in response", content, 0)


def repair_truncated_json(content: str) -> str | None:
    """Attempt to repair malformed or truncated JSON dict output."""
    content = content.strip()
    matches = _SCORE_PAIR_RE.findall(content)
    if not matches:
        return None
    pairs = [f'"{key}": {value}' for key, value in matches]
    return "{" + ", ".join(pairs) + "}"


def normalize_scores_dict(scores: dict) -> dict[str, float]:
    """Normalize score values to [0, 1] range."""
    scores = {str(k): float(v) for k, v in scores.items()}
    values = list(scores.values())
    if values and max(values) > 1.0:
        scores = {k: v / 10.0 for k, v in scores.items()}
    scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}
    return scores


def try_shift_indices(
    scores_dict: dict[str, float],
    n_sentences: int,
) -> dict[str, float] | None:
    """Try to shift indices if LLM returned offset indices (e.g., 1-indexed).

    Returns shifted dict if successful, None if indices can't be aligned.
    """
    if len(scores_dict) != n_sentences:
        return None
    try:
        int_keys = sorted(int(k) for k in scores_dict.keys())
    except ValueError:
        return None
    if int_keys != list(range(int_keys[0], int_keys[0] + n_sentences)):
        return None
    offset = int_keys[0]
    if offset == 0:
        return None
    shifted = {str(int(k) - offset): v for k, v in scores_dict.items()}
    logger.debug(f"Shifted indices by -{offset} (was {int_keys[0]}-indexed)")
    return shifted


def parse_response_to_scores(
    content: str,
    n_sentences: int,
) -> tuple[dict[str, float], set[str]]:
    """Parse LLM response content to a scores dict.

    Returns:
        Tuple of (partial_scores_dict, set of missing indices)
    """
    try:
        scores_dict = parse_json_response(content)
        scores_dict = normalize_scores_dict(scores_dict)

        expected = {str(i) for i in range(n_sentences)}
        missing = expected - set(scores_dict.keys())

        # If indices don't match but count is correct, try shifting (e.g., 1-indexed)
        if missing and len(scores_dict) == n_sentences:
            shifted = try_shift_indices(scores_dict, n_sentences)
            if shifted is not None:
                scores_dict = shifted
                missing = set()

        return scores_dict, missing
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.debug(f"Failed to parse response: {e}")
        return {}, {str(i) for i in range(n_sentences)}
