"""Async scoring engine for parallel LLM calls."""

import asyncio
import logging
import re
import time
from collections import deque
from typing import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from tqdm.auto import tqdm

from conformal_relevance.prompts import build_relevancy_prompt
from conformal_relevance.prompts.templates import (
    RELEVANCY_MISSING_SCORES_PROMPT,
    RELEVANCY_MISSING_SCORES_PROMPT_NO_INTENT,
)
from conformal_relevance.scoring._response_parser import parse_response_to_scores
from conformal_relevance.scoring.profiling import ProfileCollector

logger = logging.getLogger(__name__)

# Type aliases for callable prompt builders (used by ICL-free scoring)
PromptBuilder = Callable[[str, Sequence[str]], str]
MissingPromptBuilder = Callable[[str, Sequence[str], set[str]], str]

# Default rate limit settings
DEFAULT_RATE_LIMIT_BASE_DELAY = 60.0
DEFAULT_RATE_LIMIT_MAX_DELAY = 300.0
DEFAULT_RATE_LIMIT_BUFFER = 5.0  # Extra buffer added to suggested retry delays

_THREAD_FALLBACK_TIMEOUT = 3600  # 1 hour -- safety net for thread fallback

# Pre-compiled retry delay regex patterns
_RETRY_DELAY_PATTERNS = [
    re.compile(r"retry in (\d+\.?\d*)\s*s", re.IGNORECASE),
    re.compile(r"retry_delay.*?seconds:\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+\.?\d*)\s*seconds?", re.IGNORECASE),
]


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    rate_limit_indicators = [
        "429", "rate limit", "ratelimit", "quota",
        "resourceexhausted", "too many requests", "exceeded",
    ]
    return any(indicator in error_str or indicator in error_type
               for indicator in rate_limit_indicators)


def _extract_retry_delay(error: Exception) -> float | None:
    """Extract suggested retry delay from error message if available."""
    error_str = str(error)
    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(error_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


class _RateLimitState:
    """Shared state for coordinating rate limit pauses across concurrent requests."""

    def __init__(self):
        self.pause_until: float = 0.0
        self.lock = asyncio.Lock()
        self.consecutive_rate_limits: int = 0

    async def wait_if_paused(self) -> None:
        now = time.time()
        if now < self.pause_until:
            wait_time = self.pause_until - now
            msg = f"Rate limited - waiting {wait_time:.1f}s before next request"
            logger.info(msg)
            print(f"\n⏳ {msg}", flush=True)
            await asyncio.sleep(wait_time)

    async def handle_rate_limit(self, error: Exception) -> float:
        async with self.lock:
            self.consecutive_rate_limits += 1
            delay = _extract_retry_delay(error)
            if delay is None:
                delay = min(
                    DEFAULT_RATE_LIMIT_BASE_DELAY * (2 ** (self.consecutive_rate_limits - 1)),
                    DEFAULT_RATE_LIMIT_MAX_DELAY,
                )
            else:
                delay = delay + DEFAULT_RATE_LIMIT_BUFFER
            new_pause_until = time.time() + delay
            if new_pause_until > self.pause_until:
                self.pause_until = new_pause_until
                msg = (
                    f"Rate limit hit (attempt {self.consecutive_rate_limits}). "
                    f"Pausing all requests for {delay:.1f}s"
                )
                logger.warning(msg)
                print(f"\n⚠️  {msg}", flush=True)
            return delay

    def reset_consecutive(self) -> None:
        self.consecutive_rate_limits = 0


def build_missing_scores_prompt(
    intent: str,
    sentences: Sequence[str],
    missing_indices: set[str],
    icl_examples: str | None = None,
    has_intent: bool = True,
    task: str = "",
) -> str:
    """Build a prompt to request only missing sentence scores."""
    missing_idx_ints = sorted(int(i) for i in missing_indices)
    missing_lines = [f"{i}. {sentences[i]}" for i in missing_idx_ints]
    missing_sentences_text = "\n".join(missing_lines)
    icl_text = icl_examples if icl_examples else ""
    task_line = f"Task: {task}\n" if task else ""

    if has_intent:
        return RELEVANCY_MISSING_SCORES_PROMPT.format(
            icl_examples=icl_text,
            task_line=task_line,
            intent=intent,
            missing_sentences=missing_sentences_text,
        )
    else:
        return RELEVANCY_MISSING_SCORES_PROMPT_NO_INTENT.format(
            icl_examples=icl_text,
            task_line=task_line,
            missing_sentences=missing_sentences_text,
        )


def scores_dict_to_list(
    scores_dict: dict[str, float],
    n_sentences: int,
    max_missing_rate: float = 0.25,
) -> list[float] | None:
    """Convert scores dict to ordered list, or None if incomplete.

    If some indices are missing but the missing fraction is at or below
    ``max_missing_rate``, the gaps are filled with 0.0 and the list is
    returned.  Otherwise returns ``None``.
    """
    expected = {str(i) for i in range(n_sentences)}
    missing = expected - set(scores_dict.keys())
    if not missing:
        return [scores_dict[str(i)] for i in range(n_sentences)]
    if n_sentences > 0 and len(missing) / n_sentences <= max_missing_rate:
        return [scores_dict.get(str(i), 0.0) for i in range(n_sentences)]
    return None


async def score_single_sample(
    llm: BaseChatModel,
    sample_idx: int,
    intent: str,
    sentences: Sequence[str],
    icl_examples: str | Sequence[dict] | None,
    default_intent: str,
    semaphore: asyncio.Semaphore,
    profile_collector: ProfileCollector | None = None,
    prompt_builder: PromptBuilder | None = None,
    rate_limit_state: _RateLimitState | None = None,
    has_intent: bool = True,
    dataset_task: str = "",
) -> tuple[int, dict[str, float], set[str]]:
    """Score a single sample with semaphore-controlled concurrency."""
    n_sentences = len(sentences)

    if rate_limit_state:
        await rate_limit_state.wait_if_paused()

    async with semaphore:
        if prompt_builder is not None:
            prompt = prompt_builder(intent, sentences)
        else:
            prompt = build_relevancy_prompt(
                intent=intent,
                sentences=sentences,
                icl_examples=icl_examples,
                default_intent=default_intent,
                has_intent=has_intent,
                task=dataset_task,
            )
        try:
            start_time = time.perf_counter()
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            duration_ms = (time.perf_counter() - start_time) * 1000

            if profile_collector:
                profile_collector.record_llm_call(sample_idx, duration_ms)

            scores_dict, missing = parse_response_to_scores(
                response.content, n_sentences
            )
            if rate_limit_state:
                rate_limit_state.reset_consecutive()
            return sample_idx, scores_dict, missing
        except Exception as e:
            if rate_limit_state and _is_rate_limit_error(e):
                await rate_limit_state.handle_rate_limit(e)
                print(f"\n⚠️  Sample {sample_idx}: rate limit error, will retry", flush=True)
            else:
                logger.debug(f"Sample {sample_idx} failed: {e}")
            return sample_idx, {}, {str(i) for i in range(n_sentences)}


async def score_missing_indices(
    llm: BaseChatModel,
    sample_idx: int,
    intent: str,
    sentences: Sequence[str],
    missing_indices: set[str],
    semaphore: asyncio.Semaphore,
    profile_collector: ProfileCollector | None = None,
    missing_prompt_builder: MissingPromptBuilder | None = None,
    retry_num: int = 1,
    icl_examples: str | None = None,
    rate_limit_state: _RateLimitState | None = None,
    has_intent: bool = True,
    dataset_task: str = "",
) -> tuple[int, dict[str, float], set[str]]:
    """Score only missing indices for a sample (targeted retry)."""
    n_sentences = len(sentences)

    if rate_limit_state:
        await rate_limit_state.wait_if_paused()

    async with semaphore:
        if missing_prompt_builder is not None:
            prompt = missing_prompt_builder(intent, sentences, missing_indices)
        else:
            prompt = build_missing_scores_prompt(
                intent, sentences, missing_indices, icl_examples,
                has_intent=has_intent,
                task=dataset_task,
            )
        if retry_num > 1:
            prompt = f"{prompt}\n\n(Retry attempt {retry_num})"
        try:
            start_time = time.perf_counter()
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            duration_ms = (time.perf_counter() - start_time) * 1000

            if profile_collector:
                profile_collector.record_llm_call(
                    sample_idx, duration_ms, is_targeted_retry=True
                )

            scores_dict, still_missing = parse_response_to_scores(
                response.content, n_sentences
            )
            filtered_scores = {k: v for k, v in scores_dict.items() if k in missing_indices}
            still_missing = missing_indices - set(filtered_scores.keys())
            if rate_limit_state:
                rate_limit_state.reset_consecutive()
            return sample_idx, filtered_scores, still_missing
        except Exception as e:
            if rate_limit_state and _is_rate_limit_error(e):
                await rate_limit_state.handle_rate_limit(e)
            logger.debug(f"Sample {sample_idx} retry failed: {e}")
            return sample_idx, {}, missing_indices


async def score_all_async(
    llm: BaseChatModel,
    intents: list[str],
    sentences_list: list[Sequence[str]],
    icl_examples_list: list[str | Sequence[dict] | None],
    default_intent: str,
    max_concurrency: int,
    max_retries: int,
    show_progress: bool,
    profile_collector: ProfileCollector | None = None,
    prompt_builder: PromptBuilder | None = None,
    missing_prompt_builder: MissingPromptBuilder | None = None,
    has_intent: bool = True,
    dataset_task: str = "",
    max_missing_rate: float = 0.25,
) -> list[list[float] | None]:
    """Score all samples asynchronously with sliding window concurrency."""
    n_samples = len(intents)
    semaphore = asyncio.Semaphore(max_concurrency)
    rate_limit_state = _RateLimitState()

    partial_scores: list[dict[str, float]] = [{} for _ in range(n_samples)]
    retry_counts: list[int] = [0] * n_samples
    attempted: set[int] = set()
    completed: set[int] = set()

    if profile_collector:
        for idx in range(n_samples):
            profile_collector.start_sample(idx, n_sentences=len(sentences_list[idx]))

    pbar = tqdm(total=n_samples, desc="Scoring samples", disable=not show_progress)

    def create_task_for_idx(idx: int) -> asyncio.Task:
        n_sentences = len(sentences_list[idx])
        all_indices = {str(i) for i in range(n_sentences)}
        current_keys = set(partial_scores[idx].keys())

        if idx not in attempted:
            attempted.add(idx)
            coro = score_single_sample(
                llm, idx, intents[idx], sentences_list[idx],
                icl_examples_list[idx], default_intent, semaphore,
                profile_collector, prompt_builder, rate_limit_state,
                has_intent=has_intent,
                dataset_task=dataset_task,
            )
        else:
            missing_indices = all_indices - current_keys
            retry_num = retry_counts[idx] + 1
            icl_for_retry = icl_examples_list[idx] if isinstance(icl_examples_list[idx], str) else None
            coro = score_missing_indices(
                llm, idx, intents[idx], sentences_list[idx],
                missing_indices, semaphore, profile_collector,
                missing_prompt_builder, retry_num, icl_for_retry,
                rate_limit_state, has_intent=has_intent,
                dataset_task=dataset_task,
            )
        return asyncio.create_task(coro)

    pending_queue: deque[int] = deque(range(n_samples))
    active_tasks: dict[asyncio.Task, int] = {}

    while pending_queue and len(active_tasks) < max_concurrency:
        idx = pending_queue.popleft()
        task = create_task_for_idx(idx)
        active_tasks[task] = idx

    while active_tasks:
        done, _ = await asyncio.wait(
            active_tasks.keys(),
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            idx = active_tasks.pop(task)

            try:
                result = task.result()
                sample_idx, scores_dict, missing = result

                partial_scores[sample_idx].update(scores_dict)

                n_sentences = len(sentences_list[sample_idx])
                all_indices = {str(i) for i in range(n_sentences)}
                still_missing = all_indices - set(partial_scores[sample_idx].keys())

                if not still_missing:
                    if profile_collector:
                        profile_collector.end_sample(sample_idx, success=True)
                    completed.add(sample_idx)
                    pbar.update(1)
                    pbar.refresh()
                else:
                    retry_counts[sample_idx] += 1
                    if retry_counts[sample_idx] < max_retries:
                        pending_queue.append(sample_idx)
                    else:
                        n_sentences = len(sentences_list[sample_idx])
                        n_missing = len(still_missing)
                        missing_rate = n_missing / n_sentences if n_sentences > 0 else 1.0
                        if missing_rate <= max_missing_rate:
                            logger.info(
                                f"Row {sample_idx}: Filling {n_missing}/{n_sentences} "
                                f"missing scores with 0.0 ({missing_rate:.1%} missing)"
                            )
                            if profile_collector:
                                profile_collector.end_sample(sample_idx, success=True)
                        else:
                            logger.warning(
                                f"Row {sample_idx}: Max retries ({max_retries}) reached "
                                f"with {n_missing} missing scores"
                            )
                            if profile_collector:
                                profile_collector.end_sample(sample_idx, success=False)
                        completed.add(sample_idx)
                        pbar.update(1)
                        pbar.refresh()

            except Exception as e:
                logger.debug(f"Sample {idx} failed with exception: {e}")
                retry_counts[idx] += 1
                if retry_counts[idx] < max_retries:
                    pending_queue.append(idx)
                else:
                    logger.warning(f"Row {idx}: Max retries reached due to exception")
                    if profile_collector:
                        profile_collector.end_sample(idx, success=False)
                    completed.add(idx)
                    pbar.update(1)
                    pbar.refresh()

            if pending_queue and len(active_tasks) < max_concurrency:
                next_idx = pending_queue.popleft()
                new_task = create_task_for_idx(next_idx)
                active_tasks[new_task] = next_idx

    pbar.close()

    results_list: list[list[float] | None] = []
    for i in range(n_samples):
        n_sentences = len(sentences_list[i])
        result = scores_dict_to_list(partial_scores[i], n_sentences, max_missing_rate)
        results_list.append(result)

    return results_list


def run_async(coro: object) -> object:
    """Run an async coroutine, handling Jupyter/IPython environments."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Running inside an existing event loop (e.g. Jupyter).
    # Use nest_asyncio to allow reentrant event loop usage.
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except (ImportError, RuntimeError):
        pass

    # Thread fallback: run a fresh event loop in a separate thread.
    import threading

    result = None
    exception = None

    def _run():
        nonlocal result, exception
        try:
            result = asyncio.run(coro)
        except BaseException as exc:
            exception = exc

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=_THREAD_FALLBACK_TIMEOUT)

    if thread.is_alive():
        raise TimeoutError(
            f"Async operation did not complete within {_THREAD_FALLBACK_TIMEOUT}s. "
            "This likely indicates a hung LLM request. Consider reducing max_concurrency."
        )

    if exception is not None:
        raise exception
    return result
