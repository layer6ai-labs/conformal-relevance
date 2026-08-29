"""Profiling utilities for tracking scoring performance."""

import time
from dataclasses import dataclass, field


@dataclass
class SampleProfile:
    """Profiling data for a single sample."""
    sample_idx: int
    start_time_ms: float = 0.0
    end_time_ms: float = 0.0
    total_duration_ms: float = 0.0
    llm_calls: list[float] = field(default_factory=list)
    total_llm_time_ms: float = 0.0
    retry_count: int = 0
    targeted_retry_count: int = 0
    n_sentences: int = 0
    success: bool = False

    @property
    def overhead_ms(self) -> float:
        return self.total_duration_ms - self.total_llm_time_ms


@dataclass
class RunProfile:
    """Aggregated profiling data for a complete scoring run."""
    total_duration_ms: float = 0.0
    total_llm_time_ms: float = 0.0
    n_samples: int = 0
    n_successful: int = 0
    n_failed: int = 0
    n_llm_calls: int = 0
    n_retries: int = 0
    n_targeted_retries: int = 0
    avg_sample_time_ms: float = 0.0
    avg_llm_latency_ms: float = 0.0
    samples: list[SampleProfile] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_duration_ms": self.total_duration_ms,
            "total_llm_time_ms": self.total_llm_time_ms,
            "n_samples": self.n_samples,
            "n_successful": self.n_successful,
            "n_failed": self.n_failed,
            "n_llm_calls": self.n_llm_calls,
            "n_retries": self.n_retries,
            "n_targeted_retries": self.n_targeted_retries,
            "avg_sample_time_ms": self.avg_sample_time_ms,
            "avg_llm_latency_ms": self.avg_llm_latency_ms,
        }


class ProfileCollector:
    """Collects profiling data during scoring runs."""

    def __init__(self) -> None:
        self._run_start_ms: float = 0.0
        self._run_end_ms: float = 0.0
        self._samples: dict[int, SampleProfile] = {}
        self._enabled: bool = True

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def start_run(self) -> None:
        if not self._enabled:
            return
        self._run_start_ms = time.perf_counter() * 1000
        self._samples.clear()

    def start_sample(self, sample_idx: int, n_sentences: int = 0) -> None:
        if not self._enabled:
            return
        self._samples[sample_idx] = SampleProfile(
            sample_idx=sample_idx,
            start_time_ms=time.perf_counter() * 1000,
            n_sentences=n_sentences,
        )

    def record_llm_call(
        self,
        sample_idx: int,
        duration_ms: float,
        is_targeted_retry: bool = False,
    ) -> None:
        if not self._enabled:
            return
        if sample_idx not in self._samples:
            return
        profile = self._samples[sample_idx]
        profile.llm_calls.append(duration_ms)
        profile.total_llm_time_ms += duration_ms
        if is_targeted_retry:
            profile.targeted_retry_count += 1
        elif len(profile.llm_calls) > 1:
            profile.retry_count += 1

    def end_sample(self, sample_idx: int, success: bool = True) -> None:
        if not self._enabled:
            return
        if sample_idx not in self._samples:
            return
        profile = self._samples[sample_idx]
        profile.end_time_ms = time.perf_counter() * 1000
        profile.total_duration_ms = profile.end_time_ms - profile.start_time_ms
        profile.success = success

    def get_sample_profile(self, sample_idx: int) -> SampleProfile | None:
        return self._samples.get(sample_idx)

    def finalize(self) -> RunProfile:
        self._run_end_ms = time.perf_counter() * 1000
        total_duration_ms = self._run_end_ms - self._run_start_ms

        if not self._enabled or not self._samples:
            return RunProfile(total_duration_ms=total_duration_ms)

        samples = list(self._samples.values())
        n_samples = len(samples)
        n_successful = sum(1 for s in samples if s.success)
        n_failed = n_samples - n_successful

        total_llm_time_ms = sum(s.total_llm_time_ms for s in samples)
        n_llm_calls = sum(len(s.llm_calls) for s in samples)
        n_retries = sum(s.retry_count for s in samples)
        n_targeted_retries = sum(s.targeted_retry_count for s in samples)

        avg_sample_time_ms = (
            sum(s.total_duration_ms for s in samples) / n_samples
            if n_samples > 0 else 0.0
        )

        all_llm_calls = [call for s in samples for call in s.llm_calls]
        avg_llm_latency_ms = (
            sum(all_llm_calls) / len(all_llm_calls)
            if all_llm_calls else 0.0
        )

        return RunProfile(
            total_duration_ms=total_duration_ms,
            total_llm_time_ms=total_llm_time_ms,
            n_samples=n_samples,
            n_successful=n_successful,
            n_failed=n_failed,
            n_llm_calls=n_llm_calls,
            n_retries=n_retries,
            n_targeted_retries=n_targeted_retries,
            avg_sample_time_ms=avg_sample_time_ms,
            avg_llm_latency_ms=avg_llm_latency_ms,
            samples=samples,
        )
