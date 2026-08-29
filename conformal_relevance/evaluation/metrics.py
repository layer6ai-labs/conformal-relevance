"""Metrics computation for evaluation framework."""

from dataclasses import dataclass, field
from typing import Any, Sequence

import polars as pl


@dataclass
class IntentMetrics:
    """Metrics for a specific intent."""

    intent: str
    n_samples: int
    mean_ap: float
    std_ap: float
    min_ap: float
    max_ap: float
    n_success: int
    n_failed: int


@dataclass
class FailedSampleInfo:
    """Information about a failed sample."""

    sample_id: int
    error_message: str | None
    retry_count: int


@dataclass
class RunReport:
    """Comprehensive report for a scoring run."""

    run_id: int
    run_name: str | None
    dataset_name: str
    model_name: str
    status: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    duration_seconds: float | None

    # Overall metrics
    n_samples: int
    n_success: int
    n_failed: int
    mean_ap: float
    std_ap: float
    median_ap: float
    min_ap: float
    max_ap: float
    ci_lower: float
    ci_upper: float

    # Profiling
    total_duration_ms: int | None
    total_llm_time_ms: int | None
    avg_sample_time_ms: float | None

    # ICL strategy
    icl_strategy_name: str | None
    icl_strategy_params: dict | None

    # Intent breakdown
    intent_metrics: list[IntentMetrics] = field(default_factory=list)

    # Failed samples
    failed_samples: list[FailedSampleInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "n_samples": self.n_samples,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "mean_ap": self.mean_ap,
            "std_ap": self.std_ap,
            "median_ap": self.median_ap,
            "min_ap": self.min_ap,
            "max_ap": self.max_ap,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "total_duration_ms": self.total_duration_ms,
            "total_llm_time_ms": self.total_llm_time_ms,
            "avg_sample_time_ms": self.avg_sample_time_ms,
            "icl_strategy_name": self.icl_strategy_name,
            "icl_strategy_params": self.icl_strategy_params,
            "intent_metrics": [
                {
                    "intent": m.intent,
                    "n_samples": m.n_samples,
                    "mean_ap": m.mean_ap,
                    "std_ap": m.std_ap,
                    "n_success": m.n_success,
                    "n_failed": m.n_failed,
                }
                for m in self.intent_metrics
            ],
            "failed_samples": [
                {
                    "sample_id": f.sample_id,
                    "error_message": f.error_message,
                    "retry_count": f.retry_count,
                }
                for f in self.failed_samples
            ],
        }


def compute_run_metrics(
    ap_values: Sequence[float],
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute metrics from a list of AP values.

    Args:
        ap_values: List of average precision scores.
        confidence: Confidence level for CI (default: 0.95).

    Returns:
        Dictionary with computed metrics.
    """
    import statistics

    if not ap_values:
        return {
            "mean_ap": float("nan"),
            "std_ap": float("nan"),
            "median_ap": float("nan"),
            "min_ap": float("nan"),
            "max_ap": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    n = len(ap_values)
    mean_ap = statistics.mean(ap_values)
    std_ap = statistics.stdev(ap_values) if n > 1 else 0.0
    median_ap = statistics.median(ap_values)
    min_ap = min(ap_values)
    max_ap = max(ap_values)

    # Compute confidence interval
    if n > 1:
        try:
            from scipy import stats
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_value * (std_ap / (n ** 0.5))
            ci_lower = mean_ap - margin
            ci_upper = mean_ap + margin
        except ImportError:
            # Fallback without scipy
            margin = 1.96 * (std_ap / (n ** 0.5))
            ci_lower = mean_ap - margin
            ci_upper = mean_ap + margin
    else:
        ci_lower = mean_ap
        ci_upper = mean_ap

    return {
        "mean_ap": mean_ap,
        "std_ap": std_ap,
        "median_ap": median_ap,
        "min_ap": min_ap,
        "max_ap": max_ap,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def compute_intent_metrics(
    df: pl.DataFrame,
    intent_col: str = "intent",
    ap_col: str = "average_precision",
    status_col: str = "status",
) -> list[IntentMetrics]:
    """Compute metrics broken down by intent.

    Args:
        df: DataFrame with sample scores.
        intent_col: Column name for intent.
        ap_col: Column name for average precision.
        status_col: Column name for status.

    Returns:
        List of IntentMetrics, one per unique intent.
    """
    if intent_col not in df.columns:
        return []

    agg = (
        df.group_by(intent_col)
        .agg([
            pl.len().alias("n_samples"),
            (pl.col(status_col) == "success").sum().alias("n_success"),
            pl.col(ap_col).drop_nulls().mean().alias("mean_ap"),
            pl.col(ap_col).drop_nulls().std().alias("std_ap"),
            pl.col(ap_col).drop_nulls().min().alias("min_ap"),
            pl.col(ap_col).drop_nulls().max().alias("max_ap"),
        ])
    )

    metrics = []
    for row in agg.iter_rows(named=True):
        mean_ap = row["mean_ap"] if row["mean_ap"] is not None else float("nan")
        std_ap = row["std_ap"] if row["std_ap"] is not None else 0.0
        min_ap = row["min_ap"] if row["min_ap"] is not None else float("nan")
        max_ap = row["max_ap"] if row["max_ap"] is not None else float("nan")
        n_samples = row["n_samples"]
        n_success = row["n_success"]
        metrics.append(IntentMetrics(
            intent=row[intent_col],
            n_samples=n_samples,
            mean_ap=mean_ap,
            std_ap=std_ap,
            min_ap=min_ap,
            max_ap=max_ap,
            n_success=n_success,
            n_failed=n_samples - n_success,
        ))

    return sorted(metrics, key=lambda m: m.mean_ap if m.mean_ap == m.mean_ap else 0, reverse=True)


def generate_run_report(
    run_data: dict[str, Any],
    scores_df: pl.DataFrame,
    include_failed_samples: bool = True,
) -> RunReport:
    """Generate a comprehensive run report.

    Args:
        run_data: Dictionary with run metadata.
        scores_df: DataFrame with sample scores (joined with samples).
        include_failed_samples: Whether to include failed sample info.

    Returns:
        RunReport instance.
    """
    # Extract AP values
    ap_values = [
        v for v in scores_df["average_precision"].to_list()
        if v is not None and not (isinstance(v, float) and v != v)
    ]

    # Compute overall metrics
    metrics = compute_run_metrics(ap_values)

    # Compute intent metrics
    intent_metrics = compute_intent_metrics(scores_df)

    # Gather failed samples
    failed_samples = []
    if include_failed_samples:
        failed_df = scores_df.filter(pl.col("status") == "failed")
        for i, row in enumerate(failed_df.iter_rows(named=True)):
            failed_samples.append(FailedSampleInfo(
                sample_id=row.get("sample_id", i),
                error_message=row.get("error_message"),
                retry_count=row.get("retry_count", 0),
            ))

    # Calculate duration
    duration_seconds = None
    if run_data.get("started_at") and run_data.get("completed_at"):
        try:
            from datetime import datetime
            started = run_data["started_at"]
            completed = run_data["completed_at"]
            if isinstance(started, str):
                started = datetime.fromisoformat(started)
            if isinstance(completed, str):
                completed = datetime.fromisoformat(completed)
            duration_seconds = (completed - started).total_seconds()
        except (ValueError, TypeError):
            pass

    n_samples = scores_df.shape[0]
    n_success = scores_df.filter(pl.col("status") == "success").shape[0]
    n_failed = n_samples - n_success

    return RunReport(
        run_id=run_data.get("run_id", 0),
        run_name=run_data.get("run_name"),
        dataset_name=run_data.get("dataset_name", "unknown"),
        model_name=run_data.get("model_name", "unknown"),
        status=run_data.get("status", "unknown"),
        created_at=str(run_data.get("created_at", "")),
        started_at=str(run_data.get("started_at", "")) if run_data.get("started_at") else None,
        completed_at=str(run_data.get("completed_at", "")) if run_data.get("completed_at") else None,
        duration_seconds=duration_seconds,
        n_samples=n_samples,
        n_success=n_success,
        n_failed=n_failed,
        mean_ap=metrics["mean_ap"],
        std_ap=metrics["std_ap"],
        median_ap=metrics["median_ap"],
        min_ap=metrics["min_ap"],
        max_ap=metrics["max_ap"],
        ci_lower=metrics["ci_lower"],
        ci_upper=metrics["ci_upper"],
        total_duration_ms=run_data.get("total_duration_ms"),
        total_llm_time_ms=run_data.get("total_llm_time_ms"),
        avg_sample_time_ms=run_data.get("avg_sample_time_ms"),
        icl_strategy_name=run_data.get("icl_strategy_name"),
        icl_strategy_params=run_data.get("icl_strategy_params"),
        intent_metrics=intent_metrics,
        failed_samples=failed_samples,
    )
