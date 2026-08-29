"""Run ORM model for scoring experiment runs."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ARRAY, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from conformal_relevance.db._base import Base, TimestampMixin

if TYPE_CHECKING:
    from conformal_relevance.db.models.dataset import Dataset
    from conformal_relevance.db.models.llm_model import LLMModel
    from conformal_relevance.db.models.run_sample import RunSample


class Run(Base, TimestampMixin):
    """Scoring experiment runs."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(256))
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    llm_model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("llm_models.id", ondelete="CASCADE"), nullable=False
    )
    split: Mapped[str] = mapped_column(String(32), default="test")
    max_concurrency: Mapped[Optional[int]] = mapped_column(Integer)
    max_retries: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    started_at: Mapped[Optional[datetime]] = mapped_column()
    completed_at: Mapped[Optional[datetime]] = mapped_column()
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB)

    # Profiling columns
    total_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    total_llm_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    avg_sample_time_ms: Mapped[Optional[float]] = mapped_column(Float)

    # ICL strategy columns
    icl_strategy_name: Mapped[Optional[str]] = mapped_column(String(64))
    icl_strategy_params: Mapped[Optional[dict]] = mapped_column(JSONB)
    icl_permutation_id: Mapped[Optional[int]] = mapped_column(Integer)
    icl_diversity_metrics: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Run-level aggregate columns
    mean_ap: Mapped[Optional[float]] = mapped_column(Float)
    std_ap: Mapped[Optional[float]] = mapped_column(Float)
    n_samples_scored: Mapped[Optional[int]] = mapped_column(Integer)
    n_samples_failed: Mapped[Optional[int]] = mapped_column(Integer)
    n_retries_total: Mapped[Optional[int]] = mapped_column(Integer)
    error_rate: Mapped[Optional[float]] = mapped_column(Float)
    ap_percentiles: Mapped[Optional[dict]] = mapped_column(JSONB)
    intent_metrics: Mapped[Optional[dict]] = mapped_column(JSONB)
    prompt_content: Mapped[Optional[dict]] = mapped_column(JSONB)
    error_summary: Mapped[Optional[dict]] = mapped_column(JSONB)
    tags: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String(64)))

    # Relationships
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="runs")
    llm_model: Mapped["LLMModel"] = relationship("LLMModel", back_populates="runs")
    samples: Mapped[list["RunSample"]] = relationship(
        "RunSample", back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_runs_dataset_model", "dataset_id", "llm_model_id"),
        Index("ix_runs_status", "status", "created_at"),
        Index("ix_runs_strategy", "icl_strategy_name", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Run(id={self.id}, status='{self.status}', dataset_id={self.dataset_id})>"

    def start(self) -> None:
        """Mark the run as started."""
        self.status = "running"
        self.started_at = datetime.now(timezone.utc)

    def complete(self) -> None:
        """Mark the run as completed."""
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc)

    def fail(self, error_message: Optional[str] = None) -> None:
        """Mark the run as failed."""
        self.status = "failed"
        self.completed_at = datetime.now(timezone.utc)
        if error_message:
            self.error_message = error_message

    def set_profiling(
        self,
        total_duration_ms: int,
        total_llm_time_ms: int,
        avg_sample_time_ms: float,
    ) -> None:
        """Set profiling metrics."""
        self.total_duration_ms = total_duration_ms
        self.total_llm_time_ms = total_llm_time_ms
        self.avg_sample_time_ms = avg_sample_time_ms

    def set_icl_strategy(
        self,
        strategy_name: str,
        strategy_params: Optional[dict] = None,
        permutation_id: Optional[int] = None,
        strategy_metrics: Optional[dict] = None,
    ) -> None:
        """Set ICL strategy information.

        Args:
            strategy_name: Name of the ICL strategy.
            strategy_params: Strategy configuration parameters.
            permutation_id: Permutation ID for permutation testing strategies.
            strategy_metrics: Strategy-specific quality metrics (stored in
                icl_diversity_metrics column for backwards compatibility).
                For DPP: {"log_det_score": float}
                For representative: {"distances": list[float]}
                For stratified: {"per_intent_log_det": {...}} or {"per_intent_distances": {...}}
        """
        self.icl_strategy_name = strategy_name
        self.icl_strategy_params = strategy_params
        self.icl_permutation_id = permutation_id
        self.icl_diversity_metrics = strategy_metrics

    def set_aggregates(
        self,
        mean_ap: Optional[float] = None,
        std_ap: Optional[float] = None,
        n_samples_scored: Optional[int] = None,
        n_samples_failed: Optional[int] = None,
        n_retries_total: Optional[int] = None,
        error_rate: Optional[float] = None,
        ap_percentiles: Optional[dict] = None,
        intent_metrics: Optional[dict] = None,
        prompt_content: Optional[dict] = None,
        error_summary: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        """Set all run-level aggregate columns at once.

        Args:
            mean_ap: Mean Average Precision across scored samples.
            std_ap: Standard deviation of Average Precision.
            n_samples_scored: Number of samples successfully scored.
            n_samples_failed: Number of samples that failed scoring.
            n_retries_total: Total number of retries across all samples.
            error_rate: Fraction of samples that failed (n_failed / n_total).
            ap_percentiles: Percentile breakdown of AP scores,
                e.g. {"p25": 0.6, "p50": 0.75, "p75": 0.9}.
            intent_metrics: Per-intent aggregate metrics,
                e.g. {"intent_name": {"mean_ap": 0.8, "n_samples": 10}}.
            prompt_content: Prompt template and ICL content used for the run,
                e.g. {"system_prompt": "...", "icl_examples": "..."}.
            error_summary: Summary of errors encountered during the run,
                e.g. {"parse_error": 3, "rate_limit": 1}.
            tags: List of tags for categorizing/filtering runs.
        """
        self.mean_ap = mean_ap
        self.std_ap = std_ap
        self.n_samples_scored = n_samples_scored
        self.n_samples_failed = n_samples_failed
        self.n_retries_total = n_retries_total
        self.error_rate = error_rate
        self.ap_percentiles = ap_percentiles
        self.intent_metrics = intent_metrics
        self.prompt_content = prompt_content
        self.error_summary = error_summary
        self.tags = tags
