"""RunSample ORM model for per-sample score storage."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Double, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from conformal_relevance.db._base import Base

if TYPE_CHECKING:
    from conformal_relevance.db.models.run import Run


class RunSample(Base):
    """Per-sample scores for a scoring run.

    Does NOT inherit TimestampMixin: sample rows are immutable (no updated_at
    needed) and use a SQL-side server_default for created_at.
    """

    __tablename__ = "run_samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    sample_index: Mapped[int] = mapped_column(nullable=False)
    intent: Mapped[str | None] = mapped_column(Text, nullable=True)
    ap: Mapped[float | None] = mapped_column(Double, nullable=True)
    scores: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    labels: Mapped[list] = mapped_column(JSONB, nullable=False)
    n_sentences: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="scored")
    split: Mapped[str] = mapped_column(String(8), nullable=False, default="test")
    sub_scores: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped["Run"] = relationship("Run", back_populates="samples")
