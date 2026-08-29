"""LLMModel ORM model."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from conformal_relevance.db._base import Base

if TYPE_CHECKING:
    from conformal_relevance.db.models.run import Run


class LLMModel(Base):
    """LLM model registry table."""

    __tablename__ = "llm_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    config: Mapped[Optional[dict]] = mapped_column(JSONB)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    runs: Mapped[list["Run"]] = relationship(
        "Run", back_populates="llm_model", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_llm_models_name", "name"),
        Index("ix_llm_models_provider", "provider"),
    )

    def __repr__(self) -> str:
        return f"<LLMModel(id={self.id}, name='{self.name}', provider='{self.provider}')>"
