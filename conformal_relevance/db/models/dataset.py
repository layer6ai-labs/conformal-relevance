"""Dataset ORM model."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from conformal_relevance.db._base import Base, TimestampMixin

if TYPE_CHECKING:
    from conformal_relevance.db.models.run import Run


class Dataset(Base, TimestampMixin):
    """Dataset metadata table."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(256))
    has_intent: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB)

    # Relationships
    runs: Mapped[list["Run"]] = relationship(
        "Run", back_populates="dataset", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_datasets_name", "name"),)

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name='{self.name}')>"
