"""SQLAlchemy base configuration with naming conventions and common mixins."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import MetaData, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Naming conventions for constraints (required for Alembic migrations)
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Mixin that adds created_at and updated_at columns."""

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# Register event listener for all models with TimestampMixin
@event.listens_for(Base, "before_update", propagate=True)
def receive_before_update(mapper: Any, connection: Any, target: Any) -> None:
    """Update updated_at timestamp before any update."""
    if hasattr(target, "updated_at"):
        target.updated_at = datetime.now(timezone.utc)
