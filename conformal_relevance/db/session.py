"""Database session management and connection handling."""

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from conformal_relevance.db._base import Base

# Global engine instance (lazy initialization)
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_database_url() -> str:
    """Get database URL from environment variables.

    Supports two formats:
    1. SUPABASE_DATABASE_URL: Full connection string
    2. Individual DB_* variables: DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DBNAME

    Returns:
        Database connection URL.

    Raises:
        ValueError: If neither SUPABASE_DATABASE_URL nor DB_* variables are set.
    """
    from urllib.parse import quote_plus

    # First try the full URL
    url = os.environ.get("SUPABASE_DATABASE_URL")
    if url:
        return url

    # Try to construct from individual variables
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_DBNAME", "postgres")

    if db_user and db_password and db_host:
        # URL-encode the password to handle special characters
        encoded_password = quote_plus(db_password)
        url = f"postgresql+psycopg2://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        return url

    raise ValueError(
        "Database connection not configured. Set either:\n"
        "  - SUPABASE_DATABASE_URL environment variable, or\n"
        "  - Individual DB_USER, DB_PASSWORD, DB_HOST (and optionally DB_PORT, DB_DBNAME)"
    )


def get_engine(database_url: str | None = None, **kwargs) -> Engine:
    """Get or create the database engine.

    Args:
        database_url: Optional database URL. If not provided, uses SUPABASE_DATABASE_URL.
        **kwargs: Additional arguments passed to create_engine.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine

    if _engine is not None and database_url is None:
        return _engine

    url = database_url or get_database_url()

    # Default engine configuration
    engine_kwargs = {
        "pool_pre_ping": True,  # Verify connections before use
        "pool_size": 5,
        "max_overflow": 10,
        "echo": False,  # Set to True for SQL debugging
    }
    engine_kwargs.update(kwargs)

    engine = create_engine(url, **engine_kwargs)

    if database_url is None:
        _engine = engine

    return engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Get or create the session factory.

    Args:
        engine: Optional engine instance. If not provided, uses default engine.

    Returns:
        Session factory.
    """
    global _SessionLocal

    if _SessionLocal is not None and engine is None:
        return _SessionLocal

    eng = engine or get_engine()
    factory = sessionmaker(bind=eng, autocommit=False, autoflush=False)

    if engine is None:
        _SessionLocal = factory

    return factory


@contextmanager
def get_session(
    engine: Engine | None = None,
) -> Generator[Session, None, None]:
    """Context manager for database sessions.

    Automatically commits on success and rolls back on exception.

    Args:
        engine: Optional engine instance. If not provided, uses default engine.

    Yields:
        Database session.

    Example:
        with get_session() as session:
            dataset = get_or_create_dataset(session, "subsume")
            session.commit()  # Optional - auto-commits on exit
    """
    factory = get_session_factory(engine)
    session = factory()
    try:
        # Ensure we start with a fresh DB snapshot (important when the
        # connection is reused from the pool and a prior transaction on
        # another session committed changes we need to see).
        session.expire_all()
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database(engine: Engine | None = None, drop_existing: bool = False) -> None:
    """Initialize database tables.

    Creates all tables defined in the ORM models. This is primarily for
    development and testing. In production, use Alembic migrations.

    Args:
        engine: Optional engine instance. If not provided, uses default engine.
        drop_existing: If True, drops all existing tables first.

    Warning:
        Setting drop_existing=True will DELETE ALL DATA. Use with caution.
    """
    eng = engine or get_engine()

    # Import models to ensure they're registered with Base
    from conformal_relevance.db import models  # noqa: F401

    if drop_existing:
        Base.metadata.drop_all(eng)

    Base.metadata.create_all(eng)


def reset_connection() -> None:
    """Reset global engine and session factory.

    Useful for testing or when changing database connections.
    """
    global _engine, _SessionLocal

    if _engine is not None:
        _engine.dispose()
        _engine = None

    _SessionLocal = None
