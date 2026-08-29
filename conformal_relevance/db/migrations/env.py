"""Alembic environment configuration."""

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# Load environment variables
load_dotenv()

# Import models to register them with Base.metadata
from conformal_relevance.db._base import Base
from conformal_relevance.db import models  # noqa: F401

# Alembic Config object
config = context.config

# Set database URL from environment
# Supports SUPABASE_DATABASE_URL or individual DB_* variables
from urllib.parse import quote_plus

database_url = os.environ.get("SUPABASE_DATABASE_URL")
if not database_url:
    # Try to construct from individual variables
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_DBNAME", "postgres")
    if db_user and db_password and db_host:
        # URL-encode the password to handle special characters
        encoded_password = quote_plus(db_password)
        database_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"

if database_url:
    # Escape % characters for ConfigParser interpolation
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

# Setup Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a
    connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
