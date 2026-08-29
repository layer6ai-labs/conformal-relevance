"""LLM response caching utilities."""

import logging
import threading
from pathlib import Path

from conformal_relevance.types import DEFAULT_CACHE_PATH

logger = logging.getLogger(__name__)

# Serializes SQLiteCache init so concurrent worker threads (e.g. multi-GPU
# Ollama sharding) can't race SQLAlchemy's non-atomic CREATE TABLE IF NOT
# EXISTS and raise `sqlite3.OperationalError: table full_llm_cache already exists`.
_CACHE_INIT_LOCK = threading.Lock()


def enable_llm_cache(cache_path: str | Path = DEFAULT_CACHE_PATH) -> None:
    """Enable LangChain LLM caching with SQLite backend."""
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    cache_path = Path(cache_path)
    with _CACHE_INIT_LOCK:
        set_llm_cache(SQLiteCache(database_path=str(cache_path)))
    logger.info(f"LLM cache enabled: {cache_path}")


