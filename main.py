"""Strategy-driven experiment runner with database storage.

Runs intent-based relevance scoring experiments using the ICL strategy
framework and optionally persists results to the database.

Usage:
    uv run python main.py --dataset subsume --n-samples 5
    uv run python main.py --strategy dpp
    uv run python main.py --provider openai --model gpt-4o-mini --no-db
    uv run python main.py --list-strategies
    uv run python main.py --no-icl --dataset puma --n-samples 5
"""

import argparse
import logging
import sys

import polars as pl
from pydantic_settings import BaseSettings, SettingsConfigDict

from conformal_relevance import (
    compute_average_precision,
    score_dataframe,
    score_dataframe_no_icl,
    summarize_results,
    DEFAULT_N_ICL,
    DEFAULT_SEED,
    ICLResult,
    LocalICLResult,
    select_icl_examples,
    select_local_icl_examples,
    get_local_strategy,
    is_local_strategy,
)
from conformal_relevance.scoring.fusion import (
    score_dataframe_moe,
    DEFAULT_MOE_K,
    resolve_moe_strategies,
)
from conformal_relevance.icl.registry import DATASET_REGISTRY, get_dataset_config
from conformal_relevance.icl.strategies import StrategyRegistry
from conformal_relevance.icl.strategies.base import StratifiedICLSelectionResult
from conformal_relevance.scoring.pipeline import DEFAULT_MAX_CONCURRENCY

# Default provider/model for CLI
DEFAULT_PROVIDER = "google"
DEFAULT_MODEL = "gemini-2.5-flash-lite"

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    supabase_database_url: str | None = None

    # Ollama — no API key required; configure connection and inference params via .env
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int | None = None
    ollama_timeout: int = 300

    # Individual DB connection fields (used if supabase_database_url is not set)
    db_user: str | None = None
    db_password: str | None = None
    db_host: str | None = None
    db_port: str = "5432"
    db_dbname: str = "postgres"

    @property
    def database_url(self) -> str | None:
        """Resolve database URL from either full URL or individual fields."""
        if self.supabase_database_url:
            return self.supabase_database_url
        if self.db_user and self.db_password and self.db_host:
            from urllib.parse import quote_plus

            password = quote_plus(self.db_password)
            return f"postgresql+psycopg2://{self.db_user}:{password}@{self.db_host}:{self.db_port}/{self.db_dbname}"
        return None


class ExperimentResult(dict):
    """Experiment results that behave as a dict for backward compatibility.

    Existing callers use this as ``dict[str, float]`` (MAP values keyed by
    score column name).  New callers can also access the scored DataFrames
    via ``.df_scored``, ``.cal_scored``, and ``.output_col``.
    """

    def __init__(self, metrics: dict[str, float], **kwargs):
        super().__init__(metrics)
        self.df_scored: pl.DataFrame | None = kwargs.get("df_scored")
        self.cal_scored: pl.DataFrame | None = kwargs.get("cal_scored")
        self.output_col: str | None = kwargs.get("output_col")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


_LLM_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "google": ("google_api_key", "GOOGLE_API_KEY"),
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
}


_DEFAULT_LLM_TIMEOUT = 120  # seconds -- generous for long-doc scoring prompts


def _set_timeout_preserving_cache_key(llm, timeout_attr: str, value: float):
    """Set HTTP timeout on an LLM without changing its LangChain cache key.

    LangChain's SQLiteCache keys on ``llm._get_llm_string()`` which serializes
    all non-default Pydantic fields.  Passing timeout in the constructor changes
    the serialized representation and invalidates the entire cache.  Instead we
    construct without timeout, snapshot the cache key, set timeout, then patch
    ``_get_llm_string`` to return the original key.
    """
    original_llm_string = llm._get_llm_string()
    setattr(llm, timeout_attr, value)
    llm._get_llm_string = lambda **kwargs: original_llm_string


def _patch_ollama_cache_key(llm) -> None:
    """Patch ChatOllama's cache key to include model name and output-affecting params.

    Works around langchain-ai/langchain#25712 where ChatOllama._get_llm_string()
    omits critical fields, causing cross-model cache collisions in SQLiteCache.
    """
    import json as _json

    key_parts = {
        "provider": "ollama",
        "model": llm.model,
    }
    if getattr(llm, "num_ctx", None) is not None:
        key_parts["num_ctx"] = llm.num_ctx
    if getattr(llm, "format", None) is not None:
        key_parts["format"] = llm.format
    if getattr(llm, "temperature", None) is not None:
        key_parts["temperature"] = llm.temperature

    cache_key = f"ChatOllama\n{_json.dumps(key_parts, sort_keys=True)}"
    llm._get_llm_string = lambda **kwargs: cache_key


def _make_ollama_llm(model: str, base_url: str, num_ctx: int | None, timeout: int):
    """Instantiate a ChatOllama instance with correct cache key and timeout."""
    import httpx as _httpx
    from langchain_ollama import ChatOllama

    ollama_kwargs = {
        "format": "json",
        "base_url": base_url,
        # Disable thinking mode (Qwen3, etc.) — thinking tags break JSON parsing
        # and waste tokens.  The `reasoning` param maps to Ollama's `think` API field.
        "reasoning": False,
        # ChatOllama has no top-level timeout field; pass via httpx client kwargs
        "client_kwargs": {"timeout": _httpx.Timeout(float(timeout))},
    }
    if num_ctx is not None:
        ollama_kwargs["num_ctx"] = num_ctx

    llm = ChatOllama(model=model, **ollama_kwargs)
    _patch_ollama_cache_key(llm)
    return llm


def _make_llm(provider: str, model: str, api_key: str):
    """Instantiate the appropriate LangChain chat model."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, api_key=api_key)
        _set_timeout_preserving_cache_key(llm, "timeout", _DEFAULT_LLM_TIMEOUT)
        return llm
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
        _set_timeout_preserving_cache_key(llm, "timeout", _DEFAULT_LLM_TIMEOUT)
        return llm
    # anthropic
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=model, api_key=api_key)
    _set_timeout_preserving_cache_key(llm, "default_request_timeout", _DEFAULT_LLM_TIMEOUT)
    return llm


def create_llm(provider: str, model: str, settings: Settings):
    """Create an LLM instance.

    Args:
        provider: LLM provider ('openai', 'google', 'anthropic', 'ollama').
        model: Model name (e.g., 'gpt-4o-mini', 'gemini-2.5-flash-lite', 'qwen3:8b').
        settings: Application settings with API keys and Ollama config.

    Returns:
        A LangChain chat model instance.
    """
    if provider == "ollama":
        return _make_ollama_llm(
            model=model,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
            timeout=settings.ollama_timeout,
        )

    if provider not in _LLM_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Supported: {', '.join([*_LLM_PROVIDERS, 'ollama'])}")
    settings_key, env_var = _LLM_PROVIDERS[provider]
    api_key = getattr(settings, settings_key)
    if not api_key:
        raise ValueError(f"{env_var} not set in environment or .env file.")
    return _make_llm(provider, model, api_key)


def _print_strategy_metadata(metadata: dict) -> None:
    """Print strategy-specific metadata in a readable format."""
    if metadata.get("log_det_score") is not None:
        print(f"  Log-det score: {metadata['log_det_score']:.4f}")
    if metadata.get("distances") is not None:
        dists = metadata["distances"]
        print(f"  Centroid distances: mean={sum(dists)/len(dists):.4f}, max={max(dists):.4f}")
    if metadata.get("n_intents") is not None:
        print(f"  Number of intents: {metadata['n_intents']}")


def _save_scored_df(args: argparse.Namespace, df_scored: pl.DataFrame) -> None:
    """Save scored DataFrame to parquet if --save-scores was specified."""
    save_path = getattr(args, "save_scores", None)
    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df_scored.write_parquet(save_path)
        print(f"  Saved scored DataFrame to {save_path}")


def _store_moe_run_to_db(
        settings: Settings,
        dataset_name: str,
        has_intent: bool,
        provider: str,
        model_name: str,
        scoring_result,
        df_scored: pl.DataFrame,
        output_col: str,
        max_concurrency: int,
        run_name: str | None,
        n_icl: int,
        icl_results: dict[str, ICLResult] | None = None,
        tags: list[str] | None = None,
        store_sub_strategies: bool = True,
        cal_scored: pl.DataFrame | None = None,
        strategy_name_suffix: str | None = None,
) -> int | None:
    """Persist MoE run results to the database.

    Delegates to store_experiment_run for the main MoE run, then stores each
    sub-strategy as an independent DB record.

    Returns the MoE run ID, or None if storage was skipped/failed.
    """
    from conformal_relevance.db.run_storage import store_experiment_run, RunParams

    moe_run_id = store_experiment_run(RunParams(
        settings=settings, dataset_name=dataset_name, has_intent=has_intent,
        provider=provider, model_name=model_name, scoring_result=scoring_result,
        df_scored=df_scored, output_col=output_col, max_concurrency=max_concurrency,
        run_name=run_name, is_moe=True, n_icl=n_icl, tags=tags, cal_scored=cal_scored,
        strategy_name_suffix=strategy_name_suffix,
    ))

    if store_sub_strategies:
        meta = scoring_result.metadata or {}
        sub_scoring_results = meta.get("sub_scoring_results", {})
        sub_col_names = meta.get("sub_strategy_col_names", {})
        icl_results_map = icl_results or {}
        for strategy_name, sub_result in sub_scoring_results.items():
            sub_col = sub_col_names.get(strategy_name)
            if not sub_col or sub_col not in sub_result.df.columns:
                logger.warning(
                    "Sub-strategy col '%s' not found in df, skipping DB storage",
                    sub_col,
                )
                continue
            sub_icl = icl_results_map.get(strategy_name)
            sub_df = compute_average_precision(sub_result.df, sub_col)
            print(f"  Storing sub-strategy run: {strategy_name} ...")
            is_local = isinstance(sub_icl, LocalICLResult)
            store_experiment_run(RunParams(
                settings=settings, dataset_name=dataset_name, has_intent=has_intent,
                provider=provider, model_name=model_name, scoring_result=sub_result,
                df_scored=sub_df, output_col=sub_col, max_concurrency=max_concurrency,
                run_name=f"icl{n_icl}_{strategy_name}",
                strategy_result=None if is_local else (sub_icl.strategy_result if sub_icl else None),
                icl_examples=None if is_local else (sub_icl.icl_examples if sub_icl else None),
                display_strategy_name=strategy_name, n_icl=n_icl, tags=tags,
            ))

    return moe_run_id



def _print_run_stats(scoring_result) -> None:
    """Print duration, retries, and failed count for a scoring run."""
    n_failed = scoring_result.n_failed
    profile = scoring_result.profile
    duration_str = f"{profile.total_duration_ms / 60000:.1f}min" if profile else "N/A"
    retries = profile.n_retries if profile else 0
    print(f"\n  Duration: {duration_str} | Retries: {retries} | Failed: {n_failed}")


def _run_icl_free(
    args: argparse.Namespace,
    settings: Settings,
    llm,
    test_df: pl.DataFrame,
    cal_df: pl.DataFrame,
    multi_intent: bool,
) -> ExperimentResult:
    """Run ICL-free baseline experiment."""
    output_col = "scores_icl0"
    print("\n" + "-" * 40)
    print("Scoring samples (ICL-free baseline)...")
    print("-" * 40)

    scoring_result = score_dataframe_no_icl(
        df=test_df, llm=llm, dataset=args.dataset, output_col=output_col,
        max_concurrency=args.max_concurrency, show_progress=True, verbose=args.verbose,
    )
    df_scored = compute_average_precision(scoring_result.df, output_col)

    print(f"\n--- Scoring calibration set (all {len(cal_df)} samples, no ICL exclusion) ---")
    cal_scoring_result = score_dataframe_no_icl(
        df=cal_df, llm=llm, dataset=args.dataset, output_col=output_col,
        max_concurrency=args.max_concurrency, show_progress=True, verbose=args.verbose,
    )
    cal_scored = compute_average_precision(cal_scoring_result.df, output_col)
    print(f"  Calibration MAP: {cal_scored[f'{output_col}_ap'].drop_nulls().mean():.4f}")

    print("\n" + "-" * 40)
    print("Evaluation Results")
    print("-" * 40)
    results = summarize_results(df_scored, score_cols=[output_col])

    if not args.no_db:
        from conformal_relevance.db.run_storage import store_experiment_run, RunParams
        store_experiment_run(RunParams(
            settings=settings, dataset_name=args.dataset, has_intent=multi_intent,
            provider=args.provider, model_name=args.model,
            scoring_result=scoring_result, df_scored=df_scored, output_col=output_col,
            max_concurrency=args.max_concurrency, run_name=args.run_name,
            is_icl_free=True, tags=args.tag, cal_scored=cal_scored,
            strategy_name_suffix=getattr(args, "strategy_name_suffix", None),
        ))

    _save_scored_df(args, df_scored)
    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)
    return ExperimentResult(results, df_scored=df_scored, cal_scored=cal_scored, output_col=output_col)


def _run_moe(
    args: argparse.Namespace,
    settings: Settings,
    llm,
    test_df: pl.DataFrame,
    cal_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    multi_intent: bool,
) -> ExperimentResult:
    """Run MoE ensemble experiment."""
    moe_k = getattr(args, "moe_k", None) or DEFAULT_MOE_K
    output_col = f"scores_icl{args.n_icl}_moe"
    moe_strategies = resolve_moe_strategies(strategies=getattr(args, "moe_strategies", None), k=moe_k)
    print(f"\n  MoE ensemble: K={len(moe_strategies)}, strategies={moe_strategies}")
    print("-" * 40)
    print("Selecting ICL examples for MoE sub-strategies...")

    moe_icl_results: dict[str, ICLResult | LocalICLResult] = {}
    moe_local_strategies: dict[str, object] = {}

    for strategy_name in moe_strategies:
        if is_local_strategy(strategy_name):
            local_strategy_obj = get_local_strategy(strategy_name)
            moe_icl_results[strategy_name] = select_local_icl_examples(
                pool_df=pool_df, inference_df=test_df, strategy=local_strategy_obj,
                dataset=args.dataset, n_icl=args.n_icl, seed=args.seed,
                dataset_task=args.dataset_task, context_window=args.context_window,
                n_remote_neg_ratio=args.n_remote_neg_ratio, verbose=args.verbose,
            )
            moe_local_strategies[strategy_name] = local_strategy_obj
        else:
            strategy_obj = StrategyRegistry.get(strategy_name, seed=args.seed)
            moe_icl_results[strategy_name] = select_icl_examples(
                pool_df=pool_df, strategy=strategy_obj, dataset=args.dataset,
                n_icl=args.n_icl, seed=args.seed, dataset_task=args.dataset_task,
                context_window=args.context_window, n_remote_neg_ratio=args.n_remote_neg_ratio,
                verbose=args.verbose,
            )

    icl_by_strategy = {name: r.icl_examples for name, r in moe_icl_results.items()}
    print("-" * 40)
    print("Scoring samples (MoE fusion)...")
    print("-" * 40)

    scoring_result = score_dataframe_moe(
        df=test_df, llm=llm, icl_by_strategy=icl_by_strategy, dataset=args.dataset,
        n_icl=args.n_icl, output_col=output_col, max_concurrency=args.max_concurrency,
        show_progress=True, verbose=args.verbose, dataset_task=args.dataset_task,
    )
    df_scored = compute_average_precision(scoring_result.df, output_col)

    # Score calibration set: local sub-strategies need fresh per-sample ICL
    print("\n  Scoring calibration set (MoE)...")
    cal_icl_by_strategy: dict[str, str | dict[str, str] | list[str]] = {}
    for strategy_name in moe_strategies:
        if is_local_strategy(strategy_name) and strategy_name in moe_local_strategies:
            cal_local_result = select_local_icl_examples(
                pool_df=pool_df, inference_df=cal_df,
                strategy=moe_local_strategies[strategy_name],
                dataset=args.dataset, n_icl=args.n_icl, seed=args.seed,
                dataset_task=args.dataset_task, context_window=args.context_window,
                n_remote_neg_ratio=args.n_remote_neg_ratio, verbose=False,
            )
            cal_icl_by_strategy[strategy_name] = cal_local_result.icl_examples
        else:
            cal_icl_by_strategy[strategy_name] = moe_icl_results[strategy_name].icl_examples

    cal_scored = compute_average_precision(
        score_dataframe_moe(
            df=cal_df, llm=llm, icl_by_strategy=cal_icl_by_strategy, dataset=args.dataset,
            n_icl=args.n_icl, output_col=output_col, max_concurrency=args.max_concurrency,
            show_progress=True, verbose=args.verbose, dataset_task=args.dataset_task,
        ).df,
        output_col,
    )
    print(f"  Calibration MAP (MoE): {cal_scored[f'{output_col}_ap'].drop_nulls().mean():.4f}")

    print("\n" + "-" * 40)
    print("Evaluation Results")
    print("-" * 40)
    results = summarize_results(df_scored, score_cols=[output_col])

    meta = scoring_result.metadata or {}
    if meta.get("per_strategy_n_failed"):
        for name, n_fail in meta.get("per_strategy_n_failed", {}).items():
            if n_fail > 0:
                print(f"  Warning: {name} failed on {n_fail} samples")
    _print_run_stats(scoring_result)

    if not args.no_db:
        _store_moe_run_to_db(
            settings=settings, dataset_name=args.dataset, has_intent=multi_intent,
            provider=args.provider, model_name=args.model, scoring_result=scoring_result,
            df_scored=df_scored, output_col=output_col, max_concurrency=args.max_concurrency,
            run_name=args.run_name, n_icl=args.n_icl, icl_results=moe_icl_results,
            tags=args.tag, store_sub_strategies=args.store_sub_strategies, cal_scored=cal_scored,
            strategy_name_suffix=getattr(args, "strategy_name_suffix", None),
        )

    _save_scored_df(args, df_scored)
    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)
    return ExperimentResult(results, df_scored=df_scored, cal_scored=cal_scored, output_col=output_col)


def _run_local_strategy(
    args: argparse.Namespace,
    settings: Settings,
    llm,
    test_df: pl.DataFrame,
    cal_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    multi_intent: bool,
) -> ExperimentResult:
    """Run local (per-sample) ICL strategy experiment."""
    print(f"\nSelecting per-sample ICL examples: strategy={args.strategy}"
          f", n={args.n_icl}, seed={args.seed}")
    local_strategy = get_local_strategy(args.strategy)
    local_result = select_local_icl_examples(
        pool_df=pool_df, inference_df=test_df, strategy=local_strategy,
        dataset=args.dataset, n_icl=args.n_icl, seed=args.seed,
        dataset_task=args.dataset_task, context_window=args.context_window,
        n_remote_neg_ratio=args.n_remote_neg_ratio, verbose=args.verbose,
    )
    output_col = local_result.output_col

    print("\n" + "-" * 40)
    print("Scoring samples (local ICL)...")
    print("-" * 40)
    scoring_result = score_dataframe(
        df=test_df, llm=llm, output_col=output_col, icl_examples=local_result.icl_examples,
        dataset=args.dataset, dataset_task=args.dataset_task,
        max_concurrency=args.max_concurrency, show_progress=True, verbose=args.verbose,
    )
    df_scored = compute_average_precision(scoring_result.df, output_col)

    print(f"\n--- Scoring calibration set (all {len(cal_df)} samples, local ICL) ---")
    cal_local_result = select_local_icl_examples(
        pool_df=pool_df, inference_df=cal_df, strategy=local_strategy,
        dataset=args.dataset, n_icl=args.n_icl, seed=args.seed,
        dataset_task=args.dataset_task, context_window=args.context_window,
        n_remote_neg_ratio=args.n_remote_neg_ratio, verbose=args.verbose,
    )
    cal_scored = compute_average_precision(
        score_dataframe(
            df=cal_df, llm=llm, output_col=output_col, icl_examples=cal_local_result.icl_examples,
            dataset=args.dataset, dataset_task=args.dataset_task,
            max_concurrency=args.max_concurrency, show_progress=True, verbose=args.verbose,
        ).df,
        output_col,
    )
    print(f"  Calibration MAP: {cal_scored[f'{output_col}_ap'].drop_nulls().mean():.4f}")

    print("\n" + "-" * 40)
    print("Evaluation Results")
    print("-" * 40)
    results = summarize_results(df_scored, score_cols=[output_col])
    _print_run_stats(scoring_result)

    if not args.no_db:
        from conformal_relevance.db.run_storage import store_experiment_run, RunParams
        store_experiment_run(RunParams(
            settings=settings, dataset_name=args.dataset, has_intent=multi_intent,
            provider=args.provider, model_name=args.model,
            scoring_result=scoring_result, df_scored=df_scored, output_col=output_col,
            max_concurrency=args.max_concurrency, run_name=args.run_name,
            tags=args.tag, display_strategy_name=args.strategy, cal_scored=cal_scored,
        ))

    _save_scored_df(args, df_scored)
    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)
    return ExperimentResult(results, df_scored=df_scored, cal_scored=cal_scored, output_col=output_col)


def _run_strategy(
    args: argparse.Namespace,
    settings: Settings,
    llm,
    test_df: pl.DataFrame,
    cal_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    multi_intent: bool,
) -> ExperimentResult:
    """Run global ICL strategy experiment."""
    resolved_strategy = StrategyRegistry.resolve_strategy_name(args.strategy, multi_intent=multi_intent)
    print(f"\nSelecting ICL examples: strategy={args.strategy}"
          f"{' (-> ' + resolved_strategy + ')' if resolved_strategy != args.strategy else ''}"
          f", n={args.n_icl}, seed={args.seed}")
    strategy = StrategyRegistry.get(resolved_strategy, seed=args.seed)

    icl_result = select_icl_examples(
        pool_df=pool_df, strategy=strategy, dataset=args.dataset, n_icl=args.n_icl,
        seed=args.seed, dataset_task=args.dataset_task, context_window=args.context_window,
        n_remote_neg_ratio=args.n_remote_neg_ratio, verbose=args.verbose,
    )
    output_col = icl_result.output_col
    strategy_result = icl_result.strategy_result

    if strategy_result is not None:
        if isinstance(strategy_result, StratifiedICLSelectionResult):
            print(f"\n  Strategy: {strategy_result.strategy_name}")
            print(f"  Intents: {strategy_result.intents}")
            for intent_val, intent_result in strategy_result.by_intent.items():
                print(f"    {intent_val}: indices={intent_result.sample_indices}")
        else:
            print(f"\n  Strategy: {strategy_result.strategy_name}")
            print(f"  Selected indices: {strategy_result.sample_indices}")
        _print_strategy_metadata(strategy_result.metadata)

    print("\n" + "-" * 40)
    print("Scoring samples...")
    print("-" * 40)
    scoring_result = score_dataframe(
        df=test_df, llm=llm, output_col=output_col, icl_examples=icl_result.icl_examples,
        dataset=args.dataset, dataset_task=args.dataset_task,
        max_concurrency=args.max_concurrency, show_progress=True, verbose=args.verbose,
    )
    df_scored = compute_average_precision(scoring_result.df, output_col)

    print(f"\n--- Scoring calibration set (all {len(cal_df)} samples) ---")
    cal_scored = compute_average_precision(
        score_dataframe(
            df=cal_df, llm=llm, icl_examples=icl_result.icl_examples, dataset=args.dataset,
            output_col=output_col, dataset_task=args.dataset_task,
            max_concurrency=args.max_concurrency, show_progress=True, verbose=args.verbose,
        ).df,
        output_col,
    )
    print(f"  Calibration MAP: {cal_scored[f'{output_col}_ap'].drop_nulls().mean():.4f}")

    print("\n" + "-" * 40)
    print("Evaluation Results")
    print("-" * 40)
    results = summarize_results(df_scored, score_cols=[output_col])
    _print_run_stats(scoring_result)

    if not args.no_db:
        from conformal_relevance.db.run_storage import store_experiment_run, RunParams
        store_experiment_run(RunParams(
            settings=settings, dataset_name=args.dataset, has_intent=multi_intent,
            provider=args.provider, model_name=args.model,
            strategy_result=icl_result.strategy_result, scoring_result=scoring_result,
            df_scored=df_scored, output_col=output_col, max_concurrency=args.max_concurrency,
            run_name=args.run_name, icl_examples=icl_result.icl_examples,
            tags=args.tag, display_strategy_name=args.strategy, cal_scored=cal_scored,
        ))

    _save_scored_df(args, df_scored)

    print("\n" + "-" * 40)
    print("Sample Output (first row)")
    print("-" * 40)
    first_row = df_scored.row(0, named=True)
    print(f"Intent: {first_row.get('intent', 'N/A')}")
    print(f"Sentences: {len(first_row['input_units'])} total")
    scores = first_row.get(output_col)
    if scores:
        print(f"Scores (first 5): {[f'{s:.2f}' for s in scores[:5]]}")
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        print("\nTop 3 most relevant sentences:")
        for rank, idx in enumerate(sorted_indices[:3], 1):
            sent = first_row["input_units"][idx]
            print(f"  {rank}. [{scores[idx]:.2f}] {sent[:80]}...")

    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)
    return ExperimentResult(results, df_scored=df_scored, cal_scored=cal_scored, output_col=output_col)


def run_experiment(
    args: argparse.Namespace,
    settings: Settings,
    test_df: pl.DataFrame | None = None,
    cal_df: pl.DataFrame | None = None,
) -> ExperimentResult:
    """Run a scoring experiment.

    Loads data, creates the LLM, then dispatches to the appropriate mode function:
    - ICL-free baseline (_run_icl_free)
    - MoE ensemble (_run_moe)
    - Local per-sample ICL strategy (_run_local_strategy)
    - Global ICL strategy (_run_strategy)

    Args:
        test_df: Optional pre-split test DataFrame. If None, loads from parquet.
        cal_df: Optional pre-split calibration DataFrame. If None, loads from parquet.
    """
    config = get_dataset_config(args.dataset)
    multi_intent = bool(config["multi_intent"])

    def _load(path: str, label: str) -> pl.DataFrame:
        print(f"\nLoading {label}: {path}")
        try:
            return pl.read_parquet(path)
        except FileNotFoundError:
            print(f"Error: {label.capitalize()} not found at {path}")
            print("Run 'python -m conformal_relevance.data_loading.split' first.")
            sys.exit(1)

    pool_df = _load(str(config["pool_path"]), "pool data")
    if cal_df is None:
        cal_df = _load(str(config["cal_path"]), "calibration data")
    else:
        print(f"\nUsing provided calibration data ({cal_df.shape[0]} samples)")
    if test_df is None:
        test_df = _load(str(config["test_path"]), "test data")
    else:
        print(f"\nUsing provided test data ({test_df.shape[0]} samples)")

    if args.n_samples > 0 and args.n_samples < test_df.shape[0]:
        test_df = test_df.head(args.n_samples)
        print(f"Using {args.n_samples} test samples")
    else:
        print(f"Using all {test_df.shape[0]} test samples")

    print(f"\nInitializing LLM: {args.provider}/{args.model}")
    try:
        llm = create_llm(args.provider, args.model, settings)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\nError: Missing dependency. {e}")
        print("Install the required package with: uv add <package-name>")
        sys.exit(1)

    if args.no_icl:
        return _run_icl_free(args, settings, llm, test_df, cal_df, multi_intent)
    if args.strategy == "moe":
        return _run_moe(args, settings, llm, test_df, cal_df, pool_df, multi_intent)
    if is_local_strategy(args.strategy):
        return _run_local_strategy(args, settings, llm, test_df, cal_df, pool_df, multi_intent)
    return _run_strategy(args, settings, llm, test_df, cal_df, pool_df, multi_intent)


def list_strategies() -> None:
    """Print available ICL strategies and exit."""
    from conformal_relevance.icl.strategies.registry import LOCAL_STRATEGY_REGISTRY
    print("Available ICL Strategies:")
    print("-" * 60)
    print("  Global strategies (shared ICL block for all samples):")
    for name in StrategyRegistry.list_strategies():
        info = StrategyRegistry.get_strategy_info(name)
        desc = info.get("description", "")
        print(f"    {name:<25} {info['class']:<30} {desc}")
    print("  Local strategies (per-sample ICL selection):")
    for name, cls in LOCAL_STRATEGY_REGISTRY.items():
        desc = (cls.__doc__ or "").split("\n")[0].strip()
        print(f"    {name:<25} {cls.__name__:<30} {desc}")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run relevance scoring experiments with ICL strategy framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py --dataset subsume --n-samples 5
  uv run python main.py --strategy dpp
  uv run python main.py --provider anthropic --model claude-3-5-sonnet --strategy rep
  uv run python main.py --provider openai --model gpt-4o-mini --no-db
  uv run python main.py --no-icl --dataset puma --n-samples 5
  uv run python main.py --list-strategies
  uv run python main.py --provider ollama --model qwen3:8b --dataset ectsum --n-samples 5
  uv run python main.py --provider ollama --model qwen3:8b --dataset subsume --ollama-num-ctx 8192

Environment Variables (or set in .env):
  OPENAI_API_KEY         Required for openai provider
  GOOGLE_API_KEY         Required for google provider
  ANTHROPIC_API_KEY      Required for anthropic provider
  SUPABASE_DATABASE_URL  For storing experiment results (optional)
  OLLAMA_BASE_URL        Ollama server URL (default: http://localhost:11434)
  OLLAMA_NUM_CTX         Ollama context window size in tokens (optional)
  OLLAMA_TIMEOUT         Ollama request timeout in seconds (default: 300)
        """,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "google", "anthropic", "ollama"],
        default=DEFAULT_PROVIDER,
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama server base URL. Overrides OLLAMA_BASE_URL from .env (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=None,
        help="Ollama context window size in tokens. Overrides OLLAMA_NUM_CTX from .env. "
             "Increase for long-document datasets (e.g., 8192 for SubSumE).",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=None,
        help="Ollama request timeout in seconds. Overrides OLLAMA_TIMEOUT from .env (default: 300).",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="subsume",
        help="Dataset to use (default: subsume)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        help="ICL strategy name (default: random). Use --list-strategies to see options.",
    )
    parser.add_argument(
        "--n-icl",
        type=int,
        default=DEFAULT_N_ICL,
        help=f"Number of ICL examples to select (default: {DEFAULT_N_ICL})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=0,
        help="Number of test samples to score, 0 for all (default: 0)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Max concurrent LLM requests (default: {DEFAULT_MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for strategy (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Windowed ICL: max context window per positive (asymmetric U(0,W) each side). None=disabled.",
    )
    parser.add_argument(
        "--n-remote-neg-ratio",
        type=float,
        default=0.5,
        help="Ratio of remote negatives per positive in windowed ICL examples (default: 0.5)",
    )
    parser.add_argument(
        "--dataset-task",
        type=str,
        default=None,
        dest="dataset_task",
        help=(
            "Uniform task description for all samples (e.g., 'PHI detection', 'Question answering'). "
            "Appears as 'Task: ...' in every ICL example and inference prompt. Overrides the registry "
            "default. Separate from intent (per-sample category in multi-intent datasets). "
            "Pass empty string to suppress the registry default."
        ),
    )
    parser.add_argument("--no-db", action="store_true", help="Skip database storage")
    parser.add_argument(
        "--no-icl",
        action="store_true",
        help="Run ICL-free baseline using score_dataframe_no_icl()",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional name for the run")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print available strategies and exit",
    )
    parser.add_argument(
        "--tag",
        type=str,
        action="append",
        default=None,
        help="Tag for the run (repeatable, e.g. --tag baseline --tag v2)",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Delete all runs from the database and exit",
    )
    parser.add_argument(
        "--save-scores",
        type=str,
        default=None,
        metavar="PATH",
        help="Save scored DataFrame to parquet file after scoring",
    )
    parser.add_argument(
        "--moe-k",
        type=int,
        choices=[2, 3, 4, 5, 6],
        default=None,
        help="MoE ensemble size (2–6). Only used when --strategy=moe. Default: 4 "
             "(random+anchor_dpp+pattern_dpp+bm25).",
    )
    parser.add_argument(
        "--moe-strategies",
        type=str,
        default=None,
        help="Comma-separated sub-strategies for MoE (e.g., 'bm25,anchor_dpp'). "
             "Overrides --moe-k. Must be 2–6 strategies.",
    )
    parser.add_argument(
        "--no-store-sub-strategies",
        dest="store_sub_strategies",
        action="store_false",
        default=True,
        help="When using --strategy moe, skip storing sub-strategy runs as separate DB runs.",
    )
    return parser


def _handle_early_exits(args: argparse.Namespace) -> None:
    """Handle --list-strategies and --clear-db flags, exiting if triggered."""
    if args.list_strategies:
        list_strategies()
        sys.exit(0)

    if args.clear_db:
        settings = Settings()
        db_url = settings.database_url
        if not db_url:
            print("Error: No database URL configured.")
            sys.exit(1)
        from conformal_relevance.db import get_engine, get_session
        from conformal_relevance.db.operations import delete_all_runs
        engine = get_engine(database_url=db_url)
        with get_session(engine) as session:
            count = delete_all_runs(session, include_datasets=True)
        print(f"Deleted {count} run(s) and all associated data.")
        sys.exit(0)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate strategy name, DPP constraints, and MoE-k range. Exits on error."""
    if not args.no_icl and args.strategy != "moe":
        available = StrategyRegistry.list_strategies()
        if not is_local_strategy(args.strategy) and args.strategy not in available:
            print(f"Error: Unknown strategy '{args.strategy}'")
            print(f"Available global: {available}")
            print(f"Available local: bm25, knn")
            sys.exit(1)
        if args.strategy == "dpp" and args.n_icl < 2:
            print(f"Error: DPP strategies require at least 2 ICL examples (got {args.n_icl}).")
            print("Diversity is meaningless with a single sample. Use 'random' or 'rep' instead.")
            sys.exit(1)

    if args.strategy == "moe":
        moe_k = getattr(args, "moe_k", None)
        if moe_k is not None and not (2 <= moe_k <= 6):
            print(f"Error: --moe-k must be 2–6 (got {moe_k})")
            sys.exit(1)


def _print_experiment_banner(args: argparse.Namespace, settings: Settings) -> None:
    """Print the experiment configuration banner."""
    print("=" * 60)
    print("RELEVANCY SCORING EXPERIMENT")
    print("=" * 60)
    print(f"  Provider:    {args.provider}")
    print(f"  Model:       {args.model}")
    if args.provider == "ollama":
        print(f"  Ollama URL:  {settings.ollama_base_url}")
        if settings.ollama_num_ctx:
            print(f"  Num CTX:     {settings.ollama_num_ctx}")
    print(f"  Dataset:     {args.dataset}")
    if args.no_icl:
        print(f"  Mode:        ICL-free baseline")
    else:
        print(f"  Strategy:    {args.strategy}")
        print(f"  N ICL:       {args.n_icl}")
        if args.strategy == "moe":
            moe_k = getattr(args, "moe_k", None) or DEFAULT_MOE_K
            print(f"  MoE K:       {moe_k}")
        if args.context_window is not None:
            print(f"  Windowed:    context_window={args.context_window}, "
                  f"n_remote_neg_ratio={args.n_remote_neg_ratio}")
    if args.dataset_task:
        task_preview = args.dataset_task[:60] + ("..." if len(args.dataset_task) > 60 else "")
        print(f"  Task:        {task_preview}")
    print(f"  N Samples:   {args.n_samples if args.n_samples > 0 else 'all'}")
    print(f"  Concurrency: {args.max_concurrency}")
    print(f"  Seed:        {args.seed}")
    print(f"  DB Storage:  {'disabled' if args.no_db else 'enabled'}")
    if args.run_name:
        print(f"  Run Name:    {args.run_name}")
    if args.tag:
        print(f"  Tags:        {', '.join(args.tag)}")
    if getattr(args, "save_scores", None):
        print(f"  Save to:     {args.save_scores}")


def main() -> None:
    """Main entry point."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    if args.moe_strategies:
        args.moe_strategies = args.moe_strategies.split(",")

    _handle_early_exits(args)
    setup_logging(args.verbose)
    settings = Settings()
    _validate_args(args)

    # Merge CLI overrides into settings before banner (so banner shows effective values)
    if args.provider == "ollama":
        if getattr(args, "ollama_base_url", None) is not None:
            settings.ollama_base_url = args.ollama_base_url
        if getattr(args, "ollama_num_ctx", None) is not None:
            settings.ollama_num_ctx = args.ollama_num_ctx
        if getattr(args, "ollama_timeout", None) is not None:
            settings.ollama_timeout = args.ollama_timeout

    # Ollama: default to 1 concurrent request (local GPU constraint)
    if args.provider == "ollama" and args.max_concurrency == DEFAULT_MAX_CONCURRENCY:
        args.max_concurrency = 1
        logger.info("Ollama provider: defaulting max_concurrency=1 (override with --max-concurrency)")

    _print_experiment_banner(args, settings)

    try:
        run_experiment(args, settings)
    except Exception as e:
        logger.exception("Experiment failed")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
