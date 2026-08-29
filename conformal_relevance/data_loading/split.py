"""
Dataset splitting utilities for conformal prediction calibration.

This module provides functions to split processed datasets into pool, calibration,
and test sets with consistent column naming across all datasets.

Three-way split design:
- Pool: ICL example selection only. Never scored by the LLM.
- Calibration: Conformal threshold selection. Scored identically to test. No exclusions.
- Test: Performance evaluation. Scored identically to calibration.

The standardized output format has columns:
- intent (optional): The query intent or perspective
- input: Full input text
- input_units: List of text units (sentences or claims)
- input_unit_labels: Binary labels for each unit
- output_text: Summary/output text
- output_units: List of output text units
"""

from pathlib import Path

import polars as pl

from conformal_relevance.types import DEFAULT_SEED

from ._common import get_project_data_dir


# =============================================================================
# Constants
# =============================================================================

# Pool set sizes
DEFAULT_POOL_SIZE = 50          # Uni-intent: 50 total, uniform random
DEFAULT_POOL_PER_INTENT = 20    # Multi-intent: 20 per intent, stratified

# Calibration set size (uniform random for ALL datasets)
DEFAULT_CAL_SIZE = 100

# HotpotQA test cap (too large otherwise)
HOTPOTQA_TEST_CAP = 10_000

# Standard column mapping for each dataset
# Maps dataset-specific column names to standardized names
COLUMN_MAPPINGS = {
    "puma": {
        "intent": "intent",
        "input": "input",
        "input_sentences": "input_units",
        "input_sentences_labels": "input_unit_labels",
        "summary": "output_text",
        "summary_sentences": "output_units",
    },
    "subsume": {
        "intent": "intent",
        "input": "input",
        "input_sentences": "input_units",
        "input_sentences_labels": "input_unit_labels",
        "summary": "output_text",
        "summary_sentences": "output_units",
    },
    "ectsum": {
        "input": "input",
        "input_sentences": "input_units",
        "input_sentences_labels": "input_unit_labels",
        "summary": "output_text",
        "summary_sentences": "output_units",
    },
    "physionet": {
        "input": "input",
        "input_claims": "input_units",
        "input_claims_labels": "input_unit_labels",
        "summary": "output_text",
        "summary_claims": "output_units",
    },
    "hotpotqa": {
        "intent": "intent",
        "input_sentences": "input_units",
        "input_sentences_labels": "input_unit_labels",
        "summary": "output_text",
        "summary_sentences": "output_units",
    },
    "evidence_inference": {
        "intent": "intent",
        "input": "input",
        "input_sentences": "input_units",
        "input_sentences_labels": "input_unit_labels",
        "summary": "output_text",
        "summary_sentences": "output_units",
    },
    "contractnli": {
        "intent": "intent",
        "input": "input",
        "input_sentences": "input_units",
        "input_sentences_labels": "input_unit_labels",
        "summary": "output_text",
        "summary_sentences": "output_units",
    },
}

# Datasets that have an intent column suitable for stratified pool sampling
# Note: HotpotQA has intent (question) but with ~90K unique questions,
# stratified sampling is not practical, so we use uniform sampling.
DATASETS_WITH_INTENT = {"puma", "subsume", "evidence_inference", "contractnli"}


# =============================================================================
# Helper Functions
# =============================================================================

def _standardize_columns(df: pl.DataFrame, dataset_name: str) -> pl.DataFrame:
    """
    Rename columns to standardized format and select only required columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with dataset-specific column names.
    dataset_name : str
        Name of the dataset (puma, subsume, ectsum, physionet, hotpotqa, evidence_inference).

    Returns
    -------
    pl.DataFrame
        DataFrame with standardized column names.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in COLUMN_MAPPINGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {list(COLUMN_MAPPINGS.keys())}"
        )

    mapping = COLUMN_MAPPINGS[dataset_name]

    # Build rename mapping for columns that exist in the DataFrame
    rename_dict = {}
    select_cols = []

    for old_name, new_name in mapping.items():
        if old_name in df.columns:
            if old_name != new_name:
                rename_dict[old_name] = new_name
            select_cols.append(new_name)

    # Apply rename if needed
    if rename_dict:
        df = df.rename(rename_dict)

    # Select only the standardized columns
    return df.select(select_cols)


def _uniform_three_way_split(
    df: pl.DataFrame,
    pool_size: int,
    cal_size: int,
    seed: int = DEFAULT_SEED,
    dataset_name: str = "",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split data uniformly into pool, calibration, and test sets.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    pool_size : int
        Number of samples for pool set.
    cal_size : int
        Number of samples for calibration set.
    seed : int
        Random seed for reproducibility.
    dataset_name : str
        Dataset name; used to apply HotpotQA test cap.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (pool_df, calibration_df, test_df) DataFrames.
    """
    # Shuffle the full dataset
    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)

    # Cap pool and cal at available size
    actual_pool = min(pool_size, df_shuffled.shape[0])
    remaining_after_pool = df_shuffled.shape[0] - actual_pool
    actual_cal = min(cal_size, remaining_after_pool)

    pool_df = df_shuffled.head(actual_pool)
    remainder = df_shuffled.tail(df_shuffled.shape[0] - actual_pool)
    cal_df = remainder.head(actual_cal)
    test_df = remainder.tail(remainder.shape[0] - actual_cal)

    # Apply HotpotQA test cap
    if dataset_name == "hotpotqa" and test_df.shape[0] > HOTPOTQA_TEST_CAP:
        test_df = test_df.head(HOTPOTQA_TEST_CAP)

    return pool_df, cal_df, test_df


def _stratified_pool_uniform_cal_split(
    df: pl.DataFrame,
    pool_per_intent: int,
    cal_size: int,
    intent_col: str = "intent",
    seed: int = DEFAULT_SEED,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split data with stratified pool and uniform calibration.

    Pool: stratified across intents (pool_per_intent samples per intent).
    Calibration: uniform random from remainder (preserves natural intent distribution).
    Test: remainder after pool and cal.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with intent column.
    pool_per_intent : int
        Number of pool samples per intent.
    cal_size : int
        Total calibration set size (uniform random from remainder).
    intent_col : str
        Name of the intent column.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (pool_df, calibration_df, test_df) DataFrames.
    """
    intents = sorted(df[intent_col].unique().to_list())

    pool_dfs = []
    remainder_dfs = []

    for i, intent in enumerate(intents):
        intent_df = df.filter(pl.col(intent_col) == intent)
        # Different seed per intent for variety (matches original approach)
        intent_df_shuffled = intent_df.sample(
            fraction=1.0, shuffle=True, seed=seed + i
        )
        n_pool = min(pool_per_intent, intent_df_shuffled.shape[0])
        pool_dfs.append(intent_df_shuffled.head(n_pool))
        remainder_dfs.append(intent_df_shuffled.tail(intent_df_shuffled.shape[0] - n_pool))

    pool_df = pl.concat(pool_dfs)
    remainder_df = pl.concat(remainder_dfs)

    # Uniform random calibration from remainder (preserves natural distribution)
    remainder_shuffled = remainder_df.sample(fraction=1.0, shuffle=True, seed=seed)
    actual_cal = min(cal_size, remainder_shuffled.shape[0])
    cal_df = remainder_shuffled.head(actual_cal)
    test_df = remainder_shuffled.tail(remainder_shuffled.shape[0] - actual_cal)

    return pool_df, cal_df, test_df



# =============================================================================
# Public API
# =============================================================================

def split_dataset(
    df: pl.DataFrame,
    dataset_name: str,
    pool_size: int | None = None,
    cal_size: int | None = None,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a dataset into pool, calibration, and test sets with standardized columns.

    Pool: used only for ICL example selection, never scored.
    Calibration: uniform random sample (100), scored identically to test.
    Test: all remaining samples after pool and cal are taken.

    For multi-intent datasets (puma, subsume, evidence_inference): pool is stratified
    (DEFAULT_POOL_PER_INTENT per intent). Calibration is always uniform random.

    For other datasets (ectsum, physionet, hotpotqa): pool is uniform random.
    HotpotQA test is capped at HOTPOTQA_TEST_CAP samples.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame (processed dataset).
    dataset_name : str
        Name of the dataset (puma, subsume, ectsum, physionet, hotpotqa, evidence_inference).
    pool_size : int | None
        Pool set size. None uses defaults: DEFAULT_POOL_PER_INTENT per intent (multi-intent)
        or DEFAULT_POOL_SIZE total (uni-intent).
    cal_size : int | None
        Number of samples for calibration set. None uses DEFAULT_CAL_SIZE (100).
    seed : int
        Random seed for reproducibility. Default is DEFAULT_SEED (3).
    verbose : bool
        Print progress information. Default is True.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (pool_df, cal_df, test_df) with standardized column names.

    Examples
    --------
    >>> df = pl.read_parquet("data/PUMA/PUMA_processed.parquet")
    >>> pool_df, cal_df, test_df = split_dataset(df, "puma")
    >>> pool_df.shape[0], cal_df.shape[0]
    (100, 100)
    """
    dataset_name = dataset_name.lower()
    effective_cal_size = cal_size if cal_size is not None else DEFAULT_CAL_SIZE

    if verbose:
        print(f"Splitting {dataset_name} dataset...")
        print(f"  Input shape: {df.shape}")

    # Standardize column names
    df_std = _standardize_columns(df, dataset_name)

    if verbose:
        print(f"  Standardized columns: {df_std.columns}")

    # Choose sampling strategy based on whether dataset has intent
    is_multi_intent = dataset_name in DATASETS_WITH_INTENT and "intent" in df_std.columns

    if is_multi_intent:
        intents = df_std["intent"].unique().to_list()
        effective_pool_per_intent = pool_size if pool_size is not None else DEFAULT_POOL_PER_INTENT
        if verbose:
            print(f"  Stratified pool across {len(intents)} intents ({effective_pool_per_intent} per intent)")
            print(f"  Uniform random cal ({effective_cal_size} samples)")
        pool_df, cal_df, test_df = _stratified_pool_uniform_cal_split(
            df_std,
            pool_per_intent=effective_pool_per_intent,
            cal_size=effective_cal_size,
            seed=seed,
        )
    else:
        effective_pool_size = pool_size if pool_size is not None else DEFAULT_POOL_SIZE
        if verbose:
            print(f"  Uniform random pool ({effective_pool_size} samples)")
            print(f"  Uniform random cal ({effective_cal_size} samples)")
        pool_df, cal_df, test_df = _uniform_three_way_split(
            df_std,
            pool_size=effective_pool_size,
            cal_size=effective_cal_size,
            seed=seed,
            dataset_name=dataset_name,
        )

    if verbose:
        print(f"  Pool set: {pool_df.shape[0]} samples")
        print(f"  Calibration set: {cal_df.shape[0]} samples")
        print(f"  Test set: {test_df.shape[0]} samples")
        if is_multi_intent:
            pool_intents = pool_df["intent"].value_counts().sort("intent")
            print(f"  Pool intent distribution:")
            for row in pool_intents.iter_rows():
                print(f"    {row[0]}: {row[1]}")

    return pool_df, cal_df, test_df


def split_and_save_dataset(
    df: pl.DataFrame,
    dataset_name: str,
    output_dir: Path | None = None,
    pool_size: int | None = None,
    cal_size: int | None = None,
    seed: int = DEFAULT_SEED,
    save_output: bool = True,
    verbose: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a dataset and optionally save pool, calibration, and test sets.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame (processed dataset).
    dataset_name : str
        Name of the dataset (puma, subsume, ectsum, physionet).
    output_dir : Path, optional
        Directory to save output files. Defaults to the dataset's data directory.
    pool_size : int | None
        Pool set size. None uses defaults based on multi_intent setting.
    cal_size : int | None
        Number of samples for calibration set. None uses DEFAULT_CAL_SIZE (100).
    seed : int
        Random seed for reproducibility. Default is DEFAULT_SEED (3).
    save_output : bool
        If True, save splits to parquet files. Default is True.
    verbose : bool
        Print progress information. Default is True.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (pool_df, cal_df, test_df) with standardized column names.

    Output Files
    ------------
    If save_output is True:
    - {output_dir}/{dataset_name}_pool.parquet
    - {output_dir}/{dataset_name}_cal.parquet
    - {output_dir}/{dataset_name}_test.parquet
    """
    dataset_name = dataset_name.lower()

    # Determine output directory
    if output_dir is None:
        data_dir = get_project_data_dir()
        dir_mapping = {
            "puma": "PUMA",
            "subsume": "SubSumE",
            "ectsum": "ECTSum",
            "physionet": "PhysionetGoldCorpus",
            "hotpotqa": "HotpotQA",
            "evidence_inference": "EvidenceInference_v2",
        }
        if dataset_name not in dir_mapping:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        output_dir = data_dir / dir_mapping[dataset_name]

    output_dir = Path(output_dir)

    if verbose:
        print("=" * 60)
        print(f"Dataset Splitting: {dataset_name}")
        print("=" * 60)

    # Split the dataset
    pool_df, cal_df, test_df = split_dataset(
        df,
        dataset_name,
        pool_size=pool_size,
        cal_size=cal_size,
        seed=seed,
        verbose=verbose
    )

    # Save output files
    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)

        pool_path = output_dir / f"{dataset_name}_pool.parquet"
        cal_path = output_dir / f"{dataset_name}_cal.parquet"
        test_path = output_dir / f"{dataset_name}_test.parquet"

        pool_df.write_parquet(pool_path)
        cal_df.write_parquet(cal_path)
        test_df.write_parquet(test_path)

        if verbose:
            print(f"\nSaved pool set to: {pool_path}")
            print(f"Saved calibration set to: {cal_path}")
            print(f"Saved test set to: {test_path}")
    elif verbose:
        print("\nOutput not saved (save_output=False)")

    return pool_df, cal_df, test_df


# =============================================================================
# Dataset-Specific Convenience Functions
# =============================================================================

# Registry: dataset_name -> (default_input_filename, data_subdir, process_fn_hint)
_SPLIT_CONFIG: dict[str, tuple[str, str, str]] = {
    "puma": ("PUMA_processed.parquet", "PUMA", "process_puma_dataset()"),
    "subsume": ("SubSumE_processed.parquet", "SubSumE", "process_subsume_dataset()"),
    "ectsum": ("ECTSum_processed.parquet", "ECTSum", "process_ectsum_dataset()"),
    "physionet": (
        "physionet_goldcorpus_i2b2deid_claimlevel.parquet",
        "PhysionetGoldCorpus",
        "process_physionet_goldcorpus()",
    ),
    "hotpotqa": ("HotpotQA_processed.parquet", "HotpotQA", "process_hotpotqa_dataset()"),
    "evidence_inference": (
        "EvidenceInference_processed.parquet",
        "EvidenceInference_v2",
        "process_evidence_inference_dataset()",
    ),
    "contractnli": (
        "ContractNLI_processed.parquet",
        "ContractNLI",
        "process_contractnli_dataset()",
    ),
}


def split_named_dataset(
    dataset_name: str,
    data_dir: Path | None = None,
    input_filename: str | None = None,
    pool_size: int | None = None,
    cal_size: int | None = None,
    seed: int = DEFAULT_SEED,
    save_output: bool = True,
    verbose: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split any named dataset into pool, calibration, and test sets.

    Parameters
    ----------
    dataset_name : str
        One of: "puma", "subsume", "ectsum", "physionet", "hotpotqa", "evidence_inference".
    data_dir : Path, optional
        Directory containing the processed parquet file. Defaults to dataset-specific dir.
    input_filename : str, optional
        Name of input parquet file. Defaults to dataset-specific filename.
    pool_size, cal_size, seed, save_output, verbose
        Passed through to split_and_save_dataset().
    """
    if dataset_name not in _SPLIT_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. Available: {list(_SPLIT_CONFIG)}")

    default_filename, data_subdir, process_fn = _SPLIT_CONFIG[dataset_name]
    data_dir = Path(data_dir) if data_dir else get_project_data_dir() / data_subdir
    filename = input_filename or default_filename

    input_path = data_dir / filename
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {input_path}\n"
            f"Run {process_fn} first."
        )

    df = pl.read_parquet(input_path)
    return split_and_save_dataset(
        df, dataset_name, output_dir=data_dir,
        pool_size=pool_size, cal_size=cal_size, seed=seed,
        save_output=save_output, verbose=verbose,
    )


def split_puma_dataset(data_dir=None, input_filename=None, pool_size=None, cal_size=None, seed=DEFAULT_SEED, save_output=True, verbose=True):
    return split_named_dataset("puma", data_dir, input_filename, pool_size, cal_size, seed, save_output, verbose)


def split_subsume_dataset(data_dir=None, input_filename=None, pool_size=None, cal_size=None, seed=DEFAULT_SEED, save_output=True, verbose=True):
    return split_named_dataset("subsume", data_dir, input_filename, pool_size, cal_size, seed, save_output, verbose)


def split_ectsum_dataset(data_dir=None, input_filename=None, pool_size=None, cal_size=None, seed=DEFAULT_SEED, save_output=True, verbose=True):
    return split_named_dataset("ectsum", data_dir, input_filename, pool_size, cal_size, seed, save_output, verbose)


def split_physionet_dataset(data_dir=None, input_filename=None, pool_size=None, cal_size=None, seed=DEFAULT_SEED, save_output=True, verbose=True):
    return split_named_dataset("physionet", data_dir, input_filename, pool_size, cal_size, seed, save_output, verbose)


def split_hotpotqa_dataset(data_dir=None, input_filename=None, pool_size=None, cal_size=None, seed=DEFAULT_SEED, save_output=True, verbose=True):
    return split_named_dataset("hotpotqa", data_dir, input_filename, pool_size, cal_size, seed, save_output, verbose)


def split_evidence_inference_dataset(data_dir=None, input_filename=None, pool_size=None, cal_size=None, seed=DEFAULT_SEED, save_output=True, verbose=True):
    return split_named_dataset("evidence_inference", data_dir, input_filename, pool_size, cal_size, seed, save_output, verbose)


def split_contractnli_dataset(data_dir=None, input_filename=None, pool_size=None, cal_size=None, seed=DEFAULT_SEED, save_output=True, verbose=True):
    return split_named_dataset("contractnli", data_dir, input_filename, pool_size, cal_size, seed, save_output, verbose)


def split_all_datasets(
    pool_size: int | None = None,
    cal_size: int | None = None,
    seed: int = DEFAULT_SEED,
    save_output: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Split all available datasets into pool, calibration, and test sets.

    Processes each dataset that has a processed parquet file available.
    Skips datasets with missing files and reports them.

    Parameters
    ----------
    pool_size : int | None
        Pool set size. None uses defaults (per-intent for multi-intent, total for uni-intent).
    cal_size : int | None
        Number of samples for calibration set. None uses DEFAULT_CAL_SIZE (100).
    seed : int
        Random seed for reproducibility. Default is DEFAULT_SEED (3).
    save_output : bool
        If True, save splits to parquet files. Default is True.
    verbose : bool
        Print progress information. Default is True.

    Returns
    -------
    dict
        Dictionary mapping dataset names to (pool_df, cal_df, test_df) tuples.
        Only includes successfully processed datasets.
    """
    results = {}

    split_functions = {
        "puma": split_puma_dataset,
        "subsume": split_subsume_dataset,
        "ectsum": split_ectsum_dataset,
        "physionet": split_physionet_dataset,
        "hotpotqa": split_hotpotqa_dataset,
        "evidence_inference": split_evidence_inference_dataset,
        "contractnli": split_contractnli_dataset,
    }

    for name, split_fn in split_functions.items():
        if verbose:
            print("\n" + "=" * 60)
        try:
            pool_df, cal_df, test_df = split_fn(
                pool_size=pool_size,
                cal_size=cal_size,
                seed=seed,
                save_output=save_output,
                verbose=verbose,
            )
            results[name] = (pool_df, cal_df, test_df)
        except FileNotFoundError as e:
            if verbose:
                print(f"Skipping {name}: {e}")

    if verbose:
        print("\n" + "=" * 60)
        print(f"Successfully split {len(results)} dataset(s): {list(results.keys())}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for dataset splitting."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Split datasets into pool, calibration, and test sets"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=["puma", "subsume", "ectsum", "physionet", "hotpotqa", "evidence_inference", "contractnli", "all"],
        default="all",
        help="Dataset to split (default: all)"
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help=(
            f"Pool set size. For multi-intent datasets (puma, subsume, evidence_inference), "
            f"this is per-intent (default: {DEFAULT_POOL_PER_INTENT}). "
            f"For others, this is total (default: {DEFAULT_POOL_SIZE})."
        )
    )
    parser.add_argument(
        "--cal-size",
        type=int,
        default=None,
        help=f"Calibration set size (default: {DEFAULT_CAL_SIZE})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Process data but don't save to file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    kwargs = {
        "pool_size": args.pool_size,
        "cal_size": args.cal_size,
        "seed": args.seed,
        "save_output": not args.no_save,
        "verbose": not args.quiet,
    }

    if args.dataset == "all":
        split_all_datasets(**kwargs)
    elif args.dataset == "puma":
        split_puma_dataset(**kwargs)
    elif args.dataset == "subsume":
        split_subsume_dataset(**kwargs)
    elif args.dataset == "ectsum":
        split_ectsum_dataset(**kwargs)
    elif args.dataset == "physionet":
        split_physionet_dataset(**kwargs)
    elif args.dataset == "hotpotqa":
        split_hotpotqa_dataset(**kwargs)
    elif args.dataset == "evidence_inference":
        split_evidence_inference_dataset(**kwargs)
    elif args.dataset == "contractnli":
        split_contractnli_dataset(**kwargs)


if __name__ == "__main__":
    main()
