"""
ECTSum Dataset (Earnings Call Transcript Summarization) processing.

This module provides functions to download, load, and process the ECTSum dataset
from HuggingFace into a structured parquet format suitable for conformal prediction tasks.

The ECTSum dataset contains earnings call transcripts with extractive summarization labels.

Data source:
    - HuggingFace: https://huggingface.co/datasets/nyamuda/ECTSum

Required input files (downloaded automatically or manually placed in data/ECTSum/):
    - train.json: Training set
    - val.json: Validation set
    - test.json: Test set

Output file:
    - ECTSum_processed.parquet
"""

from pathlib import Path

import polars as pl

from ._common import download_dataset_files, get_project_data_dir


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = get_project_data_dir() / "ECTSum"
OUTPUT_FILENAME = "ECTSum_processed.parquet"

# HuggingFace URLs for ECTSum dataset
DOWNLOAD_URLS = {
    "train.json": "https://huggingface.co/datasets/nyamuda/ECTSum/resolve/main/train.json?download=true",
    "val.json": "https://huggingface.co/datasets/nyamuda/ECTSum/resolve/main/val.json?download=true",
    "test.json": "https://huggingface.co/datasets/nyamuda/ECTSum/resolve/main/test.json?download=true",
}

# Required input files
REQUIRED_FILES = ["train.json", "val.json", "test.json"]


# =============================================================================
# Download Functions
# =============================================================================

def download_ectsum_data(
    data_dir: Path | None = None,
    verbose: bool = True,
) -> None:
    """Download ECTSum dataset files from HuggingFace. Skips files that already exist."""
    download_dataset_files(data_dir, DATA_DIR, DOWNLOAD_URLS, "ECTSum", verbose)


# =============================================================================
# Loading Functions
# =============================================================================

def _load_raw_data(data_dir: Path) -> pl.DataFrame:
    """
    Load raw ECTSum JSON files and concatenate into single DataFrame.

    Parameters
    ----------
    data_dir : Path
        Directory containing train.json, val.json, test.json.

    Returns
    -------
    pl.DataFrame
        Combined raw data from all splits.
    """
    dfs = []
    for filename in REQUIRED_FILES:
        filepath = data_dir / filename
        df = pl.read_ndjson(filepath)
        split_name = filename.replace('.json', '')
        df = df.with_columns(pl.lit(split_name).alias("split"))
        dfs.append(df)

    return pl.concat(dfs)


# =============================================================================
# Processing Functions
# =============================================================================

def _process_raw_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Process raw ECTSum DataFrame into standardized format.

    Transformations:
    1. Rename columns: summaries→summary, doc→input, labels→input_sentences_labels
    2. Split input text by newlines → input_sentences (list)
    3. Split summary text by newlines → summary_sentences (list)
    4. Convert labels from newline-separated string → list of ints

    Parameters
    ----------
    df : pl.DataFrame
        Raw DataFrame with columns: doc, summaries, labels.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with standardized columns.
    """
    df_processed = (
        df
        # Rename columns to match standard format
        .rename({
            "summaries": "summary",
            "doc": "input",
            "labels": "input_sentences_labels_str",
        })
        # Split input document into sentences (by newline)
        .with_columns([
            pl.col("input").str.split("\n").alias("input_sentences"),
            pl.col("summary").str.split("\n").alias("summary_sentences"),
            # Convert labels from newline-separated string to list of integers
            pl.col("input_sentences_labels_str")
              .str.split("\n")
              .list.eval(pl.element().cast(pl.Int64))
              .alias("input_sentences_labels"),
        ])
        # Drop intermediate column and reorder
        .drop("input_sentences_labels_str")
        .select([
            "input",
            "summary",
            "input_sentences",
            "summary_sentences",
            "input_sentences_labels",
            "split",
        ])
    )

    return df_processed


# =============================================================================
# Public API
# =============================================================================

def load_ectsum_raw(
    data_dir: Path | None = None,
    auto_download: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Load raw ECTSum data from JSON files.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/ECTSum/.
    auto_download : bool
        If True, automatically download missing files.
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Raw combined DataFrame from all splits.

    Raises
    ------
    FileNotFoundError
        If required files are missing and auto_download is False.
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)

    # Check for missing files
    missing_files = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]

    if missing_files:
        if auto_download:
            if verbose:
                print(f"Missing files: {missing_files}")
            download_ectsum_data(data_dir=data_dir, verbose=verbose)
        else:
            raise FileNotFoundError(
                f"Missing required files in {data_dir}: {missing_files}\n"
                "Set auto_download=True to download automatically, or manually download from:\n"
                "https://huggingface.co/datasets/nyamuda/ECTSum"
            )

    if verbose:
        print(f"Loading ECTSum data from: {data_dir}")

    df = _load_raw_data(data_dir)

    if verbose:
        print(f"  Loaded {df.shape[0]} total records")
        for split in df["split"].unique().to_list():
            count = df.filter(pl.col("split") == split).shape[0]
            print(f"    {split}: {count} records")

    return df


def process_ectsum_data(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """
    Process raw ECTSum DataFrame into standardized format.

    Parameters
    ----------
    df : pl.DataFrame
        Raw DataFrame from load_ectsum_raw().
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with columns:
        - input: Full document text
        - summary: Summary text
        - input_sentences: List of sentences from document
        - summary_sentences: List of sentences from summary
        - input_sentences_labels: Binary labels for each sentence
        - split: Data split (train/val/test)
    """
    if verbose:
        print("Processing ECTSum data...")

    df_processed = _process_raw_data(df)

    if verbose:
        print(f"  Processed {df_processed.shape[0]} records")
        print(f"  Columns: {df_processed.columns}")

    return df_processed


def process_ectsum_dataset(
    data_dir: Path | None = None,
    output_filename: str = OUTPUT_FILENAME,
    auto_download: bool = True,
    save_output: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process ECTSum dataset into structured parquet format.

    Main entry point for ECTSum dataset processing. Downloads (if needed),
    loads, processes, and optionally saves the dataset.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/ECTSum/.
    output_filename : str
        Name of output parquet file. Defaults to "ECTSum_processed.parquet".
    auto_download : bool
        If True, automatically download missing files from HuggingFace.
    save_output : bool
        If True, save result to parquet file. Defaults to True.
    verbose : bool
        If True, print progress information. Defaults to True.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with sentence-level labels.

    Examples
    --------
    >>> df = process_ectsum_dataset(verbose=True)
    >>> df.shape
    (2425, 6)

    >>> # Without saving
    >>> df = process_ectsum_dataset(save_output=False)

    >>> # Without auto-download (files must exist)
    >>> df = process_ectsum_dataset(auto_download=False)
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)
    output_path = data_dir / output_filename

    if verbose:
        print("=" * 60)
        print("ECTSum Dataset Processing Pipeline")
        print("=" * 60)
        print(f"\nData directory: {data_dir}")
        print(f"Output: {output_path}")

    # Step 1: Load raw data (with optional download)
    if verbose:
        print("\n[Step 1/2] Loading raw ECTSum data...")

    df_raw = load_ectsum_raw(
        data_dir=data_dir,
        auto_download=auto_download,
        verbose=verbose,
    )

    # Step 2: Process data
    if verbose:
        print("\n[Step 2/2] Processing data...")

    df_result = process_ectsum_data(df_raw, verbose=verbose)

    # Save output
    if save_output:
        data_dir.mkdir(parents=True, exist_ok=True)
        df_result.write_parquet(output_path)
        if verbose:
            print("\n" + "=" * 60)
            print(f"Output saved to: {output_path}")
    elif verbose:
        print("\n" + "=" * 60)
        print("Output not saved (save_output=False)")

    if verbose:
        print(f"Final shape: {df_result.shape}")
        print(f"Columns: {df_result.columns}")

    return df_result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for ECTSum data processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ECTSum dataset into parquet format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory for input/output files (default: data/ECTSum/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILENAME,
        help=f"Output parquet filename (default: {OUTPUT_FILENAME})"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't auto-download missing files"
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

    process_ectsum_dataset(
        data_dir=args.data_dir,
        output_filename=args.output,
        auto_download=not args.no_download,
        save_output=not args.no_save,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
