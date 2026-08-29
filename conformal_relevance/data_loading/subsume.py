"""
SubSumE Dataset (Subjective Summarization Evaluation) processing.

This module provides functions to process the SubSumE dataset from raw files
into a structured parquet format suitable for conformal prediction tasks.

The SubSumE dataset contains Wikipedia articles about US states with
user-generated extractive summaries based on different intents/queries.

Required input files (in data/SubSumE/):
    - processed_state_sentences.csv: All sentences from 50 US state articles
    - user_summary_jsons/*.txt: 275 JSON files with user intents and summaries

Output file:
    - SubSumE_processed.parquet
"""

import glob
import os
import zipfile
from pathlib import Path
from typing import Any

import polars as pl

from ._common import download_file, get_project_data_dir


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = get_project_data_dir() / "SubSumE"
OUTPUT_FILENAME = "SubSumE_processed.parquet"

# Required input files/directories
SENTENCES_FILE = "processed_state_sentences.csv"
USER_SUMMARIES_DIR = "user_summary_jsons"

# Download source: SubSumE_Data.zip hosted on Google Drive
# https://drive.google.com/file/d/1tEDDHzZM_idnv-_PfRE5BmJU5E8yKLRH/view
_GDRIVE_FILE_ID = "1tEDDHzZM_idnv-_PfRE5BmJU5E8yKLRH"
_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={_GDRIVE_FILE_ID}"
_ZIP_FILENAME = "SubSumE_Data.zip"


# =============================================================================
# Download Functions
# =============================================================================



def download_subsume_data(
    data_dir: Path | None = None,
    verbose: bool = True,
) -> None:
    """
    Download SubSumE dataset from Google Drive and extract to data directory.

    Downloads SubSumE_Data.zip (~4.3 MB) from Google Drive and extracts
    ``processed_state_sentences.csv`` and ``user_summary_jsons/`` into
    ``data_dir``.  Handles ZIP archives with or without a top-level prefix
    directory.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to save extracted files. Defaults to data/SubSumE/.
    verbose : bool
        Print progress information.

    Notes
    -----
    Skips download if both required files already exist.
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if (data_dir / SENTENCES_FILE).exists() and (data_dir / USER_SUMMARIES_DIR).exists():
        if verbose:
            print("SubSumE data files already exist, skipping download.")
        return

    if verbose:
        print("Downloading SubSumE dataset from Google Drive...")

    zip_path = data_dir / _ZIP_FILENAME
    download_file(_DOWNLOAD_URL, zip_path, show_progress=verbose)

    if verbose:
        print(f"  Extracting {_ZIP_FILENAME}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find SENTENCES_FILE inside the ZIP to determine the prefix directory.
        # The archive may place files at root ("processed_state_sentences.csv")
        # or inside a subdirectory ("SubSumE_Data/processed_state_sentences.csv").
        csv_member = next(
            (name for name in zf.namelist() if name.endswith(SENTENCES_FILE)), None
        )
        if csv_member is None:
            raise ValueError(
                f"'{SENTENCES_FILE}' not found inside {_ZIP_FILENAME}. "
                "The archive may have unexpected contents."
            )

        prefix = csv_member[: -len(SENTENCES_FILE)]  # "" or "SubSumE_Data/"

        for member in zf.namelist():
            if not member.startswith(prefix):
                continue
            relative = member[len(prefix):]  # path relative to prefix root
            if not relative:
                continue  # skip the prefix directory entry itself
            target = data_dir / relative
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    zip_path.unlink()

    if verbose:
        print(f"  Extracted to: {data_dir}")
        print("Download complete.")


# =============================================================================
# Loading Functions
# =============================================================================

def _load_state_sentences(data_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Load state sentences and create a lookup dictionary.

    Parameters
    ----------
    data_dir : Path
        Directory containing processed_state_sentences.csv.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping state names to:
        - sentences: list of sentences
        - concatenated: all sentences joined with space
        - sids: list of sentence IDs
    """
    filepath = data_dir / SENTENCES_FILE
    df = pl.read_csv(filepath)

    # Ensure sentence column is string type
    df = df.with_columns(pl.col("sentence").cast(pl.Utf8))

    # Group by state name and aggregate
    agg_df = (
        df
        .group_by("name")
        .agg([
            pl.col("sentence").alias("sentences"),
            pl.col("sentence").str.concat(" ").alias("concatenated"),
            pl.col("sid").alias("sids"),
        ])
    )

    # Convert to dictionary for fast lookup
    result = {}
    for row in agg_df.iter_rows(named=True):
        result[row["name"]] = {
            "sentences": row["sentences"],
            "concatenated": row["concatenated"],
            "sids": row["sids"],
        }

    return result


def _load_user_summaries(data_dir: Path) -> pl.DataFrame:
    """
    Load all user summary JSON files and concatenate into single DataFrame.

    Parameters
    ----------
    data_dir : Path
        Directory containing user_summary_jsons subdirectory.

    Returns
    -------
    pl.DataFrame
        Combined data from all user summary files.
    """
    json_dir = data_dir / USER_SUMMARIES_DIR
    pattern = str(json_dir / "*.txt")
    file_list = glob.glob(pattern)

    if not file_list:
        raise FileNotFoundError(
            f"No user summary files found in {json_dir}\n"
            "Expected files matching pattern: userInput_*.txt"
        )

    dfs = []
    for fp in file_list:
        df = pl.read_json(fp)
        # Tag each row with its source file
        df = df.with_columns(pl.lit(os.path.basename(fp)).alias("source_file"))
        dfs.append(df)

    return pl.concat(dfs)


# =============================================================================
# Processing Functions
# =============================================================================

def _expand_summaries_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Expand the nested 'summaries' column into separate columns.

    The summaries column contains a List of structs, where each struct has:
    - state_name
    - state_id
    - sentence_ids
    - sentences
    - used_keywords

    Each row in the input has a list of 8 summaries (one per state).
    This function explodes the list so each summary becomes its own row.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with 'summaries' column containing List(Struct).

    Returns
    -------
    pl.DataFrame
        DataFrame with summaries exploded and expanded into separate columns.
    """
    # First explode the list so each summary struct becomes its own row
    df_exploded = df.explode("summaries")

    # Then extract fields from the summaries struct
    df_expanded = df_exploded.with_columns([
        pl.col("summaries").struct.field("state_name").alias("state_name"),
        pl.col("summaries").struct.field("state_id").alias("state_id"),
        pl.col("summaries").struct.field("sentence_ids").alias("summary_sentences_id"),
        pl.col("summaries").struct.field("sentences").alias("summary_sentences"),
        pl.col("summaries").struct.field("used_keywords").alias("used_keywords"),
    ]).drop("summaries")

    return df_expanded


def _add_state_data(
    df: pl.DataFrame,
    state_lookup: dict[str, dict[str, Any]],
) -> pl.DataFrame:
    """
    Add input sentences and IDs from state lookup dictionary.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with state_name column.
    state_lookup : Dict[str, Dict[str, Any]]
        Dictionary mapping state names to sentences data.

    Returns
    -------
    pl.DataFrame
        DataFrame with added input_sentences, input, and input_sentences_ids columns.
    """
    # Create lists for new columns
    input_sentences_list = []
    input_list = []
    input_sentences_ids_list = []

    for state_name in df["state_name"].to_list():
        state_data = state_lookup.get(state_name, {})
        input_sentences_list.append(state_data.get("sentences", []))
        input_list.append(state_data.get("concatenated", ""))
        input_sentences_ids_list.append(state_data.get("sids", []))

    # Add new columns
    df = df.with_columns([
        pl.Series("input_sentences", input_sentences_list),
        pl.Series("input", input_list),
        pl.Series("input_sentences_ids", input_sentences_ids_list),
    ])

    return df


def _create_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create binary labels indicating which input sentences are in the summary.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with input_sentences_ids and summary_sentences_id columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with added input_sentences_labels column.
    """
    # Create labels for each row
    labels_list = []

    for row in df.iter_rows(named=True):
        summary_ids_set = set(row["summary_sentences_id"]) if row["summary_sentences_id"] else set()
        input_ids = row["input_sentences_ids"] if row["input_sentences_ids"] else []
        labels = [1 if sid in summary_ids_set else 0 for sid in input_ids]
        labels_list.append(labels)

    df = df.with_columns(pl.Series("input_sentences_labels", labels_list))

    return df


def _process_raw_data(
    df_summaries: pl.DataFrame,
    state_lookup: dict[str, dict[str, Any]],
) -> pl.DataFrame:
    """
    Process raw SubSumE data into standardized format.

    Parameters
    ----------
    df_summaries : pl.DataFrame
        Raw DataFrame from user summary files.
    state_lookup : Dict[str, Dict[str, Any]]
        State sentences lookup dictionary.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with standardized columns.
    """
    # Step 1: Expand summaries column
    df = _expand_summaries_column(df_summaries)

    # Step 2: Add state data (input sentences, etc.)
    df = _add_state_data(df, state_lookup)

    # Step 3: Create summary by joining summary_sentences
    df = df.with_columns(
        pl.col("summary_sentences").list.join(" ").alias("summary")
    )

    # Step 4: Create binary labels
    df = _create_labels(df)

    # Step 5: Drop unnecessary columns and reorder
    df = df.select([
        "intent",
        "input",
        "input_sentences",
        "input_sentences_labels",
        "summary",
        "summary_sentences",
    ])

    return df


# =============================================================================
# Public API
# =============================================================================

def load_state_sentences(
    data_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Load state sentences and create lookup dictionary.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/SubSumE/.
    verbose : bool
        Print progress information.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping state names to sentences data.
    """
    data_dir = data_dir or DATA_DIR

    if verbose:
        print(f"Loading state sentences from: {data_dir / SENTENCES_FILE}")

    state_lookup = _load_state_sentences(data_dir)

    if verbose:
        print(f"  Loaded {len(state_lookup)} states")

    return state_lookup


def load_user_summaries(
    data_dir: Path | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Load all user summary files.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/SubSumE/.
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Combined DataFrame from all user summary files.
    """
    data_dir = data_dir or DATA_DIR

    if verbose:
        print(f"Loading user summaries from: {data_dir / USER_SUMMARIES_DIR}")

    df = _load_user_summaries(data_dir)

    if verbose:
        print(f"  Loaded {df.shape[0]} records from user summary files")

    return df


def process_subsume_data(
    df_summaries: pl.DataFrame,
    state_lookup: dict[str, dict[str, Any]],
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process raw SubSumE data into standardized format.

    Parameters
    ----------
    df_summaries : pl.DataFrame
        Raw DataFrame from load_user_summaries().
    state_lookup : Dict[str, Dict[str, Any]]
        State lookup from load_state_sentences().
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with columns:
        - intent: User's query/intent
        - input: Full article text (sentences joined)
        - input_sentences: List of sentences from article
        - input_sentences_labels: Binary labels (1 if in summary)
        - summary: Summary text (summary sentences joined)
        - summary_sentences: List of summary sentences
    """
    if verbose:
        print("Processing SubSumE data...")

    df_processed = _process_raw_data(df_summaries, state_lookup)

    if verbose:
        print(f"  Processed {df_processed.shape[0]} records")
        print(f"  Columns: {df_processed.columns}")

    return df_processed


def process_subsume_dataset(
    data_dir: Path | None = None,
    output_filename: str = OUTPUT_FILENAME,
    save_output: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process SubSumE dataset into structured parquet format.

    Main entry point for SubSumE dataset processing. Loads state sentences,
    user summaries, processes them, and optionally saves the result.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/SubSumE/.
    output_filename : str
        Name of output parquet file. Defaults to "SubSumE_processed.parquet".
    save_output : bool
        If True, save result to parquet file. Defaults to True.
    verbose : bool
        If True, print progress information. Defaults to True.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with sentence-level labels.

    Raises
    ------
    FileNotFoundError
        If required input files are missing.

    Examples
    --------
    >>> df = process_subsume_dataset(verbose=True)
    >>> df.shape
    (2200, 6)

    >>> # Without saving
    >>> df = process_subsume_dataset(save_output=False)
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)
    output_path = data_dir / output_filename

    # Check for required files
    sentences_file = data_dir / SENTENCES_FILE
    summaries_dir = data_dir / USER_SUMMARIES_DIR

    if not sentences_file.exists():
        raise FileNotFoundError(
            f"Required file not found: {sentences_file}\n"
            "Run download_subsume_data() first, or manually place "
            "processed_state_sentences.csv in the data directory."
        )

    if not summaries_dir.exists():
        raise FileNotFoundError(
            f"Required directory not found: {summaries_dir}\n"
            "Run download_subsume_data() first, or manually place "
            "user_summary_jsons/ in the data directory."
        )

    if verbose:
        print("=" * 60)
        print("SubSumE Dataset Processing Pipeline")
        print("=" * 60)
        print(f"\nData directory: {data_dir}")
        print(f"Output: {output_path}")

    # Step 1: Load state sentences
    if verbose:
        print("\n[Step 1/3] Loading state sentences...")

    state_lookup = load_state_sentences(data_dir=data_dir, verbose=verbose)

    # Step 2: Load user summaries
    if verbose:
        print("\n[Step 2/3] Loading user summaries...")

    df_summaries = load_user_summaries(data_dir=data_dir, verbose=verbose)

    # Step 3: Process data
    if verbose:
        print("\n[Step 3/3] Processing data...")

    df_result = process_subsume_data(df_summaries, state_lookup, verbose=verbose)

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
        # Show unique intents
        intents = df_result["intent"].unique().to_list()
        print(f"Unique intents ({len(intents)}): {intents[:3]}...")

    return df_result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for SubSumE data processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process SubSumE dataset into parquet format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory for input/output files (default: data/SubSumE/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILENAME,
        help=f"Output parquet filename (default: {OUTPUT_FILENAME})"
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

    process_subsume_dataset(
        data_dir=args.data_dir,
        output_filename=args.output,
        save_output=not args.no_save,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
