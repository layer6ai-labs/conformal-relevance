"""
HotpotQA Dataset processing (distractor setting).

This module provides functions to download, load, and process the HotpotQA dataset
into a structured parquet format suitable for conformal prediction tasks.

The HotpotQA dataset contains multi-hop reasoning questions with 10 paragraphs
per question (2 gold + 8 distractors). Each paragraph contains multiple sentences,
and `supporting_facts` annotations indicate which sentences are needed to answer
the question.

Data source:
    - Training set: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    - Dev distractor: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

Required input files (downloaded automatically or manually placed in data/HotpotQA/):
    - hotpot_train_v1.1.json: Training set (~90K questions)
    - hotpot_dev_distractor_v1.json: Development set (~7.4K questions)

Output file:
    - HotpotQA_processed.parquet
"""

import json
from pathlib import Path

import polars as pl
from tqdm import tqdm

from ._common import download_dataset_files, get_project_data_dir


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = get_project_data_dir() / "HotpotQA"
OUTPUT_FILENAME = "HotpotQA_processed.parquet"

# Download URLs for HotpotQA dataset
DOWNLOAD_URLS = {
    "hotpot_train_v1.1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "hotpot_dev_distractor_v1.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
}

# Required input files
REQUIRED_FILES = ["hotpot_train_v1.1.json", "hotpot_dev_distractor_v1.json"]


# =============================================================================
# Download Functions
# =============================================================================


def download_hotpotqa_data(
    data_dir: Path | None = None,
    verbose: bool = True,
) -> None:
    """Download HotpotQA dataset files from CMU servers. Skips files that already exist."""
    download_dataset_files(data_dir, DATA_DIR, DOWNLOAD_URLS, "HotpotQA", verbose)


# =============================================================================
# Loading Functions
# =============================================================================

def _load_raw_data(data_dir: Path, filename: str) -> list[dict]:
    """
    Load raw HotpotQA JSON file.

    Parameters
    ----------
    data_dir : Path
        Directory containing the JSON file.
    filename : str
        Name of the JSON file.

    Returns
    -------
    list[dict]
        List of question objects.
    """
    filepath = data_dir / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# Processing Functions
# =============================================================================

def _process_single_sample(sample: dict) -> dict:
    """
    Process a single HotpotQA sample into standardized format.

    The HotpotQA format:
    - question: The question text (becomes intent)
    - supporting_facts: List of [title, sent_idx] pairs indicating relevant sentences
    - context: List of [title, [sentences]] pairs for each paragraph (10 total)

    Parameters
    ----------
    sample : dict
        A single HotpotQA sample.

    Returns
    -------
    dict
        Processed sample with standardized columns.
    """
    question = sample["question"]

    # Extract supporting facts as a set of (title, sent_idx) tuples
    supporting_facts = set()
    for title, sent_idx in sample.get("supporting_facts", []):
        supporting_facts.add((title, sent_idx))

    # Process context: flatten all sentences while tracking labels
    input_sentences = []
    input_labels = []
    summary_sentences = []

    for title, sentences in sample.get("context", []):
        for sent_idx, sentence in enumerate(sentences):
            input_sentences.append(sentence)
            # Label is 1 if this sentence is a supporting fact
            is_supporting = (title, sent_idx) in supporting_facts
            input_labels.append(1 if is_supporting else 0)
            if is_supporting:
                summary_sentences.append(sentence)

    # Create summary by joining supporting fact sentences
    summary = " ".join(summary_sentences)

    return {
        "intent": question,
        "input_sentences": input_sentences,
        "input_sentences_labels": input_labels,
        "summary": summary,
        "summary_sentences": summary_sentences,
    }


def _process_raw_data(
    data: list[dict],
    desc: str = "Processing",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process raw HotpotQA data into a DataFrame.

    Parameters
    ----------
    data : list[dict]
        Raw HotpotQA data (list of samples).
    desc : str
        Description for progress bar.
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame.
    """
    processed_samples = []

    iterator = data
    if verbose:
        iterator = tqdm(data, desc=desc)

    for sample in iterator:
        processed = _process_single_sample(sample)
        processed_samples.append(processed)

    return pl.DataFrame(processed_samples)


# =============================================================================
# Public API
# =============================================================================

def load_hotpotqa_raw(
    data_dir: Path | None = None,
    splits: list[str] | None = None,
    auto_download: bool = True,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """
    Load raw HotpotQA data from JSON files.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/HotpotQA/.
    splits : list[str] | None
        Which splits to load. Options: ['train', 'dev']. If None, loads all.
    auto_download : bool
        If True, automatically download missing files.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict[str, list[dict]]
        Dictionary mapping split names to raw data lists.

    Raises
    ------
    FileNotFoundError
        If required files are missing and auto_download is False.
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)

    if splits is None:
        splits = ["train", "dev"]

    # Map split names to filenames
    filename_mapping = {
        "train": "hotpot_train_v1.1.json",
        "dev": "hotpot_dev_distractor_v1.json",
    }

    required_files = [filename_mapping[s] for s in splits]
    missing_files = [f for f in required_files if not (data_dir / f).exists()]

    if missing_files:
        if auto_download:
            if verbose:
                print(f"Missing files: {missing_files}")
            download_hotpotqa_data(data_dir=data_dir, verbose=verbose)
        else:
            raise FileNotFoundError(
                f"Missing required files in {data_dir}: {missing_files}\n"
                "Set auto_download=True to download automatically, or manually download from:\n"
                "http://curtis.ml.cmu.edu/datasets/hotpot/"
            )

    if verbose:
        print(f"Loading HotpotQA data from: {data_dir}")

    result = {}
    for split in splits:
        filename = filename_mapping[split]
        data = _load_raw_data(data_dir, filename)
        result[split] = data
        if verbose:
            print(f"  {split}: {len(data)} samples")

    return result


def process_hotpotqa_data(
    raw_data: dict[str, list[dict]],
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process raw HotpotQA data into standardized DataFrame format.

    Parameters
    ----------
    raw_data : dict[str, list[dict]]
        Dictionary mapping split names to raw data lists.
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with columns:
        - intent: Question text (serves as the intent/query)
        - input_sentences: List of all sentences from all paragraphs
        - input_sentences_labels: Binary labels (1 if supporting fact)
        - summary: Supporting fact sentences joined with space
        - summary_sentences: List of supporting fact sentences
    """
    if verbose:
        print("Processing HotpotQA data...")

    dfs = []
    for split, data in raw_data.items():
        df = _process_raw_data(data, desc=f"Processing {split}", verbose=verbose)
        dfs.append(df)

    df_combined = pl.concat(dfs)

    if verbose:
        print(f"  Total processed: {df_combined.shape[0]} samples")
        print(f"  Columns: {df_combined.columns}")

    return df_combined


def process_hotpotqa_dataset(
    data_dir: Path | None = None,
    output_filename: str = OUTPUT_FILENAME,
    splits: list[str] | None = None,
    auto_download: bool = True,
    save_output: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process HotpotQA dataset into structured parquet format.

    Main entry point for HotpotQA dataset processing. Downloads (if needed),
    loads, processes, and optionally saves the dataset.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/HotpotQA/.
    output_filename : str
        Name of output parquet file. Defaults to "HotpotQA_processed.parquet".
    splits : list[str] | None
        Which splits to process. Options: ['train', 'dev']. If None, processes all.
    auto_download : bool
        If True, automatically download missing files from CMU servers.
    save_output : bool
        If True, save result to parquet file. Defaults to True.
    verbose : bool
        If True, print progress information. Defaults to True.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with columns:
        - intent: Question text (serves as the intent/query)
        - input_sentences: List of all sentences from all paragraphs
        - input_sentences_labels: Binary labels (1 if supporting fact)
        - summary: Supporting fact sentences joined with space
        - summary_sentences: List of supporting fact sentences

    Examples
    --------
    >>> df = process_hotpotqa_dataset(verbose=True)
    >>> df.shape
    (97416, 5)

    >>> # Process only dev set
    >>> df = process_hotpotqa_dataset(splits=["dev"])

    >>> # Without auto-download (files must exist)
    >>> df = process_hotpotqa_dataset(auto_download=False)
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)
    output_path = data_dir / output_filename

    if verbose:
        print("=" * 60)
        print("HotpotQA Dataset Processing Pipeline")
        print("=" * 60)
        print(f"\nData directory: {data_dir}")
        print(f"Output: {output_path}")
        if splits:
            print(f"Splits: {splits}")

    # Step 1: Load raw data (with optional download)
    if verbose:
        print("\n[Step 1/2] Loading raw HotpotQA data...")

    raw_data = load_hotpotqa_raw(
        data_dir=data_dir,
        splits=splits,
        auto_download=auto_download,
        verbose=verbose,
    )

    # Step 2: Process data
    if verbose:
        print("\n[Step 2/2] Processing data...")

    df_result = process_hotpotqa_data(raw_data, verbose=verbose)

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

        # Show statistics
        n_samples = df_result.shape[0]
        avg_sentences = (
            df_result.select(pl.col("input_sentences").list.len().mean())
            .item()
        )
        avg_supporting = (
            df_result.select(
                pl.col("input_sentences_labels").list.eval(pl.element().sum()).list.first().mean()
            )
            .item()
        )

        print(f"\nDataset statistics:")
        print(f"  Total samples: {n_samples}")
        print(f"  Avg sentences per sample: {avg_sentences:.1f}")
        print(f"  Avg supporting facts per sample: {avg_supporting:.1f}")

    return df_result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for HotpotQA data processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process HotpotQA dataset into parquet format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory for input/output files (default: data/HotpotQA/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILENAME,
        help=f"Output parquet filename (default: {OUTPUT_FILENAME})"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "dev"],
        default=None,
        help="Which splits to process (default: all)"
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

    process_hotpotqa_dataset(
        data_dir=args.data_dir,
        output_filename=args.output,
        splits=args.splits,
        auto_download=not args.no_download,
        save_output=not args.no_save,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
