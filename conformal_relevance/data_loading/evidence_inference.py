"""
Evidence Inference v2.0 Dataset processing.

This module provides functions to download, load, and process the Evidence Inference
dataset into a structured parquet format suitable for conformal prediction tasks.

The Evidence Inference dataset contains clinical trial abstracts with annotations
for evidence inference tasks - determining whether interventions have effects on
outcomes based on the evidence in the text.

Data source:
    - https://evidence-inference.ebm-nlp.com/v2.0.tar.gz

Required input files (downloaded and extracted automatically or manually placed):
    - annotations_merged.csv: Merged annotations with evidence spans
    - prompts_merged.csv: Prompts with intervention/comparator/outcome info
    - txt_files/: Directory containing article text files
    - splits/: Directory containing train/dev/test split definitions

Output file:
    - EvidenceInference_processed.parquet
"""

import tarfile
from pathlib import Path

import polars as pl
from tqdm import tqdm

from ._common import download_file, get_project_data_dir, split_sentences_with_spans


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = get_project_data_dir() / "EvidenceInference_v2"
OUTPUT_FILENAME = "EvidenceInference_processed.parquet"

# Download URL for Evidence Inference v2.0
DOWNLOAD_URL = "https://evidence-inference.ebm-nlp.com/v2.0.tar.gz"
TAR_FILENAME = "v2.0.tar.gz"

# Required input files/directories after extraction
REQUIRED_FILES = ["annotations_merged.csv", "prompts_merged.csv"]
REQUIRED_DIRS = ["txt_files", "splits"]


# =============================================================================
# Download Functions
# =============================================================================


def _extract_tarball(tar_path: Path, extract_dir: Path, verbose: bool = True) -> None:
    """
    Extract a tar.gz file to the specified directory.

    Parameters
    ----------
    tar_path : Path
        Path to the tar.gz file.
    extract_dir : Path
        Directory to extract files to.
    verbose : bool
        Print progress information.
    """
    if verbose:
        print(f"  Extracting: {tar_path.name}")

    with tarfile.open(tar_path, 'r:gz') as tar:
        # Get all members
        members = tar.getmembers()

        if verbose:
            members = tqdm(members, desc="Extracting")

        for member in members:
            # Strip the top-level directory from the path
            # (tar file contains v2.0/ prefix)
            if member.name.startswith("v2.0/"):
                member.name = member.name[5:]  # Remove "v2.0/" prefix

            if member.name:  # Skip empty names
                tar.extract(member, extract_dir)

    if verbose:
        print(f"    Extracted to: {extract_dir}")


def download_evidence_inference_data(
    data_dir: Path | None = None,
    verbose: bool = True,
) -> None:
    """
    Download and extract Evidence Inference v2.0 dataset.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to save files. Defaults to data/EvidenceInference_v2/.
    verbose : bool
        Print progress information.

    Notes
    -----
    Skips download if files already exist.
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    all_exist = (
        all((data_dir / f).exists() for f in REQUIRED_FILES) and
        all((data_dir / d).exists() for d in REQUIRED_DIRS)
    )

    if all_exist:
        if verbose:
            print("Evidence Inference data already exists, skipping download.")
        return

    if verbose:
        print("Downloading Evidence Inference v2.0 dataset...")

    tar_path = data_dir / TAR_FILENAME

    # Download if tar file doesn't exist
    if not tar_path.exists():
        download_file(DOWNLOAD_URL, tar_path, show_progress=verbose)
    elif verbose:
        print(f"  Using existing tar file: {tar_path}")

    # Extract
    _extract_tarball(tar_path, data_dir, verbose=verbose)

    # Optionally remove tar file to save space
    # tar_path.unlink()

    if verbose:
        print("Download and extraction complete.")


# =============================================================================
# Loading Functions
# =============================================================================

def load_annotations(data_dir: Path) -> pl.DataFrame:
    """
    Load annotations_merged.csv.

    Parameters
    ----------
    data_dir : Path
        Directory containing the CSV file.

    Returns
    -------
    pl.DataFrame
        Annotations DataFrame.
    """
    filepath = data_dir / "annotations_merged.csv"
    return pl.read_csv(filepath)


def load_prompts(data_dir: Path) -> pl.DataFrame:
    """
    Load prompts_merged.csv.

    Parameters
    ----------
    data_dir : Path
        Directory containing the CSV file.

    Returns
    -------
    pl.DataFrame
        Prompts DataFrame.
    """
    filepath = data_dir / "prompts_merged.csv"
    return pl.read_csv(filepath)


def load_article_text(data_dir: Path, pmcid: int | str) -> str:
    """
    Load article text from txt_files directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing txt_files subdirectory.
    pmcid : int | str
        PubMed Central ID of the article.

    Returns
    -------
    str
        Article text content.
    """
    txt_path = data_dir / "txt_files" / f"PMC{pmcid}.txt"
    if not txt_path.exists():
        return ""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_splits(data_dir: Path) -> dict[str, list[int]]:
    """
    Load train/dev/test split definitions.

    Parameters
    ----------
    data_dir : Path
        Directory containing splits subdirectory.

    Returns
    -------
    dict[str, list[int]]
        Dictionary mapping split names to lists of prompt IDs.
    """
    splits_dir = data_dir / "splits"
    splits = {}

    for split_file in splits_dir.glob("*.txt"):
        split_name = split_file.stem  # e.g., "train", "dev", "test"
        with open(split_file, 'r') as f:
            # Each line contains a prompt ID
            ids = [int(line.strip()) for line in f if line.strip()]
        splits[split_name] = ids

    return splits


# =============================================================================
# Public API
# =============================================================================

def load_evidence_inference_raw(
    data_dir: Path | None = None,
    auto_download: bool = True,
    verbose: bool = True,
) -> dict[str, pl.DataFrame | dict]:
    """
    Load raw Evidence Inference data from CSV files.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/EvidenceInference_v2/.
    auto_download : bool
        If True, automatically download missing files.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict[str, pl.DataFrame | dict]
        Dictionary containing:
        - 'annotations': DataFrame with annotations
        - 'prompts': DataFrame with prompts
        - 'splits': Dictionary with train/dev/test split IDs

    Raises
    ------
    FileNotFoundError
        If required files are missing and auto_download is False.
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)

    # Check for required files
    missing_files = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    missing_dirs = [d for d in REQUIRED_DIRS if not (data_dir / d).exists()]

    if missing_files or missing_dirs:
        if auto_download:
            if verbose:
                print(f"Missing files: {missing_files}")
                print(f"Missing directories: {missing_dirs}")
            download_evidence_inference_data(data_dir=data_dir, verbose=verbose)
        else:
            raise FileNotFoundError(
                f"Missing required files in {data_dir}: {missing_files}\n"
                f"Missing required directories: {missing_dirs}\n"
                "Set auto_download=True to download automatically."
            )

    if verbose:
        print(f"Loading Evidence Inference data from: {data_dir}")

    annotations = load_annotations(data_dir)
    prompts = load_prompts(data_dir)
    splits = load_splits(data_dir)

    if verbose:
        print(f"  Annotations: {annotations.shape[0]} rows")
        print(f"  Prompts: {prompts.shape[0]} rows")
        print(f"  Splits: {list(splits.keys())}")
        for split_name, ids in splits.items():
            print(f"    {split_name}: {len(ids)} prompts")

    return {
        'annotations': annotations,
        'prompts': prompts,
        'splits': splits,
    }


# =============================================================================
# Processing Functions
# =============================================================================

def _get_evidence_spans_by_article_user(
    annotations: pl.DataFrame,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """
    Group evidence spans by (article, user), taking union of annotations per user.

    Parameters
    ----------
    annotations : pl.DataFrame
        Annotations DataFrame with PMCID, UserID, Evidence Start, Evidence End columns.

    Returns
    -------
    dict[tuple[int, int], list[tuple[int, int]]]
        Dictionary mapping (PMCID, UserID) to list of (start, end) evidence spans.
    """
    evidence_by_article_user: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for row in annotations.iter_rows(named=True):
        pmcid = row["PMCID"]
        user_id = row["UserID"]
        start = row["Evidence Start"]
        end = row["Evidence End"]

        # Skip invalid spans (marked as -1 in the dataset)
        if start < 0 or end < 0:
            continue

        key = (pmcid, user_id)
        if key not in evidence_by_article_user:
            evidence_by_article_user[key] = []

        evidence_by_article_user[key].append((start, end))

    return evidence_by_article_user


def _spans_overlap(span1: tuple[int, int], span2: tuple[int, int]) -> bool:
    """
    Check if two character spans overlap.

    Parameters
    ----------
    span1 : tuple[int, int]
        First span (start, end).
    span2 : tuple[int, int]
        Second span (start, end).

    Returns
    -------
    bool
        True if spans overlap, False otherwise.
    """
    start1, end1 = span1
    start2, end2 = span2
    return start1 < end2 and start2 < end1


def _label_sentences(
    sentences_with_spans: list[tuple[str, int, int]],
    evidence_spans: list[tuple[int, int]],
) -> list[int]:
    """
    Label sentences based on overlap with evidence spans.

    Parameters
    ----------
    sentences_with_spans : list[tuple[str, int, int]]
        List of (sentence_text, start, end) tuples.
    evidence_spans : list[tuple[int, int]]
        List of (start, end) evidence spans.

    Returns
    -------
    list[int]
        Binary labels (1 if sentence overlaps with any evidence span, 0 otherwise).
    """
    labels = []
    for sent_text, sent_start, sent_end in sentences_with_spans:
        # Check if this sentence overlaps with any evidence span
        is_evidence = any(
            _spans_overlap((sent_start, sent_end), ev_span)
            for ev_span in evidence_spans
        )
        labels.append(1 if is_evidence else 0)
    return labels


def _process_single_article_user(
    pmcid: int,
    user_id: int,
    article_text: str,
    evidence_spans: list[tuple[int, int]],
) -> dict | None:
    """
    Process a single article for a specific user into standardized format.

    Parameters
    ----------
    pmcid : int
        PubMed Central ID.
    user_id : int
        User ID who provided the annotations.
    article_text : str
        Full article text.
    evidence_spans : list[tuple[int, int]]
        List of evidence span (start, end) tuples from this user.

    Returns
    -------
    dict | None
        Processed sample with standardized columns, or None if article is empty.
    """
    if not article_text or not article_text.strip():
        return None

    # Split article into sentences with character spans
    sentences_with_spans = split_sentences_with_spans(article_text)

    if not sentences_with_spans:
        return None

    # Label sentences based on overlap with evidence spans
    labels = _label_sentences(sentences_with_spans, evidence_spans)

    # Extract sentence texts
    input_sentences = [sent for sent, start, end in sentences_with_spans]

    # Build summary from evidence sentences
    summary_sentences = [
        sent for sent, label in zip(input_sentences, labels) if label == 1
    ]
    summary = " ".join(summary_sentences)

    return {
        "intent": f"User{user_id}",
        "input": article_text,
        "input_sentences": input_sentences,
        "input_sentences_labels": labels,
        "summary": summary,
        "summary_sentences": summary_sentences,
    }


def process_evidence_inference_data(
    raw_data: dict[str, pl.DataFrame | dict],
    data_dir: Path | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process raw Evidence Inference data into standardized DataFrame format.

    Each (article, user) pair becomes a separate sample, with intent set to
    "User{N}" based on the annotator's UserID.

    Parameters
    ----------
    raw_data : dict[str, pl.DataFrame | dict]
        Dictionary from load_evidence_inference_raw().
    data_dir : Path | None
        Directory containing txt_files. If None, uses DATA_DIR.
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with columns:
        - intent: User perspective ("User0", "User1", etc.)
        - input: Full article text
        - input_sentences: List of sentences
        - input_sentences_labels: Binary labels (1 if evidence)
        - summary: Evidence sentences joined with space
        - summary_sentences: List of evidence sentences
    """
    data_dir = data_dir or DATA_DIR

    if verbose:
        print("Processing Evidence Inference data...")

    annotations = raw_data["annotations"]

    # Get evidence spans grouped by (article, user)
    evidence_by_article_user = _get_evidence_spans_by_article_user(annotations)

    # Get unique users for reporting
    unique_users = sorted(set(user_id for _, user_id in evidence_by_article_user.keys()))

    if verbose:
        print(f"  Found {len(evidence_by_article_user)} (article, user) pairs")
        print(f"  Unique users: {len(unique_users)} (User{min(unique_users)}-User{max(unique_users)})")

    # Cache article texts to avoid re-reading
    article_cache: dict[int, str] = {}

    # Process each (article, user) pair
    processed_samples = []
    keys = list(evidence_by_article_user.keys())

    iterator = keys
    if verbose:
        iterator = tqdm(keys, desc="Processing (article, user) pairs")

    for pmcid, user_id in iterator:
        # Load article text (with caching)
        if pmcid not in article_cache:
            article_cache[pmcid] = load_article_text(data_dir, pmcid)

        article_text = article_cache[pmcid]

        if not article_text:
            continue

        # Get evidence spans for this (article, user) pair
        evidence_spans = evidence_by_article_user[(pmcid, user_id)]

        # Process the article for this user
        processed = _process_single_article_user(
            pmcid, user_id, article_text, evidence_spans
        )

        if processed:
            processed_samples.append(processed)

    df = pl.DataFrame(processed_samples)

    if verbose:
        print(f"  Total processed: {df.shape[0]} samples")
        print(f"  Columns: {df.columns}")

        # Show intent distribution
        intent_counts = df["intent"].value_counts().sort("intent")
        print(f"  Intent distribution:")
        for row in intent_counts.iter_rows():
            print(f"    {row[0]}: {row[1]}")

    return df


def process_evidence_inference_dataset(
    data_dir: Path | None = None,
    output_filename: str = OUTPUT_FILENAME,
    auto_download: bool = True,
    save_output: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process Evidence Inference dataset into structured parquet format.

    Main entry point for Evidence Inference dataset processing. Downloads (if needed),
    loads, processes, and optionally saves the dataset.

    Each (article, user) pair becomes a separate sample, with intent set to
    "User{N}" based on the annotator's UserID (0-7).

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/EvidenceInference_v2/.
    output_filename : str
        Name of output parquet file. Defaults to "EvidenceInference_processed.parquet".
    auto_download : bool
        If True, automatically download missing files.
    save_output : bool
        If True, save result to parquet file. Defaults to True.
    verbose : bool
        If True, print progress information. Defaults to True.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with columns:
        - intent: User perspective ("User0", "User1", ..., "User7")
        - input: Full article text
        - input_sentences: List of sentences
        - input_sentences_labels: Binary labels (1 if evidence)
        - summary: Evidence sentences joined with space
        - summary_sentences: List of evidence sentences

    Examples
    --------
    >>> df = process_evidence_inference_dataset(verbose=True)
    >>> df.shape
    (2381, 6)
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)
    output_path = data_dir / output_filename

    if verbose:
        print("=" * 60)
        print("Evidence Inference Dataset Processing Pipeline")
        print("=" * 60)
        print(f"\nData directory: {data_dir}")
        print(f"Output: {output_path}")

    # Step 1: Load raw data (with optional download)
    if verbose:
        print("\n[Step 1/2] Loading raw Evidence Inference data...")

    raw_data = load_evidence_inference_raw(
        data_dir=data_dir,
        auto_download=auto_download,
        verbose=verbose,
    )

    # Step 2: Process data
    if verbose:
        print("\n[Step 2/2] Processing data...")

    df_result = process_evidence_inference_data(raw_data, data_dir=data_dir, verbose=verbose)

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
        n_intents = df_result["intent"].n_unique()
        avg_sentences = (
            df_result.select(pl.col("input_sentences").list.len().mean())
            .item()
        )
        avg_evidence = (
            df_result.select(
                pl.col("input_sentences_labels").list.eval(pl.element().sum()).list.first().mean()
            )
            .item()
        )

        print(f"\nDataset statistics:")
        print(f"  Total samples: {n_samples}")
        print(f"  Unique intents (users): {n_intents}")
        print(f"  Avg sentences per sample: {avg_sentences:.1f}")
        print(f"  Avg evidence sentences per sample: {avg_evidence:.1f}")

    return df_result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for Evidence Inference data processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Evidence Inference v2.0 dataset into parquet format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory for input/output files (default: data/EvidenceInference_v2/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILENAME,
        help=f"Output parquet filename (default: {OUTPUT_FILENAME})"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download data, don't process"
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

    if args.download_only:
        download_evidence_inference_data(
            data_dir=args.data_dir,
            verbose=not args.quiet,
        )
    else:
        process_evidence_inference_dataset(
            data_dir=args.data_dir,
            output_filename=args.output,
            auto_download=not args.no_download,
            save_output=not args.no_save,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
