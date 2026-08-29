"""
Common utilities shared across dataset processing modules.

This module provides helper functions used by multiple dataset processors,
including NLTK utilities and general-purpose helper functions.
"""

import re
from pathlib import Path
from typing import List, Tuple

import polars as pl


# ---------- Sentence splitting utilities ----------
def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK's sent_tokenize.

    Handles edge cases like decimal numbers (102.0), abbreviations (Dr., Mr.),
    and other patterns that simple regex-based splitting would break.

    Parameters
    ----------
    text : str
        Input text to split into sentences.

    Returns
    -------
    List[str]
        List of sentences.

    Examples
    --------
    >>> split_sentences("Dr. Smith recommended 2.5 mg of medicine.")
    ['Dr. Smith recommended 2.5 mg of medicine.']

    >>> split_sentences("First sentence. Second sentence.")
    ['First sentence.', 'Second sentence.']
    """
    if not text or not text.strip():
        return []

    ensure_nltk_data()
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)

    # Post-process: merge isolated punctuation with previous sentence
    # This handles cases like "question.?" where NLTK might split the "?"
    result = []
    for sent in sentences:
        stripped = sent.strip()
        # If this is just punctuation and we have a previous sentence, merge
        if stripped and all(c in '.!?,' for c in stripped) and result:
            result[-1] = result[-1] + stripped
        elif stripped:
            result.append(stripped)

    return result


def split_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences and track their character positions.

    Uses NLTK's sent_tokenize for robust sentence boundary detection,
    then finds each sentence's position in the original text.

    Parameters
    ----------
    text : str
        Input text to split into sentences.

    Returns
    -------
    List[Tuple[str, int, int]]
        List of (sentence_text, start_index, end_index) tuples.

    Examples
    --------
    >>> result = split_sentences_with_spans("First. Second.")
    >>> [(s, start, end) for s, start, end in result]
    [('First.', 0, 6), ('Second.', 7, 14)]
    """
    if not text or not text.strip():
        return []

    sentences = split_sentences(text)
    result = []
    search_start = 0

    for sent in sentences:
        # Find the sentence in the original text
        idx = text.find(sent, search_start)
        if idx >= 0:
            result.append((sent, idx, idx + len(sent)))
            search_start = idx + len(sent)
        else:
            # Fallback: if exact match not found, try to find approximate position
            # This can happen if NLTK modifies the sentence slightly
            # In this case, append at current position with estimated span
            result.append((sent, search_start, search_start + len(sent)))
            search_start += len(sent)

    return result


# ---------- Helpers for record_id parsing ----------
_SENT_RE = re.compile(r"_sent_(\d+)$")


def base_id(rid: str) -> str:
    """Extract base ID from record_id by removing _sent_N suffix."""
    return _SENT_RE.sub("", rid)


def sent_idx(rid: str) -> int:
    """Extract sentence index from record_id."""
    m = _SENT_RE.search(rid)
    return int(m.group(1)) if m else -1


def groupby(df: pl.DataFrame, *keys, **kwargs):
    """Version-safe group_by for Polars (handles API changes)."""
    return df.group_by(*keys, **kwargs) if hasattr(df, "group_by") else df.groupby(*keys, **kwargs)


# ---------- NLTK utilities ----------
def ensure_nltk_data():
    """Download required NLTK data if not present."""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)


# ---------- Path utilities ----------
def get_project_data_dir() -> Path:
    """Get the project's data directory path."""
    return Path(__file__).parent.parent.parent / "data"


# ---------- Download utilities ----------
def download_file(url: str, dest: Path, show_progress: bool = True) -> None:
    """Download a file from URL to dest with optional tqdm progress bar.

    Parameters
    ----------
    url : str
        URL to download from.
    dest : Path
        Destination path where the file will be saved.
    show_progress : bool
        Print progress information and show tqdm progress bar (default True).
    """
    import requests
    from tqdm import tqdm

    if show_progress:
        print(f"  Downloading: {dest.name}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        if show_progress and total_size > 0:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    if show_progress:
        print(f"    Saved to: {dest}")


def download_dataset_files(
    data_dir: Path | None,
    default_dir: Path,
    download_urls: dict[str, str],
    dataset_label: str,
    verbose: bool = True,
) -> Path:
    """Download a set of dataset files, skipping any that already exist.

    Parameters
    ----------
    data_dir : Path or None
        Target directory. Uses default_dir if None.
    default_dir : Path
        Default directory when data_dir is not specified.
    download_urls : dict[str, str]
        Mapping of filename → URL.
    dataset_label : str
        Human-readable dataset name used in progress messages.
    verbose : bool
        Print progress information.

    Returns
    -------
    Path
        Resolved target directory.
    """
    resolved = Path(data_dir) if data_dir is not None else default_dir
    resolved.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Downloading {dataset_label} dataset...")
    for filename, url in download_urls.items():
        target_path = resolved / filename
        if target_path.exists():
            if verbose:
                print(f"  Skipping {filename} (already exists)")
            continue
        download_file(url, target_path, show_progress=verbose)
    if verbose:
        print("Download complete.")
    return resolved
