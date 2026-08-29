"""
PUMA Dataset (Patient-generated User-centered Medical Answers) processing.

This module provides functions to process the PUMA dataset from raw JSON files
into a structured parquet format suitable for conformal prediction tasks.

Required input files (in data/PUMA/):
    - PUMA_complete_Data.json: Raw PUMA dataset

Output file:
    - PUMA_processed.parquet
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from ._common import get_project_data_dir, split_sentences


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = get_project_data_dir() / "PUMA"
INPUT_FILENAME = "PUMA_complete_Data.json"
OUTPUT_FILENAME = "PUMA_processed.parquet"

# All possible perspectives in the PUMA dataset (these become the 5 distinct intents)
ALL_PERSPECTIVES = ["CAUSE", "SUGGESTION", "INFORMATION", "EXPERIENCE", "QUESTION"]

# Known corrections for data quality issues in the raw dataset
DEFAULT_CORRECTIONS = {
    89: {
        'labelled_summaries': {
            'EXPERIENCE_SUMMARY': 'In users experience, an individual knows someone who had shingles and came in direct contact with a child who already had the vaccine, that child later broke out with the chicken pox. It is recommended to stay away from kids if you have shingles.',
            'INFORMATION_SUMMARY': 'A child can still get chicken pox even with the vaccine so it is better to stay away.'
        }
    }
}


# =============================================================================
# Private Helper Functions
# =============================================================================

def _get_answer_sentences_with_spans(raw_text: str) -> list[tuple[str, int, int]]:
    """
    Extract sentences from answer blocks along with their character spans.

    Parses the raw_text field from PUMA records, extracting individual sentences
    from each answer_N: block using NLTK's sentence tokenizer for robust boundary
    detection. Tracks character positions for span-based label alignment.

    Parameters
    ----------
    raw_text : str
        Raw text containing answer blocks in format "answer_N: <text>".

    Returns
    -------
    List[Tuple[str, int, int]]
        List of (sentence_text, start_char_index, end_char_index) tuples.

    Notes
    -----
    Uses NLTK's sent_tokenize which correctly handles:
    - Decimal numbers (e.g., 102.0, 26.7°)
    - Abbreviations (e.g., Dr., Mr., etc.)
    - Currency amounts (e.g., $750.00)
    """
    sentences_with_spans = []

    # Find each answer block and its span in the raw_text
    answer_blocks = re.finditer(r'answer_\d+:\s*(.*?)(?:\n|$)', raw_text, re.DOTALL)

    for block_match in answer_blocks:
        block_text = block_match.group(1)
        block_start_offset = block_match.start(1)

        if not block_text.strip():
            continue

        # Use NLTK-based sentence splitting
        sentences = split_sentences(block_text)

        # Track spans by finding each sentence in the original block
        search_start = 0
        for sent in sentences:
            # Find the sentence in the original block text
            idx = block_text.find(sent, search_start)
            if idx >= 0:
                sentence_start_index = block_start_offset + idx
                sentence_end_index = block_start_offset + idx + len(sent)
                sentences_with_spans.append(
                    (sent, sentence_start_index, sentence_end_index)
                )
                search_start = idx + len(sent)
            else:
                # Fallback: if exact match not found (rare), use approximate position
                # This can happen if NLTK slightly modifies whitespace
                sentence_start_index = block_start_offset + search_start
                sentence_end_index = sentence_start_index + len(sent)
                sentences_with_spans.append(
                    (sent, sentence_start_index, sentence_end_index)
                )
                search_start += len(sent)

    return sentences_with_spans


def _create_answer_labels(
    raw_text: str,
    labelled_spans: dict[str, list[dict[str, Any]]],
) -> dict[str, list[int]]:
    """
    Generate ground truth relevance labels for answer sentences across all perspectives.

    Parameters
    ----------
    raw_text : str
        Raw text containing answer blocks.
    labelled_spans : Dict[str, List[Dict[str, Any]]]
        Dictionary mapping perspective names to lists of span annotations.

    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping each perspective to a list of binary labels
        (0 or 1) aligned with the answer sentences extracted from raw_text.
    """
    sentences_with_spans = _get_answer_sentences_with_spans(raw_text)

    ground_truth_labels = {}

    for perspective in ALL_PERSPECTIVES:
        spans_list = labelled_spans.get(perspective, [])
        labels = [0] * len(sentences_with_spans)

        if spans_list:
            for labeled_span in spans_list:
                start_span, end_span = labeled_span['label_spans']

                for i, (sentence, sentence_start, sentence_end) in enumerate(
                    sentences_with_spans
                ):
                    # Check for any overlap between the labeled span and sentence span
                    if max(start_span, sentence_start) < min(end_span, sentence_end):
                        labels[i] = 1

        ground_truth_labels[perspective] = labels

    return ground_truth_labels


def _transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw PUMA DataFrame by splitting rows per perspective.

    For each row in the input DataFrame:
    1. Extracts sentences from question, context, and answers
    2. Generates ground truth labels for all 5 perspectives
    3. Creates extractive summaries by selecting labeled sentences
    4. Creates a new row for each perspective with at least one positive label

    Parameters
    ----------
    df : pd.DataFrame
        Raw PUMA DataFrame with columns: question, context, raw_text,
        labelled_answer_spans, labelled_summaries.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with columns:
        - intent: Perspective name (CAUSE, SUGGESTION, INFORMATION, EXPERIENCE, QUESTION)
        - input: Full text (question + context + answers joined)
        - input_sentences: List of all sentences
        - input_sentences_labels: Binary labels (0 for question/context, 0/1 for answers)
        - summary: Extractive summary (selected sentences joined)
        - summary_sentences: List of selected sentences (where label=1)
    """
    new_rows = []

    for row in df.to_dict('records'):
        question = row['question'] or ''
        context = row['context'] or ''
        raw_text = row['raw_text']
        labelled_spans = row['labelled_answer_spans']

        # Split question and context into sentences using NLTK
        question_sentences = split_sentences(question)
        context_sentences = split_sentences(context)

        # Extract answer sentences with spans (for label alignment)
        answer_sentences_with_spans = _get_answer_sentences_with_spans(raw_text)
        answer_sentences = [s[0] for s in answer_sentences_with_spans]

        # Combine all sentences: question + context + answers
        all_sentences = question_sentences + context_sentences + answer_sentences

        # Number of prefix sentences (question + context) - these get label 0
        num_prefix_sentences = len(question_sentences) + len(context_sentences)

        # Get answer-only labels for all perspectives
        answer_labels_by_perspective = _create_answer_labels(raw_text, labelled_spans)

        # Create a row for each perspective that has at least one positive label
        for perspective, answer_labels in answer_labels_by_perspective.items():
            if any(label == 1 for label in answer_labels):
                # Prepend 0s for question/context sentences
                full_labels = [0] * num_prefix_sentences + answer_labels

                # Create extractive summary from selected sentences
                summary_sentences = [
                    sent for sent, label in zip(all_sentences, full_labels)
                    if label == 1
                ]
                summary = ' '.join(summary_sentences)

                # Create full input text
                input_text = ' '.join(all_sentences)

                new_row = {
                    'intent': perspective,
                    'input': input_text,
                    'input_sentences': all_sentences,
                    'input_sentences_labels': full_labels,
                    'summary': summary,
                    'summary_sentences': summary_sentences,
                }
                new_rows.append(new_row)

    return pd.DataFrame(new_rows)


# =============================================================================
# Public API
# =============================================================================

def get_sentences_with_spans(raw_text: str) -> list[tuple[str, int, int]]:
    """
    Extract sentences from answer blocks along with their character spans.

    This is the public interface for sentence extraction from PUMA raw_text.

    Parameters
    ----------
    raw_text : str
        Raw text containing answer blocks in format "answer_N: <text>".

    Returns
    -------
    List[Tuple[str, int, int]]
        List of (sentence_text, start_char_index, end_char_index) tuples.

    Examples
    --------
    >>> raw_text = "answer_0: First sentence. Second sentence.\\nanswer_1: Third."
    >>> sentences = get_sentences_with_spans(raw_text)
    >>> [s[0] for s in sentences]
    ['First sentence.', 'Second sentence.', 'Third.']
    """
    return _get_answer_sentences_with_spans(raw_text)


def create_ground_truth_labels(
    raw_text: str,
    labelled_spans: dict[str, list[dict[str, Any]]],
) -> dict[str, list[int]]:
    """
    Generate ground truth relevance labels for answer sentences across all perspectives.

    This is the public interface for label generation.

    Parameters
    ----------
    raw_text : str
        Raw text containing answer blocks.
    labelled_spans : Dict[str, List[Dict[str, Any]]]
        Dictionary mapping perspective names to lists of span annotations.

    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping each perspective to a list of binary labels.

    Examples
    --------
    >>> labels = create_ground_truth_labels(raw_text, labelled_spans)
    >>> labels['INFORMATION']
    [1, 0, 0, 1, 0]
    """
    return _create_answer_labels(raw_text, labelled_spans)


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw PUMA DataFrame by splitting rows per perspective.

    This is the public interface for DataFrame transformation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PUMA DataFrame.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with perspective-level rows.
    """
    return _transform_dataframe(df)


def process_puma_dataset(
    data_dir: Path | None = None,
    input_filename: str = INPUT_FILENAME,
    output_filename: str = OUTPUT_FILENAME,
    corrections: dict[int, dict[str, Any]] | None = DEFAULT_CORRECTIONS,
    save_output: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process raw PUMA dataset into structured parquet format.

    Main entry point for PUMA dataset processing. Loads the raw JSON file,
    applies optional corrections, transforms rows by perspective, and
    optionally saves the result.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/PUMA/.
    input_filename : str
        Name of input JSON file. Defaults to "PUMA_complete_Data.json".
    output_filename : str
        Name of output parquet file. Defaults to "PUMA_processed.parquet".
    corrections : Dict[int, Dict[str, Any]], optional
        Dictionary mapping row indices to corrected field values.
        Defaults to DEFAULT_CORRECTIONS which fixes known data quality issues.
        Pass None or {} to skip corrections.
    save_output : bool
        If True, save result to parquet file. Defaults to True.
    verbose : bool
        If True, print progress information. Defaults to True.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with columns:
        - intent: Perspective name (5 distinct values)
        - input: Full text (question + context + answers)
        - input_sentences: List of all sentences
        - input_sentences_labels: Binary labels
        - summary: Extractive summary text
        - summary_sentences: List of selected sentences

    Raises
    ------
    FileNotFoundError
        If required input files are missing.

    Examples
    --------
    >>> df = process_puma_dataset(verbose=True)
    >>> df.shape
    (6282, 6)

    >>> # Without default corrections
    >>> df = process_puma_dataset(corrections=None)
    """
    data_dir = data_dir or DATA_DIR
    input_path = data_dir / input_filename
    output_path = data_dir / output_filename

    # Check for required input file
    if not input_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {input_path}\n"
            "Please ensure PUMA_complete_Data.json is present in the data directory."
        )

    if verbose:
        print("=" * 60)
        print("PUMA Dataset Processing Pipeline")
        print("=" * 60)
        print(f"\nInput: {input_path}")
        print(f"Output: {output_path}")

    # Step 1: Load raw JSON data
    if verbose:
        print("\n[Step 1/3] Loading raw PUMA data...")

    df_puma = pd.read_json(input_path, orient='records')

    if verbose:
        print(f"  Loaded {df_puma.shape[0]} records with {df_puma.shape[1]} columns")

    # Step 2: Apply corrections if provided
    if corrections:
        if verbose:
            print(f"\n[Step 2/3] Applying {len(corrections)} correction(s)...")

        for idx, correction_dict in corrections.items():
            for field, value in correction_dict.items():
                df_puma.at[idx, field] = value
                if verbose:
                    print(f"  Corrected row {idx}, field '{field}'")
    elif verbose:
        print("\n[Step 2/3] No corrections to apply")

    # Step 3: Transform DataFrame
    if verbose:
        print("\n[Step 3/3] Transforming DataFrame (splitting by perspective)...")

    df_processed = _transform_dataframe(df_puma)

    if verbose:
        print(f"  Transformed {df_puma.shape[0]} rows -> {df_processed.shape[0]} rows")
        print(f"  Columns: {list(df_processed.columns)}")
        # Show unique intents
        intents = df_processed['intent'].unique().tolist()
        print(f"  Unique intents ({len(intents)}): {intents}")

    # Convert to Polars for consistency with other functions
    df_result = pl.from_pandas(df_processed)

    # Save output
    if save_output:
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
    """Command-line entry point for PUMA data processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process PUMA dataset into parquet format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing input files (default: data/PUMA/)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_FILENAME,
        help=f"Input JSON filename (default: {INPUT_FILENAME})"
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

    process_puma_dataset(
        data_dir=args.data_dir,
        input_filename=args.input,
        output_filename=args.output,
        save_output=not args.no_save,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
