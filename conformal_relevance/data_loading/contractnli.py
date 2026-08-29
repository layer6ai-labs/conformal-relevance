"""
ContractNLI Dataset processing.

Processes the ContractNLI dataset (Koreeda & Manning, EMNLP Findings 2021) into
a structured parquet format suitable for conformal prediction tasks.

Dataset: 607 NDAs annotated against 17 fixed hypotheses (intents). Each (contract,
hypothesis) pair has a label: Entailment, Contradiction, or NotMentioned, plus
a list of evidence span indices (pre-segmented sentences/list items).

Only Entailment and Contradiction pairs are kept (NotMentioned have no positive
spans; AP is undefined for all-zero labels).

Source:
    - Paper: https://aclanthology.org/2021.findings-emnlp.164
    - Data:  https://stanfordnlp.github.io/contract-nli/
    - License: Apache 2.0 (Hitachi America)

Input files (pre-downloaded, placed in data/ContractNLI/contract-nli/):
    - train.json
    - dev.json
    - test.json

Output file:
    - data/ContractNLI/ContractNLI_processed.parquet
"""

import json
from pathlib import Path

import polars as pl
from tqdm import tqdm

from ._common import get_project_data_dir


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = get_project_data_dir() / "ContractNLI"
RAW_SUBDIR = "contract-nli"
OUTPUT_FILENAME = "ContractNLI_processed.parquet"

# Split filenames inside RAW_SUBDIR
SPLIT_FILES = ["train.json", "dev.json", "test.json"]


# =============================================================================
# Loading Functions
# =============================================================================

def load_raw_split(json_path: Path) -> dict:
    """Load a single ContractNLI JSON split file."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Processing Functions
# =============================================================================

def _process_document_hypothesis(
    doc: dict,
    hyp_id: str,
    hyp_text: str,
    choice: str,
    evidence_span_indices: list[int],
    source_split: str = "",
) -> dict:
    """
    Convert a single (document, hypothesis) pair into a standardized row.

    Parameters
    ----------
    doc : dict
        ContractNLI document dict with 'text' and 'spans' fields.
    hyp_id : str
        Hypothesis identifier (e.g. 'nda-11').
    hyp_text : str
        Full hypothesis text used as the intent string.
    choice : str
        NLI label: 'Entailment' or 'Contradiction'.
    evidence_span_indices : list[int]
        Indices into doc['spans'] that are evidence for this hypothesis.
    source_split : str
        Original dataset split this sample came from ('train', 'dev', or 'test').

    Returns
    -------
    dict
        Row with standardized columns matching the schema.
    """
    text = doc["text"]
    char_spans = doc["spans"]  # list of [start_char, end_char]

    # Extract span texts
    input_sentences = [text[s[0]:s[1]] for s in char_spans]

    # Build binary labels: 1 if span index is in evidence set
    evidence_set = set(evidence_span_indices)
    input_sentences_labels = [1 if i in evidence_set else 0 for i in range(len(char_spans))]

    # output_text = concatenation of evidence span texts
    summary_sentences = [input_sentences[i] for i in sorted(evidence_set)]
    summary = " ".join(summary_sentences)

    return {
        "intent": hyp_text,
        "input": text,
        "input_sentences": input_sentences,
        "input_sentences_labels": input_sentences_labels,
        "summary": summary,
        "summary_sentences": summary_sentences,
        # Metadata (not used in scoring, but useful for debugging)
        "hyp_id": hyp_id,
        "nli_label": choice,
        "doc_id": doc.get("file_name", doc.get("id", "")),
        "source_split": source_split,
    }


def process_contractnli_data(
    raw_dir: Path,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process ContractNLI JSON splits into a single standardized DataFrame.

    Loads train/dev/test splits, filters to Entailment + Contradiction pairs,
    and builds one row per (document, hypothesis) pair.

    Parameters
    ----------
    raw_dir : Path
        Directory containing train.json, dev.json, test.json.
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with columns:
        - intent: Full hypothesis text
        - input: Full contract text
        - input_sentences: Pre-segmented span texts
        - input_sentences_labels: Binary labels (1 if evidence span)
        - summary: Evidence spans joined with space
        - summary_sentences: List of evidence span texts
        - hyp_id: Hypothesis ID (e.g. 'nda-11')
        - nli_label: 'Entailment' or 'Contradiction'
        - doc_id: Contract filename
        - source_split: Original split ('train', 'dev', or 'test')
    """
    all_rows = []
    skipped_not_mentioned = 0

    for split_file in SPLIT_FILES:
        path = raw_dir / split_file
        if not path.exists():
            raise FileNotFoundError(
                f"Missing ContractNLI split file: {path}\n"
                f"Download from https://stanfordnlp.github.io/contract-nli/ "
                f"and extract to {raw_dir}/"
            )

        data = load_raw_split(path)
        documents = data["documents"]
        labels = data["labels"]  # {hyp_id: {short_description, hypothesis}}
        # Derive source split name from filename (e.g. "train.json" -> "train")
        source_split = split_file.replace(".json", "")

        if verbose:
            print(f"  {split_file}: {len(documents)} documents, {len(labels)} hypotheses")

        iterator = documents
        if verbose:
            iterator = tqdm(documents, desc=f"Processing {split_file}", leave=False)

        for doc in iterator:
            ann_set = doc["annotation_sets"][0]["annotations"]
            for hyp_id, ann in ann_set.items():
                choice = ann["choice"]
                if choice == "NotMentioned":
                    skipped_not_mentioned += 1
                    continue

                hyp_text = labels[hyp_id]["hypothesis"]
                evidence_span_indices = ann["spans"]

                row = _process_document_hypothesis(
                    doc=doc,
                    hyp_id=hyp_id,
                    hyp_text=hyp_text,
                    choice=choice,
                    evidence_span_indices=evidence_span_indices,
                    source_split=source_split,
                )
                all_rows.append(row)

    if verbose:
        print(f"  Skipped {skipped_not_mentioned} NotMentioned pairs (no positives)")
        print(f"  Total usable rows: {len(all_rows)}")

    df = pl.DataFrame(all_rows)
    return df


# =============================================================================
# Public API
# =============================================================================

def process_contractnli_dataset(
    data_dir: Path | None = None,
    output_filename: str = OUTPUT_FILENAME,
    save_output: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Process ContractNLI dataset into structured parquet format.

    Requires pre-downloaded data in data/ContractNLI/contract-nli/.
    Download from https://stanfordnlp.github.io/contract-nli/ and extract there.

    Each (contract, hypothesis) pair with Entailment or Contradiction label
    becomes one sample. Intent = full hypothesis text. Input units = pre-segmented
    contract spans. Labels = 1 for evidence spans, 0 otherwise.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the contract-nli/ subdirectory.
        Defaults to data/ContractNLI/.
    output_filename : str
        Name of output parquet file.
    save_output : bool
        If True, save result to parquet file.
    verbose : bool
        Print progress information.

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with columns:
        - intent: Full hypothesis text (17 unique values)
        - input: Full contract text
        - input_sentences: Pre-segmented span texts
        - input_sentences_labels: Binary labels (1 if evidence span)
        - summary: Evidence spans joined with space
        - summary_sentences: List of evidence span texts
        - hyp_id: Hypothesis ID (e.g. 'nda-11')
        - nli_label: 'Entailment' or 'Contradiction'
        - doc_id: Contract filename
        - source_split: Original split ('train', 'dev', or 'test')

    Examples
    --------
    >>> df = process_contractnli_dataset(verbose=True)
    >>> df.shape[0]  # ~6173 usable pairs
    6173
    """
    data_dir = data_dir or DATA_DIR
    data_dir = Path(data_dir)
    raw_dir = data_dir / RAW_SUBDIR
    output_path = data_dir / output_filename

    if verbose:
        print("=" * 60)
        print("ContractNLI Dataset Processing Pipeline")
        print("=" * 60)
        print(f"\nData directory: {data_dir}")
        print(f"Raw data directory: {raw_dir}")
        print(f"Output: {output_path}")
        print("\nProcessing splits...")

    df = process_contractnli_data(raw_dir=raw_dir, verbose=verbose)

    if save_output:
        data_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

    if verbose:
        print("\n" + "=" * 60)
        if save_output:
            print(f"Output saved to: {output_path}")
        print(f"Final shape: {df.shape}")
        print(f"Columns: {df.columns}")

        n_samples = df.shape[0]
        n_intents = df["intent"].n_unique()
        avg_sentences = df.select(pl.col("input_sentences").list.len().mean()).item()
        avg_evidence = (
            df.select(
                pl.col("input_sentences_labels").list.eval(pl.element().sum()).list.first().mean()
            ).item()
        )
        mean_pos_rate = (
            df.select(
                (
                    pl.col("input_sentences_labels").list.eval(pl.element().sum()).list.first()
                    / pl.col("input_sentences").list.len()
                ).mean()
            ).item()
        )

        print(f"\nDataset statistics:")
        print(f"  Total samples: {n_samples}")
        print(f"  Unique intents (hypotheses): {n_intents}")
        print(f"  Avg spans per sample: {avg_sentences:.1f}")
        print(f"  Avg evidence spans per sample: {avg_evidence:.1f}")
        print(f"  Mean positive rate: {mean_pos_rate:.1%}")

        label_dist = df["nli_label"].value_counts().sort("nli_label")
        print(f"\n  NLI label distribution:")
        for row in label_dist.iter_rows():
            print(f"    {row[0]}: {row[1]}")

    return df


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for ContractNLI data processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ContractNLI dataset into parquet format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing contract-nli/ subdirectory (default: data/ContractNLI/)"
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

    process_contractnli_dataset(
        data_dir=args.data_dir,
        output_filename=args.output,
        save_output=not args.no_save,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
