"""
Physionet Gold Corpus (I2B2 2014 De-identification) dataset processing.

This module provides functions to process the I2B2 2014 De-identification dataset
from raw files into a structured parquet format suitable for conformal prediction tasks.

Required input files (in data/PhysionetGoldCorpus/):
    - id.text: Raw medical text records (requires PhysioNet license)
    - I2B2-2014-Relabeled-PhysionetGoldCorpus.csv: Annotation file
    - corrections.txt: Manual annotation corrections (optional)

Output file:
    - physionet_goldcorpus_i2b2deid_claimlevel.parquet
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import polars as pl

from ._common import base_id, sent_idx, groupby, ensure_nltk_data, get_project_data_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = get_project_data_dir() / "PhysionetGoldCorpus"
DEFAULT_OUTPUT_FILENAME = "physionet_goldcorpus_i2b2deid_claimlevel.parquet"

# Required input files
TEXT_FILE = "id.text"
ANNOTATIONS_FILE = "I2B2-2014-Relabeled-PhysionetGoldCorpus.csv"
CORRECTIONS_FILE = "corrections.txt"

# Delimiter Configuration
# Hard delimiters always create segment breaks (colon removed - handled contextually)
DEFAULT_HARD_DELIMS = {";", "—", "--", "|"}

# Soft delimiters used for overflow splitting
DEFAULT_SOFT_DELIMS = {","}

# Conjunctions for splitting long segments
DEFAULT_CONJUNCTIONS = {
    "AND", "BUT", "OR", "NOR", "YET", "SO", "THEN",
    "HOWEVER", "THEREFORE", "ALSO", "MEANWHILE", "FURTHERMORE"
}

# Common clinical headers (base forms without colon)
COMMON_HEADERS = {
    "HPI", "PMH", "PSH", "MEDS", "MEDICATIONS", "ALLERGIES", "ROS",
    "PE", "LABS", "ASSESSMENT", "PLAN", "DX", "RX", "IMPRESSION",
    "HISTORY", "EXAM", "VITALS", "CC", "CHIEF", "COMPLAINT",
    "SUBJECTIVE", "OBJECTIVE", "A", "P", "S", "O",  # SOAP note components
    "DIAGNOSIS", "TREATMENT", "FINDINGS", "PROCEDURES", "RESULTS",
    "FAMILY", "SOCIAL", "REVIEW", "PHYSICAL", "MENTAL", "STATUS",
}


@dataclass
class PhysioNetConfig:
    """Configuration for PhysioNet corpus processing (delimiter/header defaults)."""

    hard_delims: set = field(default_factory=lambda: DEFAULT_HARD_DELIMS.copy())
    soft_delims: set = field(default_factory=lambda: DEFAULT_SOFT_DELIMS.copy())
    conjunctions: set = field(default_factory=lambda: DEFAULT_CONJUNCTIONS.copy())
    custom_headers: Optional[set] = None


# =============================================================================
# Private Helper Functions
# =============================================================================

def _load_records(filename: Path) -> List[Dict[str, str]]:
    """Load records from text file."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'(START_OF_RECORD=.*?\|\|\|\|END_OF_RECORD)'
    records = re.findall(pattern, content, flags=re.DOTALL)
    cleaned_records = []

    for record in records:
        text = record
        for marker in ["START_OF_RECORD=", "||||END_OF_RECORD"]:
            text = text.replace(marker, "")

        match_res = re.match(r'^(\d+\|\|\|\|\d+\|\|\|\|)', text)
        record_id = match_res.group(1) if match_res else None
        text = re.sub(r'^\d+\|\|\|\|\d+\|\|\|\|', '', text)
        cleaned_records.append({"record_id": record_id, "text": text})

    return cleaned_records


def _load_annotations(file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load annotations from CSV file."""
    annotations = pd.read_csv(file_path)
    grouped_annotations = annotations.groupby('record_id')

    id2annotations = {}
    for record_id, group in grouped_annotations:
        if record_id not in id2annotations:
            id2annotations[record_id] = []

        for el in group.itertuples():
            annotation = {
                'start': el.begin,
                'end': el.begin + el.length,
                'type': el.type
            }
            id2annotations[record_id].append(annotation)
    return id2annotations


def _map_annotations_to_records(
    id2annotations: Dict[str, List[Dict[str, Any]]],
    records_txt: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Map annotations to records."""
    ds_annotations = []
    for item in records_txt:
        text = item["text"]
        ann_dict = {
            "record_id": item["record_id"],
            "text": text,
            "annotations": []
        }

        if item["record_id"] in id2annotations:
            annotations = id2annotations[item["record_id"]]

            for annotation in annotations:
                start = annotation['start'] + 1  # first char not included
                end = annotation['end'] + 1  # last char included
                ent_type = annotation['type']
                ann_dict["annotations"].append({
                    "start": start,
                    "end": end,
                    "span": text[start:end],
                    "type": ent_type
                })
            ds_annotations.append(ann_dict)
    return ds_annotations


def _split_long_sentence(sentence: str, max_tokens: int = 150) -> List[str]:
    """Split a long sentence into smaller chunks based on punctuation and conjunctions."""
    import nltk

    if len(nltk.word_tokenize(sentence)) <= max_tokens:
        return [sentence]

    split_patterns = [
        r'\n+',
        r';\s+',
        r',\s+(?:and|but|or|yet|so|for|nor)\s+',
        r',\s+(?:however|therefore|moreover|furthermore|nevertheless|meanwhile|consequently)\s+',
        r',\s+(?:which|that|who|where|when)\s+',
        r',\s+(?:with|without|including|excluding|during|after|before)\s+',
        r',\s+',
        r'\s+(?:and|but|or)\s+',
    ]

    chunks = [sentence]

    for pattern in split_patterns:
        new_chunks = []
        split_made = False

        for chunk in chunks:
            if len(nltk.word_tokenize(chunk)) <= max_tokens:
                new_chunks.append(chunk)
                continue

            parts = re.split(f'({pattern})', chunk)
            if len(parts) > 1:
                current_segment = ""
                for i, part in enumerate(parts):
                    if re.match(pattern, part):
                        current_segment += part
                    else:
                        if current_segment:
                            current_segment += part
                        else:
                            current_segment = part

                        if (i == len(parts) - 1 or
                            len(nltk.word_tokenize(current_segment)) >= max_tokens * 0.7):
                            new_chunks.append(current_segment.strip())
                            current_segment = ""
                            split_made = True

                if current_segment.strip():
                    new_chunks.append(current_segment.strip())
            else:
                new_chunks.append(chunk)

        chunks = new_chunks
        if split_made and all(len(nltk.word_tokenize(chunk)) <= max_tokens for chunk in chunks):
            break

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_into_sentences(text: str, max_tokens: int = 150) -> List[str]:
    """Split text into sentences using NLTK sentence tokenizer."""
    import nltk

    sentences = nltk.sent_tokenize(text)

    final_sentences = []
    for sentence in sentences:
        chunks = _split_long_sentence(sentence, max_tokens)
        final_sentences.extend(chunks)

    return final_sentences


def _map_annotations_to_sentences(
    original_text: str,
    sentences: List[str],
    annotations: List[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    """Map annotations to individual sentences."""
    sentence_annotations = []
    current_pos = 0

    for sentence in sentences:
        sentence_start = original_text.find(sentence, current_pos)
        if sentence_start == -1:
            sentence_start = original_text.find(sentence, current_pos)
            if sentence_start == -1:
                sentence_annotations.append([])
                continue

        sentence_end = sentence_start + len(sentence)

        sentence_anns = []
        for ann in annotations:
            ann_start = ann['start']
            ann_end = ann['end']

            if (ann_start < sentence_end and ann_end > sentence_start):
                new_start = max(0, ann_start - sentence_start)
                new_end = min(len(sentence), ann_end - sentence_start)

                if new_end > new_start:
                    sentence_anns.append({
                        'start': new_start,
                        'end': new_end,
                        'span': sentence[new_start:new_end],
                        'type': ann['type']
                    })

        sentence_annotations.append(sentence_anns)
        current_pos = sentence_end

    return sentence_annotations


def _convert_to_iob_format(
    record_id: str,
    text: str,
    annotations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Convert text and annotations to IOB format."""
    import nltk

    tokens = nltk.word_tokenize(text)
    ner_tags = ['O'] * len(tokens)

    char_to_token = {}
    current_pos = 0

    for token_idx, token in enumerate(tokens):
        token_start = text.find(token, current_pos)
        if token_start != -1:
            for char_pos in range(token_start, token_start + len(token)):
                char_to_token[char_pos] = token_idx
            current_pos = token_start + len(token)

    for annotation in annotations:
        start_char = annotation['start']
        end_char = annotation['end']
        label = annotation['type']

        overlapping_tokens = set()
        for char_pos in range(start_char, end_char):
            if char_pos in char_to_token:
                overlapping_tokens.add(char_to_token[char_pos])

        overlapping_tokens = sorted(list(overlapping_tokens))

        for i, token_idx in enumerate(overlapping_tokens):
            if i == 0:
                ner_tags[token_idx] = f'B-{label}'
            else:
                ner_tags[token_idx] = f'I-{label}'

    return {
        'record_id': record_id,
        'tokens': tokens,
        'ner_tags': ner_tags
    }


def _process_records_to_sentences(
    records_with_annotations: List[Dict[str, Any]],
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Process records by splitting into sentences."""
    from tqdm import tqdm

    processed_sentences = []
    iterator = tqdm(records_with_annotations) if verbose else records_with_annotations

    for record in iterator:
        record_id = record['record_id']
        text = record['text']
        annotations = record['annotations']

        sentences = _split_into_sentences(text)
        sentence_annotations = _map_annotations_to_sentences(text, sentences, annotations)

        for i, (sentence, sentence_anns) in enumerate(zip(sentences, sentence_annotations)):
            sentence_record_id = f"{record_id}_sent_{i}"
            iob_data = _convert_to_iob_format(sentence_record_id, sentence, sentence_anns)
            processed_sentences.append(iob_data)

    return processed_sentences


def _load_corrections(corrections_file: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load corrections from file."""
    if not corrections_file.exists():
        return {}

    with open(corrections_file) as f:
        corrections = f.readlines()
        corrections = [eval(line) for line in corrections if line.strip()]

    corrections_dict = {}
    for correction in corrections:
        record_id = correction['id']
        if record_id not in corrections_dict:
            corrections_dict[record_id] = []
        corrections_dict[record_id].append(correction)
    return corrections_dict


def _parse_correction(correction_str: str) -> List[List[tuple]]:
    """Parse a correction string into a list of valid token-tag groups."""
    try:
        parsed = ast.literal_eval(correction_str)
        if not isinstance(parsed, list):
            return []
        return [
            group for group in parsed
            if isinstance(group, list) and group
            and all(isinstance(p, tuple) and len(p) == 2 for p in group)
        ]
    except Exception:
        return []


def _find_token_sequence(tokens: List[str], group: List[tuple], start: int) -> bool:
    """Return True if group's tokens appear consecutively in tokens starting at start."""
    if start + len(group) > len(tokens):
        return False
    return all(tokens[start + i] == group[i][0] for i in range(len(group)))


def _apply_corrections(
    dataset: List[Dict[str, Any]],
    corrections_dict: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Apply corrections to the dataset."""
    corrected_dataset = []
    for item in dataset:
        tokens = item['tokens'][:]
        tags = item['ner_tags'][:]
        for correction in corrections_dict.get(item['record_id'], []):
            for group in _parse_correction(correction['correction']):
                first_token = group[0][0]
                for idx, tok in enumerate(tokens):
                    if tok == first_token and _find_token_sequence(tokens, group, idx):
                        for i, (_, tag) in enumerate(group):
                            tags[idx + i] = tag
                        break
        corrected_dataset.append({'record_id': item['record_id'], 'tokens': tokens, 'ner_tags': tags})
    return corrected_dataset


# =============================================================================
# Claim Segmentation Helpers
# =============================================================================

def _is_header_token(tok: str, custom_headers: Optional[set] = None) -> bool:
    """Check if token is a clinical header (e.g., 'PMH:', 'HPI:', 'History:').

    Recognizes:
    - Known clinical headers (case-insensitive): PMH:, Hpi:, history:
    - All-caps tokens with colon: ASSESSMENT:, LABS:
    """
    if not tok.endswith(":") or len(tok) < 2:
        return False

    base = tok[:-1]  # Remove colon
    headers = custom_headers if custom_headers is not None else COMMON_HEADERS

    # Check against known headers (case-insensitive)
    if base.upper() in headers:
        return True

    # All-caps tokens with colon (length >= 3 to avoid "A:" false positives unless in headers)
    if len(base) >= 2 and base.isupper() and base.isalpha():
        return True

    return False


def _is_hard_delim(tok: str, hard_delims: Optional[set] = None) -> bool:
    """Check if token is a hard delimiter."""
    delims = hard_delims if hard_delims is not None else DEFAULT_HARD_DELIMS
    return tok in delims


def _is_structural_colon(tokens: List[str], idx: int) -> bool:
    """Check if colon at position idx is structural (header) vs. part of value.

    Returns False for:
    - Time patterns: 10:30, 2:15
    - Ratios: 1:2, 1:100

    Returns True for:
    - Header patterns: word followed by colon
    """
    if idx <= 0 or idx >= len(tokens):
        return False

    tok = tokens[idx]
    if tok != ":":
        return False

    prev_tok = tokens[idx - 1]

    # Check if there's a next token
    if idx < len(tokens) - 1:
        next_tok = tokens[idx + 1]
        # Time/ratio pattern: digit:digit
        if prev_tok.replace(".", "").isdigit() and next_tok.replace(".", "").isdigit():
            return False

    # Header pattern: alphabetic word before colon
    if prev_tok.isalpha() and len(prev_tok) >= 2:
        return True

    return False


def _find_hard_splits(
    tokens: List[str],
    hard_delims: Optional[set] = None,
    custom_headers: Optional[set] = None,
) -> List[int]:
    """
    Return split positions as token indices where a segment should END.

    For headers, we start a segment AT the header token.
    For hard delimiters, we end the segment at the delimiter.
    For standalone colons, we check if they're structural (headers) vs. values (times/ratios).
    """
    ends = []
    last_end = -1

    for i, tok in enumerate(tokens):
        if _is_header_token(tok, custom_headers):
            # Header token - end previous segment before it
            if i - 1 > last_end:
                ends.append(i - 1)
            last_end = i - 1
        elif _is_hard_delim(tok, hard_delims):
            # Hard delimiter - end segment at delimiter
            ends.append(i)
            last_end = i
        elif tok == ":" and _is_structural_colon(tokens, i):
            # Structural colon (not time/ratio) - treat as delimiter
            ends.append(i)
            last_end = i

    if not ends or ends[-1] != len(tokens) - 1:
        ends.append(len(tokens) - 1)

    ends = sorted(set(e for e in ends if e >= 0))
    return ends


def _apply_max_len(
    tokens: List[str],
    segment_ranges: List[Tuple[int, int]],
    max_len: int,
    soft_delims: Optional[set] = None,
    conjunctions: Optional[set] = None,
) -> List[Tuple[int, int]]:
    """
    Split segments longer than max_len near soft delimiters or conjunctions.
    Falls back to hard split at max_len if no good split point found.

    Split priority (highest to lowest):
    1. Comma followed by conjunction (e.g., ", and")
    2. Standalone conjunction
    3. Comma alone
    4. Hard split at max_len (last resort)
    """
    delims = soft_delims if soft_delims is not None else DEFAULT_SOFT_DELIMS
    conjs = conjunctions if conjunctions is not None else DEFAULT_CONJUNCTIONS

    out = []
    for (s, e) in segment_ranges:
        L = e - s + 1
        if L <= max_len:
            out.append((s, e))
            continue

        start = s
        while start <= e:
            if e - start + 1 <= max_len:
                out.append((start, e))
                break

            target = start + max_len
            best = None
            best_priority = 999

            # Search backwards from target for best split point
            for j in range(min(e, target), start, -1):
                tj = tokens[j].upper()

                # Priority 1: Comma followed by conjunction
                if tokens[j] in delims and j + 1 <= e:
                    next_tok = tokens[j + 1].upper()
                    if next_tok in conjs and best_priority > 1:
                        best = j
                        best_priority = 1
                        break  # Best possible, stop searching

                # Priority 2: Standalone conjunction
                if tj in conjs and best_priority > 2:
                    best = j
                    best_priority = 2

                # Priority 3: Comma/soft delimiter alone
                if tokens[j] in delims and best_priority > 3:
                    best = j
                    best_priority = 3

            # Priority 4: Hard split at target (last resort)
            if best is None:
                best = min(target, e)

            out.append((start, best))
            start = best + 1

    return out


def _merge_tiny_negatives(
    tokens: List[str],
    ranges: List[Tuple[int, int]],
    labels: List[int],
    min_len: int
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Merge any label==0 segment whose length < min_len into its neighbor.
    Prefers right merge.
    """
    if not ranges:
        return ranges, labels

    segs = ranges[:]
    labs = labels[:]
    i = 0

    while i < len(segs):
        s, e = segs[i]
        length = e - s + 1

        if labs[i] == 0 and length < min_len and len(segs) > 1:
            if i < len(segs) - 1:
                sR, eR = segs[i + 1]
                segs[i] = (s, eR)
                labs[i] = max(labs[i], labs[i + 1])
                del segs[i + 1]
                del labs[i + 1]
                continue
            else:
                sL, eL = segs[i - 1]
                segs[i - 1] = (sL, e)
                labs[i - 1] = max(labs[i - 1], labs[i])
                del segs[i]
                del labs[i]
                i -= 1
                continue
        i += 1

    return segs, labs


def _sentence_to_claims(
    tokens: List[str],
    tags: List[str],
    max_len: int = 48,
    min_len: int = 3,
    hard_delims: Optional[set] = None,
    soft_delims: Optional[set] = None,
    conjunctions: Optional[set] = None,
    custom_headers: Optional[set] = None,
) -> List[Tuple[int, int, int]]:
    """
    Convert one sentence into non-overlapping claim segments using token/tags.

    Returns list of (start_idx, end_idx, label) where label=1 if any tag != 'O' in the span.

    Parameters
    ----------
    tokens : List[str]
        List of tokens in the sentence.
    tags : List[str]
        List of NER tags corresponding to tokens.
    max_len : int
        Maximum tokens per claim segment.
    min_len : int
        Minimum tokens for non-PHI segments (smaller ones get merged).
    hard_delims : set, optional
        Hard delimiters that always create segment breaks.
    soft_delims : set, optional
        Soft delimiters used for overflow splitting.
    conjunctions : set, optional
        Conjunctions for splitting long segments.
    custom_headers : set, optional
        Additional clinical header base forms to recognize.
    """
    assert len(tokens) == len(tags), "tokens and ner_tags must be same length"

    ends = _find_hard_splits(tokens, hard_delims=hard_delims, custom_headers=custom_headers)

    ranges = []
    start = 0
    for end in ends:
        if end >= start:
            ranges.append((start, end))
            start = end + 1
    if start < len(tokens):
        ranges.append((start, len(tokens) - 1))

    ranges = _apply_max_len(
        tokens, ranges, max_len=max_len,
        soft_delims=soft_delims, conjunctions=conjunctions
    )
    labels = [1 if any(t != "O" for t in tags[s:e + 1]) else 0 for (s, e) in ranges]
    ranges, labels = _merge_tiny_negatives(tokens, ranges, labels, min_len=min_len)

    return [(s, e, lab) for (s, e), lab in zip(ranges, labels)]


# =============================================================================
# Public API
# =============================================================================

def load_phi_dataset(
    data_dir: Optional[Path] = None,
    save_parquet_path: Optional[Path] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Load the PHI dataset from raw files and return as a Polars DataFrame.

    This function directly processes the raw data files without relying on
    the HuggingFace datasets infrastructure.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/PhysionetGoldCorpus/.
    save_parquet_path : Path, optional
        If provided, the DataFrame will be saved to this path as Parquet.
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with columns: 'record_id', 'tokens', 'ner_tags'.

    Required Files
    --------------
    - id.text: Raw medical text records
    - I2B2-2014-Relabeled-PhysionetGoldCorpus.csv: Annotation file
    - corrections.txt: Manual annotation corrections (optional)
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir)

    # Ensure NLTK data is available
    ensure_nltk_data()

    text_file = data_dir / TEXT_FILE
    annotations_file = data_dir / ANNOTATIONS_FILE
    corrections_file = data_dir / CORRECTIONS_FILE

    if not text_file.exists():
        raise FileNotFoundError(
            f"Required file '{TEXT_FILE}' not found at {text_file}. "
            "Please download it from PhysioNet (requires license)."
        )

    if not annotations_file.exists():
        raise FileNotFoundError(
            f"Required file '{ANNOTATIONS_FILE}' not found at {annotations_file}."
        )

    if verbose:
        print(f"Loading records from: {text_file}")

    # Step 1: Load records and annotations
    records_txt = _load_records(text_file)
    if verbose:
        print(f"  Loaded {len(records_txt)} records")

    id2annotations = _load_annotations(annotations_file)
    if verbose:
        print(f"  Loaded annotations for {len(id2annotations)} records")

    # Step 2: Map annotations to records
    ds_annotations = _map_annotations_to_records(id2annotations, records_txt)
    if verbose:
        print(f"  Mapped annotations to {len(ds_annotations)} records")

    # Step 3: Process into sentences with IOB tags
    if verbose:
        print("  Processing records into sentences...")

    processed_data = _process_records_to_sentences(ds_annotations, verbose=verbose)
    if verbose:
        print(f"  Generated {len(processed_data)} sentence rows")

    # Step 4: Apply corrections if available
    if corrections_file.exists():
        corrections_dict = _load_corrections(corrections_file)
        if corrections_dict:
            processed_data = _apply_corrections(processed_data, corrections_dict)
            if verbose:
                print(f"  Applied corrections from {corrections_file}")

    # Convert to Polars DataFrame
    if not processed_data:
        if verbose:
            print("No data processed. Returning empty DataFrame.")
        return pl.DataFrame(schema={
            "record_id": pl.Utf8,
            "tokens": pl.List(pl.Utf8),
            "ner_tags": pl.List(pl.Utf8)
        })

    df = pl.DataFrame(processed_data)

    if verbose:
        print(f"Final DataFrame shape: {df.shape}")

    if save_parquet_path:
        df.write_parquet(save_parquet_path)
        if verbose:
            print(f"Wrote Parquet to: {save_parquet_path}")

    return df


def sentences_to_claims(
    df_sent: pl.DataFrame,
    record_col: str = "record_id",
    tokens_col: str = "tokens",
    tags_col: str = "ner_tags",
    max_len: int = 100,
    min_len: int = 3,
    hard_delims: Optional[set] = None,
    soft_delims: Optional[set] = None,
    conjunctions: Optional[set] = None,
    custom_headers: Optional[set] = None,
) -> pl.DataFrame:
    """
    Convert each sentence row into 1..N claims based on tokens & BIO tags.

    Parameters
    ----------
    df_sent : pl.DataFrame
        DataFrame with sentence-level data (record_id, tokens, ner_tags).
    record_col : str
        Column name for record ID.
    tokens_col : str
        Column name for token list.
    tags_col : str
        Column name for NER tag list.
    max_len : int
        Maximum tokens per claim segment.
    min_len : int
        Minimum tokens for non-PHI segments (smaller ones get merged).
    hard_delims : set, optional
        Hard delimiters that always create segment breaks.
        Defaults to {";", "—", "--", "|"}.
    soft_delims : set, optional
        Soft delimiters used for overflow splitting.
        Defaults to {","}.
    conjunctions : set, optional
        Conjunctions for splitting long segments.
        Defaults to {"AND", "BUT", "OR", "NOR", "YET", "SO", "THEN", "HOWEVER", ...}.
    custom_headers : set, optional
        Additional clinical header base forms to recognize (without colon).
        These are added to the default COMMON_HEADERS set.

    Returns
    -------
    pl.DataFrame
        Claim-level DataFrame with columns:
        - base_id: Record ID without _sent_N suffix
        - sentence_index: Sentence index within document
        - claim_index: Claim index within sentence
        - claim_tokens: List of tokens in the claim
        - claim_tags: List of NER tags in the claim
        - claim_text: Tokens joined as string
        - claim_label: 1 if any PHI tag present, else 0
    """
    # Merge custom headers with defaults if provided
    effective_headers = None
    if custom_headers is not None:
        effective_headers = COMMON_HEADERS | {h.upper() for h in custom_headers}
    dfp = (
        df_sent
        .with_columns([
            pl.col(record_col).map_elements(base_id, return_dtype=pl.Utf8).alias("base_id"),
            pl.col(record_col).map_elements(sent_idx, return_dtype=pl.Int64).alias("sentence_index"),
        ])
        .drop_nulls(["sentence_index"])
        .sort(["base_id", "sentence_index"])
    )

    out_rows = []
    for rid, sidx, toks, tags in dfp.select(["base_id", "sentence_index", tokens_col, tags_col]).iter_rows():
        segments = _sentence_to_claims(
            list(toks), list(tags),
            max_len=max_len,
            min_len=min_len,
            hard_delims=hard_delims,
            soft_delims=soft_delims,
            conjunctions=conjunctions,
            custom_headers=effective_headers,
        )
        for j, (a, b, lab) in enumerate(segments):
            claim_toks = toks[a:b + 1]
            claim_tags = tags[a:b + 1]
            claim_txt = " ".join(claim_toks)
            out_rows.append({
                "base_id": rid,
                "sentence_index": sidx,
                "claim_index": j,
                "claim_tokens": claim_toks,
                "claim_tags": claim_tags,
                "claim_text": claim_txt,
                "claim_label": lab,
            })

    if not out_rows:
        return pl.DataFrame(
            schema={
                "base_id": pl.Utf8,
                "sentence_index": pl.Int64,
                "claim_index": pl.Int64,
                "claim_tokens": pl.List(pl.Utf8),
                "claim_tags": pl.List(pl.Utf8),
                "claim_text": pl.Utf8,
                "claim_label": pl.Int64,
            }
        )

    df_claim = pl.DataFrame(out_rows).sort(["base_id", "sentence_index", "claim_index"])
    return df_claim


def claims_to_doclevel(
    df_claim: pl.DataFrame,
    id_col: str = "base_id",
    text_col: str = "claim_text",
    label_col: str = "claim_label",
    sent_idx_col: str = "sentence_index",
    claim_idx_col: str = "claim_index",
    join_with_space: bool = False,
) -> pl.DataFrame:
    """
    Aggregate claim rows into a document-level table.

    Parameters
    ----------
    df_claim : pl.DataFrame
        Claim-level DataFrame from sentences_to_claims().
    id_col : str
        Column name for document ID.
    text_col : str
        Column name for claim text.
    label_col : str
        Column name for claim label.
    sent_idx_col : str
        Column name for sentence index.
    claim_idx_col : str
        Column name for claim index.
    join_with_space : bool
        If True, join claims with space; otherwise join directly.

    Returns
    -------
    pl.DataFrame
        Document-level DataFrame with columns:
        - id: Document ID
        - input: Full document text (claims concatenated)
        - input_claims: List of claim texts
        - input_claims_labels: List of claim labels (0/1)
        - summary_claims: List of claims with label=1
        - summary: Summary text (positive claims concatenated)
    """
    joiner = " " if join_with_space else ""

    grouped = (
        groupby(df_claim, id_col, maintain_order=True)
        .agg([
            pl.col(text_col).sort_by([sent_idx_col, claim_idx_col]).alias("input_claims"),
            pl.col(label_col).sort_by([sent_idx_col, claim_idx_col]).alias("input_claims_labels"),
        ])
        .with_columns([
            pl.col("input_claims").list.join(joiner).alias("input"),
            pl.struct(["input_claims", "input_claims_labels"])
              .map_elements(
                  lambda s: [t for t, l in zip(s["input_claims"], s["input_claims_labels"]) if l == 1],
                  return_dtype=pl.List(pl.Utf8)
              )
              .alias("summary_claims"),
        ])
        .with_columns([
            pl.col("summary_claims").list.join(joiner).alias("summary")
        ])
        .select([
            pl.col(id_col).alias("id"),
            "input",
            "input_claims",
            "input_claims_labels",
            "summary_claims",
            "summary",
        ])
    )

    return grouped


def process_physionet_goldcorpus(
    data_dir: Optional[Path] = None,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    max_len: int = 100,
    min_len: int = 3,
    join_with_space: bool = False,
    config: Optional["PhysioNetConfig"] = None,
) -> pl.DataFrame:
    """
    Run the complete data processing pipeline for Physionet Gold Corpus.

    This function:
    1. Loads the PHI dataset from raw input files
    2. Converts sentences to claims using BIO tag segmentation
    3. Aggregates claims to document-level format

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing input files. Defaults to data/PhysionetGoldCorpus/.
    output_filename : str
        Name of the output parquet file. Caller is responsible for saving if needed.
    max_len : int
        Maximum tokens per claim segment.
    min_len : int
        Minimum tokens for non-PHI segments.
    join_with_space : bool
        If True, join claims with space in output text fields.
    config : PhysioNetConfig, optional
        Delimiter and header configuration. Defaults to PhysioNetConfig().

    Returns
    -------
    pl.DataFrame
        The processed document-level DataFrame.

    Required Files
    --------------
    The following files must exist in data_dir:
    - id.text: Raw medical text records (requires PhysioNet license)
    - I2B2-2014-Relabeled-PhysionetGoldCorpus.csv: Annotation file
    - corrections.txt: Manual annotation corrections (optional)

    Example
    -------
    >>> from conformal_relevance.data_loading import process_physionet_goldcorpus
    >>> df = process_physionet_goldcorpus()
    >>> print(df.shape)
    (787, 6)
    """
    if config is None:
        config = PhysioNetConfig()

    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir)

    # Check required files exist
    missing_files = [f for f in [TEXT_FILE, ANNOTATIONS_FILE] if not (data_dir / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {data_dir}: {missing_files}\n"
            "Please ensure all input files are present before running."
        )

    logger.info("Processing Physionet Gold Corpus from: %s", data_dir)

    # Step 1: Load PHI dataset
    logger.info("[Step 1/3] Loading PHI dataset...")
    df_phi = load_phi_dataset(data_dir=data_dir, verbose=False)
    logger.info("  Loaded %d sentence rows", df_phi.shape[0])

    # Step 2: Convert sentences to claims
    logger.info("[Step 2/3] Converting sentences to claims...")
    df_claim = sentences_to_claims(
        df_sent=df_phi,
        record_col="record_id",
        tokens_col="tokens",
        tags_col="ner_tags",
        max_len=max_len,
        min_len=min_len,
        hard_delims=config.hard_delims,
        soft_delims=config.soft_delims,
        conjunctions=config.conjunctions,
        custom_headers=config.custom_headers,
    )
    logger.info("  Generated %d claim rows", df_claim.shape[0])

    # Step 3: Aggregate to document level
    logger.info("[Step 3/3] Aggregating to document level...")
    df_doc = claims_to_doclevel(df_claim=df_claim, join_with_space=join_with_space)
    logger.info("  Created %d document rows", df_doc.shape[0])

    logger.info("Final shape: %s, columns: %s", df_doc.shape, df_doc.columns)
    return df_doc


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for Physionet data processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Physionet Gold Corpus data into parquet format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing input files (default: data/PhysionetGoldCorpus/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output filename (default: {DEFAULT_OUTPUT_FILENAME})"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=100,
        help="Maximum tokens per claim segment (default: 100)"
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=3,
        help="Minimum tokens for non-PHI segments (default: 3)"
    )
    parser.add_argument(
        "--join-with-space",
        action="store_true",
        help="Join claims with space instead of directly"
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

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    df = process_physionet_goldcorpus(
        data_dir=args.data_dir,
        output_filename=args.output,
        max_len=args.max_len,
        min_len=args.min_len,
        join_with_space=args.join_with_space,
    )

    if not args.no_save:
        data_dir = args.data_dir or DATA_DIR
        output_path = data_dir / args.output
        df.write_parquet(output_path)
        logger.info("Output saved to: %s", output_path)


if __name__ == "__main__":
    main()
