"""
Data loading and processing utilities for various datasets.

This package provides functions to process datasets from raw files into
structured parquet formats suitable for conformal prediction tasks.

Supported Datasets
------------------

1. **Physionet Gold Corpus** (I2B2 2014 De-identification)
   - Main function: :func:`process_physionet_goldcorpus`
   - Module: :mod:`.physionet`

2. **PUMA Dataset** (Patient-generated User-centered Medical Answers)
   - Main function: :func:`process_puma_dataset`
   - Module: :mod:`.puma`

3. **ECTSum Dataset** (Earnings Call Transcript Summarization)
   - Main function: :func:`process_ectsum_dataset`
   - Module: :mod:`.ectsum`

4. **SubSumE Dataset** (Subjective Summarization Evaluation)
   - Main function: :func:`process_subsume_dataset`
   - Module: :mod:`.subsume`

5. **HotpotQA Dataset** (Multi-hop Question Answering)
   - Main function: :func:`process_hotpotqa_dataset`
   - Module: :mod:`.hotpotqa`

6. **Evidence Inference Dataset** (Clinical Trial Evidence Detection)
   - Main function: :func:`process_evidence_inference_dataset`
   - Module: :mod:`.evidence_inference`

7. **ContractNLI Dataset** (Legal NDA Analysis)
   - Main function: :func:`process_contractnli_dataset`
   - Module: :mod:`.contractnli`

Dataset Splitting
-----------------

After processing, datasets can be split into pool, calibration, and test sets
using the splitting utilities. All datasets are converted to a standardized
column format:

- **intent** (optional): The query intent or perspective
- **input_units**: List of text units (sentences or claims)
- **input_unit_labels**: Binary labels for each unit
- **output_text**: Summary or output text
- **output_units**: List of output text units

For datasets with intent column (PUMA, SubSumE, Evidence Inference, ContractNLI): pool is
stratified by intent; calibration and test are uniform random.

For datasets without intent (ECTSum, Physionet, HotpotQA): all splits use uniform random sampling.

Quick Start
-----------

Process Physionet Gold Corpus::

    from conformal_relevance.data_loading import process_physionet_goldcorpus
    df = process_physionet_goldcorpus()

Process PUMA dataset::

    from conformal_relevance.data_loading import process_puma_dataset
    df = process_puma_dataset()

Process ECTSum dataset::

    from conformal_relevance.data_loading import process_ectsum_dataset
    df = process_ectsum_dataset()  # Auto-downloads from HuggingFace if needed

Process SubSumE dataset::

    from conformal_relevance.data_loading import process_subsume_dataset
    df = process_subsume_dataset()

Process HotpotQA dataset::

    from conformal_relevance.data_loading import process_hotpotqa_dataset
    df = process_hotpotqa_dataset(splits=["dev"])  # Dev set only

Process Evidence Inference dataset::

    from conformal_relevance.data_loading import process_evidence_inference_dataset
    df = process_evidence_inference_dataset()  # Auto-downloads from ebm-nlp.com

Split datasets into pool, calibration, and test sets::

    from conformal_relevance.data_loading import split_puma_dataset
    pool_df, cal_df, test_df = split_puma_dataset()

    # Or split all datasets at once
    from conformal_relevance.data_loading import split_all_datasets
    results = split_all_datasets()  # dict[str, tuple[pool, cal, test]]

Process ContractNLI dataset::

    from conformal_relevance.data_loading import process_contractnli_dataset
    df = process_contractnli_dataset()

For more control, import from specific modules::

    from conformal_relevance.data_loading.physionet import (
        load_phi_dataset,
        sentences_to_claims,
        claims_to_doclevel,
    )

    from conformal_relevance.data_loading.puma import (
        get_sentences_with_spans,
        create_ground_truth_labels,
    )

    from conformal_relevance.data_loading.ectsum import (
        download_ectsum_data,
        load_ectsum_raw,
        process_ectsum_data,
    )

    from conformal_relevance.data_loading.subsume import (
        load_state_sentences,
        load_user_summaries,
        process_subsume_data,
    )

    from conformal_relevance.data_loading.hotpotqa import (
        download_hotpotqa_data,
        load_hotpotqa_raw,
        process_hotpotqa_data,
    )

    from conformal_relevance.data_loading.evidence_inference import (
        download_evidence_inference_data,
        load_evidence_inference_raw,
        process_evidence_inference_data,
    )

    from conformal_relevance.data_loading.split import (
        split_dataset,
        split_and_save_dataset,
    )
"""

# Common utilities
from ._common import (
    split_sentences,
    split_sentences_with_spans,
)

# Physionet Gold Corpus exports
from .physionet import (
    # Main entry point
    process_physionet_goldcorpus,
    # Individual pipeline steps
    load_phi_dataset,
    sentences_to_claims,
    claims_to_doclevel,
    # Constants
    DATA_DIR as PHYSIONET_DATA_DIR,
    DEFAULT_OUTPUT_FILENAME as PHYSIONET_OUTPUT_FILENAME,
    COMMON_HEADERS,
    DEFAULT_HARD_DELIMS,
    DEFAULT_SOFT_DELIMS,
    DEFAULT_CONJUNCTIONS,
)

# PUMA dataset exports
from .puma import (
    # Main entry point
    process_puma_dataset,
    # Individual functions
    get_sentences_with_spans as get_puma_sentences_with_spans,
    create_ground_truth_labels as create_puma_ground_truth_labels,
    transform_dataframe as transform_puma_dataframe,
    # Constants
    DATA_DIR as PUMA_DATA_DIR,
    INPUT_FILENAME as PUMA_INPUT_FILENAME,
    OUTPUT_FILENAME as PUMA_OUTPUT_FILENAME,
    ALL_PERSPECTIVES as PUMA_ALL_PERSPECTIVES,
    DEFAULT_CORRECTIONS as PUMA_DEFAULT_CORRECTIONS,
)

# ECTSum dataset exports
from .ectsum import (
    # Main entry point
    process_ectsum_dataset,
    # Individual functions
    download_ectsum_data,
    load_ectsum_raw,
    process_ectsum_data,
    # Constants
    DATA_DIR as ECTSUM_DATA_DIR,
    OUTPUT_FILENAME as ECTSUM_OUTPUT_FILENAME,
    DOWNLOAD_URLS as ECTSUM_DOWNLOAD_URLS,
)

# SubSumE dataset exports
from .subsume import (
    # Main entry point
    process_subsume_dataset,
    # Individual functions
    load_state_sentences,
    load_user_summaries,
    process_subsume_data,
    # Constants
    DATA_DIR as SUBSUME_DATA_DIR,
    OUTPUT_FILENAME as SUBSUME_OUTPUT_FILENAME,
)

# HotpotQA dataset exports
from .hotpotqa import (
    # Main entry point
    process_hotpotqa_dataset,
    # Individual functions
    download_hotpotqa_data,
    load_hotpotqa_raw,
    process_hotpotqa_data,
    # Constants
    DATA_DIR as HOTPOTQA_DATA_DIR,
    OUTPUT_FILENAME as HOTPOTQA_OUTPUT_FILENAME,
    DOWNLOAD_URLS as HOTPOTQA_DOWNLOAD_URLS,
)

# Evidence Inference dataset exports
from .evidence_inference import (
    # Main entry point
    process_evidence_inference_dataset,
    # Individual functions
    download_evidence_inference_data,
    load_evidence_inference_raw,
    process_evidence_inference_data,
    # Constants
    DATA_DIR as EVIDENCE_INFERENCE_DATA_DIR,
    OUTPUT_FILENAME as EVIDENCE_INFERENCE_OUTPUT_FILENAME,
    DOWNLOAD_URL as EVIDENCE_INFERENCE_DOWNLOAD_URL,
)

# ContractNLI dataset exports
from .contractnli import (
    # Main entry point
    process_contractnli_dataset,
    # Individual functions
    process_contractnli_data,
    # Constants
    DATA_DIR as CONTRACTNLI_DATA_DIR,
    OUTPUT_FILENAME as CONTRACTNLI_OUTPUT_FILENAME,
)

# Dataset splitting exports — includes split_contractnli_dataset
from .split import (
    # Main functions
    split_dataset,
    split_and_save_dataset,
    split_all_datasets,
    # Dataset-specific functions
    split_puma_dataset,
    split_subsume_dataset,
    split_ectsum_dataset,
    split_physionet_dataset,
    split_hotpotqa_dataset,
    split_evidence_inference_dataset,
    split_contractnli_dataset,
    # Constants
    DEFAULT_POOL_SIZE,
    DEFAULT_POOL_PER_INTENT,
    DEFAULT_CAL_SIZE,
    DEFAULT_SEED,
    COLUMN_MAPPINGS,
    DATASETS_WITH_INTENT,
)

__all__ = [
    # Common utilities
    "split_sentences",
    "split_sentences_with_spans",
    # Physionet Gold Corpus
    "process_physionet_goldcorpus",
    "load_phi_dataset",
    "sentences_to_claims",
    "claims_to_doclevel",
    "PHYSIONET_DATA_DIR",
    "PHYSIONET_OUTPUT_FILENAME",
    "COMMON_HEADERS",
    "DEFAULT_HARD_DELIMS",
    "DEFAULT_SOFT_DELIMS",
    "DEFAULT_CONJUNCTIONS",
    # PUMA Dataset
    "process_puma_dataset",
    "get_puma_sentences_with_spans",
    "create_puma_ground_truth_labels",
    "transform_puma_dataframe",
    "PUMA_DATA_DIR",
    "PUMA_INPUT_FILENAME",
    "PUMA_OUTPUT_FILENAME",
    "PUMA_ALL_PERSPECTIVES",
    "PUMA_DEFAULT_CORRECTIONS",
    # ECTSum Dataset
    "process_ectsum_dataset",
    "download_ectsum_data",
    "load_ectsum_raw",
    "process_ectsum_data",
    "ECTSUM_DATA_DIR",
    "ECTSUM_OUTPUT_FILENAME",
    "ECTSUM_DOWNLOAD_URLS",
    # SubSumE Dataset
    "process_subsume_dataset",
    "load_state_sentences",
    "load_user_summaries",
    "process_subsume_data",
    "SUBSUME_DATA_DIR",
    "SUBSUME_OUTPUT_FILENAME",
    # HotpotQA Dataset
    "process_hotpotqa_dataset",
    "download_hotpotqa_data",
    "load_hotpotqa_raw",
    "process_hotpotqa_data",
    "HOTPOTQA_DATA_DIR",
    "HOTPOTQA_OUTPUT_FILENAME",
    "HOTPOTQA_DOWNLOAD_URLS",
    # Evidence Inference Dataset
    "process_evidence_inference_dataset",
    "download_evidence_inference_data",
    "load_evidence_inference_raw",
    "process_evidence_inference_data",
    "EVIDENCE_INFERENCE_DATA_DIR",
    "EVIDENCE_INFERENCE_OUTPUT_FILENAME",
    "EVIDENCE_INFERENCE_DOWNLOAD_URL",
    # ContractNLI Dataset
    "process_contractnli_dataset",
    "process_contractnli_data",
    "CONTRACTNLI_DATA_DIR",
    "CONTRACTNLI_OUTPUT_FILENAME",
    # Dataset Splitting
    "split_dataset",
    "split_and_save_dataset",
    "split_all_datasets",
    "split_puma_dataset",
    "split_subsume_dataset",
    "split_ectsum_dataset",
    "split_physionet_dataset",
    "split_hotpotqa_dataset",
    "split_evidence_inference_dataset",
    "split_contractnli_dataset",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_POOL_PER_INTENT",
    "DEFAULT_CAL_SIZE",
    "DEFAULT_SEED",
    "COLUMN_MAPPINGS",
    "DATASETS_WITH_INTENT",
]
