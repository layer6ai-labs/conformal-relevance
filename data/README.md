# Data Directory

This directory contains data processing scripts and instructions for various datasets used in the conformal relevance project.

## Quick Start

### Processing All Datasets

```bash
# Process individual datasets (generates *_processed.parquet files)
uv run python -m conformal_relevance.data_loading.puma
uv run python -m conformal_relevance.data_loading.subsume
uv run python -m conformal_relevance.data_loading.ectsum
uv run python -m conformal_relevance.data_loading.physionet
uv run python -m conformal_relevance.data_loading.hotpotqa --splits dev
uv run python -m conformal_relevance.data_loading.evidence_inference
uv run python -m conformal_relevance.data_loading.contractnli

# Split all datasets into pool/calibration/test sets
uv run python -m conformal_relevance.data_loading.split
```

### From Python/Jupyter

```python
from conformal_relevance.data_loading import (
    split_all_datasets,
    split_puma_dataset,
    DEFAULT_SEED,
    DEFAULT_CAL_SIZE,
)

# Split all available datasets (returns dict of 3-tuples: pool, cal, test)
results = split_all_datasets(save_output=True, verbose=True)

# Access individual results
puma_pool, puma_cal, puma_test = results["puma"]
```

---

## Dataset Splitting

After processing, datasets are split into **three** sets for experiments:

1. **Pool** — ICL example selection only (never scored)
2. **Calibration** — calibration samples (scored separately)
3. **Test** — primary evaluation set

### Standardized Output Format

All datasets are converted to a consistent column format:

| Column | Type | Description |
|--------|------|-------------|
| `intent` | str (optional) | Query intent or perspective |
| `input` | str | Full input text |
| `input_units` | list[str] | Text units (sentences or claims) |
| `input_unit_labels` | list[int] | Binary labels (0 or 1) for each unit |
| `output_text` | str | Output text (summary, summary sentences, etc.) |
| `output_units` | list[str] | Output text units |

### Sampling Strategy

- **Stratified by intent** (PUMA, SubSumE, Evidence Inference, ContractNLI): Pool stratified per intent (20 per intent by default); calibration and test are uniform random
- **Uniform random** (ECTSum, PhysioNet, HotpotQA): Pool is 50 samples total; calibration and test are uniform random

### Reproducibility

All splits are reproducible via constants:
- `DEFAULT_POOL_SIZE = 50` - Default pool set size (uni-intent datasets)
- `DEFAULT_POOL_PER_INTENT = 20` - Default pool samples per intent (multi-intent datasets)
- `DEFAULT_CAL_SIZE = 100` - Default calibration set size
- `DEFAULT_SEED = 3` - Default random seed

```python
from conformal_relevance.data_loading import DEFAULT_SEED, split_puma_dataset

# Same seed = identical splits
pool1, cal1, test1 = split_puma_dataset(seed=DEFAULT_SEED)
pool2, cal2, test2 = split_puma_dataset(seed=DEFAULT_SEED)
assert cal1.equals(cal2)  # True
```

### Output Files

When `save_output=True`, splits are saved as:
- `data/{DATASET}/{dataset}_pool.parquet` - Pool set (ICL selection only)
- `data/{DATASET}/{dataset}_cal.parquet` - Calibration set (100 samples by default)
- `data/{DATASET}/{dataset}_test.parquet` - Test set (remaining samples)

---

## Datasets

### PUMA Dataset (Patient-generated User-centered Medical Answers)

**Location:** `data/PUMA/`

A dataset of patient-generated medical Q&A with 5 perspective intents:
- CAUSE, SUGGESTION, INFORMATION, EXPERIENCE, QUESTION

#### Required Files

```
data/PUMA/
    PUMA_data_raw.parquet     # Raw dataset (required)
    PUMA_processed.parquet    # Generated after processing
    puma_pool.parquet         # Pool set (after splitting)
    puma_cal.parquet          # Calibration set (after splitting)
    puma_test.parquet         # Test set (after splitting)
```

#### Processing

```python
from conformal_relevance.data_loading import process_puma_dataset, split_puma_dataset

# Process raw data
df = process_puma_dataset(verbose=True)

# Split into pool/calibration/test sets (pool stratified by intent)
pool_df, cal_df, test_df = split_puma_dataset()
```

---

### SubSumE Dataset (Subjective Summarization Evaluation)

**Location:** `data/SubSumE/`

Wikipedia state summaries with 10 different query intents.

#### Required Files

```
data/SubSumE/
    Raw_Wiki_Text_Files/           # Raw Wikipedia text files
    user_summary_jsons/            # User summary JSON files
    SubSumE_processed.parquet      # Generated after processing
    subsume_pool.parquet           # Pool set (after splitting)
    subsume_cal.parquet            # Calibration set (after splitting)
    subsume_test.parquet           # Test set (after splitting)
```

#### Processing

```python
from conformal_relevance.data_loading import process_subsume_dataset, split_subsume_dataset

# Process raw data
df = process_subsume_dataset(verbose=True)

# Split into pool/calibration/test sets (pool stratified by intent)
pool_df, cal_df, test_df = split_subsume_dataset()
```

---

### ECTSum Dataset (Earnings Call Transcript Summarization)

**Location:** `data/ECTSum/`

Earnings call transcripts with extractive summarization labels.

#### Data Access

The dataset is automatically downloaded from HuggingFace on first use.

#### Files

```
data/ECTSum/
    train.json                    # Downloaded automatically
    val.json                      # Downloaded automatically
    test.json                     # Downloaded automatically
    ECTSum_processed.parquet      # Generated after processing
    ectsum_pool.parquet           # Pool set (after splitting)
    ectsum_cal.parquet            # Calibration set (after splitting)
    ectsum_test.parquet           # Test set (after splitting)
```

#### Processing

```python
from conformal_relevance.data_loading import process_ectsum_dataset, split_ectsum_dataset

# Process (auto-downloads if needed)
df = process_ectsum_dataset(verbose=True)

# Split into pool/calibration/test sets (uniform random sampling)
pool_df, cal_df, test_df = split_ectsum_dataset()
```

---

### PhysioNet Gold Corpus (I2B2 2014 De-identification)

**Location:** `data/PhysionetGoldCorpus/`

Medical records with PHI (Protected Health Information) annotations for de-identification.

#### Data Access Requirements

The PhysioNet De-identified Medical Text dataset requires a signed data use agreement before access is granted.

1. Visit: https://physionet.org/content/deidentifiedmedicaltext/1.0/
2. Sign the required data use agreement
3. Download the following files:
   - `id.text` - Raw medical text records
   - `id-phi.phrase` - PHI phrase annotations (optional)
4. Place the downloaded files in `data/PhysionetGoldCorpus/`

#### Required Files

```
data/PhysionetGoldCorpus/
    id.text                                          # Downloaded from PhysioNet (required)
    I2B2-2014-Relabeled-PhysionetGoldCorpus.csv      # Included in repo
    corrections.txt                                   # Included in repo (optional)
    physionet_goldcorpus_i2b2deid_claimlevel.parquet # Generated after processing
    physionet_pool.parquet                           # Pool set (after splitting)
    physionet_cal.parquet                            # Calibration set (after splitting)
    physionet_test.parquet                           # Test set (after splitting)
```

#### Processing

```python
from conformal_relevance.data_loading import process_physionet_goldcorpus, split_physionet_dataset

# Process and save to parquet
df = process_physionet_goldcorpus(verbose=True)

# Split into pool/calibration/test sets (uniform random sampling)
pool_df, cal_df, test_df = split_physionet_dataset()
```

#### Processing Options

The processing pipeline supports customizable claim segmentation:

```python
df = process_physionet_goldcorpus(
    max_len=100,              # Max tokens per claim
    min_len=3,                # Min tokens for non-PHI segments
    join_with_space=False,    # How to join claims in output
    save_output=True,         # Set False to skip saving
    hard_delims={";", "|"},   # Custom hard delimiters
    conjunctions={"AND", "BUT", "OR"},  # Custom conjunctions
    custom_headers={"IMPRESSION"},      # Additional clinical headers
    verbose=True,
)
```

---

### HotpotQA Dataset (Multi-hop Question Answering)

**Location:** `data/HotpotQA/`

Multi-hop QA dataset with supporting fact annotations. Each question requires reasoning across multiple paragraphs.

#### Data Access

The dataset is downloaded from the official HotpotQA website on first use.

#### Files

```
data/HotpotQA/
    hotpot_dev_distractor_v1.json    # Downloaded automatically
    hotpot_train_v1.1.json           # Downloaded automatically (optional)
    HotpotQA_processed.parquet       # Generated after processing
    hotpotqa_pool.parquet            # Pool set (after splitting)
    hotpotqa_cal.parquet             # Calibration set (after splitting)
    hotpotqa_test.parquet            # Test set (after splitting)
```

#### Processing

```bash
# Process dev set only (recommended - faster)
uv run python -m conformal_relevance.data_loading.hotpotqa --splits dev

# Process both train and dev
uv run python -m conformal_relevance.data_loading.hotpotqa --splits train dev
```

```python
from conformal_relevance.data_loading import split_hotpotqa_dataset

# Split into pool/calibration/test sets (uniform random sampling)
pool_df, cal_df, test_df = split_hotpotqa_dataset()
```

---

### Evidence Inference Dataset (Clinical Trial Evidence)

**Location:** `data/EvidenceInference_v2/`

Clinical trial evidence detection with 8 user perspectives (User0-User7).

#### Data Access

The dataset is automatically downloaded from the official repository on first use.

#### Files

```
data/EvidenceInference_v2/
    splits/                              # Downloaded automatically
    txt_files/                           # Downloaded automatically
    evidence_inference_processed.parquet # Generated after processing
    evidence_inference_pool.parquet      # Pool set (after splitting)
    evidence_inference_cal.parquet       # Calibration set (after splitting)
    evidence_inference_test.parquet      # Test set (after splitting)
```

#### Processing

```bash
# Process (auto-downloads if needed)
uv run python -m conformal_relevance.data_loading.evidence_inference
```

```python
from conformal_relevance.data_loading import split_evidence_inference_dataset

# Split into pool/calibration/test sets (pool stratified by user perspective)
pool_df, cal_df, test_df = split_evidence_inference_dataset()
```

---

### ContractNLI Dataset (Legal NDA Analysis)

**Location:** `data/ContractNLI/`

Non-disclosure agreements with 17 hypothesis intents for contract clause classification.

#### Data Access

Download the ContractNLI dataset and extract to `data/ContractNLI/contract-nli/`.

#### Files

```
data/ContractNLI/
    contract-nli/                        # Raw dataset directory
    ContractNLI_processed.parquet        # Generated after processing
    contractnli_pool.parquet             # Pool set (after splitting)
    contractnli_cal.parquet              # Calibration set (after splitting)
    contractnli_test.parquet             # Test set (after splitting)
```

#### Processing

```bash
uv run python -m conformal_relevance.data_loading.contractnli
```

```python
from conformal_relevance.data_loading import split_contractnli_dataset

# Split into pool/calibration/test sets (pool stratified by hypothesis)
pool_df, cal_df, test_df = split_contractnli_dataset()
```

---

## Adding New Datasets

When adding a new dataset, please:

1. **Create a processing module** in `conformal_relevance/data_loading/`:
   - Follow the pattern of existing modules (puma.py, ectsum.py, etc.)
   - Include a `process_{dataset}_dataset()` main function
   - Add CLI entry point via `main()` function

2. **Update the split module** (`split.py`):
   - Add column mapping in `COLUMN_MAPPINGS`
   - Add to `DATASETS_WITH_INTENT` if applicable
   - Add split config in `_SPLIT_CONFIG`
   - Create `split_{dataset}_dataset()` convenience function

3. **Update `__init__.py`**:
   - Export processing and splitting functions
   - Export any relevant constants

4. **Update the ICL registry** (`icl/registry.py`):
   - Add dataset entry with pool/cal/test paths, multi_intent flag, and dataset_task

5. **Update the ICL-free prompt registry** (`prompts/icl_free.py`):
   - Add dataset-specific prompt templates

6. **Add documentation**:
   - Add a section in this README
   - Include data access/licensing requirements
   - Document required file structure
   - Provide processing examples

7. **Ensure consistent output format**:
   - Use standardized column names (see table above)
   - Include pool, calibration, and test parquet files
