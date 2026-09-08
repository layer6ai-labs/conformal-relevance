<div align="center">

<a href="https://layer6.ai"><img src="assets/layer6.png" alt="Layer 6 AI" width="220"></a>

[![arXiv](https://img.shields.io/badge/arXiv-2609.03005-b31b1b.svg)](https://arxiv.org/abs/2609.03005)

# Conformal Relevance

</div>

A Python library for intent-based relevancy scoring using LLMs with in-context learning (ICL). Score text units (sentences/claims) for relevancy to a specific intent/query and evaluate using Average Precision.

## Installation

```bash
# Install from source
uv sync
```

## Quick Start

### Strategy-Based ICL Scoring (Recommended)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from conformal_relevance import (
    score_dataframe,
    summarize_results,
    compute_average_precision,
    select_icl_examples,
    StrategyRegistry,
    DEFAULT_SEED,
)
import polars as pl

# Load data (3-way split: pool for ICL selection, cal for calibration, test for evaluation)
pool_df = pl.read_parquet("data/SubSumE/subsume_pool.parquet")
test_df = pl.read_parquet("data/SubSumE/subsume_test.parquet")

# Select ICL examples from pool via strategy
strategy = StrategyRegistry.get("random", seed=DEFAULT_SEED)
icl_result = select_icl_examples(
    pool_df=pool_df, strategy=strategy, dataset="subsume", n_icl=2, seed=DEFAULT_SEED,
)

# Create LLM and score
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
scoring_result = score_dataframe(
    df=test_df,
    llm=llm,
    output_col=icl_result.output_col,
    icl_examples=icl_result.icl_examples,
    dataset="subsume",
)

# Evaluate
df_scored = compute_average_precision(scoring_result.df, icl_result.output_col)
results = summarize_results(df_scored, score_cols=[icl_result.output_col])
```

### ICL-Free Baseline

```python
from conformal_relevance import score_dataframe_no_icl, summarize_results

scoring_result = score_dataframe_no_icl(
    df=test_df,
    llm=llm,
    dataset="puma",
    output_col="scores_icl0",
)
results = summarize_results(scoring_result.df, score_cols=["scores_icl0"])
```

## End-to-End Workflow

### Step 1: Process Raw Datasets

```bash
# Process individual datasets
uv run python -m conformal_relevance.data_loading.puma
uv run python -m conformal_relevance.data_loading.subsume
uv run python -m conformal_relevance.data_loading.ectsum
uv run python -m conformal_relevance.data_loading.physionet
uv run python -m conformal_relevance.data_loading.hotpotqa --splits dev
uv run python -m conformal_relevance.data_loading.evidence_inference
uv run python -m conformal_relevance.data_loading.contractnli
```

### Step 2: Split into Pool/Calibration/Test Sets

```bash
# Split all datasets (creates *_pool.parquet, *_cal.parquet, and *_test.parquet files)
uv run python -m conformal_relevance.data_loading.split
```

### Step 3: Run Experiments

```bash
# Basic run
uv run python main.py --dataset subsume --n-samples 5

# Use a specific strategy
uv run python main.py --strategy dpp --n-icl 3

# ICL-free baseline
uv run python main.py --no-icl --dataset puma --n-samples 5

# MoE ensemble scoring (default K=4: random+anchor_dpp+pattern_dpp+bm25)
uv run python main.py --strategy moe --dataset ectsum --n-samples 5

# Windowed ICL for long-document datasets
uv run python main.py --dataset subsume --strategy anchor_dpp --context-window 2

# Local Ollama models
uv run python main.py --provider ollama --model qwen3:8b --dataset ectsum --n-samples 5

# Skip database storage
uv run python main.py --no-db

# List available strategies
uv run python main.py --list-strategies
```

### CLI Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--provider` | `openai`, `google`, `anthropic`, `ollama` | `google` | LLM provider |
| `--model` | string | `gemini-2.5-flash-lite` | LLM model name |
| `--dataset` | `subsume`, `puma`, `ectsum`, `physionet`, `hotpotqa`, `evidence_inference`, `contractnli` | `subsume` | Dataset to use |
| `--strategy` | string | `random` | ICL strategy name (use `moe` for ensemble) |
| `--n-icl` | integer | `2` | Number of ICL examples |
| `--n-samples` | integer | `0` (all) | Number of test samples |
| `--max-concurrency` | integer | `10` | Max concurrent LLM requests |
| `--seed` | integer | `3` | Random seed for strategy |
| `--context-window` | integer | `None` | Windowed ICL: max context window per positive |
| `--dataset-task` | string | registry default | Uniform task description for all samples |
| `--moe-k` | `2`–`6` | `4` | Number of sub-strategies for MoE ensemble |
| `--moe-strategies` | comma-separated | `None` | Custom MoE sub-strategies (overrides --moe-k) |
| `--tag` | string (repeatable) | `None` | Tag(s) for the run |
| `--save-scores` | path | `None` | Save scored DataFrame to parquet |
| `--no-db` | flag | `False` | Skip database storage |
| `--no-icl` | flag | `False` | ICL-free baseline mode |
| `--run-name` | string | `None` | Optional name for the run |
| `--verbose` | flag | `False` | Enable debug logging |
| `--list-strategies` | flag | | Print strategies and exit |
| `--clear-db` | flag | `False` | Delete all runs from DB and exit |
| `--ollama-base-url` | URL | `http://localhost:11434` | Ollama server URL |
| `--ollama-num-ctx` | integer | `None` | Ollama context window size in tokens |
| `--ollama-timeout` | integer | `300` | Ollama request timeout in seconds |

## ICL Selection Strategies

### Global Strategies (shared ICL block for all samples)

| Strategy | Name | Description |
|----------|------|-------------|
| Random | `random` | Uniform random sampling (baseline) |
| Most Positive | `most_positive` | Samples with highest positive label count |
| DPP | `dpp` | Determinantal Point Process for maximum diversity (requires `--n-icl 2` or higher) |
| Representative | `rep` | Samples closest to centroid (most typical) |
| Anchored DPP | `anchor_dpp` | Centroid anchor + conditional DPP expansion |
| Boundary | `boundary` | Examples where positive/negative sentences are hardest to distinguish (sentence-level) |
| Pattern DPP | `pattern_dpp` | DPP diversity over relevancy-direction vectors (requires `--n-icl 2` or higher) |
| Clarity | `clarity` | Examples with most coherent within-class sentence structure (sentence-level) |
| Fisher | `fisher` | Examples with highest Fisher discriminant ratio (sentence-level) |

### Local Strategies (per-sample ICL selection)

| Strategy | Name | Description |
|----------|------|-------------|
| BM25 | `bm25` | BM25 lexical similarity to each test sample |
| KNN | `knn` | K-nearest-neighbor embedding similarity |

For multi-intent datasets (SubSumE, PUMA, Evidence Inference, ContractNLI), strategies automatically use per-intent stratified variants.

## Supported Datasets

| Dataset | Domain | Intent | Stratified ICL |
|---------|--------|--------|----------------|
| SubSumE | Wikipedia summaries | 10 query intents | Yes |
| PUMA | Medical Q&A | 5 perspectives | Yes |
| ECTSum | Earnings calls | None | No |
| PhysioNet | Medical de-id | None | No |
| HotpotQA | Multi-hop QA | Question as intent | No |
| Evidence Inference | Clinical trials | 8 user perspectives | Yes |
| ContractNLI | Legal (NDA) | 17 hypotheses | Yes |

## Supported LLMs

Any LangChain-compatible chat model:

```python
# Google (default)
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Ollama (local models, no API key needed)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:8b", format="json")
```

## Configuration

Set API keys in a `.env` file or as environment variables:

```bash
cp .env.example .env
# Edit .env and add your API key(s)
```

Required (set one based on your provider):
- `OPENAI_API_KEY` - For GPT models
- `GOOGLE_API_KEY` - For Gemini models
- `ANTHROPIC_API_KEY` - For Claude models

Optional:
- `SUPABASE_DATABASE_URL` - Full PostgreSQL connection string (for experiment storage)
- `OLLAMA_BASE_URL` - Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_NUM_CTX` - Ollama context window size in tokens
- `OLLAMA_TIMEOUT` - Ollama request timeout in seconds (default: `300`)

## Key Features

### LLM Response Caching

LLM responses are cached by default using SQLite (`.langchain_cache.db`):

```python
# Caching enabled by default (uses .langchain_cache.db)
result = score_dataframe(df, llm, output_col="scores")

# Disable caching
result = score_dataframe(df, llm, output_col="scores", cache_path=None)

# Custom cache path
result = score_dataframe(df, llm, output_col="scores", cache_path="my_cache.db")
```

### Async Processing with Rate Limiting

The scoring pipeline uses async processing with configurable concurrency and automatic rate limit handling:

```python
result = score_dataframe(df, llm, output_col="scores", max_concurrency=20)
```

### ScoringResult

`score_dataframe()` returns a `ScoringResult` dataclass:

```python
result = score_dataframe(df, llm, output_col="scores")
result.df           # The scored DataFrame
result.n_scored     # Number successfully scored
result.n_failed     # Number that failed
result.profile      # RunProfile with timing data
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=conformal_relevance
```

## Project Structure

```
conformal_relevance/
    __init__.py          # Public API exports
    types.py             # TypedDict definitions, shared constants, helpers
    models.py            # Pydantic model for structured LLM output
    utils.py             # Utility re-exports (split_sentences)
    prompts/             # Prompt templates and formatting
    scoring/             # Scoring pipeline, MoE fusion, and LLM integration
    icl/                 # Strategy-based on-the-fly ICL generation
    evaluation/          # Metrics, statistics, error analysis
    db/                  # Database layer for experiment storage
    dashboard/           # Dash-based experiment dashboard
    data_loading/        # Dataset processing and splitting (7 datasets)
tests/                   # Test suite
data/                    # Processed datasets
notebooks/               # Paper-relevant analysis notebooks (see notebooks/README.md)
    dataset_inspection/  # Per-dataset raw→processed exploration
experimental_results/    # Notebook-derived parquets + MD reports (parquets gitignored; see experimental_results/README.md)
scripts/                 # Analysis + figure-regeneration utilities (see scripts/README.md)
    figures/             # Standalone figure regeneration scripts (regenerate_*.py)
    conformal_analysis/  # §5.4 conformal coverage pipeline
paper/                   # Research paper artifacts
    currentresults/      # Synthesized findings + investigations/ (root-cause analyses)
    figures/             # Paper-ready figure assets (see paper/figures/README.md)
PlanDocs/                # Workflow plan docs (YYYY-MM-DD-<name>.md)
main.py                  # Strategy-driven experiment runner
```

## Citation

If you find this repository useful, please cite the paper as follows

```bibtex
@article{huang2026unifying,
      title={Unifying Conformal Language Tasks with In-Context Ensembles}, 
      author={Xiao Shi Huang and Chen-Yuan Lin and Bruce Kuwahara and Kin Kwan Leung and Jesse C. Cresswell},
      year={2026},
      journal={arXiv:2609.03005}
}
```

## License

MIT
