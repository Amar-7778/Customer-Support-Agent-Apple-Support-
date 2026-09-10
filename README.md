# Customer Support AI Agent: Data Ingestion & Brand-Filtering Pipeline (Stages 1–2)

A production-grade, highly optimized data ingestion and brand-filtering pipeline for the Kaggle **Customer Support on Twitter** dataset (`twcs.csv`, ~2.81M rows).

Every output is derived directly from the actual raw dataset. Zero synthetic, sample, or mocked data are used.

---

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Architecture & Design](#architecture--design)
   - [Stage 1: Raw Ingestion & Schema Validation](#stage-1-raw-ingestion--schema-validation)
   - [Stage 1: Thread Reconstruction & Conservation](#stage-1-thread-reconstruction--conservation)
   - [Stage 2: Brand Candidate Benchmark](#stage-2-brand-candidate-benchmark)
   - [Stage 2: Brand Filter & Heuristic Resolution](#stage-2-brand-filter--heuristic-resolution)
3. [Setup & Quickstart from Fresh Clone](#setup--quickstart-from-fresh-clone)
4. [Execution Commands](#execution-commands)
5. [Outputs & Artifacts](#outputs--artifacts)
6. [Testing](#testing)

---

## Pipeline Overview

This repository implements the first two foundational stages of an end-to-end Customer Support AI agent:
- **Stage 1 (Ingestion & Reconstruction)**: Ingests all 2.81M raw tweets, validates schema integrity, and reconstructs multi-turn conversation threads using vectorized graph algorithms. Guarantees 100% row count conservation with zero dropped tweets, handling branches, broken links, and single-sided messages.
- **Stage 2 (Brand Filter & Resolution Heuristic)**: Benchmarks top brands by volume to ensure data-driven brand selection, filters complete customer-brand threads for a chosen target brand (e.g. `AppleSupport`), applies a configurable resolution heuristic (inactivity threshold + customer gratitude/closure language), and outputs high-performance Parquet artifacts.

---

## Architecture & Design

### Stage 1: Raw Ingestion & Schema Validation (`src/ingest/load_raw.py`)
- **Schema Validation**: Strictly checks for required columns (`tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`). Fails loudly with `SchemaValidationError` if any columns are missing or malformed.
- **Statistical Audit**: Audits row counts, inbound vs outbound distribution, timestamp range, null values per column, and malformed rows.
- **Persistence**: Records baseline metrics to `reports/pipeline_stats.json`.

### Stage 1: Thread Reconstruction & Conservation (`src/ingest/reconstruct_threads.py`)
- **Vectorized Graph Formulation**:
  - Treats each tweet as a node in an undirected reply graph.
  - Generates reply edges from `in_response_to_tweet_id` and `response_tweet_id`.
  - Uses `scipy.sparse.csgraph.connected_components` in C to partition 2.81M tweets into conversation trees in seconds.
- **Robust Edge Case Handling**:
  - **Broken Links**: If a parent tweet ID is missing from the dataset, the referenced edge is omitted. The earliest observed tweet in that branch becomes the local root; no tweets are lost.
  - **Branching**: When a tweet receives multiple replies, all branches belong to the same connected component. All replies are preserved in chronological order.
  - **Single-Sided Messages**: Isolated tweets (with no parent or reply in the dataset) form 1-message threads.
- **Row Count Conservation**:
  - Guarantees: $\sum_{\text{threads}} \text{thread\_length} = \text{total raw rows}$.
  - Every tweet in `twcs.csv` belongs to exactly one conversation thread.
- **Output**: Persisted as `data/processed/threads.parquet`.

### Stage 2: Brand Candidate Benchmark (`src/ingest/filter_brand.py`)
- Computes metrics across the top $N$ brands by volume:
  - Total brand tweets
  - Total threads involved
  - Average thread length
  - % of threads resolved
- Generates `reports/brand_candidates.csv` so brand selection is empirical and data-driven.

### Stage 2: Brand Filter & Heuristic Resolution (`src/ingest/heuristics.py` & `src/ingest/filter_brand.py`)
- **Target Brand Extraction**: Pulls full multi-turn conversation threads containing at least one interaction by the target brand (both customer inquiries and brand responses).
- **Configurable Resolution Heuristic**:
  - Fully encapsulated in `evaluate_thread_resolution()` and `ResolutionConfig`.
  - Configurable parameters: `inactivity_hours` (default: 24h) and `gratitude_keywords`.
  - **Decision Rules**:
    1. Customer inquired, brand never replied $\to$ `resolved = false` (`unanswered_by_brand`).
    2. Customer's final message contains gratitude/closure ("thank you", "resolved", "fixed", "worked", etc.) $\to$ `resolved = true` (`customer_gratitude_closure`).
    3. Thread ends with brand reply, and observation window $> N$ hours with no customer reply $\to$ `resolved = true` (`brand_final_reply_dormant_24h`).
    4. Thread ends with customer message awaiting response $\to$ `resolved = false` (`pending_brand_reply`).
    5. Single outbound announcement or indeterminate $\to$ `resolved = unknown`.
- **Output**: Persisted as `data/processed/{brand}_threads.parquet`.

---

## Setup & Quickstart from Fresh Clone

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Raw dataset `twcs.csv` (downloaded from Kaggle: `thoughtvector/customer-support-on-twitter`) placed at `data/raw/twcs.csv`.

### Step 1: Clone and Install Dependencies
```bash
# Clone the repository
git clone https://github.com/Amar-7778/Customer-Support-Agent-Apple-Support-.git
cd Customer-Support-Agent-Apple-Support-

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Ensure Raw Data is Present
Place the Kaggle `twcs.csv` file into `data/raw/twcs.csv`:
```
data/
└── raw/
    └── twcs.csv
```

**Option A (Automated Download via Kagglehub):**
```bash
python -c "import kagglehub, shutil, os, glob; p = kagglehub.dataset_download('thoughtvector/customer-support-on-twitter'); os.makedirs('data/raw', exist_ok=True); f = glob.glob(p + '/**/twcs.csv', recursive=True)[0]; shutil.copy(f, 'data/raw/twcs.csv'); print('twcs.csv placed at data/raw/twcs.csv')"
```

**Option B (Manual Download):**
Download `twcs.csv` directly from [Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place it inside `data/raw/`.

---

## Execution Commands

### Run End-to-End Pipeline
To run Stage 1 and Stage 2 for `AppleSupport`:
```bash
# Using Makefile
make ingest BRAND=AppleSupport

# OR directly using Python CLI
python run_pipeline.py --brand AppleSupport --input data/raw/twcs.csv
```

### Run Stage 1 Separately
```bash
# Step 1.1: Schema validation & raw load
python -m src.ingest.load_raw --input data/raw/twcs.csv --stats reports/pipeline_stats.json

# Step 1.2: Thread reconstruction
python -m src.ingest.reconstruct_threads --input data/raw/twcs.csv --output data/processed/threads.parquet --stats reports/pipeline_stats.json
```

### Run Stage 2 Separately
```bash
python -m src.ingest.filter_brand --brand AppleSupport --threads data/processed/threads.parquet --output data/processed/AppleSupport_threads.parquet --candidates reports/brand_candidates.csv --stats reports/pipeline_stats.json --inactivity_hours 24.0
```

---

## Directory Structure

```
.
├── data/
│   ├── raw/                  # twcs.csv (gitignored, download documented below)
│   └── processed/            # threads.parquet, AppleSupport_threads.parquet
├── src/
│   ├── __init__.py
│   ├── ingest/               # load_raw.py, reconstruct_threads.py, filter_brand.py, heuristics.py
│   ├── taxonomy/             # Stage 3 scripts (once built)
│   ├── retrieval/            # Stage 4 scripts (once built)
│   ├── agent/                # Stage 5 scripts (once built)
│   └── eval/                 # Stage 6 scripts (once built)
├── notebooks/                # Exploratory analysis only (not required for reproduction)
├── reports/                  # brand_candidates.csv, pipeline_stats.json, cleanup_audit.md
├── eval_set/                 # Golden evaluation sets (once built)
├── tests/                    # test_ingest.py (mirroring /src structure)
├── DECISIONS.md              # Running architecture & engineering decisions log
├── README.md                 # Project documentation & reproduction instructions
├── Makefile                  # CLI build/run shortcuts
├── requirements.txt          # Python dependencies
└── run_pipeline.py           # Cross-platform CLI runner
```

---

## Outputs & Artifacts

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **Pipeline Audit Stats** | `reports/pipeline_stats.json` | Exact row counts, step runtimes, and conservation audit metrics. |
| **Brand Benchmark** | `reports/brand_candidates.csv` | Empirical comparison of top candidate brands (volume, thread length, % resolved). |
| **Cleanup Audit Report** | `reports/cleanup_audit.md` | Inventory, categorization, and justification of repository files. |
| **Decisions Log** | `DECISIONS.md` | Chronological record of architectural decisions and trade-offs. |
| **Reconstructed Threads** | `data/processed/threads.parquet` | Full dataset threads with ordered `tweet_ids`, `author_ids`, `brand_handles_involved`, `thread_length`, `start_time`, `end_time`. |
| **Target Brand Threads** | `data/processed/AppleSupport_threads.parquet` | Cleaned, multi-turn conversations for the target brand with `resolved` and `resolution_reason` tags. |

---

## Testing

Run the automated test suite with pytest:
```bash
pytest tests/ -v
```

The test suite validates:
1. **Schema Validation**: Ensures missing columns raise `SchemaValidationError` loudly.
2. **Row Count Conservation**: Guarantees zero tweets are dropped during reconstruction.
3. **Graph Branching & Broken Links**: Verifies multi-reply branches and broken parent links are properly resolved.
4. **Heuristic Determinism**: Verifies resolution evaluation produces identical, deterministic tags across runs.
5. **Brand Filter Integrity**: Verifies brand filtering produces non-empty output with all required schema fields.
