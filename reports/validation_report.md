# Pipeline Validation & Verification Report (Stages 1–2)

**Project**: Customer Support AI Agent (AppleSupport)  
**Date**: 2026-09-10  
**Audit Scope**: End-to-end audit of data ingestion, thread reconstruction, brand filtering, heuristic classification, clean reproduction, and repository security.  

---

## Overall Gate Decision

### **READY FOR STAGE 3: YES (with documented heuristic limitations)**

> [!NOTE]
> The data pipeline, schema integrity, graph-based thread reconstruction, row conservation (100%), fresh clone reproduction (2.57 minutes vs 5.0 minute target), and test suite (6/6 passing) are **fully verified and production-ready**.  
> The rule-based resolution heuristic shows known linguistic edge cases (sarcasm and polysemy with words like *"fixed"*), which are documented in detail below and will serve as benchmark ground truth for Stage 3/4 ML and LLM-based categorization.

---

## Scorecard & Summary Table

| Step | Area | Verdict | Key Finding |
| :--- | :--- | :---: | :--- |
| **Step 1** | Manual Thread Spot-Check | **NEEDS-ATTENTION** | **15/15 (100%)** chronological coherence; **11/15 (73.3%)** human agreement on resolution tags. 4 edge-case discrepancies documented (3 false positive gratitude, 1 false negative pending). |
| **Step 2** | Fresh-Clone Reproduction | **PASS** | Cloned from GitHub into clean environment. Pipeline executed in **154.04s (2.57 min)**, beating the 5-min target. All 6 tests passed (100%). README updated with verbatim instructions. |
| **Step 3** | Cleanup Audit Verification | **PASS** | Verified that no AmazonHelp parquet artifacts were ever committed to Git; `brand_candidates.csv` preserved as empirical evidence. All 18 "keep" files verified on disk. Zero cache tracked. |
| **Step 4** | Secrets & Credentials Scan | **PASS** | Full Git history scan (`git log -p --all`) found **0 credentials, API keys, or tokens**. `.gitignore` strictly guards `.env`, `.venv/`, `__pycache__/`, and credential files. `.env.example` added. |

---

## Detailed Findings by Step

### Step 1: Manual Thread Spot-Check
- **Artifact**: Full citable transcripts and judgments saved in [reports/manual_spotcheck.md](file:///d:/Academic%20Projects/Hiver/reports/manual_spotcheck.md).
- **Sampling**: 15 threads sampled via fixed seed (42) from `data/processed/AppleSupport_threads.parquet` across 3 strata:
  - 5 `brand_final_reply_dormant_24h`
  - 5 gratitude-tagged (`customer_gratitude_confirmed_by_brand` / `customer_gratitude_closure`)
  - 5 unresolved / unknown (`pending_brand_reply` / `brand_final_reply_recent_insufficient_observation_window`)
- **Key Results**:
  1. **Thread Reconstruction Integrity**: **100% Pass**. Across both simple 2-turn dialogs and complex 13-turn branching trees (e.g. `T_1149745`), all parent-child reply relationships were ordered chronologically without orphaned or mismatched tweets.
  2. **Resolution Tagging Accuracy (11/15 Agreement)**:
     - **3 False Positives in Gratitude Rules**:
       - `T_543708`: Customer said *"if this isn't fixed soon..."* (conditional complaint). Regex matched `"fixed"`.
       - `T_663919`: Customer said *"I fixed the I thing once before, I'm not doing it again"* (workaround fatigue). Regex matched `"fixed"`.
       - `T_1862049`: Customer said *"Cheers @AppleSupport for deleting all my notes! 😒"* (sarcastic complaint). Regex matched `"cheers"`.
     - **1 False Negative in Unresolved Rules**:
       - `T_360269`: Apple provided the solution, customer thanked Apple (*"Thank you Apple!!"*), but subsequent customer-to-customer banter caused the thread to end on a customer tweet without gratitude, tagging it `pending_brand_reply`.
- **Heuristic Validation Verdict**: The heuristic functions as designed for a deterministic, rule-based baseline, successfully identifying true dormancy and explicit thank-yous, but exhibits standard NLP limitations (polysemy, sarcasm, and conversational trailing). These edge cases are documented for refinement in Stage 3 taxonomy modeling.

---

### Step 2: Fresh-Clone Reproduction Test
- **Repository Tested**: `https://github.com/Amar-7778/Customer-Support-Agent-Apple-Support-`
- **Reproduction Environment**: Clean temporary directory, fresh Python 3.11 virtual environment (`.venv`), dependencies installed from `requirements.txt`.
- **Raw Data Setup**: Tested automated placement of Kaggle `twcs.csv` into `data/raw/twcs.csv`.
- **Benchmark Metrics**:
  - **Raw Load & Validation**: 2,811,774 rows loaded in **23.48s**.
  - **Thread Reconstruction**: 798,197 threads reconstructed in **49.20s** (100% row conservation verified).
  - **Brand Benchmark**: Top 20 brands benchmarked in **56.29s**.
  - **Brand Filter & Resolution Tagging**: 80,717 AppleSupport threads extracted in **10.15s**.
  - **Total Pipeline Wall-Clock Time**: **154.04 seconds (2.57 minutes)**.
  - **Performance Targets**:
    - Target: < 5.0 minutes $\to$ **PASSED** (2.57 min, 48% under budget).
    - README Promise: < 15.0 minutes $\to$ **PASSED**.
- **Automated Tests**:
  ```text
  tests/test_ingest.py::test_schema_validation_success PASSED
  tests/test_ingest.py::test_schema_validation_missing_column PASSED
  tests/test_ingest.py::test_thread_reconstruction_conservation_and_branching PASSED
  tests/test_ingest.py::test_resolved_heuristic_determinism PASSED
  tests/test_ingest.py::test_brand_filtering_non_empty_output PASSED
  tests/test_ingest.py::test_real_dataset_output_artifact PASSED
  ====== 6 passed in 4.37s ======
  ```
- **README Fixes Applied**:
  - Replaced `<repository-url>` placeholder with the actual GitHub clone URL.
  - Added an automated one-line `kagglehub` command to fetch and place `twcs.csv` automatically into `data/raw/`.

---

### Step 3: Cleanup Audit Verification
- **Artifact Isolation**:
  - Searched full commit history for any AmazonHelp-related parquet files (`git log --all --name-only`). Zero were ever committed.
  - `reports/brand_candidates.csv` correctly retains `AmazonHelp` (and 18 other brands) purely as comparative benchmark context for Decision 1.
- **Audit File Verification**:
  - Confirmed that all 18 files listed as "keep" in [reports/cleanup_audit.md](file:///d:/Academic%20Projects/Hiver/reports/cleanup_audit.md) are present on disk.
  - Confirmed that all cache directories (`__pycache__`, `.pytest_cache`) are excluded from Git tracking.

---

### Step 4: Security & Secrets Check
- **Commit History Scan**:
  - Executed automated regex scanning across all diffs in Git history (`git log -p --all`) for API keys, secret keys, tokens, Bearer headers, and private key blocks.
  - **Result**: **0 secrets found in Git history**.
- **Exclusion Verification**:
  - Confirmed [.gitignore](file:///d:/Academic%20Projects/Hiver/.gitignore) includes `.env`, `.env.*`, `.venv/`, `env/`, `venv/`, `__pycache__/`, `*.pem`, `*.key`, and `credentials.json`.
  - Added a clean [.env.example](file:///d:/Academic%20Projects/Hiver/.env.example) template for upcoming API integrations.

---

## Action Items for Stage 3 (Taxonomy & Classification)

1. **Taxonomy Ground Truth**: When training or prompting classifiers in Stage 3, do not rely solely on the regex gratitude heuristic for resolution labels; incorporate conversational context to handle sarcasm and trailing banter.
2. **Conversation Pruning**: Consider stripping conversational post-scripts (friend-to-friend banter) following a confirmed brand resolution turn to improve downstream embedding quality.
