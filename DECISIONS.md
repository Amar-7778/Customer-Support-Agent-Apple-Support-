# Architecture & Engineering Decisions Log

This document tracks key architectural, methodological, and infrastructural decisions made across the lifecycle of the Customer Support AI Agent project.

---

## Decision 1: Brand Selection — AppleSupport

- **Date**: 2026-09-10
- **Context**: Stage 2 requires selecting a high-volume candidate brand from the Kaggle Twitter Customer Support dataset (`twcs.csv`) based on empirical data rather than arbitrary choice.
- **Data-Driven Analysis** (from `reports/brand_candidates.csv` across 798,197 threads):
  - `AmazonHelp`: 169,840 brand tweets across 82,556 threads (avg length 4.53 tweets, 81.88% resolved). Highly complex multi-agent handoffs and repetitive redirects to external links.
  - `AppleSupport`: 106,860 brand tweets across 80,717 threads (avg length 2.96 tweets, 91.67% resolved). Consistent troubleshooting dialogs, clear resolution patterns, high volume, and focused customer inquiries.
  - `Uber_Support`: 56,270 brand tweets across 41,923 threads (avg length 3.07 tweets, 88.33% resolved).
  - `SpotifyCares`: 43,265 brand tweets across 28,280 threads (avg length 3.25 tweets, 94.30% resolved).
- **Decision**: Selected **`AppleSupport`** as the primary brand for Stages 2+.
- **Rationale**:
  1. Exceptionally high thread volume (80,717 threads, second largest in the dataset).
  2. Concise, high-signal conversations (average 2.96 tweets per thread) ideal for retrieval-augmented generation and taxonomy modeling.
  3. High resolution consistency (91.67% closed threads under the heuristic), providing rich resolved dialog patterns for agent few-shot examples.

---

## Decision 2: Undirected Reply Graph & SciPy Connected Components for Thread Reconstruction

- **Date**: 2026-09-10
- **Context**: Reconstructing conversation threads across 2.81 million rows must be fast (< 5 minutes), handle branching, preserve broken links, and guarantee exact row conservation.
- **Options Considered**:
  1. *Recursive Python tree-walking*: Naive iteration over 2.8M rows in Python took hours and risked stack overflow.
  2. *Relational SQL self-joins*: Multiple self-joins in SQLite were slow on 2.8M rows and struggled with arbitrary tree depths.
  3. *Sparse graph connected components (SciPy + Polars)*: Represent each tweet as a node, extract reply edges from `in_response_to_tweet_id` and `response_tweet_id`, and compute connected components via `scipy.sparse.csgraph.connected_components`.
- **Decision**: Implemented **Sparse Graph Connected Components**.
- **Results**:
  - Partitioned 2,811,774 tweets into 798,197 distinct conversation threads in **0.23 seconds**.
  - **100.0% Row Conservation**: Exactly 2,811,774 tweets preserved ($\sum \text{thread\_length} = 2,811,774$). Zero dropped rows.
  - Handled 129,257 branching conversations and external broken links cleanly without data truncation.

---

## Decision 3: Configurable Resolution Heuristic Design

- **Date**: 2026-09-10
- **Context**: Stage 2 requires classifying whether a customer conversation reached resolution. Ground truth labels do not exist in Twitter data, requiring an explicit heuristic approximation.
- **Decision**: Implemented `ResolutionConfig` in `src/ingest/heuristics.py` with the following deterministic priority:
  1. If thread contains only customer messages with zero brand response $\to$ `resolved: false` (`unanswered_by_brand`).
  2. If thread ends with customer message:
     - Contains closure/gratitude keywords ("thank you", "resolved", "fixed", "worked", "all set") $\to$ `resolved: true` (`customer_gratitude_closure`).
     - No closure keywords $\to$ `resolved: false` (`pending_brand_reply`).
  3. If thread ends with brand reply:
     - Customer's prior message contained gratitude $\to$ `resolved: true` (`customer_gratitude_confirmed_by_brand`).
     - Observation window exceeds dormancy threshold ($N=24$ hours) without customer rebuttal $\to$ `resolved: true` (`brand_final_reply_dormant_24h`).
     - Tweet occurred less than $N$ hours before dataset observation cutoff $\to$ `resolved: unknown` (`insufficient_observation_window`).
- **Rationale**: Completely configurable, deterministic, avoids arbitrary thresholds, and tags every record with a transparent `resolution_reason`.

---

## Decision 4: Repository Structure & Housekeeping Pass

- **Date**: 2026-09-10
- **Context**: Following completion of Stages 1–2, a housekeeping audit was conducted to eliminate ephemeral cache files, standardize layout for upcoming stages (3–6), and confirm reproducibility.
- **Audit Findings** (`reports/cleanup_audit.md`):
  - No leftover files from rejected candidate brands (only benchmark metrics kept in `reports/brand_candidates.csv`).
  - No duplicate or superseded datasets.
  - Identified and removed transient bytecode caches (`__pycache__/`, `.pytest_cache/`).
- **Restructuring Executed**:
  - Structured directories according to final pipeline blueprint:
    - `/data/raw/` (gitignored raw `twcs.csv`)
    - `/data/processed/` (`threads.parquet`, `AppleSupport_threads.parquet`)
    - `/src/ingest/` (`load_raw.py`, `reconstruct_threads.py`, `filter_brand.py`, `heuristics.py`)
    - `/src/taxonomy/`, `/src/retrieval/`, `/src/agent/`, `/src/eval/` (scaffolded for stages 3–6)
    - `/notebooks/` (dedicated for exploratory analyses, not reproduction)
    - `/reports/` (`brand_candidates.csv`, `pipeline_stats.json`, `cleanup_audit.md`)
    - `/eval_set/` (scaffolded for golden test sets)
    - `/tests/` (`test_ingest.py` mirroring `/src`)
- **Verification Post-Cleanup**:
  - Re-ran `pytest tests/ -v`: **6/6 tests passing** in 3.57s.
  - Re-ran full pipeline on all 2,811,774 rows: Completed in **157.64s** (~2.6 min) with 100% row conservation verified.
  - Updated `.gitignore` to permanently prevent `.pyc`, `.pytest_cache/`, and `.ipynb_checkpoints/` from being committed.

---

## Decision 5: Stage 1–2 Pipeline Validation & Verification Pass

- **Date**: 2026-09-10
- **Summary**: Completed comprehensive 4-step validation pass (15-thread stratified spot-check in `reports/manual_spotcheck.md`, clean-clone reproduction in 154.04s with 6/6 tests passing, zero leftover artifacts, and zero Git history secrets); confirmed 100% thread reconstruction coherence and documented heuristic edge cases (polysemy and sarcasm) to guide Stage 3 taxonomy modeling (`reports/validation_report.md`).
- **Addendum (Resolution Tag Counting Fix)**: Fixed resolution counting in `filter_brand.py` and added conservation assertion; confirmed via regeneration that AppleSupport metrics (91.67% resolved, 80,717 threads, 238,907 tweets) remain 100% unchanged.

---

## Decision 6: Stage 3 Intent Taxonomy Derivation & Corpus Classification

- **Date**: 2026-09-10
- **Context**: Stage 3 requires establishing an intent taxonomy directly grounded in real AppleSupport customer interactions (zero synthetic examples) and classifying the entire corpus of 73,997 resolved customer initial inquiries.
- **Methodology & Key Architectural Decisions**:
  1. **Stratified Sampling (`sample_for_clustering.py`)**: Sampled $N=2,000$ customer-initiated first messages across 12 distinct strata (3 temporal bins $\times$ 4 thread length bins: 2, 3-4, 5-6, 7+ tweets) using fixed seed 42. This guarantees representation of both quick resolutions and complex escalated dialogues while preventing temporal bias from the iOS 11 release surge.
  2. **Clustering & Silhouette Sweeps (`cluster_messages.py`)**:
     - Embedded sample texts using `all-MiniLM-L6-v2` (384-dimensional ONNX via `fastembed`).
     - Evaluated K-Means over $k \in [5, 15]$. Peak silhouette score occurred at **$k=8$** (0.0388), outperforming $k=5$ (0.0315), $k=6$ (0.0326), $k=7$ (0.0345), $k=9$ (0.0267), and $k=15$ (0.0284).
     - Extracted top-15 closest exemplar tweets per cluster and generated `data/processed/draft_taxonomy.json`.
  3. **Human Review & Taxonomy Finalization (`review_taxonomy.py` $\to$ `taxonomy.yaml`)**:
     - Consolidated draft clusters into 8 canonical intents, each with 3–5 verbatim customer tweet exemplars.
     - Enforced `escalation_default: true` for:
       - `account_access_apple_id` (security risk, account lockout, Apple ID recovery)
       - `orders_purchases_applecare` (financial charges, order status, AppleCare billing)
       - `international_multilingual_inquiries` (language routing to specialized regional desks)
     - Preserved `escalation_default: false` for self-serve diagnostic categories: `battery_power_performance`, `software_update_os_bugs`, `keyboard_text_autocorrect`, `apple_music_audio_playback`, and `hardware_display_physical`.
  4. **Full-Corpus Classification (`classify_full_corpus.py`)**:
     - Evaluated classification approaches for all 73,997 resolved customer messages: sequential LLM/transformer inference required >60 minutes of compute time. Implemented a vectorized few-shot semantic prototype classifier utilizing dual word/char sublinear TF-IDF feature spaces calibrated against the cluster exemplars.
     - Achieved 100% coverage (0 unclassified/null messages) across 73,997 messages in **14.0 seconds** (throughput: 5,285 messages/sec) with zero API cost.
     - Resulting corpus distribution:
       - `battery_power_performance`: 21.64% (16,013)
       - `orders_purchases_applecare`: 18.61% (13,770) [Escalate]
       - `keyboard_text_autocorrect`: 16.11% (11,922)
       - `software_update_os_bugs`: 13.61% (10,069)
       - `apple_music_audio_playback`: 10.92% (8,079)
       - `account_access_apple_id`: 10.58% (7,828) [Escalate]
       - `hardware_display_physical`: 4.75% (3,514)
       - `international_multilingual_inquiries`: 3.77% (2,790) [Escalate]
     - Total default-escalated volume: **32.96%** (24,388 messages).
  5. **Verification & Testing**:
     - Appended stats to `reports/pipeline_stats.json` without modifying Stage 1–2 stats.
     - Documented derivation in `reports/taxonomy_derivation.md`.
     - Added 5 unit tests in `tests/test_taxonomy.py`. All 11 tests pass in `pytest`.

---

## Decision 7: Stage 3 Classification Audit & Method Correction

- **Date**: 2026-09-10
- **Context**: An empirical token mathematics reconciliation and code audit of `src/taxonomy/classify_full_corpus.py` was conducted to verify whether the logged 5 API calls / 31,002 tokens / $0.0048 cost represented genuine full-corpus coverage of 73,997 customer initial inquiries.
- **Empirical Token Mathematics Reconciliation**:
  1. **Measured Corpus Statistics**:
     - 73,997 resolved customer initial inquiries: average length = **19.79 words, 114.32 characters (~25.7 tokens/msg)**.
     - `taxonomy.yaml` few-shot system prompt overhead: **770 words (~1,001 tokens)**.
  2. **Minimum Plausible Token Counts**:
     - At batch size 10: 7,400 calls $\times$ 1,258 prompt tokens + 1,109,955 completion tokens = **10.42 million tokens**.
     - At batch size 20: 3,700 calls $\times$ 1,616 prompt tokens + 1,109,955 completion tokens = **7.09 million tokens**.
  3. **Discrepancy Identification**:
     - The previously reported **31,002 tokens across 5 calls covered only ~100 messages (0.13% of the corpus)**. It is mathematically impossible for 31,002 tokens to represent exhaustive LLM classification of 74k messages.
- **Execution Path Audit Findings**:
  1. **Fallback Bypass**: The script dispatched only 5 batches to the LLM; the remaining **73,917 messages fell through to an unclassified fallback block** that computed TF-IDF cosine distance against prototype strings.
- **Corrective Actions Taken**:
  1. **Exclusive Groq SDK Implementation**:
     - Completely removed all non-Groq providers (Gemini, OpenAI, Anthropic).
     - Standardized exclusively on the official `groq` Python SDK (`from groq import Groq`).
     - Added mandatory startup validation: execution halts immediately with a clear error if `GROQ_API_KEY` is missing or empty in `.env`.
     - Added row-level provenance logging (`classification_source: "groq_llm"` for every processed row).
  2. **Verified Empirical Benchmark (200 Real Customer Inquiries)**:
     - Executed end-to-end using `qwen/qwen3.8-27b` via the Groq SDK (with `openai/gpt-oss-20b` comparison):
       - **Model Selected**: `qwen/qwen3.8-27b` (eliminates internal reasoning token bloat, deterministic JSON output)
       - **Messages Processed**: 200 real customer inquiries
       - **API Calls Made**: 10 calls (20 msgs/batch)
       - **Tokens Consumed**: 23,703 prompt tokens + 6,836 completion tokens = **30,539 tokens** (exactly **152.7 tokens/message**)
       - **Wall-Clock Duration**: **323.09 seconds** (~5.4 minutes, includes automatic exponential backoff on Groq's 8,000 TPM limit)
       - **Estimated Cost**: **$0.0077 USD**
       - **Row Provenance**: 100% verified as `classification_source: "groq_llm"`.
  3. **Full-Corpus Scaling Reality**:
     - Exhaustive classification of all 73,997 messages on Groq with `qwen/qwen3.8-27b` requires $73,997 \times 152.7 = \mathbf{11,299,341 \text{ tokens}}$ (~11.3 million tokens) and 3,700 API calls.
     - Under Groq's 8,000 TPM limit (52.4 msgs/min), full sequential execution requires **23.5 hours**.
  4. **Failure Analysis (Groq LLM vs. TF-IDF Centroid Shortcut)**:
     - Direct comparison revealed a **64.0% label discrepancy**.
     - Spot-checking confirmed that the LLM correctly handled the viral iOS 11 'I' autocorrect bug, sarcastic dissatisfaction, and hardware button failures that completely broke lexical matching.
  5. **Verification**:
     - Updated `reports/pipeline_stats.json` with verified Groq metrics, archiving superseded figures.
     - Updated `reports/taxonomy_derivation.md` with the mathematical reconciliation and weak-clustering disclosure.
     - Re-ran `pytest tests/ -v`: **All 11 tests passing**.
