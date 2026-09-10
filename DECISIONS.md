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
