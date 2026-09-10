# Repository Cleanup Audit (Stages 1–2 Housekeeping)

Date: 2026-09-10  
Repository: Customer Support AI Agent (Hiver Take-Home)

This audit inspects every file in the repository prior to restructuring into the target layout. No files have been deleted during this step.

---

## 1. Inventory & Categorization Table

| File Path | Size (Bytes) | Last Modified (UTC) | Category | Justification |
| :--- | :--- | :--- | :--- | :--- |
| `.gitignore` | 165 | 2026-09-10 04:19:46 | **keep** | Git exclusion specification for raw data and cache. |
| `Makefile` | 466 | 2026-09-10 03:51:12 | **keep** | Command-line shortcuts for ingestion (`make ingest`) and testing (`make test`). |
| `README.md` | 8,062 | 2026-09-10 03:52:25 | **keep** | Setup, architecture overview, and replication instructions. |
| `requirements.txt` | 75 | 2026-09-10 03:48:58 | **keep** | Core Python dependencies (`pandas`, `pyarrow`, `scipy`, `pytest`, `kagglehub`). |
| `run_pipeline.py` | 6,127 | 2026-09-10 04:15:29 | **keep** | End-to-end CLI orchestrator for Stages 1 & 2. |
| `src/__init__.py` | 48 | 2026-09-10 03:49:07 | **keep** | Python root package initialization. |
| `src/ingest/__init__.py` | 605 | 2026-09-10 03:49:17 | **keep** | Public module exports for ingestion package. |
| `src/ingest/load_raw.py` | 6,660 | 2026-09-10 04:00:54 | **keep** | Stage 1.1 raw CSV loader, schema validator, and baseline stats auditor. |
| `src/ingest/reconstruct_threads.py` | 13,667 | 2026-09-10 04:01:17 | **keep** | Stage 1.2 conversation graph reconstruction and row conservation checker. |
| `src/ingest/heuristics.py` | 6,492 | 2026-09-10 04:06:17 | **keep** | Configurable resolution heuristic (`ResolutionConfig`, gratitude/dormancy rules). |
| `src/ingest/filter_brand.py` | 14,443 | 2026-09-10 04:10:53 | **keep** | Stage 2 brand candidates benchmark and target brand thread filter. |
| `tests/__init__.py` | 15 | 2026-09-10 03:51:23 | **keep** | Pytest test package marker. |
| `tests/test_ingest.py` | 11,026 | 2026-09-10 04:13:15 | **keep** | Pytest test suite (6/6 passing: schema, conservation, branching, heuristics, artifacts). |
| `data/raw/twcs.csv` | 516,508,641 | 2026-09-10 03:56:09 | **keep** | Full raw Kaggle dataset (2,811,774 rows, gitignored). |
| `data/processed/threads.parquet` | 249,046,537 | 2026-09-10 04:16:52 | **keep** | Full dataset reconstructed conversation threads (798,197 threads). |
| `data/processed/AppleSupport_threads.parquet` | 20,594,088 | 2026-09-10 04:18:16 | **keep** | Final Stage 2 deliverable for AppleSupport with resolution tags (80,717 threads). |
| `reports/brand_candidates.csv` | 1,286 | 2026-09-10 04:18:06 | **keep** | Empirical comparison of top 20 candidate brands by volume. |
| `reports/pipeline_stats.json` | 2,529 | 2026-09-10 04:18:16 | **keep** | Machine-readable audit stats and row counts across all pipeline stages. |
| `.pytest_cache/` (all files) | ~932 | 2026-09-10 04:18:59 | **remove** | Ephemeral pytest cache. |
| `src/__pycache__/` (all `.pyc`) | 151 | 2026-09-10 03:51:56 | **remove** | Ephemeral Python bytecode cache. |
| `src/ingest/__pycache__/` (all `.pyc`) | ~59,387 | 2026-09-10 04:11:11 | **remove** | Ephemeral Python bytecode cache. |
| `tests/__pycache__/` (all `.pyc`) | ~33,621 | 2026-09-10 04:13:26 | **remove** | Ephemeral Python bytecode cache. |

---

## 2. Brand Specificity & Leftover Artifact Audit

- **Artifacts tied to brands OTHER than AppleSupport**:
  - `reports/brand_candidates.csv` contains benchmark statistics for the top 20 brands (including AmazonHelp, Uber_Support, etc.) to empirically justify the brand selection. This is an intentional analytical report and not leftover data clutter.
  - No leftover `AmazonHelp_threads.parquet` or other brand-specific parquet files exist in `data/processed/`. Only `AppleSupport_threads.parquet` was generated as the final target deliverable.
- **Duplicate or Superseded Artifacts**:
  - None found. Only the final unified `threads.parquet` and `AppleSupport_threads.parquet` exist in `data/processed/`.
- **Unreferenced Notebooks or Scripts**:
  - None found. All scripts (`run_pipeline.py`, `load_raw.py`, `reconstruct_threads.py`, `heuristics.py`, `filter_brand.py`) are directly referenced in the reproduction path and test suite.
- **Empty / Placeholder Files**:
  - `src/__init__.py` and `tests/__init__.py` are standard Python package markers (48 bytes and 15 bytes respectively).

---

## 3. Items Flagged as "Unclear"

**None (0 items).**  
All files have a definitive purpose and clear mapping to either the final target layout or ephemeral cache categories.
