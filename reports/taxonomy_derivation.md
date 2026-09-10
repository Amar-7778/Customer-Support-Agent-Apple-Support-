# Stage 3: Intent Taxonomy Derivation & Corpus Classification Report

- **Target Brand**: `AppleSupport`
- **Source Artifact**: `data/processed/AppleSupport_threads.parquet` (80,717 threads)
- **Resolved Corpus Analyzed**: 73,997 resolved customer inquiries
- **Output Taxonomy**: `taxonomy.yaml`
- **Classified Corpus**: `data/processed/AppleSupport_classified_corpus.parquet`
- **Date**: 2026-09-10

---

## 1. Stratified Sampling Methodology

To ensure reproducible, unbiased representation without synthetic data:
- **Input Population**: 80,717 conversation threads containing `AppleSupport`.
- **Target Message**: The initial customer inquiry (first inbound message with `inbound == True`) from each thread.
- **Stratification Strata**:
  1. `time_period`: Temporal monthly buckets (`2017_10`, `2017_11`, `2017_12`, `pre_2017_10`).
  2. `thread_length_bin`: Conversation depth (short: 2 tweets, medium: 3–4 tweets, long: 5+ tweets).
- **Sample Drawn**: Exactly $N = 2,000$ messages sampled proportionally across all 12 composite strata with fixed random seed (`seed = 42`).
- **Saved Sample**: `data/processed/taxonomy_sample.parquet`.

---

## 2. Silhouette Score Sweep & Statistical Clustering Limitations

The 2,000 sampled customer inquiries were embedded into a dense 384-dimensional semantic space using `sentence-transformers/all-MiniLM-L6-v2` via ONNX Runtime and L2-normalized.

K-Means clustering was evaluated across a parameter sweep of $k \in [5, 15]$ (`random_state = 42`, 10 initializations per $k$):

| Number of Clusters ($k$) | Silhouette Score | Interpretation |
| :---: | :---: | :--- |
| $k = 5$ | 0.0283 | Coarse grouping; merges audio, screen, and battery into broad hardware bucket. |
| $k = 6$ | 0.0338 | Better separation, but clusters overlap between update errors and autocorrect. |
| $k = 7$ | 0.0348 | Strong secondary peak; separates OS updates and general support. |
| **$k = 8$** | **0.0388** | **Global Silhouette Peak**; optimal relative cluster cohesion. |
| $k = 9$ | 0.0327 | Score drops; creates fragmented sub-clusters within battery complaints. |
| $k = 10$ | 0.0340 | Slight recovery, but introduces redundant update splits. |
| $k = 11$ | 0.0296 | Degrading cluster boundaries; over-segmentation. |
| $k = 12$ | 0.0327 | Splits iOS 11.1 updates into redundant sub-versions. |
| $k = 13$ | 0.0283 | High intra-cluster dispersion. |
| $k = 14$ | 0.0295 | Excessive fragmentation. |
| $k = 15$ | 0.0273 | Lowest cohesion; over-partitioned. |

> [!WARNING]
> **Misleading Headline Number — Weak Cluster Separation Disclosure**:
> The silhouette scores across all values of $k$ are objectively low (peaking at only **0.0388** for $k=8$). In embedding space, values near zero indicate substantial boundary overlap and lack of dense, well-isolated clusters. This is an expected artifact of raw Twitter customer service data: messages are short (averaging ~15–25 tokens), informal, syntactically noisy, and dominated by shared generic tokens ("phone", "help", "iOS", "update"). 
> 
> Therefore, **the derived 8-intent taxonomy must be recognized as a practical, human-reviewed domain grouping informed by topic clusters, rather than a statistically strong geometric partition of the embedding space**. Clustering served as an exploratory lens to surface real exemplar topics, which human domain review then consolidated.

---

## 3. Human Review & Taxonomy Consolidation Decisions

Review of the $k=8$ raw clusters revealed natural domain boundaries as well as necessary consolidation and escalation decisions:

1. **Autocorrect 'I' Bug Separation**:
   - *Observation*: In late 2017, Apple released iOS 11.1, triggering the infamous letter "I" autocorrect bug (where typing "I" rendered as "[?] A"). Clusters 3 and 6 captured hundreds of tweets specifically about this anomaly.
   - *Decision*: Maintained a dedicated intent `keyboard_text_autocorrect` distinct from generic OS updates, enabling focused FAQ and text-replacement troubleshooting.
2. **Apple Music Playback & Library Recovery**:
   - *Observation*: Cluster 1 contained inquiries about the Apple Music app failing to open, car audio playback failing, and offline music libraries disappearing post-update.
   - *Decision*: Consolidated into `apple_music_audio_playback` (`escalation_default: false`).
3. **Escalation Policy Alignment**:
   - *Mandatory Escalation*:
     - `orders_purchases_applecare`: Pre-orders, billing errors, refunds, and warranty disputes require human agent intervention (`escalation_default: true`).
     - `account_access_apple_id`: Password resets, 2FA recovery, and account compromises pose high security risks and must always escalate (`escalation_default: true`).
     - `international_multilingual_inquiries`: Non-English tweets (Spanish, Portuguese, Italian, French) require immediate routing to native-language queues (`escalation_default: true`).

---

## 4. Finalized 8-Intent Taxonomy (`taxonomy.yaml`)

| Intent Name | Escalation Default | Sample Volume | Key Customer Grievance Focus |
| :--- | :---: | :---: | :--- |
| **`battery_power_performance`** | `standard` | 21.2% | Sudden battery drain, device dying at 30%, overheating, charging failures. |
| **`orders_purchases_applecare`** | **`ALWAYS`** | 18.9% | iPhone X pre-orders, reservations, AppleCare buying, billing disputes, refunds. |
| **`keyboard_text_autocorrect`** | `standard` | 16.1% | Keyboard glitch when typing "I", question mark box symbol, autocorrect lag. |
| **`software_update_os_bugs`** | `standard` | 13.7% | iOS 11 update failures, device freezing, app crash loops, boot loops. |
| **`apple_music_audio_playback`** | `standard` | 11.0% | Apple Music crashing, missing offline playlists, car audio connectivity. |
| **`account_access_apple_id`** | **`ALWAYS`** | 10.6% | Apple ID disabled, forgotten passcode, 2FA code missing, phishing alerts. |
| **`hardware_display_physical`** | `standard` | 4.7% | Cracked screen, unresponsive touch digitizer, black screen freeze. |
| **`international_multilingual_inquiries`** | **`ALWAYS`** | 3.8% | Spanish, Portuguese, Italian, and French support requests. |

---

## 5. Classification Audit & Token Mathematics Reconciliation

### 5.1 Empirical Token Mathematics Reconciliation
To determine whether logged API usage is consistent with full-corpus LLM classification:
- **Measured Message Length (Actual Corpus)**: 73,997 customer initial inquiries were empirically analyzed:
  - Average word count: **19.79 words**
  - Average character count: **114.32 characters**
  - Average token count: **~25.7 tokens per customer message**
- **Measured Prompt Overhead (`taxonomy.yaml`)**:
  - Full system prompt containing all 8 intent definitions, allowed schema, and 24 real customer exemplars: **770 words (~1,001 tokens)**.
- **Minimum Plausible Token Counts for 73,997 Messages**:
  - At **Batch Size 10**: 7,400 API calls $\times$ 1,258 prompt tokens + 1,109,955 completion tokens = **10,419,175 tokens (~10.4M tokens)**.
  - At **Batch Size 20**: 3,700 API calls $\times$ 1,616 prompt tokens + 1,109,955 completion tokens = **7,087,882 tokens (~7.1M tokens)**.
  - At **Batch Size 25**: 2,960 API calls $\times$ 1,769 prompt tokens + 1,109,955 completion tokens = **6,347,142 tokens (~6.3M tokens)**.
  - At **Batch Size 50**: 1,480 API calls $\times$ 2,538 prompt tokens + 1,109,955 completion tokens = **4,865,662 tokens (~4.9M tokens)**.
- **Comparison Against Logged Stats (31,002 tokens / 5 calls)**:
  - **The logged 31,002 tokens across 5 calls covers at most ~100 messages (0.13% of the corpus). It is mathematically impossible for 31,002 tokens / 5 calls to represent exhaustive LLM classification of 73,997 messages.**

### 5.2 What Actually Ran: Execution Path Diagnostics
Investigation of the code paths in `src/taxonomy/classify_full_corpus.py` revealed:
1. **Fallback Bypass**: The script ran only 5 batches (100 messages) through the LLM. The remaining **73,917 messages fell through to an unclassified fallback block** (lines 208–232) that computed TF-IDF cosine similarity against prototype strings.
2. **Provider Leakage**: Leftover non-Groq code paths (Gemini) existed in the script, violating the requirement for Groq-exclusive inference.
3. **Parquet Timestamp Alignment**: The parquet artifact was indeed written by the script, but 99.87% of its rows were populated by the TF-IDF cosine distance fallback rather than the LLM.

### 5.3 Model Provider Constraint & Groq SDK Re-Implementation
`classify_full_corpus.py` was rebuilt strictly for Groq using the official `groq` Python SDK (`from groq import Groq`):
- All non-Groq code paths (Gemini, OpenAI, Anthropic) were completely removed.
- `GROQ_API_KEY` is loaded from `.env` via `python-dotenv` and validated at startup; execution fails immediately if unset or empty.
- Row-level provenance tracking was added (`classification_source: "groq_llm"` for every processed row).

### 5.4 Verified Groq Empirical Benchmarks (200 Real Customer Inquiries)
Verified test runs were executed on 200 real customer inquiries using both available Groq models via the official Groq SDK:

1. **`qwen/qwen3.8-27b` (Current Primary Pipeline Model)**:
   - **Messages Processed**: Exactly 200 real customer inquiries
   - **API Calls Made**: **10 calls** (20 messages per batch)
   - **Prompt Tokens**: **23,703 tokens**
   - **Completion Tokens**: **6,836 tokens** (compact, strictly structured JSON output)
   - **Total Tokens Consumed**: **30,539 tokens** (exactly **152.7 tokens per message**)
   - **Wall-Clock Duration**: **323.09 seconds** (~5.4 minutes)
   - **Throughput**: **0.62 messages/sec** (includes automated exponential backoff on Groq 8,000 TPM rate limit)
   - **Estimated Cost**: **$0.0077 USD**
   - **Row Provenance**: 100% verified as `classification_source: "groq_llm"` with monotonic logging.

2. **`openai/gpt-oss-20b` (Comparison)**:
   - Required 20 calls (10 messages/batch) due to internal reasoning tokens consuming completion budget: 62,397 tokens (312.0 tokens/msg), duration 415.74s, cost $0.0062.
   - **Model Selection Decision**: `qwen/qwen3.8-27b` is selected as the default pipeline model because it produces clean, deterministic JSON without reasoning token overhead, cutting prompt/completion tokens per message in half (152.7 vs 312.0 tokens/msg).

### 5.5 Full-Corpus Scaling Reality on Groq
Extrapolating the verified `qwen/qwen3.8-27b` benchmark (152.7 tokens/msg) to all 73,997 resolved customer messages:
- **Total Required Tokens**: $73,997 \times 152.7 = \mathbf{11,299,341 \text{ tokens}}$ (~11.3 million tokens).
- **Total Required API Calls**: $73,997 / 20 = \mathbf{3,700 \text{ calls}}$.
- **Operational Duration**: Under Groq's active rate limits (8,000 TPM and 1,000 output tokens per minute [OTPM]):
  - At 1,000 OTPM limit (~35.7 messages/min):
    $$\frac{73,997 \text{ messages}}{35.7 \text{ msgs/min}} = 2,072 \text{ minutes} \approx \mathbf{34.5 \text{ hours}}$$
This empirical calculation proves that claims of exhaustive 74k-message LLM classification in minutes on standard API tiers are physically impossible.

### 5.6 Scoping to a Representative 6,000-Message Stratified Sample
To maintain rigorous, genuine LLM classification without unscientific shortcuts or multi-day runtime blocking:
- **Scoping Decision**: Full-corpus classification was officially scoped down to a **6,000-message stratified sample** (`data/processed/AppleSupport_sample_6000.parquet`).
- **Stratification Design**:
  - Sampled proportionally across the **8 draft clusters from stage 3's K-Means output** AND across **time period** (`2017_10`, `2017_11`, `2017_12`, `pre_2017_10`) across 31 composite strata (`seed = 42`).
  - Overlap with earlier 2,000-message clustering sample: exactly **150 messages (2.50% of the 6,000 sample)**.
- **Execution Architecture (`src/taxonomy/classify_stratified_sample.py`)**:
  - Exclusively powered by the official Groq Python SDK (`qwen/qwen3.8-27b`).
  - Batch size: 25 messages per call.
  - Zero fallback: Removed TF-IDF shortcut completely. Every row classified by Groq (`classification_source: "groq_llm"`).
  - Checkpointing: Saves intermediate progress after every batch to `data/processed/AppleSupport_classified_sample_checkpoint.parquet` for instant resume capability and crash resilience.
- **Downstream Pipeline Role**:
  - The classified sample (`data/processed/AppleSupport_classified_sample.parquet`) serves as:
    1. **Stage 4 Source**: Precedent retrieval index and intent-partitioned vector store for candidate response drafting.
    2. **Stage 6 Candidate Pool**: Candidate pool for golden evaluation test set curation.

---

## 6. Failure Analysis: Real LLM Understanding vs. TF-IDF Centroid Shortcut

Comparing the genuine Groq LLM predictions against the old TF-IDF centroid shortcut revealed a **64.0% discrepancy rate**. Spot-checking confirmed that the LLM is overwhelmingly more accurate because it understands syntactic nuance, colloquial expressions, and domain context:

| Tweet ID | Customer Message Text | Old TF-IDF Intent (conf) | New LLM Intent (conf) | Analysis of Ground Truth |
| :---: | :--- | :--- | :--- | :--- |
| `700` | `"@AppleSupport why are my I⍰s changing not showing up correctly on any of my social media platforms?"` | `orders_purchases_applecare` (0.029) | `keyboard_text_autocorrect` (0.96) | **LLM Correct**. TF-IDF latched onto noise; LLM correctly identified the viral iOS 11 "I" symbol autocorrect glitch. |
| `711` | `"What is up with this ? I⍰ ?"` | `apple_music_audio_playback` (0.038) | `keyboard_text_autocorrect` (0.96) | **LLM Correct**. Extremely short tweet with Unicode glitch character; TF-IDF had near-zero confidence, LLM understood the glitch. |
| `719` | `"Tf is wrong with my keyboard @115858"` | `account_access_apple_id` (0.081) | `keyboard_text_autocorrect` (0.96) | **LLM Correct**. Direct complaint about keyboard malfunction; TF-IDF misclassified due to absence of specific keywords. |
| `736` | `"Thank you @AppleSupport I updated my phone and now it is even slower and barely works. Thank you for ruining my phone."` | `apple_music_audio_playback` (0.097) | `software_update_os_bugs` (0.95) | **LLM Correct**. Sarcastic gratitude ("Thank you... for ruining my phone") completely confused TF-IDF; LLM identified post-update OS throttling. |
| `752` | `"@AppleSupport my Apple TV works fine with my phone, but when playing videos on Mac air it stutters so much & sound is still on Mac not tv?"` | `orders_purchases_applecare` (0.069) | `software_update_os_bugs` (0.90) | **LLM Correct**. Streaming stutter / AirPlay bug; TF-IDF falsely matched "Apple TV" to purchasing/store orders. |
| `756` | `"MY HOME BUTTON DOESN'T WORK #IOS11 @AppleSupport"` | `apple_music_audio_playback` (0.062) | `hardware_display_physical` (0.96) | **LLM Correct**. Physical button malfunction; TF-IDF erroneously assigned to audio playback. |
| `758` | `"Hey @115858! Last time I downloaded an update my freaking phone gave me hell. Any recommendations?"` | `orders_purchases_applecare` (0.040) | `software_update_os_bugs` (0.95) | **LLM Correct**. Apprehension about OS updates; TF-IDF failed due to slang ("gave me hell"). |
| `765` | `"After update #ios1103 no spotify on my lock screen?@AppleSupport"` | `hardware_display_physical` (0.034) | `software_update_os_bugs` (0.90) | **LLM Correct**. iOS 11 lock screen media widget glitch; TF-IDF matched "screen" to physical display damage. |
| `1755` | `"Why does my I⍰ not work ?! @115858 please fix this!!!"` | `software_update_os_bugs` (0.086) | `keyboard_text_autocorrect` (0.96) | **LLM Correct**. Autocorrect Unicode symbol glitch specifically identified by LLM rather than lumped into generic OS bugs. |

---

## 7. Pipeline Statistics & Decision Log

[`reports/pipeline_stats.json`](file:///d:/Academic%20Projects/Hiver/reports/pipeline_stats.json) has been updated with the verified Groq runs:
- **Provider**: `Groq` (Official Python SDK)
- **Model**: `qwen/qwen3.8-27b`
- **Output Artifact**: `data/processed/AppleSupport_classified_sample.parquet`
- **Scoping**: Rescoped from 74k full corpus (~34.5h) to representative 6,000 stratified sample.
- **Prior Run Status**: Superseded; prior 5-call/31,002-token numbers were mathematically inconsistent with full-corpus coverage and have been archived in Decision 7 and 8.
