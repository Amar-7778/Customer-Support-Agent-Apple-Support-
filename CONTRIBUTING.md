# Contributing & Engineering Standards

## Security & Credential Handling (Standing Rule)

### 1. Zero Plaintext API Keys in Logs or Console
- **Never print a full API key to stdout, stderr, or any log file**, even during local debugging, error handling, or exception stack traces.
- All loggers, error handlers, and debug outputs that reference credentials **must always mask** the key using:
  ```python
  def mask_key(k: str) -> str:
      """Masks an API key for safe logging (e.g. gsk_R6Ns...ll2W)."""
      if len(k) > 12:
          return f"{k[:8]}...{k[-4:]}"
      return "***"
  ```
- Any PR or commit containing an unmasked key string matching `gsk_[A-Za-z0-9]{20,}` will be rejected.

### 2. Environment Variables Only
- No hardcoded API keys are permitted in source code, configuration files, test suites, or documentation.
- All credentials must be loaded at runtime exclusively through `os.getenv()` or `python-dotenv` from a local `.env` file.
- The `.env` file must strictly remain ignored by git (`.gitignore`).

### 3. Key Rotation & Compromise Protocol
- If a credential is ever exposed in a log, terminal transcript, or artifact, it must immediately be marked as **compromised**.
- Revoke the key immediately in the provider console (e.g., Groq Console).
- Generate fresh replacement keys and update `.env`.
- Audit git history (`git log -p --all | grep -i gsk_`) and repo files to verify zero persistence.

---

## Code Quality & Discipline
1. **Provider Exclusivity**: Groq is the sole LLM inference provider (`qwen/qwen3.8-27b`). No other providers.
2. **Zero Synthetic / Fallback Shortcuts**: No centroid shortcuts, simulated data, or unlogged fallbacks. All data must originate from real runs.
3. **Audit Trails & Token Accounting**: Every LLM extraction or classification run must record real token consumption, clock time, and estimated compute cost in `reports/pipeline_stats.json`.
