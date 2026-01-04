# Gemini Quick Reference for Agents

This guide covers the usage, best practices, and architecture of the Gemini-based processing pipeline in this project.

***

## 🔧 Architecture: Multi-Provider LLM System

The project uses a dual-provider system to ensure reliability and bypass quota limitations of the Google Gemini free tier.

*   **Primary Provider**: `Google Gemini API` (Direct)
*   **Fallback Provider**: `OpenRouter API` (via `google/gemini-2.5-flash`)

### LLM Manager & Fallback Logic
The `LLMManager` handles the transition between providers:
1.  **Try Google**: Attempts to use the Google Gemini API first.
2.  **Detect Quota**: If a `QuotaExceededError` (429) is received, it automatically enables **Fallback Mode**.
3.  **Switch to OpenRouter**: Subsequent requests are routed to OpenRouter until the manager is reset.
4.  **Embeddings**: Uses `models/gemini-embedding-001`. If it fails, falls back to OpenRouter's embedding model.

***

## 🔧 Installation & Setup

**Requirements:**
```bash
pip install google-generativeai requests
```

**Environment Variables (.env):**
```bash
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key  # Optional fallback
```

***

## 🚀 Core Component: `GeminiProcessor`

The `GeminiProcessor` is the high-level interface for all LLM tasks.

### Initialization
```python
from src.processors.gemini_processor import GeminiProcessor

processor = GeminiProcessor(
    rpm=5,  # Rate limit for Google Free Tier
    google_api_key="...",
    openrouter_api_key="..."
)
```

### Key Functionalities

#### 1. Transcript Refinement
Fixes misheard words, medical terminology, and formatting without changing meaning.
```python
refined_text = processor.refine_transcript(raw_text)
```

#### 2. Video Summarization
Generates a 3-5 sentence summary used as context for individual chunks.
```python
summary = processor.summarize_video(full_transcript)
```

#### 3. Integrated Context & Topic Extraction
Performs two tasks in one API call to save tokens and time.
```python
context, topics = processor.generate_chunk_context_and_topics(chunk_text, video_summary)
# context: 1-2 sentence description of chunk's role in the video
# topics: List of up to 5 medical keywords
```

#### 4. Embeddings (1024-dimension)
Uses `models/gemini-embedding-001` with Matryoshka Representation Learning (MRL) to truncate output to 1024 dimensions for Pinecone compatibility.
```python
vector = processor.get_embedding("Text to embed")
# Returns a normalized list[float] of length 1024
```

***

## ⏱️ Rate Limiting & Quotas

### Google Gemini Free Tier
*   **Rate Limit**: Default 5 RPM (Requests Per Minute).
*   **Daily Quota**: Limited (varies by region/account).
*   **Handling**: The `RateLimiter` class ensures requests stay within the RPM limit by introducing `time.sleep()`.

### Implementation
```python
# RateLimiter logic used internally by GoogleProvider
limiter = RateLimiter(rpm=5)
limiter.wait_if_needed() # Called before every API request
```

***

## 🚨 Common Patterns & Best Practices

### 1. Always Use `GeminiProcessor`
Avoid calling `GoogleProvider` or `OpenRouterProvider` directly. `GeminiProcessor` provides the prompts and fallback safety.

### 2. MRL Embeddings (1024-dim)
The project's Pinecone index is configured for **1024 dimensions**.
*   **Google**: Uses `output_dimensionality=1024`.
*   **OpenRouter**: Truncates output to `[:1024]`.
*   **Normalization**: Always normalize vectors after truncation to ensure cosine similarity works correctly.

### 3. Prompting for Medical Accuracy
When modifying prompts in `GeminiProcessor`, maintain the **"Medical Expert"** persona and strict constraints:
*   "Meaning must not be changed."
*   "Do not add medical information not present in the source."
*   "Output raw text only (no preamble)."

### 4. Monitoring Stats
Check provider usage during long-running tasks:
```python
stats = processor.get_stats()
print(f"Google: {stats['google_success']} | OpenRouter: {stats['openrouter_success']}")
```

***

## 🛠️ Error Handling

| Error Class | Cause | Action |
| :--- | :--- | :--- |
| `QuotaExceededError` | Daily limit reached (Google) | Manager switches to OpenRouter automatically. |
| `RateLimitError` | RPM/TPM limit hit | Retries with exponential backoff. |
| `ProviderError` | DNS, Timeout, Authentication | Retries up to `MAX_RETRIES` then fails. |

***

## Recommended Models (2025)

| Task | Model | Reason |
| :--- | :--- | :--- |
| Text Gen | `gemini-2.5-flash` | Extremely fast, high context window, free tier available. |
| Embedding | `gemini-embedding-001` | High quality, supports MRL truncation. |
| Fallback | `google/gemini-2.5-flash` | Consistency in output style when switching providers. |
