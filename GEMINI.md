# Gemini & System Architecture Reference

> **SuperClaude Framework Active**: This project utilizes the SuperClaude operational framework.
> See [SuperClaude Context](.gemini/superclaude_context.md) for Personas, Rules, and Modes.

This guide details the **Dual-Service Architecture** of the Medical RAG Pipeline, distinguishing between the Core Processing Engine and the Standalone Monitoring Dashboard.

---

## 🇰🇷 Language & Localization

*   **Primary Language:** **Korean (한국어)**.
*   **LLM Outputs:** All generated content (Summaries, Contexts, Topic Keywords) must be in Korean.
*   **Agent Interaction:** When maintaining this project or interacting via CLI, **responses and explanations should be provided in Korean** unless explicitly requested otherwise.

---

## 🏗️ System Architecture: The "Dual-Service" Model

The project is architected as two completely independent services that share only data interfaces (Storage & Config). This decoupling ensures that the lightweight Dashboard can be deployed separately from the heavy-duty Core Pipeline.

### 1. Core Processing Service (Root)
*   **Location:** `/` (Root Directory)
*   **Role:** The backend worker responsible for heavy-lifting tasks.
*   **Key Responsibilities:**
    *   YouTube Data Collection
    *   LLM Processing (Gemini/OpenRouter)
    *   Vector Embedding & Indexing
    *   R2 Storage Management
*   **Codebase:** Uses the `src/` package structure.
*   **Dependencies:** Heavy (Torch, LangChain, etc. - inferred from complex tasks).

### 2. Dashboard Service (`/dashboard`)
*   **Location:** `/dashboard/`
*   **Role:** A lightweight, standalone web viewer for system monitoring.
*   **Design Principle:** **Zero-Dependency on Core**.
    *   It does **not** import from `src/`.
    *   It implements its own lightweight clients (`R2Client`, `PineconeClient`, `StateManager`) to remain independent.
    *   It can be deployed on serverless/PaaS platforms (like Railway) without the heavy dependencies of the core pipeline.
*   **Key Responsibilities:**
    *   Real-time status visualization.
    *   Data inspection (R2/Pinecone stats).
    *   State recovery.

---

## 📂 Project Structure & Boundaries

```text
/ (Root)
├── main.py                     # [Core] Pipeline Entry Point
├── src/                        # [Core] Logic Modules (Collectors, Processors)
├── config/                     # [Core] Configuration
├── data/                       # [Shared] Local State & Cache
├── .env                        # [Shared] Configuration Source
│
└── dashboard/                  # [Service] Standalone Dashboard
    ├── main.py                 # [Dash] FastAPI App (Self-contained logic)
    ├── templates/              # [Dash] UI Templates
    ├── railway.toml            # [Dash] Deployment Config
    ├── Procfile                # [Dash] Deployment Command
    └── requirements.txt        # [Dash] Lightweight Dependencies
```

---

## 🚀 Service 1: Core Pipeline (`src/`)

The Core Pipeline is driven by `main.py` and utilizes the `src` modules.

### Gemini Processor (`src.processors.gemini_processor`)
The central intelligence unit using a Multi-Provider Strategy:
1.  **Google Gemini API**: Primary provider (Free Tier optimization).
2.  **OpenRouter API**: Fallback provider (for `QuotaExceededError`).
3.  **Embeddings**: Generates 1024-dimension vectors (MRL) for Pinecone.

### Storage & Vector DB (`src.storage`, `src.vector_db`)
*   **R2Storage**: Handles raw JSON storage (transcripts, chunks).
*   **PineconeManager**: Manages vector indices and semantic search.

---

## 📊 Service 2: Monitoring Dashboard (`dashboard/`)

The Dashboard is a FastAPI application designed for observability.

### Independence & Duplication
To maintain isolation, the dashboard **re-implements** necessary logic instead of importing it:
*   **`R2Client`**: A lightweight Boto3 wrapper, distinct from `src.storage.r2_storage`.
*   **`PineconeClient`**: A simplified status checker, distinct from `src.vector_db.pinecone_manager`.
*   **`StateManager`**: Reads `data/state.json` or reconstructs state from R2 metadata.

### Features
*   **Pipeline Tracking**: Visualizes the lifecycle of a video (Pending -> Processing -> Completed).
*   **Storage Inspection**: Counts objects in R2 buckets directly.
*   **Recovery Tool**: Can rebuild the local `state.json` file by scanning R2 if the local state is lost.

### Deployment (Railway)
The `dashboard/` folder contains everything needed for deployment:
*   `railway.toml`: Configures the build.
*   `Procfile`: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## 🔌 Shared Interfaces

Despite code separation, both services connect to the same infrastructure via:

1.  **Environment Variables (`.env`)**:
    *   Both services load the **same** `.env` file from the project root.
    *   Keys: `GOOGLE_API_KEY`, `R2_ACCESS_KEY_ID`, `PINECONE_API_KEY`, etc.

2.  **Cloud Storage (Cloudflare R2)**:
    *   **Core**: Writes transcripts, chunks, and metadata.
    *   **Dashboard**: Reads these files to display status and details.

3.  **Vector Database (Pinecone)**:
    *   **Core**: Upserts vectors.
    *   **Dashboard**: Queries index statistics (vector counts).

4.  **Local State (`data/state.json`)**:
    *   **Core**: Writes processing status.
    *   **Dashboard**: Reads to show real-time progress (when running locally).

---

## 🛠️ Usage Guide

### Gemini CLI Commands
Use these slash commands for common tasks:

*   **/test**: Run unit tests (e.g., `/test`, `/test tests/test_pipeline.py`).
*   **/run:core**: Start the Core Pipeline.
*   **/run:dash**: Start the Monitoring Dashboard (Auto-reload enabled).
*   **/lint**: Run code linting with Ruff.
*   **/pr:review**: Generate a PR review checklist and risk analysis.

> **Tip:** The `dashboard/` directory has its own `GEMINI.md` context. When working in that folder, Gemini will automatically load those specific rules.

### Running the Core Pipeline
```bash
# Install Core Dependencies
pip install -r requirements.txt

# Run Pipeline
python main.py
```

### Running the Dashboard
```bash
# Go to Dashboard Directory
cd dashboard

# Install Dashboard Dependencies (Lightweight)
pip install -r requirements.txt

# Run Dashboard
python main.py
# Access at http://localhost:8000
```

---

## ⏱️ Rate Limiting & Quotas

*   **Core**: Strictly rate-limited (5 RPM) for Google Gemini Free Tier.
*   **Dashboard**: Uses internal **TTL Caching (60s)** for R2 and Pinecone queries to avoid API rate limits and reduce costs during monitoring.