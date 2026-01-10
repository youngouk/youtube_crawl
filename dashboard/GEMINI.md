# Dashboard Service Context

This is the **Monitoring Dashboard** service for the Medical RAG Pipeline.

## 🏗️ Architecture
*   **Role:** Lightweight, standalone web viewer.
*   **Independence:** Zero-dependency on the core `src/` pipeline. Re-implements `R2Client`, `PineconeClient`, `StateManager` locally.
*   **Platform:** Deployable on Railway (PaaS).

## 📂 Structure
*   `main.py`: FastAPI application entry point.
*   `templates/`: UI templates (Jinja2).
*   `requirements.txt`: Lightweight dependencies (FastAPI, Boto3, etc. - NO Torch).

## 🚀 Running the Dashboard
```bash
# In /dashboard directory
cd dashboard
pip install -r requirements.txt
uvicorn main:app --reload
```

## ⚠️ Conventions
*   **Language:** Korean (한국어) for all UI/Logs.
*   **ReadOnly:** This service should **only read** state/data, never modify core pipeline state (except for specific recovery tools).
