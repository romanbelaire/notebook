# Research RAG Assistant

A local-first workflow for organizing research papers and chatting with their contents using a Retrieval-Augmented Generation (RAG) pipeline powered by a Hugging Face hosted chat model.

## 1. Architecture Overview

| Layer | What it Does |
| ----- | ------------- |
| **PDF Repository** | Put every PDF you care about in `data/papers/`. |
| **Ingestion Pipeline** (`app/ingest.py`) | • Extracts text from PDFs (PyPDF2).  
• Splits text into overlapping chunks (RecursiveCharacterTextSplitter).  
• Embeds each chunk (Sentence-Transformers `all-MiniLM-L6-v2`).  
• Builds a FAISS vector index + `docs.pkl` with text + metadata. |
| **RAG Layer** (`app/rag_chat.py`) | • Loads the FAISS index and calls a Hugging Face Inference API model.  
• Retrieves top-k chunks for each query, builds a prompt, and streams a response. |
| **Streamlit UI** (`app/app.py`) | • Sidebar button to ingest PDFs.  
• Main pane is a chat interface.  
• Chat history stored in session-state so reloads are safe. |

## 2. Setup

```powershell
# 1. Create & activate a virtual environment (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Drop PDFs into data/papers/

# Provide your Hugging Face API token (one-time):
# Save token to a file
'YOUR_HF_TOKEN' | Out-File -FilePath hf_api.txt -Encoding ascii
# or set an environment variable
$Env:HF_API_TOKEN = 'YOUR_HF_TOKEN'

# (Optional) override the model ID with `$Env:HF_MODEL_ID` (defaults to `meta-llama/Llama-2-7b-chat-hf`).
```

## 3. Running the MVP

```powershell
# Run ingestion from CLI (optional)
python -m app.ingest

# OR launch the UI (recommended)
streamlit run run_app.py
```

1. Click **Ingest PDFs** once.
2. Ask questions in the chat box. Answers are grounded in retrieved context from your papers.

## 4. Extending the System

* Add **watchdog** to monitor `data/papers/` and auto-trigger ingestion.
* Swap in another embedding model or adjust chunk size in `app/ingest.py`.
* Tune llama inference params by passing them through `RAGChat`.

---
**Fail-fast philosophy**: every component raises explicit errors if prerequisites are missing so that mis-configuration surfaces immediately. 