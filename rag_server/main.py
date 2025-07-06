from __future__ import annotations

"""FastAPI wrapper around the RAG backend and auxiliary services.

This module delegates core logic to the pre-existing `app.*` modules that were
previously wired into the Streamlit UI.  No business logic is re-implemented
here – we merely expose a thin HTTP interface following the migration plan
(section 1.1).  The implementation purposefully *fails fast and loudly* in line
with the user's code-quality requirement.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import uuid
import hashlib
import base64
import re
from functools import lru_cache
import numpy as np
import glob

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Body
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.rag_chat import RAGChat
from app.task_manager import submit as submit_task, status as task_status, result as task_result, exception as task_exception, get_progress as task_progress
from app.ingest import ingest_pdfs, ingest_pdfs_with_progress
from app.insights_store import InsightsStore
from app.metadata_db import (
    get_connection,
    list_papers,
    list_collections,
    create_collection,
    add_papers_to_collection,
    replace_chunks,
    upsert_paper,
    upsert_paper_embedding,
    get_filenames_for_collection,
    rename_collection,
    delete_collection,
    remove_papers_from_collection,
    list_papers_for_collection,
    check_paper_by_sha256,
)
from datetime import datetime
import os

__all__ = ["app"]  # uvicorn entry-point: ``rag_server.main:app``

# ---------------------------------------------------------------------------
# FastAPI application instance with CORS enabled.
# ---------------------------------------------------------------------------

app = FastAPI(title="Research-RAG Server", version="1.0.0")

# Allow the React dev-server (and any Tauri WebView in production) to talk to
# this API.  During development we default to the Vite URL; in production the
# WebView loads the bundled assets from the same origin so the CORS list may be
# tightened.  Explicitly listing hosts instead of "*" keeps us in control of
# what can call the backend.

_DEV_FRONTEND_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global singletons – kept intentionally simple.  Heavy objects such as the
# RAG model are initialised lazily so the server starts up instantly.
# ---------------------------------------------------------------------------

_model_cache: dict[str, RAGChat] = {}
_insights_store: Optional[InsightsStore] = None


def _get_insights_store() -> InsightsStore:
    global _insights_store  # noqa: PLW0603 – explicit global is fine here
    if _insights_store is None:
        _insights_store = InsightsStore()
    return _insights_store


# ---------------------------------------------------------------------------
# Pydantic schemas – kept explicit instead of "Any" to validate inputs early.
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str
    # optional props used by previous UI – ignored here but accepted to avoid
    # validation errors when the new front-end sends its existing objects.
    contexts: Optional[List[str]] = None
    citations: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    query: str
    history: List[Message]
    model_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    contexts: List[str]
    citations: List[Dict[str, Any]]


class IngestRequest(BaseModel):
    pdf_dir: str = Field(..., description="Directory containing PDF files")


class IngestResponse(BaseModel):
    task_id: str


class TaskStatusResponse(BaseModel):
    status: str
    result: Optional[Any] = None  # noqa: ANN401 – may be arbitrary JSON
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None  # For tracking ingestion progress


class InsightCreateRequest(BaseModel):
    text: str
    contexts: List[str] = Field(default_factory=list)
    title: Optional[str] = None


class InsightCreateResponse(BaseModel):
    id: str


# Request model for updating insight
class InsightUpdateRequest(BaseModel):
    text: Optional[str] = None
    contexts: Optional[List[str]] = None
    title: Optional[str] = None


# ---------------------------------------------------------------------------
# Paper & Collection schemas
# ---------------------------------------------------------------------------


class Paper(BaseModel):
    id: int
    filename: str
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    sha256: Optional[str] = None
    added_at: str


class Collection(BaseModel):
    id: int
    name: str
    created_at: str
    papers: Optional[List[Paper]] = None


class CreateCollectionRequest(BaseModel):
    name: str


class AddPapersRequest(BaseModel):
    paper_ids: list[int]


class RenameCollectionRequest(BaseModel):
    name: str


class RemovePapersRequest(BaseModel):
    paper_ids: list[int]


# ---------------------------------------------------------------------------
# Notes (ScratchPad) schemas
# ---------------------------------------------------------------------------


class NoteCreateRequest(BaseModel):
    content_md: str = Field(..., description="Note content in Markdown")
    title: Optional[str] = None


class NoteCreateResponse(BaseModel):
    paper_id: int


# ---------------------------------------------------------------------------
# Utility – text cleaner for embeddings (same as ingest.py
# ---------------------------------------------------------------------------


_STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
    'is','are','was','were','be','been','being','have','has','had','do','does','did',
    'will','would','could','should','may','might','can','this','that','these','those'
}


def _clean_text_for_embedding(text: str) -> str:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    filtered = [t for t in tokens if t not in _STOPWORDS]
    return " ".join(filtered)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_rag(model_id: Optional[str]) -> RAGChat:
    """Return a cached RAG instance for *model_id* or create one lazily."""
    model_id = model_id or "meta-llama/Llama-3.2-1B-Instruct"
    if model_id in _model_cache:
        return _model_cache[model_id]
    rag = RAGChat(model_id=model_id)
    _model_cache[model_id] = rag
    return rag


# ---------------------------------------------------------------------------
# Title generation – embed-based ranking
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_embedder():
    """Return a cached MiniLM embedder to avoid re-loading every call."""
    from sentence_transformers import SentenceTransformer  # local import to keep startup fast
    return SentenceTransformer("all-MiniLM-L6-v2")


def _clean(text: str) -> str:
    text = re.sub(r'[#*_`\[\]()]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _generate_insight_title(text: str) -> str:
    """Generate a concise 2-6-word title using sentence-similarity ranking."""
    cleaned = _clean(text)

    # Short text → just title-case a trimmed version
    if len(cleaned.split()) <= 8:
        return cleaned[:60].title()

    # 1) Split into sentences (very cheap regex)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    if not sentences:
        return cleaned[:60]

    embedder = _get_embedder()
    doc_vec = embedder.encode([cleaned])[0]
    sent_vecs = embedder.encode(sentences)

    # 2) Cosine similarity of each sentence to whole doc
    sims = np.dot(sent_vecs, doc_vec) / (
        np.linalg.norm(sent_vecs, axis=1) * np.linalg.norm(doc_vec) + 1e-8
    )
    best_idx = int(np.argmax(sims))
    title_candidate = sentences[best_idx]

    # 3) Post-process → keep first 6 informative words
    words = title_candidate.split()
    meaningful = [w for w in words if w.lower() not in _STOPWORDS][:6]
    if meaningful:
        title = ' '.join(meaningful)
    else:
        title = ' '.join(words[:6])

    title = title.strip().title()
    return title[:60] if title else cleaned[:60]


# ---------------------------------------------------------------------------
# System Prompt storage helpers & schemas
# ---------------------------------------------------------------------------

_PROMPTS_FILE = Path("system_prompts.json")

_DEFAULT_PROMPT_CONTENT = (
    "You are an expert research assistant. "
    "Answer the user based solely on the given context."
)

def _ensure_prompts_file():
    """Create the prompts JSON file with a default entry if it does not exist."""
    if _PROMPTS_FILE.exists():
        return
    data = {
        "prompts": [
            {
                "id": "default",
                "name": "Default",
                "content": _DEFAULT_PROMPT_CONTENT,
                "created_at": datetime.utcnow().isoformat(),
            }
        ],
        "selected_id": "default",
    }
    _PROMPTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _load_prompts() -> Dict[str, Any]:
    _ensure_prompts_file()
    try:
        return json.loads(_PROMPTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON in system_prompts.json") from exc

def _save_prompts(data: Dict[str, Any]) -> None:
    _PROMPTS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _get_selected_system_prompt() -> str:
    data = _load_prompts()
    selected_id: str = data.get("selected_id", "default")
    for p in data.get("prompts", []):
        if p.get("id") == selected_id:
            return p.get("content", _DEFAULT_PROMPT_CONTENT)
    # fallback – return default content even if selection missing
    return _DEFAULT_PROMPT_CONTENT

class Prompt(BaseModel):
    id: str
    name: str
    content: str
    created_at: str

class PromptsResponse(BaseModel):
    prompts: List[Prompt]
    selected_id: str

class PromptCreateRequest(BaseModel):
    name: str
    content: str

class PromptUpdateRequest(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None

class PromptSelectRequest(BaseModel):
    prompt_id: str


# ---------------------------------------------------------------------------
# Routes – ordered roughly by frequency of expected access.
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Answer *req.query* using the retrieval-augmented generation backend."""
    rag = _get_rag(req.model_id)

    # Convert history objects into plain dicts as expected by RAGChat.chat.
    hist_dicts: List[dict] = [msg.dict(exclude_none=True) for msg in req.history]

    try:
        answer: str = rag.chat(req.query, hist_dicts, system_prompt=_get_selected_system_prompt())
    except Exception as exc:  # fail loudly so front-end sees detailed message
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Retrieve sidecar data captured by RAGChat for citation rendering.
    contexts: List[str] = getattr(rag, "last_contexts", [])
    citations: List[dict] = getattr(rag, "last_citation_meta", [])

    return ChatResponse(answer=answer, contexts=contexts, citations=citations)


# ---------------------------------------------------------------------------
# PDF ingestion – potentially long-running, hence executed in background.
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest(req: IngestRequest):
    """Schedule background ingestion of all PDFs in *req.pdf_dir*."""
    pdf_dir = Path(req.pdf_dir).expanduser().as_posix()
    task_id = submit_task(ingest_pdfs_with_progress, pdf_dir=pdf_dir)
    return IngestResponse(task_id=task_id)


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
def task_status_endpoint(task_id: str):
    """Return status/result information for the background task *task_id*."""
    stat = task_status(task_id)
    if stat == "unknown":
        raise HTTPException(status_code=404, detail="Unknown task id")

    # Get progress information for all statuses
    progress = task_progress(task_id)

    if stat == "running" or stat == "pending":
        return TaskStatusResponse(status=stat, progress=progress)

    # finished – either success or error
    if stat == "done":
        try:
            res = task_result(task_id)
        except Exception as exc:
            # Should not happen (*done* without result) – surface loudly.
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return TaskStatusResponse(status=stat, result=res, progress=progress)

    if stat == "error":
        err = task_exception(task_id)
        return TaskStatusResponse(status=stat, error=str(err), progress=progress)

    # Fallback – exhaustive above, but keep mypy happy.
    raise HTTPException(status_code=500, detail="Unhandled task status")


# ---------------------------------------------------------------------------
# Insights CRUD – mirrors `app.insights_store.InsightsStore` operations.
# ---------------------------------------------------------------------------

@app.get("/insight")
def list_insights() -> List[Dict[str, Any]]:  # noqa: D401
    """Return the complete list of insights (unsorted)."""
    store = _get_insights_store()
    return store.list_all()


@app.post("/insight", response_model=InsightCreateResponse, status_code=status.HTTP_201_CREATED)
def create_insight(payload: InsightCreateRequest):
    store = _get_insights_store()
    try:
        # Generate smart title if none provided
        title = payload.title or _generate_insight_title(payload.text)
        iid = store.add_insight(payload.text, payload.contexts, title=title)
        return InsightCreateResponse(id=iid)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/insight/{insight_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_insight(insight_id: str):
    store = _get_insights_store()
    try:
        store.delete_insight(insight_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/insight/search")
def search_insights(query: str, k: int = 5):
    store = _get_insights_store()
    return store.search(query, k=k)


# Update insight endpoint
@app.put("/insight/{insight_id}", status_code=status.HTTP_204_NO_CONTENT)
def update_insight_endpoint(insight_id: str, payload: InsightUpdateRequest):
    store = _get_insights_store()
    try:
        store.update_insight(insight_id, text=payload.text, contexts=payload.contexts, title=payload.title)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Papers & Collections endpoints
# ---------------------------------------------------------------------------


@app.get("/papers", response_model=list[Paper])
def get_papers():
    try:
        conn = get_connection()
        rows = list_papers(conn)
        conn.close()
        return rows
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/papers/check-hash/{sha256}")
def check_paper_hash(sha256: str):
    """Check if a paper with the given SHA256 hash already exists."""
    try:
        conn = get_connection()
        existing_paper = check_paper_by_sha256(conn, sha256)
        conn.close()
        
        if existing_paper:
            return {"exists": True, "paper": existing_paper}
        else:
            return {"exists": False}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def extract_pdf_from_firefox_document(content: bytes) -> Union[bytes, None]:
    """Extract PDF content from Firefox PDF document (HTML wrapper)."""
    try:
        text = content.decode('utf-8', errors='ignore')
        
        # Check if it's a Firefox PDF document
        if '<!DOCTYPE html' not in text or 'pdf.js' not in text:
            return None  # Not a Firefox PDF document
        
        # Look for the PDF data URL in the HTML
        match = re.search(r'data:application/pdf;base64,([A-Za-z0-9+/=]+)', text)
        if not match:
            print('Firefox PDF document detected but no PDF data found')
            return None
        
        # Decode the base64 PDF data
        base64_data = match.group(1)
        pdf_content = base64.b64decode(base64_data)
        
        return pdf_content
    except Exception as e:
        print(f'Error extracting PDF from Firefox document: {e}')
        return None


@app.post("/upload-paper", status_code=status.HTTP_201_CREATED)
async def upload_paper(file: UploadFile = File(...)):
    """Upload a PDF file to the papers directory for web mode."""
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        content = await file.read()
        
        # Handle Firefox PDF documents by extracting actual PDF content
        extracted_pdf = extract_pdf_from_firefox_document(content)
        if extracted_pdf:
            print(f"Extracted PDF content from Firefox document: {file.filename}")
            final_content = extracted_pdf
            # Ensure filename ends with .pdf
            if not file.filename.lower().endswith('.pdf'):
                file.filename = f"{file.filename}.pdf"
        else:
            final_content = content
        
        # Calculate SHA256 hash of the final content
        sha256_hash = hashlib.sha256(final_content).hexdigest()
        
        # Check if file with this hash already exists
        conn = get_connection()
        existing_paper = check_paper_by_sha256(conn, sha256_hash)
        
        if existing_paper:
            conn.close()
            return {
                "message": f"File {file.filename} already exists (duplicate detected)",
                "duplicate": True,
                "existing_paper": existing_paper
            }
        
        # Ensure papers directory exists
        papers_dir = Path("data") / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = papers_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(final_content)
        
        # Store paper metadata with SHA256 hash
        paper_id = upsert_paper(conn, filename=file.filename, sha256=sha256_hash)
        conn.close()
        
        return {
            "message": f"File {file.filename} uploaded successfully",
            "duplicate": False,
            "paper_id": paper_id,
            "sha256": sha256_hash
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/collections", response_model=list[Collection])
def get_collections():
    try:
        conn = get_connection()
        collection_rows = list_collections(conn)
        
        # Populate papers for each collection
        collections = []
        for collection_data in collection_rows:
            papers = list_papers_for_collection(conn, collection_data["id"])
            collection_with_papers = {
                **collection_data,
                "papers": papers
            }
            collections.append(collection_with_papers)
            
        conn.close()
        return collections
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/collections", response_model=Collection, status_code=status.HTTP_201_CREATED)
def create_collection_endpoint(payload: CreateCollectionRequest):
    try:
        conn = get_connection()
        cid = create_collection(conn, payload.name)
        # Fetch record to return
        rows = [r for r in list_collections(conn) if r["id"] == cid]
        conn.close()
        if not rows:
            raise ValueError("Collection could not be retrieved after creation.")
        return rows[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/collections/{collection_id}/add", status_code=status.HTTP_204_NO_CONTENT)
def add_papers_to_collection_endpoint(collection_id: int, payload: AddPapersRequest):
    try:
        conn = get_connection()
        add_papers_to_collection(conn, collection_id, payload.paper_ids)
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/collections/{collection_id}/remove", status_code=status.HTTP_204_NO_CONTENT)
def remove_papers_from_collection_endpoint(collection_id: int, payload: RemovePapersRequest):
    try:
        conn = get_connection()
        remove_papers_from_collection(conn, collection_id, payload.paper_ids)
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def rename_collection_endpoint(collection_id: int, payload: RenameCollectionRequest):
    try:
        conn = get_connection()
        rename_collection(conn, collection_id, payload.name)
        conn.close()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection_endpoint(collection_id: int):
    try:
        conn = get_connection()
        delete_collection(conn, collection_id)
        conn.close()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Notes endpoint – create note and schedule embedding
# ---------------------------------------------------------------------------


@app.post("/note", response_model=NoteCreateResponse, status_code=status.HTTP_201_CREATED)
def create_note_endpoint(payload: NoteCreateRequest):
    try:
        notes_dir = "notes"
        Path(notes_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}.md"

        # Determine title
        title = payload.title or next((line.strip('# ').strip() for line in payload.content_md.splitlines() if line.strip()), filename)

        # Write markdown to disk
        with open(Path(notes_dir) / filename, "w", encoding="utf-8") as f:
            f.write(payload.content_md)

        # Insert into metadata DB similar to run_app behaviour
        conn = get_connection()
        paper_id = upsert_paper(conn, filename=filename, title=title)
        replace_chunks(conn, paper_id, [payload.content_md])
        conn.close()

        # Background embedding
        cleaned = _clean_text_for_embedding(payload.content_md)

        def _embed_and_store():
            from sentence_transformers import SentenceTransformer as _ST
            vec = _ST("all-MiniLM-L6-v2").encode(cleaned)
            conn_bg = get_connection()
            upsert_paper_embedding(conn_bg, paper_id, vec)  # type: ignore[arg-type]
            conn_bg.close()

        submit_task(_embed_and_store)

        return NoteCreateResponse(paper_id=paper_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Context Pool – limit retrieval to a collection or all.
# ---------------------------------------------------------------------------


class ContextPoolRequest(BaseModel):
    collection_id: Optional[int] = Field(None, description="Null to use all papers")
    model_id: Optional[str] = None


@app.post("/context_pool", status_code=status.HTTP_204_NO_CONTENT)
def set_context_pool(req: ContextPoolRequest):
    rag = _get_rag(req.model_id)
    if req.collection_id is None:
        rag.set_allowed_sources(None)
        return

    try:
        conn = get_connection()
        filenames = get_filenames_for_collection(conn, req.collection_id)
        conn.close()
        rag.set_allowed_sources(set(filenames))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Static serving of exported notes when running in browser.
notes_static_dir = Path("notes")
notes_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/notes", StaticFiles(directory=notes_static_dir), name="notes")

# ---------------------------------------------------------------------------
# Notes deletion endpoint (mirror front-end expectations)
# ---------------------------------------------------------------------------

@app.delete("/note/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note_endpoint(paper_id: int):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT filename FROM papers WHERE id = ?", (paper_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Note not found")
        filename = row[0]
        # Remove DB rows – ON DELETE CASCADE handles chunks, embeddings, etc.
        cur.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        conn.commit()
        conn.close()

        # Remove the underlying markdown file if still present.
        file_path = notes_static_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                # Log error but do not fail request.
                import logging
                logging.warning("Could not delete note file %s", file_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# System Prompts CRUD
# ---------------------------------------------------------------------------

@app.get("/prompts", response_model=PromptsResponse)
def list_prompts_endpoint():
    data = _load_prompts()
    return data


@app.post("/prompts", response_model=Prompt, status_code=status.HTTP_201_CREATED)
def create_prompt_endpoint(payload: PromptCreateRequest):
    data = _load_prompts()
    new_id = uuid.uuid4().hex[:8]
    new_prompt = {
        "id": new_id,
        "name": payload.name,
        "content": payload.content,
        "created_at": datetime.utcnow().isoformat(),
    }
    data["prompts"].append(new_prompt)
    _save_prompts(data)
    return new_prompt


@app.put("/prompts/{prompt_id}", response_model=Prompt)
def update_prompt_endpoint(prompt_id: str, payload: PromptUpdateRequest):
    data = _load_prompts()
    for p in data["prompts"]:
        if p["id"] == prompt_id:
            if payload.name is not None:
                p["name"] = payload.name
            if payload.content is not None:
                p["content"] = payload.content
            _save_prompts(data)
            return p
    raise HTTPException(status_code=404, detail="Prompt not found")


@app.post("/prompts/select", status_code=status.HTTP_204_NO_CONTENT)
def select_prompt_endpoint(payload: PromptSelectRequest):
    data = _load_prompts()
    if not any(p["id"] == payload.prompt_id for p in data["prompts"]):
        raise HTTPException(status_code=404, detail="Prompt id not found")
    data["selected_id"] = payload.prompt_id
    _save_prompts(data)
    return


# ---------------------------------------------------------------------------
# Models endpoint (placeholder for settings compatibility)
# ---------------------------------------------------------------------------

@app.get("/models", response_model=list[str])
def list_models_endpoint():
    """Return available model IDs. Currently returns a default set."""
    # This is a placeholder - you can expand this to return actual available models
    return [
        "meta-llama/Llama-3.2-1B-Instruct",
        "microsoft/DialoGPT-medium", 
        "facebook/blenderbot-400M-distill",
        "gpt2"
    ]


@app.get("/models/check")
def check_model_endpoint(name: str):
    """Check if a model is available."""
    # Simple placeholder implementation
    available_models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill", 
        "gpt2"
    ]
    
    return {
        "available": name in available_models,
        "gated": False  # For now, assume no models are gated
    }


@app.delete("/prompts/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prompt_endpoint(prompt_id: str):
    data = _load_prompts()
    new_prompts = [p for p in data["prompts"] if p["id"] != prompt_id]
    if len(new_prompts) == len(data["prompts"]):
        raise HTTPException(status_code=404, detail="Prompt not found")
    data["prompts"] = new_prompts
    # If the deleted prompt was selected, fall back to default
    if data.get("selected_id") == prompt_id:
        data["selected_id"] = "default"
    _save_prompts(data)
    return


# ---------------------------------------------------------------------------
# Debug utilities – *dangerous* endpoints for local experimentation only.
# ---------------------------------------------------------------------------

@app.post("/debug/clear_db", status_code=status.HTTP_204_NO_CONTENT)
def clear_database_endpoint():
    """Dangerous helper to wipe the metadata database for debugging purposes.

    This removes the on-disk SQLite file so it is recreated empty on the next
    request.  **Do not expose in production.**
    """
    try:
        db_path = Path("db") / "metadata.db"
        if db_path.exists():
            db_path.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/pdf_dir")
def get_pdf_dir():
    config_path = Path("data/pdf_dir.txt")
    if config_path.exists():
        pdf_dir = config_path.read_text(encoding="utf-8").strip()
        return {"pdf_dir": pdf_dir}
    else:
        raise HTTPException(status_code=404, detail="No pdf_dir configured")


@app.post("/pdf_dir")
def set_pdf_dir(payload: dict = Body(...)):
    pdf_dir = payload.get("pdf_dir")
    if not pdf_dir:
        raise HTTPException(status_code=400, detail="Missing pdf_dir")
    config_path = Path("data/pdf_dir.txt")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(pdf_dir, encoding="utf-8")
    return {"pdf_dir": pdf_dir}


@app.get("/papers/{filename}")
async def serve_paper(filename: str):
    """Serve a PDF file by filename. Only serves from the configured pdf_dir."""
    print(f"*** SERVE_PAPER FUNCTION CALLED WITH: {filename} ***")
    try:
        print(f"[PDF SERVE] Starting request for: {filename}")
        
        # Clean filename for security
        if '..' in filename or '/' in filename or '\\' in filename:
            print(f"[PDF SERVE] Invalid filename rejected: {filename}")
            raise HTTPException(status_code=400, detail="Invalid filename")
        if not filename.lower().endswith('.pdf'):
            print(f"[PDF SERVE] Non-PDF filename rejected: {filename}")
            raise HTTPException(status_code=400, detail="Only PDF files allowed")
        
        print(f"[PDF SERVE] Reading config file...")
        config_path = Path("data/pdf_dir.txt")
        if not config_path.exists():
            print(f"[PDF SERVE] Config file not found: {config_path}")
            raise HTTPException(status_code=404, detail="No pdf_dir configured")
        
        pdf_dir = config_path.read_text(encoding="utf-8").strip()
        print(f"[PDF SERVE] PDF directory from config: {pdf_dir}")
        
        file_path = Path(pdf_dir) / filename
        print(f"[PDF SERVE] Checking: {file_path}")
        
        if file_path.exists() and file_path.is_file():
            print(f"[PDF SERVE] File exists, checking if it's a PDF...")
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    print(f"  ✗ File exists but is not a valid PDF: {file_path}")
                    raise HTTPException(status_code=415, detail="File is not a valid PDF")
            
            print(f"[PDF SERVE] File is valid PDF, creating FileResponse...")
            return FileResponse(
                path=str(file_path),
                media_type="application/pdf",
                filename=filename,
                headers={"Cache-Control": "public, max-age=3600"}
            )
        else:
            print(f"  ✗ Not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"PDF file '{filename}' not found in configured directory '{pdf_dir}'")
    except HTTPException:
        print(f"[PDF SERVE] HTTPException raised for {filename}")
        raise
    except Exception as exc:
        print(f"Error serving {filename}: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Debug: Print all registered routes on startup
if __name__ == "__main__":
    print("=== REGISTERED ROUTES ===")
    for route in app.routes:
        print(f"Route: {route}")
        if hasattr(route, 'path'):
            print(f"  Path: {route.path}")
        if hasattr(route, 'methods'):
            print(f"  Methods: {route.methods}")
    print("=========================") 