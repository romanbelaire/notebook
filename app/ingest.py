import os
import glob
import pickle
from pathlib import Path
from typing import List

import numpy as np
import PyPDF2
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            txt = page.extract_text() or ""
            text += txt + "\n"
    return text


def chunk_text(text: str) -> List[str]:
    """Split long text into overlapping chunks suitable for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def build_faiss_index(embeddings: List[List[float]]):
    """Create a FAISS index from the provided embeddings."""
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


def ingest_pdfs(pdf_dir: str = "data/papers", db_dir: str = "db") -> None:
    """Ingest all PDFs in `pdf_dir` into a vector store under `db_dir`.

    If *pdf_dir* does not exist, it is created automatically so users can simply
    drop files into the default location without manual setup.
    """

    # Ensure the repository directory exists so the user can add PDFs later.
    Path(pdf_dir).mkdir(parents=True, exist_ok=True)

    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'.")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors: List[List[float]] = []
    metadatas: List[dict] = []
    chunks: List[str] = []

    for pdf in pdf_files:
        raw_text = extract_text_from_pdf(pdf)
        chunked = chunk_text(raw_text)
        chunks.extend(chunked)
        for chunk in chunked:
            emb = model.encode(chunk)
            vectors.append(emb)
            metadatas.append({"source": Path(pdf).name})

    index = build_faiss_index(vectors)

    os.makedirs(db_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(db_dir, "index.faiss"))
    with open(os.path.join(db_dir, "docs.pkl"), "wb") as f:
        pickle.dump({"texts": chunks, "metadatas": metadatas}, f)

    print(
        f"Ingested {len(pdf_files)} PDFs with {len(chunks)} text chunks into '{db_dir}'."
    )


if __name__ == "__main__":
    ingest_pdfs() 