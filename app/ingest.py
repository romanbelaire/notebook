import os
import glob
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import re
import fitz  # PyMuPDF
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .metadata_db import get_connection, upsert_paper, replace_chunks, upsert_paper_embedding

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NUMERIC_RATIO_THRESHOLD = 0.4  # skip chunks with >40% digits

# Basic stopword list (can be expanded or replaced by an NLP pipeline).
STOPWORDS = {
    "the","a","an","and","or","of","to","in","for","on","with","as","by","is","are","was","were","be","been","this","that","these","those","at","from","but","into","up","out","over","after","before","between","about","because","so","than","too","very","can","cannot","could","might","may","must","shall","should","will","would","also","such","not","no","nor","do","does","did","done","if","then","else","when","while","where","which","who","whom","whose","why","how"
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract clean body text from a PDF using PyMuPDF.

    Heuristics:
    • drop header/footer blocks (top & bottom 5% of page height)
    • stop reading after we encounter a References/Bibliography heading
    """
    doc = fitz.open(pdf_path)
    collected: list[str] = []
    refs_started = False

    for page in doc:
        page_height = page.rect.height
        blocks = page.get_text("blocks")  # list of tuples
        # sort blocks top→bottom, left→right
        blocks.sort(key=lambda b: (b[1], b[0]))
        for (x0, y0, x1, y1, txt, *_rest) in blocks:
            if refs_started:
                continue
            # Skip headers/footers
            if y0 < 0.05 * page_height or y1 > 0.95 * page_height:
                continue
            line = txt.strip()
            if not line:
                continue
            if re.match(r"^references?\b", line, re.I) or re.match(r"^bibliography\b", line, re.I):
                refs_started = True
                continue
            collected.append(line)

    return "\n".join(collected)


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


def _infer_title_from_first_page(pdf_path: str) -> Optional[str]:
    """Return a best-guess title for *pdf_path* by inspecting the first page.

    Heuristic:
    1. Load the first page and extract text spans with font size information via
       PyMuPDF's ``get_text("dict")`` representation.
    2. Compute the *max* font size on the page.  Title candidates are lines
       whose average font size is within 0.5 pt of this maximum.
    3. From these candidates, pick the first one where both the preceding **and**
       subsequent line (if any) have a noticeably smaller font size (≥0.5 pt
       difference).  This approximates the lone headline pattern typical for
       academic article titles.
    4. Fallbacks: first candidate → ``None`` if no lines exist.
    """

    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            return None
        page = doc.load_page(0)
        page_width = page.rect.width
        LEFT_MARGIN_CUTOFF = 0.10 * page_width  # ignore anything left of 10% of width
        RIGHT_MARGIN_CUTOFF = 0.90 * page_width  # ignore anything right of 90%
        text_dict = page.get_text("dict")

        # Flatten into ordered line list with avg font size.
        lines: list[dict] = []
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])

                # Skip if majority of spans are not horizontal (e.g., the vertical arXiv side bar)
                horiz_spans = 0
                for sp in spans:
                    dx, dy = sp.get("dir", (1.0, 0.0))  # default to horizontal if missing
                    if abs(dx) >= abs(dy):
                        horiz_spans += 1
                if horiz_spans < len(spans) / 2:
                    continue  # mostly vertical

                # Bounding box filter – skip lines outside printable column
                x0, _y0, x1, _y1 = line.get("bbox", [0, 0, 0, 0])
                if x1 < LEFT_MARGIN_CUTOFF or x0 > RIGHT_MARGIN_CUTOFF:
                    continue

                line_text_parts: list[str] = [sp.get("text", "").strip() for sp in spans]
                line_text = " ".join(part for part in line_text_parts if part)
                if not line_text:
                    continue

                sizes = [sp.get("size", 0.0) for sp in spans]
                if not sizes:
                    continue
                avg_size = sum(sizes) / len(sizes)
                lines.append({"text": line_text, "size": avg_size})

        if not lines:
            return None

        max_size = max(l["size"] for l in lines)
        # Tolerance at 0.5pt to accommodate minor rounding differences.
        candidate_indices = [i for i, l in enumerate(lines) if l["size"] >= max_size - 0.5]
        if not candidate_indices:
            return None

        for idx in candidate_indices:
            prev_smaller = idx == 0 or lines[idx - 1]["size"] < lines[idx]["size"] - 0.5
            next_smaller = idx == len(lines) - 1 or lines[idx + 1]["size"] < lines[idx]["size"] - 0.5
            if prev_smaller and next_smaller:
                return lines[idx]["text"].strip()

        # No isolated headline found – return first max-size line.
        return lines[candidate_indices[0]]["text"].strip()
    except Exception:
        return None


def _clean_text_for_embedding(text: str) -> str:
    """Return a simplified, stop-word free version of *text* suitable for whole-paper embedding."""
    # Remove numbers and punctuation, lower case
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered)


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

    # SQLite connection for metadata tracking
    conn = get_connection(db_dir=db_dir)

    for pdf in pdf_files:
        raw_text = extract_text_from_pdf(pdf)

        # ------------------------  PDF metadata  ------------------------
        try:
            doc = fitz.open(pdf)
            meta = doc.metadata or {}
            title_meta = (meta.get("title") or "").strip()
            author_meta = (meta.get("author") or "").strip()
            year_meta = meta.get("creationDate")
            if year_meta and year_meta.startswith("D:"):
                year_meta = year_meta[2:6]
            elif year_meta and len(year_meta) >= 4:
                year_meta = year_meta[:4]
            else:
                year_meta = None
        except Exception:
            title_meta = ""
            author_meta = ""
            year_meta = None

        inferred_title = _infer_title_from_first_page(pdf)
        if inferred_title:
            paper_title = inferred_title
        elif title_meta:
            paper_title = title_meta
        else:
            paper_title = Path(pdf).stem.replace("_", " ")
        authors_field = author_meta if author_meta else None

        # Combine title with body so title words appear in at least one chunk
        combined_text = f"{paper_title}\n{raw_text}" if paper_title else raw_text

        chunked_raw = chunk_text(combined_text)
        chunked = []
        for ch in chunked_raw:
            if len(ch.strip()) == 0:
                continue
            digit_ratio = sum(c.isdigit() for c in ch) / len(ch)
            if digit_ratio > NUMERIC_RATIO_THRESHOLD:
                continue  # skip numeric-heavy chunks
            chunked.append(ch)

        # -----------------------  Whole-paper embedding  -----------------------
        cleaned_text = _clean_text_for_embedding(raw_text)
        paper_vector = model.encode(cleaned_text)

        # ----------------------  Per-chunk embeddings  ------------------------
        for chunk in chunked:
            emb = model.encode(chunk)
            vectors.append(emb)
            metadatas.append({
                "source": Path(pdf).name,
                "title": paper_title,
                "authors": authors_field,
                "year": year_meta,
            })
            chunks.append(chunk)

        # Record metadata
        paper_id = upsert_paper(conn, Path(pdf).name, title=paper_title, authors=authors_field, year=year_meta)
        replace_chunks(conn, paper_id, chunked)

        # Store paper-level embedding
        upsert_paper_embedding(conn, paper_id, paper_vector)

    index = build_faiss_index(vectors)

    os.makedirs(db_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(db_dir, "index.faiss"))
    with open(os.path.join(db_dir, "docs.pkl"), "wb") as f:
        pickle.dump({"texts": chunks, "metadatas": metadatas}, f)

    conn.close()

    print(
        f"Ingested {len(pdf_files)} PDFs with {len(chunks)} text chunks into '{db_dir}'."
    )


if __name__ == "__main__":
    ingest_pdfs() 