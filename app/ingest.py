import os
import glob
import pickle
import hashlib
import base64
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import fitz  # PyMuPDF
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .metadata_db import get_connection, upsert_paper, replace_chunks, upsert_paper_embedding, check_paper_by_sha256
from .task_manager import set_progress
import threading

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
    if not embeddings:
        # Create an empty index with default dimension
        dim = 384  # Default dimension for all-MiniLM-L6-v2
        index = faiss.IndexFlatL2(dim)
        return index
    
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


def extract_pdf_from_firefox_document(file_path: str) -> Union[bytes, None]:
    """Extract PDF content from Firefox PDF document (HTML wrapper)."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        text = content.decode('utf-8', errors='ignore')
        
        # Debug: Check what we're looking at
        print(f"[DEBUG] Checking file: {file_path}")
        print(f"[DEBUG] First 100 chars: {text[:100]}")
        
        # Check if it's a Firefox PDF document
        if '<!DOCTYPE html' not in text or 'pdf.js' not in text:
            print(f"[DEBUG] Not a Firefox PDF document: {file_path}")
            return None  # Not a Firefox PDF document
        
        print(f"[DEBUG] Detected Firefox PDF document: {file_path}")
        
        # Look for the PDF data URL in the HTML
        match = re.search(r'data:application/pdf;base64,([A-Za-z0-9+/=]+)', text)
        if not match:
            print(f'Firefox PDF document detected but no PDF data found: {file_path}')
            # Try alternative patterns
            alt_match = re.search(r'data:application/pdf[^,]*,([A-Za-z0-9+/=\s]+)', text)
            if alt_match:
                print(f'[DEBUG] Found alternative PDF data pattern: {file_path}')
                base64_data = alt_match.group(1).replace(' ', '').replace('\n', '')
                pdf_content = base64.b64decode(base64_data)
                return pdf_content
            return None
        
        print(f"[DEBUG] Found PDF data, extracting: {file_path}")
        # Decode the base64 PDF data
        base64_data = match.group(1)
        pdf_content = base64.b64decode(base64_data)
        
        return pdf_content
    except Exception as e:
        print(f'Error extracting PDF from Firefox document {file_path}: {e}')
        return None


def ingest_pdfs_with_progress(pdf_dir: str = "data/papers", db_dir: str = "db", task_id: str = None) -> dict:
    """Ingest PDFs with progress tracking. Returns stats dict (ingested_new, skipped_duplicates, failed)."""
    try:
        # Count total PDFs first
        Path(pdf_dir).mkdir(parents=True, exist_ok=True)
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        
        if not pdf_files:
            if task_id:
                set_progress(task_id, 0, 0, "No PDF files found")
            raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'.")
        
        total_files = len(pdf_files)
        if task_id:
            set_progress(task_id, 0, total_files, f"Starting ingestion of {total_files} PDFs")
        
        # Call original function with progress tracking
        return ingest_pdfs_internal(pdf_dir, db_dir, task_id)
        
    except Exception as e:
        if task_id:
            set_progress(task_id, 0, 0, f"Ingestion failed: {str(e)}")
        print(f"Ingestion error in task {task_id}: {e}")
        raise  # Re-raise so the task manager can capture it


def ingest_pdfs_internal(pdf_dir: str = "data/papers", db_dir: str = "db", task_id: str = None) -> Dict[str, Any]:
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

    total_files = len(pdf_files)
    processed_files = 0
    actually_processed = 0
    skipped_duplicates = 0
    failed_ingest: List[str] = []

    for pdf in pdf_files:
        # Check if this is a Firefox PDF document and fix it first
        extracted_pdf = extract_pdf_from_firefox_document(pdf)
        if extracted_pdf:
            print(f"Found Firefox PDF document, extracting actual PDF content: {pdf}")
            try:
                # Replace the HTML file with the extracted PDF content
                with open(pdf, 'wb') as f:
                    f.write(extracted_pdf)
                print(f"Successfully extracted and replaced Firefox PDF document: {pdf}")
            except Exception as e:
                print(f"Error replacing Firefox PDF document {pdf}: {e}")
                failed_ingest.append(f"{Path(pdf).name} — Firefox extract: {e}")
                processed_files += 1
                if task_id:
                    set_progress(task_id, processed_files, total_files, f"Error processing {processed_files}/{total_files} PDFs")
                continue
        
        # ------------------------  SHA256 duplicate check  ------------------------
        # Calculate SHA256 hash of the PDF file
        with open(pdf, 'rb') as f:
            pdf_content = f.read()
            sha256_hash = hashlib.sha256(pdf_content).hexdigest()
        
        # Check if a paper with this hash already exists
        try:
            existing_paper = check_paper_by_sha256(conn, sha256_hash)
            if existing_paper:
                print(f"Skipping {pdf} - already ingested (SHA256: {sha256_hash[:8]}...)")
                skipped_duplicates += 1
                processed_files += 1
                if task_id:
                    set_progress(task_id, processed_files, total_files, f"Skipped duplicate {processed_files}/{total_files} PDFs")
                continue
        except Exception as e:
            print(f"Error checking duplicate for {pdf}: {e}")
            # Continue processing this file if duplicate check fails
        
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

            # Extract simple TOC mapping page -> heading using PyMuPDF 1-based page numbers
            toc_entries = doc.get_toc(simple=True)  # [[lvl, title, page], ...]
            page_to_heading: dict[int, str] = {}
            for lvl, title, page_num in toc_entries:
                # Only take level 1 or 2 headings to avoid noisy subsubsections
                if lvl <= 2:
                    page_to_heading.setdefault(page_num, title.strip())
            # Build cumulative mapping so each page inherits nearest previous heading
            current_heading = None
            heading_by_page: dict[int, str] = {}
            for pnum in range(1, doc.page_count + 1):
                if pnum in page_to_heading:
                    current_heading = page_to_heading[pnum]
                if current_heading:
                    heading_by_page[pnum] = current_heading
        except Exception:
            doc = None  # type: ignore[assignment]
            title_meta = ""
            author_meta = ""
            year_meta = None
            heading_by_page = {}

        inferred_title = _infer_title_from_first_page(pdf)
        if inferred_title:
            paper_title = inferred_title
        elif title_meta:
            paper_title = title_meta
        else:
            paper_title = Path(pdf).stem.replace("_", " ")
        authors_field = author_meta if author_meta else None

        # -----------------------------  Chunk per page  -----------------------------
        for page_idx in range(doc.page_count if doc else 0):
            try:
                page = doc.load_page(page_idx)  # type: ignore[union-attr]
                page_text = page.get_text("text")
            except Exception:
                continue

            # Optionally prepend page heading to bias retrieval
            heading = heading_by_page.get(page_idx + 1)
            if heading:
                page_text = f"{heading}\n{page_text}"

            for chunk in chunk_text(page_text):
                if len(chunk.strip()) == 0:
                    continue
                digit_ratio = sum(c.isdigit() for c in chunk) / len(chunk)
                if digit_ratio > NUMERIC_RATIO_THRESHOLD:
                    continue

                emb = model.encode(chunk)
                vectors.append(emb)
                metadatas.append({
                    "source": Path(pdf).name,
                    "title": paper_title,
                    "authors": authors_field,
                    "year": year_meta,
                    "section": heading,
                    "page": page_idx + 1,
                })
                chunks.append(chunk)

        # -----------------------  Whole-paper embedding  -----------------------
        # Get all chunks for the current PDF
        current_pdf_chunks = [ch for ch, meta in zip(chunks, metadatas) if meta["source"] == Path(pdf).name]
        raw_text_combined = "\n".join(current_pdf_chunks) if current_pdf_chunks else ""
        cleaned_text = _clean_text_for_embedding(raw_text_combined)
        if cleaned_text:
            paper_vector = model.encode(cleaned_text)
        else:
            paper_vector = model.encode(paper_title)

        # Record metadata
        try:
            paper_id = upsert_paper(conn, Path(pdf).name, title=paper_title, authors=authors_field, year=year_meta, sha256=sha256_hash)
            # store only text chunks belonging to current PDF (already calculated above)
            replace_chunks(conn, paper_id, current_pdf_chunks)

            # Store paper-level embedding
            upsert_paper_embedding(conn, paper_id, paper_vector)
        except Exception as e:
            print(f"Error storing metadata for {pdf}: {e}")
            failed_ingest.append(f"{Path(pdf).name} — metadata: {e}")
            # Continue to next file if metadata storage fails
            processed_files += 1
            if task_id:
                set_progress(task_id, processed_files, total_files, f"Error processing {processed_files}/{total_files} PDFs")
            continue

        # Update progress
        processed_files += 1
        actually_processed += 1
        if task_id:
            set_progress(task_id, processed_files, total_files, f"Processed {actually_processed}/{total_files} PDFs")

    # Build index only if we have vectors
    if vectors:
        index = build_faiss_index(vectors)
        os.makedirs(db_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(db_dir, "index.faiss"))
        with open(os.path.join(db_dir, "docs.pkl"), "wb") as f:
            pickle.dump({"texts": chunks, "metadatas": metadatas}, f)
    else:
        # Create empty index and data files for consistency
        os.makedirs(db_dir, exist_ok=True)
        empty_index = build_faiss_index([])
        faiss.write_index(empty_index, os.path.join(db_dir, "index.faiss"))
        with open(os.path.join(db_dir, "docs.pkl"), "wb") as f:
            pickle.dump({"texts": [], "metadatas": []}, f)

    conn.close()

    # Final progress update
    if task_id:
        if actually_processed == 0:
            set_progress(task_id, total_files, total_files, f"All {total_files} PDFs were duplicates - skipped")
        else:
            set_progress(task_id, total_files, total_files, f"Ingestion complete! Processed {actually_processed} PDFs with {len(chunks)} text chunks")

    print(
        f"Ingested {actually_processed} new PDFs (skipped {len(pdf_files) - actually_processed} duplicates) with {len(chunks)} text chunks into '{db_dir}'."
    )

    return {
        "kind": "directory_ingest",
        "ingested_new": actually_processed,
        "duplicates_removed": skipped_duplicates,
        "failed_count": len(failed_ingest),
        "failed": failed_ingest,
    }


def ingest_pdfs(pdf_dir: str = "data/papers", db_dir: str = "db") -> None:
    """Backward compatibility wrapper for ingest_pdfs_internal."""
    ingest_pdfs_internal(pdf_dir, db_dir, task_id=None)


if __name__ == "__main__":
    ingest_pdfs() 