import os
import sqlite3
from datetime import datetime
from typing import List, Optional
import numpy as np

DB_FILENAME = "metadata.db"


def get_connection(db_dir: str = "db") -> sqlite3.Connection:
    """Return a SQLite connection creating the DB & tables if required."""
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, DB_FILENAME)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            title TEXT,
            authors TEXT,
            year TEXT,
            added_at TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER,
            chunk_index INTEGER,
            text TEXT,
            FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()

    # ------------------------------------------------------------------
    # Lightweight migration: add missing columns if DB pre-exists.
    # ------------------------------------------------------------------
    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(papers);")}
    for col_name in ["authors", "year"]:
        if col_name not in existing_cols:
            cur.execute(f"ALTER TABLE papers ADD COLUMN {col_name} TEXT;")

    # ------------------------------------------------------------------
    # New tables for collections and tagging (lightweight migration).
    # ------------------------------------------------------------------

    # Create simple lookup tables if they do not yet exist.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            created_at TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS collection_papers (
            collection_id INTEGER,
            paper_id INTEGER,
            PRIMARY KEY (collection_id, paper_id),
            FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
            FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_tags (
            paper_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (paper_id, tag_id),
            FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        );
        """
    )

    # Table for per-paper semantic vectors (stored as raw bytes)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_embeddings (
            paper_id INTEGER PRIMARY KEY,
            vector BLOB,
            dim INTEGER,
            FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
        );
        """
    )

    # Ensure migrations persist
    conn.commit()


def upsert_paper(
    conn: sqlite3.Connection,
    filename: str,
    *,
    title: Optional[str] = None,
    authors: Optional[str] = None,
    year: Optional[str] = None,
) -> int:
    """Insert the paper if new, returning its id."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM papers WHERE filename = ?", (filename,))
    row = cur.fetchone()
    if row:
        paper_id = row["id"]
    else:
        cur.execute(
            "INSERT INTO papers (filename, title, authors, year, added_at) VALUES (?,?,?,?,?)",
            (
                filename,
                title or filename,
                authors,
                year,
                datetime.utcnow().isoformat(),
            ),
        )
        paper_id = cur.lastrowid
        conn.commit()

    # If new metadata arrives later, update row.
    cur.execute("SELECT title, authors, year FROM papers WHERE id = ?", (paper_id,))
    current = cur.fetchone()
    updated_vals = {
        "title": title or current["title"],
        "authors": authors or current["authors"],
        "year": year or current["year"],
    }
    cur.execute(
        "UPDATE papers SET title = :title, authors = :authors, year = :year WHERE id = :pid",
        {**updated_vals, "pid": paper_id},
    )
    conn.commit()
    return paper_id


def replace_chunks(conn: sqlite3.Connection, paper_id: int, chunks: List[str]) -> None:
    """Delete previous chunks for paper and insert fresh ones."""
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
    cur.executemany(
        "INSERT INTO chunks (paper_id, chunk_index, text) VALUES (?,?,?)",
        [(paper_id, i, chunk) for i, chunk in enumerate(chunks)],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Convenience helpers for papers, collections & tagging
# ---------------------------------------------------------------------------


def list_papers(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT * FROM papers ORDER BY added_at DESC;")
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def create_collection(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO collections (name, created_at) VALUES (?,?)", (name, datetime.utcnow().isoformat()))
    conn.commit()
    cur.execute("SELECT id FROM collections WHERE name = ?", (name,))
    row = cur.fetchone()
    if row is None:
        raise ValueError("Failed to create or retrieve collection.")
    return row["id"]


def list_collections(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT * FROM collections ORDER BY created_at DESC;")
    return [dict(r) for r in cur.fetchall()]


def add_papers_to_collection(conn: sqlite3.Connection, collection_id: int, paper_ids: list[int]):
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO collection_papers (collection_id, paper_id) VALUES (?,?)",
        [(collection_id, pid) for pid in paper_ids],
    )
    conn.commit()


def list_papers_for_collection(conn: sqlite3.Connection, collection_id: int):
    """Return paper rows belonging to *collection_id*."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.* FROM collection_papers cp
        JOIN papers p ON p.id = cp.paper_id
        WHERE cp.collection_id = ? ORDER BY p.added_at DESC;
        """,
        (collection_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def get_filenames_for_collection(conn: sqlite3.Connection, collection_id: int):
    cur = conn.cursor()
    cur.execute(
        """SELECT p.filename FROM collection_papers cp JOIN papers p ON p.id = cp.paper_id WHERE cp.collection_id = ?;""",
        (collection_id,),
    )
    return {row[0] for row in cur.fetchall()}  # set of filenames


def upsert_paper_embedding(conn: sqlite3.Connection, paper_id: int, vector: "np.ndarray") -> None:  # type: ignore[name-defined]
    """Insert or update the semantic embedding for *paper_id*."""
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector, dtype="float32")

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO paper_embeddings (paper_id, vector, dim) VALUES (?,?,?) ON CONFLICT(paper_id) DO UPDATE SET vector=excluded.vector, dim=excluded.dim;",
        (paper_id, vector.tobytes(), vector.shape[0]),
    )
    conn.commit() 