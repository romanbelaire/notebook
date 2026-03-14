"""Persist conversation graphs and snapshots. SQLite in db/constellar.db."""

import json
import os
import sqlite3
import time
import uuid

from app.constellar.models import ActiveState, ConversationGraph, Shard, Snapshot


DB_DIR = "db"
DB_PATH = os.path.join(DB_DIR, "constellar.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    _ensure_tables(c)
    return c


def _ensure_tables(c: sqlite3.Connection) -> None:
    c.executescript("""
        CREATE TABLE IF NOT EXISTS graphs (
            id TEXT PRIMARY KEY,
            root_id TEXT NOT NULL,
            current_leaf_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shards (
            id TEXT NOT NULL,
            graph_id TEXT NOT NULL,
            parent_ids TEXT NOT NULL,
            created_at REAL NOT NULL,
            visible INTEGER NOT NULL,
            metadata TEXT,
            user_content TEXT,
            assistant_content TEXT,
            contexts TEXT,
            citations TEXT,
            content TEXT,
            role TEXT,
            PRIMARY KEY (id, graph_id),
            FOREIGN KEY (graph_id) REFERENCES graphs(id)
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY,
            graph_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            compiled_shard_ids TEXT NOT NULL,
            serialized_prompt TEXT NOT NULL,
            model TEXT NOT NULL,
            temperature REAL NOT NULL,
            response TEXT NOT NULL
        );
    """)
    c.commit()


def _shard_from_row(row: sqlite3.Row) -> Shard:
    return Shard(
        id=row["id"],
        parent_ids=json.loads(row["parent_ids"]),
        created_at=row["created_at"],
        visible=bool(row["visible"]),
        metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        user_content=row["user_content"],
        assistant_content=row["assistant_content"],
        contexts=json.loads(row["contexts"]) if row["contexts"] else [],
        citations=json.loads(row["citations"]) if row["citations"] else [],
        content=row["content"],
        role=row["role"],
    )


def create_graph() -> tuple[str, str]:
    """Create a new graph with an empty root turn shard. Returns (graph_id, root_id)."""
    graph_id = uuid.uuid4().hex
    root_id = uuid.uuid4().hex
    now = time.time()
    root = Shard(
        id=root_id,
        parent_ids=[],
        created_at=now,
        visible=True,
        user_content="",
        assistant_content="",
    )
    graph = ConversationGraph(shards={root_id: root}, root_id=root_id)
    state = ActiveState(current_leaf_id=root_id)
    save_graph(graph_id, graph, state)
    return graph_id, root_id


def load_graph(graph_id: str) -> tuple[ConversationGraph, ActiveState]:
    """Load graph and active state. Raises KeyError if graph_id not found."""
    conn = _conn()
    try:
        row = conn.execute("SELECT root_id, current_leaf_id FROM graphs WHERE id = ?", (graph_id,)).fetchone()
        if row is None:
            raise KeyError(f"Graph {graph_id} not found")
        root_id = row["root_id"]
        current_leaf_id = row["current_leaf_id"]
        shards = {}
        for r in conn.execute("SELECT * FROM shards WHERE graph_id = ?", (graph_id,)):
            s = _shard_from_row(r)
            shards[s.id] = s
        graph = ConversationGraph(shards=shards, root_id=root_id)
        state = ActiveState(current_leaf_id=current_leaf_id)
        return graph, state
    finally:
        conn.close()


def save_graph(graph_id: str, graph: ConversationGraph, state: ActiveState) -> None:
    """Upsert graph and update current_leaf_id. Overwrites all shards for this graph."""
    conn = _conn()
    try:
        now = time.time()
        conn.execute(
            """INSERT INTO graphs (id, root_id, current_leaf_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET root_id=excluded.root_id, current_leaf_id=excluded.current_leaf_id, updated_at=excluded.updated_at""",
            (graph_id, graph.root_id, state.current_leaf_id, now, now),
        )
        conn.execute("DELETE FROM shards WHERE graph_id = ?", (graph_id,))
        for s in graph.shards.values():
            conn.execute(
                """INSERT INTO shards (id, graph_id, parent_ids, created_at, visible, metadata, user_content, assistant_content, contexts, citations, content, role)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    s.id,
                    graph_id,
                    json.dumps(s.parent_ids),
                    s.created_at,
                    1 if s.visible else 0,
                    json.dumps(s.metadata) if s.metadata else None,
                    s.user_content,
                    s.assistant_content,
                    json.dumps(s.contexts),
                    json.dumps(s.citations),
                    s.content,
                    s.role,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def list_graphs() -> list[str]:
    """Return graph ids, newest first."""
    conn = _conn()
    try:
        rows = conn.execute("SELECT id FROM graphs ORDER BY updated_at DESC").fetchall()
        return [r["id"] for r in rows]
    finally:
        conn.close()


def append_snapshot(
    graph_id: str,
    compiled_shard_ids: list[str],
    serialized_prompt: str,
    model: str,
    temperature: float,
    response: str,
) -> None:
    snapshot_id = uuid.uuid4().hex
    conn = _conn()
    try:
        conn.execute(
            """INSERT INTO snapshots (id, graph_id, timestamp, compiled_shard_ids, serialized_prompt, model, temperature, response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (snapshot_id, graph_id, time.time(), json.dumps(compiled_shard_ids), serialized_prompt, model, temperature, response),
        )
        conn.commit()
    finally:
        conn.close()
