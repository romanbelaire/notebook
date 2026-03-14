"""Persistent store for pinned shards with FAISS for semantic search.

- Pin/unpin and list are synchronous (metadata only).
- Unpin marks a shard as unpinned; actual removal and index rebuild happen on cleanup (app close).
- FAISS indexing runs in a background task so the request returns immediately.
"""

import os
import pickle
import threading
import atexit
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CONTEXTS = 2


def _text_for_embedding(text: str, contexts: List[str]) -> str:
    """Combine text and contexts into a single string for embedding."""
    parts = [text.strip()]
    for ctx in contexts[:MAX_CONTEXTS]:
        if ctx and ctx.strip():
            parts.append(ctx.strip())
    return " ".join(parts)


def _migrate_shard(d: Dict[str, Any]) -> None:
    """Ensure existing records have pinned flag (default True) and notes (default [])."""
    if "pinned" not in d:
        d["pinned"] = True
    if "notes" not in d:
        d["notes"] = []


class ShardStore:
    """Persistent store for pinned shards. FAISS indexing is async; unpin is soft until cleanup."""

    def __init__(self, db_dir: str = "db"):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self._index_path = os.path.join(self.db_dir, "shards_index.faiss")
        self._meta_path = os.path.join(self.db_dir, "shards.pkl")
        self._lock = threading.RLock()
        self._embedding_model: Any = None
        self._index: Optional[Any] = None
        self._shards: List[Dict[str, Any]] = []
        self._load()

    def _get_embedding_model(self):
        """Lazy-load so sync paths (pin/unpin/list) stay fast."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embedding_model

    def _save_meta(self) -> None:
        """Write only metadata (pickle). Does not touch FAISS."""
        with self._lock:
            with open(self._meta_path, "wb") as f:
                pickle.dump(self._shards, f)

    def _save_index(self) -> None:
        """Write only FAISS index to disk. Call with _lock held by caller if needed."""
        if self._index is not None:
            faiss.write_index(self._index, self._index_path)
        elif os.path.exists(self._index_path):
            os.remove(self._index_path)

    def _save(self) -> None:
        """Write both metadata and index."""
        with self._lock:
            self._save_index()
            with open(self._meta_path, "wb") as f:
                pickle.dump(self._shards, f)

    def _load(self) -> None:
        with self._lock:
            if os.path.exists(self._meta_path):
                with open(self._meta_path, "rb") as f:
                    self._shards = pickle.load(f)
                for d in self._shards:
                    _migrate_shard(d)
            else:
                self._shards = []
            if os.path.exists(self._index_path) and self._shards:
                self._index = faiss.read_index(self._index_path)
            else:
                self._index = None

    def _pinned_list(self) -> List[Dict[str, Any]]:
        """Return list of pinned shards in stable order (same order used for index)."""
        return [d for d in self._shards if d.get("pinned", True)]

    def _rebuild_index_sync(self) -> None:
        """Encode all pinned shards and rebuild FAISS index. Call with _lock held."""
        pinned = self._pinned_list()
        if not pinned:
            self._index = None
            return
        model = self._get_embedding_model()
        embs = np.array([
            model.encode(_text_for_embedding(d["text"], d.get("contexts", [])))
            for d in pinned
        ]).astype("float32")
        dim = embs.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embs)

    def _rebuild_index_background(self) -> None:
        """Run in background: rebuild FAISS from pinned shards and save index to disk."""
        with self._lock:
            pinned = self._pinned_list()
            if not pinned:
                self._index = None
                self._save_index()
                return
        model = self._get_embedding_model()
        with self._lock:
            pinned = self._pinned_list()
        embs = np.array([
            model.encode(_text_for_embedding(d["text"], d.get("contexts", [])))
            for d in pinned
        ]).astype("float32")
        dim = embs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embs)
        with self._lock:
            self._index = index
            self._save_index()

    def upsert_shard(
        self,
        shard_id: str,
        text: str,
        contexts: List[str],
        *,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        notes: Optional[List[str]] = None,
    ) -> str:
        """Add or update a pinned shard. Saves metadata immediately; FAISS index rebuild is async."""
        if not text.strip():
            raise ValueError("Cannot save an empty shard text.")
        title = (title or text[:50]).strip()[:100]
        contexts = list(contexts[:MAX_CONTEXTS])
        notes_list = list(notes) if notes is not None else []
        now = datetime.utcnow().isoformat()

        with self._lock:
            idx = next((i for i, d in enumerate(self._shards) if d["id"] == shard_id), None)
            if idx is not None:
                self._shards[idx].update({
                    "text": text.strip(),
                    "contexts": contexts,
                    "title": title,
                    "conversation_id": conversation_id,
                    "parent_id": parent_id,
                    "pinned": True,
                    "notes": notes_list,
                })
            else:
                self._shards.append({
                    "id": shard_id,
                    "text": text.strip(),
                    "contexts": contexts,
                    "title": title,
                    "created_at": now,
                    "conversation_id": conversation_id,
                    "parent_id": parent_id,
                    "pinned": True,
                    "notes": notes_list,
                })
            self._save_meta()

        try:
            from app.task_manager import submit as submit_task
            submit_task(self._rebuild_index_background)
        except Exception as e:
            print(f"ShardStore: failed to enqueue index rebuild: {e}")
        return shard_id

    def delete_shard(self, shard_id: str) -> None:
        """Unpin: mark shard as unpinned. Actual removal happens on cleanup (app close)."""
        with self._lock:
            idx = next((i for i, d in enumerate(self._shards) if d["id"] == shard_id), None)
            if idx is None:
                raise KeyError(f"Shard id '{shard_id}' not found.")
            self._shards[idx]["pinned"] = False
            self._save_meta()

    def cleanup_unpinned(self) -> None:
        """Remove all unpinned shards and rebuild FAISS index. Call on application close."""
        with self._lock:
            self._shards = [d for d in self._shards if d.get("pinned", True)]
            self._rebuild_index_sync()
            self._save()

    def get_shard(self, shard_id: str) -> Optional[Dict[str, Any]]:
        """Return one shard by id, or None if not found or unpinned."""
        with self._lock:
            d = next((d for d in self._shards if d["id"] == shard_id), None)
            if d is None or not d.get("pinned", True):
                return None
            return dict(d)

    def update_shard(
        self,
        shard_id: str,
        *,
        text: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        title: Optional[str] = None,
        notes: Optional[List[str]] = None,
    ) -> None:
        """Update shard metadata. Re-indexing is async."""
        with self._lock:
            idx = next((i for i, d in enumerate(self._shards) if d["id"] == shard_id), None)
            if idx is None:
                raise KeyError(f"Shard id '{shard_id}' not found.")
            d = self._shards[idx]
            if not d.get("pinned", True):
                raise KeyError(f"Shard id '{shard_id}' is unpinned.")
            if text is not None and text.strip():
                d["text"] = text.strip()
            if contexts is not None:
                d["contexts"] = contexts[:MAX_CONTEXTS]
            if title is not None and title.strip():
                d["title"] = title.strip()[:100]
            if notes is not None:
                d["notes"] = list(notes)
            self._save_meta()
        try:
            from app.task_manager import submit as submit_task
            submit_task(self._rebuild_index_background)
        except Exception as e:
            print(f"ShardStore: failed to enqueue index rebuild: {e}")

    def list_all(self, pinned_only: bool = True) -> List[Dict[str, Any]]:
        """Return pinned shards only (unpinned are hidden until cleanup)."""
        with self._lock:
            if pinned_only:
                return [dict(d) for d in self._shards if d.get("pinned", True)]
            return [dict(d) for d in self._shards]

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Return up to k pinned shards most relevant to query. Filters to pinned only."""
        with self._lock:
            pinned = self._pinned_list()
            if self._index is None or not pinned:
                return []
            ntotal = self._index.ntotal
            if ntotal == 0:
                return []
            q_emb = self._get_embedding_model().encode(query)
            k_actual = min(k, ntotal)
            D, I = self._index.search(np.array([q_emb]).astype("float32"), k_actual)
        results: List[Tuple[Dict[str, Any], float]] = []
        for faiss_idx, dist in zip(I[0], D[0]):
            if faiss_idx == -1:
                continue
            if faiss_idx < len(pinned):
                results.append((dict(pinned[faiss_idx]), float(dist)))
        return results


def _cleanup_on_process_exit() -> None:
    """Run cleanup on a fresh store so unpinned shards are removed and index is rebuilt."""
    try:
        ShardStore().cleanup_unpinned()
    except Exception as e:
        print(f"ShardStore cleanup on exit: {e}")


atexit.register(_cleanup_on_process_exit)
