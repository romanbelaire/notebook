import os
import pickle
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Re-use the same embedding model as the rest of the pipeline so everything lives
# in a single vector space.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class Insight:
    """Lightweight value object representing a pinned insight."""

    def __init__(self, text: str, contexts: List[str], title: str):
        self.id: str = str(uuid.uuid4())
        self.text: str = text.strip()
        # Short human-readable label
        self.title: str = title.strip()[:100]  # cap length
        # Store at most two contexts to keep the object small.
        self.contexts: List[str] = contexts[:2]
        self.created_at: str = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "contexts": self.contexts,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Insight":
        obj = Insight(data["text"], data.get("contexts", []), data.get("title", data["text"][:50]))
        obj.id = data["id"]
        obj.created_at = data.get("created_at", datetime.utcnow().isoformat())
        return obj


class InsightsStore:
    """Persistent local store for pinned insights backed by FAISS."""

    def __init__(self, db_dir: str = "db"):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)

        self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # File paths
        self._index_path = os.path.join(self.db_dir, "insights_index.faiss")
        self._meta_path = os.path.join(self.db_dir, "insights.pkl")

        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save(self) -> None:
        if self._index is not None:
            faiss.write_index(self._index, self._index_path)
        elif os.path.exists(self._index_path):
            os.remove(self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._insights, f)

    def _load(self) -> None:
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._index = faiss.read_index(self._index_path)
            with open(self._meta_path, "rb") as f:
                self._insights: List[Dict[str, Any]] = pickle.load(f)
        else:
            # Lazy init empty structures
            self._index = None
            self._insights = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_insight(self, text: str, contexts: List[str], *, title: Optional[str] = None) -> str:
        """Add *text* (with accompanying *contexts*) as an insight and return its id."""
        if not text.strip():
            raise ValueError("Cannot save an empty insight text.")

        if title is None:
            title = text[:50]
        insight = Insight(text, contexts, title)
        emb = self._embedding_model.encode(insight.text)

        # Initialise the index lazily when the first insight is added so we
        # know the embedding dimension.
        if self._index is None:
            dim = len(emb)
            self._index = faiss.IndexFlatL2(dim)

        # Add to index and metadata list
        self._index.add(np.array([emb]).astype("float32"))
        self._insights.append(insight.to_dict())

        self._save()
        return insight.id

    def delete_insight(self, insight_id: str) -> None:
        """Delete the insight with *insight_id* if it exists."""
        idx = next((i for i, d in enumerate(self._insights) if d["id"] == insight_id), None)
        if idx is None:
            raise KeyError(f"Insight id '{insight_id}' not found.")

        # Remove from metadata list
        self._insights.pop(idx)

        # Rebuild index from scratch (simplest correct approach given deletions
        # are rare and dataset is small).
        if self._insights:
            embs = np.array([self._embedding_model.encode(d["text"]) for d in self._insights]).astype("float32")
            dim = embs.shape[1]
            self._index = faiss.IndexFlatL2(dim)
            self._index.add(embs)
        else:
            # Reset to empty state
            self._index = None

        self._save()

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Return up to *k* insights most relevant to *query* along with their distances."""
        if self._index is None or not self._insights:
            return []

        q_emb = self._embedding_model.encode(query)
        D, I = self._index.search(np.array([q_emb]).astype("float32"), min(k, len(self._insights)))

        results: List[Tuple[Dict[str, Any], float]] = []
        for faiss_idx, dist in zip(I[0], D[0]):
            if faiss_idx == -1:
                continue
            insight_dict = self._insights[faiss_idx]
            results.append((insight_dict, float(dist)))
        return results

    # Convenience for listing
    def list_all(self) -> List[Dict[str, Any]]:
        return list(self._insights) 