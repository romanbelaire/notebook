"""Shared retrieval over FAISS + MiniLM embedder. No LLM dependency."""

import os
import pickle
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class RetrievalService:
    """Loads FAISS index and embedder from db_dir. Exposes retrieve() and set_allowed_sources()."""

    def __init__(self, db_dir: str = "db") -> None:
        self.db_dir = db_dir
        self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._load_vector_store()
        self.allowed_sources: Optional[set[str]] = None

    def _load_vector_store(self) -> None:
        index_path = os.path.join(self.db_dir, "index.faiss")
        docs_path = os.path.join(self.db_dir, "docs.pkl")

        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            os.makedirs(self.db_dir, exist_ok=True)
            dim = 384
            empty_index = faiss.IndexFlatL2(dim)
            faiss.write_index(empty_index, index_path)
            with open(docs_path, "wb") as f:
                pickle.dump({"texts": [], "metadatas": []}, f)

        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            store = pickle.load(f)
        self.texts: List[str] = store["texts"]
        self.metadatas: List[dict] = store["metadatas"]

        num_vecs = self.index.ntotal
        if len(self.texts) != num_vecs or len(self.metadatas) != num_vecs:
            min_len = min(num_vecs, len(self.texts), len(self.metadatas))
            self.texts = self.texts[:min_len]
            self.metadatas = self.metadatas[:min_len]
            if self.index.ntotal != min_len:
                vectors = self.index.reconstruct_n(0, min_len)
                dim = vectors.shape[1]
                new_index = faiss.IndexFlatL2(dim)
                new_index.add(vectors)
                self.index = new_index

    def set_allowed_sources(self, sources: Optional[set[str]]) -> None:
        """Restrict retrieval to chunks whose metadata 'source' is in *sources*. Pass None to clear."""
        self.allowed_sources = sources

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[str, dict]]:
        """Return list of (text, metadata) for top-k chunks. Respects allowed_sources."""
        if self.index.ntotal == 0:
            return []

        query_emb = self._embedding_model.encode(query)
        multiplier = 3 if self.allowed_sources else 1
        search_k = min(k * multiplier, self.index.ntotal)

        D, I = self.index.search(np.array([query_emb]).astype("float32"), search_k)

        results: List[Tuple[str, dict]] = []
        for idx in I[0]:
            if idx == -1:
                continue
            meta = self.metadatas[idx]
            if self.allowed_sources is not None and meta.get("source") not in self.allowed_sources:
                continue
            results.append((self.texts[idx], meta))
            if len(results) >= k:
                break
        return results
