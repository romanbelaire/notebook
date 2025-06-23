import os
import pickle
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    pipeline,
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class RAGChat:
    """Thin wrapper providing retrieval-augmented chat via Hugging Face Inference API."""

    def __init__(self, db_dir: str = "db", model_id: Optional[str] = None, hf_token: Optional[str] = None):
        self.db_dir = db_dir

        # ---------------------------  HF token handling  ---------------------------
        if hf_token is None:
            # Priority: env var > text file
            hf_token = os.getenv("HF_API_TOKEN")
            if hf_token is None and os.path.exists("hf_api.txt"):
                with open("hf_api.txt", "r", encoding="utf-8") as f:
                    hf_token = f.read().strip()
        if hf_token is None:
            raise FileNotFoundError(
                "HuggingFace API token not found. Set HF_API_TOKEN env var or provide hf_api.txt."
            )

        # ---------------------------  Model id  ---------------------------
        if model_id is None:
            model_id = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")

        # ---------------------------  Local model load  ---------------------------
        # Basic GPU memory safeguard (fail fast instead of OOM later)
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            min_required_gb = 8  # heuristic; adjust if needed
            if free_gb < min_required_gb:
                raise RuntimeError(
                    f"Only {free_gb:.1f} GB free GPU memory; require at least {min_required_gb} GB to load the model."
                )
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
        )

        # Load and patch config to satisfy transformers 4.41 llama3 expectations
        config = AutoConfig.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
        if (
            hasattr(config, "rope_scaling")
            and isinstance(config.rope_scaling, dict)
            and config.rope_scaling.get("type") == "llama3"
        ):
            # Transformers>=4.41 expects additional keys; add sensible defaults if missing
            config.rope_scaling.setdefault("low_freq_factor", 1.0)
            config.rope_scaling.setdefault("high_freq_factor", 4.0)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            config=config,
        )

        # Create a generation pipeline for convenience
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Embeddings for retrieval
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Vector store
        self._load_vector_store()

    # ---------------------------------------------------------------------
    # Vector store helpers
    # ---------------------------------------------------------------------
    def _load_vector_store(self) -> None:
        index_path = os.path.join(self.db_dir, "index.faiss")
        docs_path = os.path.join(self.db_dir, "docs.pkl")
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(
                f"Vector store files not found under '{self.db_dir}'. Run ingestion first."
            )

        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            store = pickle.load(f)
        self.texts: List[str] = store["texts"]
        self.metadatas: List[dict] = store["metadatas"]

    def _retrieve(self, query: str, k: int = 4) -> List[str]:
        query_emb = self.embedding_model.encode(query)
        D, I = self.index.search(np.array([query_emb]).astype("float32"), k)
        return [self.texts[i] for i in I[0] if i != -1]

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    def chat(self, query: str, history: List[dict]) -> str:
        """Return an answer to *query* grounded in retrieved context using HF text-generation endpoint."""
        contexts = self._retrieve(query, k=4)
        context_block = "\n---\n".join(contexts)

        # Build messages ----------------------------------------------------------
        system_prompt = (
            "You are an expert research assistant. "
            "Answer the user based solely on the given context."
        )
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question:\n{query}\nAnswer with technical precision."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        # Convert to a proper Llama-3 prompt via the tokenizer chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # appends the assistant header automatically
        )

        # -------------------------------------------------------------------------
        out = self.generator(
            prompt_text,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,      # we only want the answer, not the prompt
        )[0]["generated_text"]
        return out.strip() 