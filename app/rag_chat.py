import os
from typing import List, Optional, TYPE_CHECKING

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    pipeline,
)

if TYPE_CHECKING:
    from app.retrieval import RetrievalService


class RAGChat:
    """Retrieval-augmented chat using a shared RetrievalService and local HF model for completion."""

    def __init__(
        self,
        db_dir: str = "db",
        model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        retrieval_service: Optional["RetrievalService"] = None,
    ):
        self.db_dir = db_dir
        self._retrieval = retrieval_service

        # ---------------------------  HF token handling  ---------------------------
        if hf_token is None:
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
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            min_required_gb = 8
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

        config = AutoConfig.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
        if (
            hasattr(config, "rope_scaling")
            and isinstance(config.rope_scaling, dict)
            and config.rope_scaling.get("type") == "llama3"
        ):
            config.rope_scaling.setdefault("low_freq_factor", 1.0)
            config.rope_scaling.setdefault("high_freq_factor", 4.0)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            config=config,
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

    def _retrieve(self, query: str, k: int = 4):
        """Return list of (text, metadata) for top-k chunks. Uses shared RetrievalService."""
        if self._retrieval is None:
            raise RuntimeError("RAGChat must be constructed with a RetrievalService")
        return self._retrieval.retrieve(query, k=k)

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    def chat(self, query: str, history: List[dict], *, window_size: int = 6, system_prompt: Optional[str] = None) -> str:
        """Answer *query* using retrieved context plus a sliding window of past contexts.

        1. Collect contexts attached to the last *window_size* messages in *history*.
        2. Retrieve fresh chunks for *query*.
        3. Deduplicate so no context appears twice.
        """

        # ------------------------------------------------------------------
        # 1. Gather recent contexts and notes from conversation history
        # ------------------------------------------------------------------
        # Use last window_size messages (excluding the just-submitted query)
        recent_messages = history[:-1][-window_size:] if len(history) > 1 else []
        recent_contexts: list[str] = []
        recent_meta: list[dict] = []
        notes_from_history: list[str] = []

        for msg in recent_messages:
            if "contexts" in msg and "citations" in msg:
                for ctext, cmeta in zip(msg["contexts"], msg["citations"]):
                    if ctext not in recent_contexts:
                        recent_contexts.append(ctext)
                        recent_meta.append(cmeta)
            note_list = msg.get("notes") or []
            if note_list:
                notes_from_history.append("[User notes on this message]: " + "; ".join(note_list))

        # ------------------------------------------------------------------
        # 2. Retrieve new context for the current query
        # ------------------------------------------------------------------
        retrieved = self._retrieve(query, k=4)

        combined_contexts: list[str] = list(recent_contexts)
        combined_meta: list[dict] = list(recent_meta)

        existing_set = set(combined_contexts)
        for text, meta in retrieved:
            if text in existing_set:
                continue  # skip duplicates to avoid double-counting
            combined_contexts.append(text)
            combined_meta.append(meta)
            existing_set.add(text)

        # Expose for UI consumption
        self.last_contexts = combined_contexts  # type: ignore[attr-defined]
        self.last_citation_meta = combined_meta  # type: ignore[attr-defined]

        context_block = "\n---\n".join(combined_contexts)
        notes_block = "\n".join(notes_from_history) if notes_from_history else ""

        # ------------------------------------------------------------------
        # Build LLM prompt including conversation summary window (optional)
        # ------------------------------------------------------------------
        # Allow overriding the default system prompt supplied by callers (e.g. via UI settings).
        system_prompt = system_prompt or (
            "You are an expert research assistant. "
            "Answer the user based solely on the given context."
        )

        user_prompt = (
            f"Context:\n{context_block}\n\n"
            + (f"Notes from conversation:\n{notes_block}\n\n" if notes_block else "")
            + f"Question:\n{query}\nAnswer with technical precision."
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

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Run local completion only (no retrieval). Used when main.py does retrieval + prompt building."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.complete_messages(messages)

    def complete_messages(
        self,
        messages: List[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """Run local completion from a full message array (e.g. from Constellar compiler)."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        out = self.generator(
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
        )[0]["generated_text"]
        return out.strip()

    # -----------------------------------------------------------------
    # Context pool control
    # -----------------------------------------------------------------

    def set_allowed_sources(self, sources: Optional[set[str]]) -> None:
        """Delegate to shared RetrievalService. Pass *None* to clear restriction."""
        if self._retrieval is not None:
            self._retrieval.set_allowed_sources(sources)