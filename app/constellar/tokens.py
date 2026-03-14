"""Token counting for compiled prompts. No silent truncation; caller raises if over limit."""

from __future__ import annotations

import os
from typing import Literal

_tokenizer_cache: dict[str, object] = {}


def _get_hf_tokenizer(model_id: str):
    if model_id not in _tokenizer_cache:
        from transformers import AutoTokenizer
        import os
        hf_token = os.getenv("HF_API_TOKEN")
        if hf_token is None and os.path.exists("hf_api.txt"):
            with open("hf_api.txt") as _f:
                hf_token = _f.read().strip()
        _tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
        )
    return _tokenizer_cache[model_id]


def count_tokens(
    messages: list[dict[str, str]],
    provider: Literal["local", "openai"],
    model_id: str | None = None,
) -> int:
    """
    Return token count for the given message array. For local uses HF tokenizer (model_id required).
    For openai uses tiktoken cl100k_base.
    """
    if provider == "local":
        if not model_id:
            raise ValueError("model_id required for provider=local")
        tokenizer = _get_hf_tokenizer(model_id)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(prompt)
        return len(ids)
    if provider == "openai":
        try:
            import tiktoken
        except ImportError as e:
            raise RuntimeError("tiktoken required for OpenAI token count. pip install tiktoken") from e
        enc = tiktoken.get_encoding("cl100k_base")
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        text = "\n\n".join(parts)
        return len(enc.encode(text))
    raise ValueError(f"Unknown provider: {provider}")
