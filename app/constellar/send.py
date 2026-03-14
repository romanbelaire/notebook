"""Graph-based send flow: compile -> (optional RAG) -> token check -> complete -> persist one turn shard -> snapshot."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

from app.constellar.compiler import compile
from app.constellar.models import ActiveState, ConversationGraph, Shard
from app.constellar.storage import append_snapshot, load_graph, save_graph
from app.constellar.tokens import count_tokens


class TokenLimitExceeded(Exception):
    def __init__(self, count: int, limit: int):
        self.count = count
        self.limit = limit
        super().__init__(f"Token count {count} exceeds limit {limit}")


def send(
    graph_id: str,
    current_leaf_id: str,
    user_draft: str,
    provider: str,
    *,
    model_id: str | None = None,
    openai_model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    model_token_limit: int = 128000,
    rag: Any = None,
    retrieval: Callable[[str, int], list[tuple[str, dict]]] | None = None,
) -> tuple[str, str, int, list[str], list[dict]]:
    """
    Compile, optionally augment with RAG context, check tokens, complete, persist one turn shard, log snapshot.
    Returns (response_text, new_leaf_id, token_count, contexts, citations). Raises TokenLimitExceeded if over limit.
    """
    graph, state = load_graph(graph_id)
    messages, compiled_shard_ids = compile(graph, current_leaf_id, user_draft)

    contexts: list[str] = []
    citations: list[dict] = []
    if retrieval is not None and user_draft.strip():
        retrieved = retrieval(user_draft.strip(), 4)
        for text, meta in retrieved:
            contexts.append(text)
            citations.append(meta)
        if contexts:
            context_block = "\n---\n".join(contexts)
            # Augment last message (user draft) with RAG context
            if messages and messages[-1].get("role") == "user":
                messages[-1] = {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\n{messages[-1]['content']}",
                }

    token_count = count_tokens(messages, provider, model_id=model_id)
    if token_count > model_token_limit:
        raise TokenLimitExceeded(token_count, model_token_limit)

    if provider == "local":
        if rag is None:
            raise ValueError("rag required for provider=local")
        response = rag.complete_messages(messages, max_new_tokens=max_tokens, temperature=temperature)
        model_used = model_id or "local"
    elif provider == "openai":
        from app.completion_openai import openai_complete_messages
        response = openai_complete_messages(
            messages,
            model=openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        model_used = openai_model or "gpt-4o"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    new_shard_id = uuid.uuid4().hex
    now = time.time()
    new_shard = Shard(
        id=new_shard_id,
        parent_ids=[current_leaf_id],
        created_at=now,
        visible=True,
        user_content=user_draft,
        assistant_content=response,
        contexts=contexts,
        citations=citations,
    )
    graph.shards[new_shard_id] = new_shard
    state.current_leaf_id = new_shard_id
    save_graph(graph_id, graph, state)

    serialized_prompt = json.dumps(messages)
    append_snapshot(
        graph_id=graph_id,
        compiled_shard_ids=compiled_shard_ids,
        serialized_prompt=serialized_prompt,
        model=model_used,
        temperature=temperature,
        response=response,
    )

    return response, new_shard_id, token_count, contexts, citations
