"""Graph-based send flow: compile -> (optional RAG) -> token check -> complete -> persist one turn shard -> snapshot."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app.constellar.compiler import compile
from app.constellar.models import ActiveState, ConversationGraph, Shard
from app.constellar.storage import append_snapshot, load_graph, save_graph
from app.constellar.tokens import count_tokens
from app.metadata_db import get_chunk_texts_for_paper, get_connection

GRAPH_MENTION_MAX_SHARDS = 8
GRAPH_MENTION_MAX_CHARS = 12_000
GRAPH_MENTION_MAX_NOTEPAD = 24_000
_GRAPH_MENTION_MAX_DEPTH = 6


class TokenLimitExceeded(Exception):
    def __init__(self, count: int, limit: int):
        self.count = count
        self.limit = limit
        super().__init__(f"Token count {count} exceeds limit {limit}")


def _format_shard_text(shard: Shard) -> str:
    parts: list[str] = []
    if shard.user_content:
        parts.append(f"User: {shard.user_content}")
    if shard.assistant_content:
        parts.append(f"Assistant: {shard.assistant_content}")
    if shard.content and shard.role:
        parts.append(f"{shard.role}: {shard.content}")
    return "\n".join(parts)


def _notepad_json_path(document_id: str) -> Path:
    return Path("data") / "documents" / f"{document_id}.json"


def parse_mentions_from_text(text: str) -> list[dict[str, Any]]:
    """Find @paper / @shard / @graph / @notepad tokens in arbitrary text (e.g. notepad JSON)."""
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add(key: tuple[str, str], item: dict[str, Any]) -> None:
        if key in seen:
            return
        seen.add(key)
        found.append(item)

    for m in re.finditer(r"@paper:(\d+)", text):
        pid = int(m.group(1))
        _add(("paper", str(pid)), {"kind": "paper", "paper_id": pid})
    for m in re.finditer(r"@shard:([^:\s]+):(\S+)", text):
        gid, sid = m.group(1), m.group(2)
        _add(("shard", f"{gid}:{sid}"), {"kind": "shard", "graph_id": gid, "shard_id": sid})
    for m in re.finditer(r"@graph:(\S+)", text):
        gid = m.group(1)
        _add(("graph", gid), {"kind": "graph", "graph_id": gid})
    for m in re.finditer(r"@notepad:(\S+)", text):
        did = m.group(1)
        _add(("notepad", did), {"kind": "notepad", "document_id": did})
    return found


def _resolve_one_mention(
    m: dict[str, Any],
    visited_notepads: set[str],
    depth: int,
) -> list[str]:
    if depth > _GRAPH_MENTION_MAX_DEPTH:
        return ["(mention expansion depth limit exceeded)"]
    kind = m.get("kind")
    if kind == "paper":
        paper_id = m["paper_id"]
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT filename FROM papers WHERE id = ?", (paper_id,))
        row = cur.fetchone()
        fname = row[0] if row else f"paper_{paper_id}"
        texts = get_chunk_texts_for_paper(conn, paper_id)
        conn.close()
        body = "\n\n".join(texts) if texts else "(no indexed text for this paper)"
        return [f"[Mentioned paper: {fname}]\n{body}"]
    if kind == "shard":
        gid = m["graph_id"]
        sid = m["shard_id"]
        graph, _state = load_graph(gid)
        shard = graph.shards.get(sid)
        if shard is None:
            raise ValueError(f"Shard not found: graph_id={gid} shard_id={sid}")
        body = _format_shard_text(shard)
        return [f"[Mentioned shard {sid} in graph {gid}]\n{body}"]
    if kind == "graph":
        gid = m["graph_id"]
        graph, _state = load_graph(gid)
        shards_sorted = sorted(
            graph.shards.values(),
            key=lambda s: s.created_at,
            reverse=True,
        )[:GRAPH_MENTION_MAX_SHARDS]
        pieces: list[str] = []
        total = 0
        for sh in shards_sorted:
            piece = _format_shard_text(sh)
            if not piece.strip():
                continue
            header = f"--- shard {sh.id} ---"
            chunk = f"{header}\n{piece}"
            if total + len(chunk) > GRAPH_MENTION_MAX_CHARS:
                break
            pieces.append(chunk)
            total += len(chunk)
        body = "\n\n".join(pieces) if pieces else "(empty graph)"
        return [f"[Mentioned graph {gid}: recent shards]\n{body}"]
    if kind == "notepad":
        doc_id = m["document_id"]
        if doc_id in visited_notepads:
            return [f"[Mentioned notepad: {doc_id}]\n(circular reference skipped)"]
        visited_notepads.add(doc_id)
        path = _notepad_json_path(doc_id)
        if not path.is_file():
            return [f"[Mentioned notepad: {doc_id}]\n(not found on server)"]
        raw = path.read_text(encoding="utf-8", errors="replace")
        body = raw[:GRAPH_MENTION_MAX_NOTEPAD]
        if len(raw) > GRAPH_MENTION_MAX_NOTEPAD:
            body += "\n… (truncated)"
        blocks: list[str] = [f"[Mentioned notepad: {doc_id}]\n{body}"]
        nested = parse_mentions_from_text(raw)
        for nm in nested:
            if nm.get("kind") == "notepad" and nm.get("document_id") in visited_notepads:
                continue
            blocks.extend(_resolve_one_mention(nm, visited_notepads, depth + 1))
        return blocks
    raise ValueError(f"Unknown mention kind: {kind!r}")


def _resolve_mention_blocks(
    mentions: list[dict[str, Any]] | None,
) -> list[str]:
    """Build human-readable context blocks from structured @-mentions (including nested notepad)."""
    if not mentions:
        return []
    visited_notepads: set[str] = set()
    blocks: list[str] = []
    for m in mentions:
        blocks.extend(_resolve_one_mention(m, visited_notepads, 0))
    return blocks


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
    system_prompt: str | None = None,
    mentions: list[dict[str, Any]] | None = None,
) -> tuple[str, str, int, list[str], list[dict]]:
    """
    Compile, optionally augment with RAG context, check tokens, complete, persist one turn shard, log snapshot.
    Returns (response_text, new_leaf_id, token_count, contexts, citations). Raises TokenLimitExceeded if over limit.
    """
    graph, state = load_graph(graph_id)
    messages, compiled_shard_ids = compile(graph, current_leaf_id, user_draft)

    if system_prompt and system_prompt.strip():
        extra = system_prompt.strip()
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": f"{extra}\n\n{messages[0]['content']}",
            }
        else:
            messages.insert(0, {"role": "system", "content": extra})

    contexts: list[str] = []
    citations: list[dict] = []
    mention_blocks = _resolve_mention_blocks(mentions)
    mention_prefix = "\n\n".join(mention_blocks) if mention_blocks else ""

    if mention_prefix and messages and messages[-1].get("role") == "user":
        messages[-1] = {
            "role": "user",
            "content": f"Referenced context:\n{mention_prefix}\n\n{messages[-1]['content']}",
        }

    if retrieval is not None and (user_draft.strip() or mention_prefix):
        query_for_retrieval = user_draft.strip()
        if mention_prefix:
            query_for_retrieval = (
                f"{query_for_retrieval}\n\n{mention_prefix}" if query_for_retrieval else mention_prefix
            )
        retrieved = retrieval(query_for_retrieval, 4)
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
