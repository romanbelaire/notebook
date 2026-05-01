"""Deterministic compiler: graph + leaf + draft -> message array + compiled shard ids."""

from app.constellar.graph_ops import collect_path_to_root
from app.constellar.models import ConversationGraph, Shard


def compile(
    graph: ConversationGraph,
    current_leaf_id: str,
    user_draft: str,
) -> tuple[list[dict[str, str]], list[str]]:
    """
    Compile activated path into a linear message array. Returns (messages, compiled_shard_ids).
    Same inputs -> same output.
    """
    path = collect_path_to_root(graph, current_leaf_id)
    visible = [s for s in path if s.visible]
    compiled_ids: list[str] = [s.id for s in visible]
    messages: list[dict[str, str]] = []

    system_parts: list[str] = []
    for s in visible:
        if s.is_special() and s.role == "system":
            system_parts.append(s.content or "")
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for s in visible:
        if s.is_special():
            if s.role == "external" or s.role == "summary":
                content = s.content or ""
                messages.append({"role": "user", "content": f"[{s.role}]\n{content}"})
            continue
        if s.is_turn():
            user_content = s.user_content or ""
            if s.contexts:
                ctx = "\n---\n".join(s.contexts)
                user_content = f"Context:\n{ctx}\n\n{user_content}"
            if s.notes:
                notes_block = "\n".join(f"- {note}" for note in s.notes)
                user_content = f"{user_content}\n\nUser notes on this message:\n{notes_block}"
            messages.append({"role": "user", "content": user_content})
            if s.assistant_content:
                messages.append({"role": "assistant", "content": s.assistant_content})

    messages.append({"role": "user", "content": user_draft})
    return messages, compiled_ids
