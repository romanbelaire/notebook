"""Graph operations: add shard, path to root, visibility, DAG validation."""

from app.constellar.models import ConversationGraph, Shard, SPECIAL_ROLES


class GraphValidationError(Exception):
    pass


def validate_shard(shard: Shard) -> None:
    """Raise if shard is neither a valid turn nor a valid special shard."""
    turn = shard.user_content is not None
    special = shard.content is not None and shard.role in SPECIAL_ROLES
    if turn and special:
        raise GraphValidationError("Shard cannot be both turn and special (content+role).")
    if not turn and not special:
        raise GraphValidationError(
            "Shard must be turn (user_content set) or special (content and role in system/external/summary)."
        )


def validate_dag(graph: ConversationGraph) -> None:
    """Raise if root missing, cycle detected, or parent_ids reference missing shards."""
    if graph.root_id not in graph.shards:
        raise GraphValidationError(f"Root id {graph.root_id} not in shards.")
    path: set[str] = set()
    visited: set[str] = set()

    def visit(sid: str) -> None:
        if sid in path:
            raise GraphValidationError(f"Cycle detected at shard {sid}.")
        if sid in visited:
            return
        path.add(sid)
        shard = graph.shards[sid]
        for pid in shard.parent_ids:
            if pid not in graph.shards:
                raise GraphValidationError(f"Parent {pid} of shard {sid} not in shards.")
            visit(pid)
        path.remove(sid)
        visited.add(sid)

    for sid in graph.shards:
        visit(sid)


def get_shard(graph: ConversationGraph, shard_id: str) -> Shard:
    return graph.shards[shard_id]


def add_shard(graph: ConversationGraph, shard: Shard) -> None:
    validate_shard(shard)
    if shard.id in graph.shards:
        raise GraphValidationError(f"Shard id {shard.id} already exists.")
    graph.shards[shard.id] = shard
    validate_dag(graph)


def set_visibility(graph: ConversationGraph, shard_id: str, visible: bool) -> None:
    graph.shards[shard_id].visible = visible


def set_notes(graph: ConversationGraph, shard_id: str, notes: list[str]) -> None:
    graph.shards[shard_id].notes = notes


def remove_shard(graph: ConversationGraph, shard_id: str) -> None:
    if shard_id == graph.root_id:
        raise GraphValidationError("Cannot remove graph root shard.")
    if shard_id not in graph.shards:
        raise GraphValidationError(f"Shard {shard_id} not in graph.")

    removed = graph.shards[shard_id]
    replacement_parents = [pid for pid in removed.parent_ids if pid != shard_id]

    for sid, shard in graph.shards.items():
        if sid == shard_id:
            continue
        if shard_id not in shard.parent_ids:
            continue

        new_parent_ids: list[str] = []
        for parent_id in shard.parent_ids:
            if parent_id == shard_id:
                for replacement in replacement_parents:
                    if replacement != sid and replacement not in new_parent_ids:
                        new_parent_ids.append(replacement)
            elif parent_id != sid and parent_id not in new_parent_ids:
                new_parent_ids.append(parent_id)
        if not new_parent_ids:
            new_parent_ids.append(graph.root_id)
        shard.parent_ids = new_parent_ids

    del graph.shards[shard_id]
    validate_dag(graph)


def collect_path_to_root(graph: ConversationGraph, leaf_id: str) -> list[Shard]:
    """Collect shards from leaf to root following parent_ids. Order: root first, then chronological by created_at."""
    if leaf_id not in graph.shards:
        raise GraphValidationError(f"Leaf {leaf_id} not in graph.")
    collected: dict[str, Shard] = {}
    stack = [leaf_id]
    while stack:
        sid = stack.pop()
        if sid in collected:
            continue
        shard = graph.shards[sid]
        collected[sid] = shard
        for pid in shard.parent_ids:
            if pid not in graph.shards:
                raise GraphValidationError(f"Parent {pid} of {sid} not in graph.")
            stack.append(pid)
    ordered = list(collected.values())
    ordered.sort(key=lambda s: s.created_at)
    return ordered
