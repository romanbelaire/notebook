"""Constellar data model: one shard = one turn (message pair) plus optional context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SPECIAL_ROLES = ("system", "external", "summary")


@dataclass
class Shard:
    """One node in the graph. Either a turn (user + assistant) or a special shard (system/external/summary)."""

    id: str
    parent_ids: list[str]
    created_at: float
    visible: bool = True
    metadata: dict[str, Any] | None = None
    # Turn shard
    user_content: str | None = None
    assistant_content: str | None = None
    contexts: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    # Special shard (system, external, summary)
    content: str | None = None
    role: str | None = None  # "system" | "external" | "summary"

    def is_turn(self) -> bool:
        return self.user_content is not None

    def is_special(self) -> bool:
        return self.content is not None and self.role in SPECIAL_ROLES


@dataclass
class ConversationGraph:
    shards: dict[str, Shard]
    root_id: str


@dataclass
class ActiveState:
    current_leaf_id: str


@dataclass
class Snapshot:
    id: str
    timestamp: float
    compiled_shard_ids: list[str]
    serialized_prompt: str
    model: str
    temperature: float
    response: str
