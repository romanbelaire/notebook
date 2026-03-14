# Shard (state)

The `state/shard.rs` module defines the core data structure for conversation content: **shards**. A shard is a blob of context (e.g. one chat message) and is the first-class citizen in the data model. Conversations store `Vec<Shard>` instead of raw messages; the chat window converts shards to/from `ChatMessage` for display.

## Shard

```rust
pub struct Shard {
    pub id: String,
    pub text: String,
    pub embedding: Option<Vec<f32>>,   // 384 dims, all-MiniLM-L6-v2
    pub sources: Vec<ShardSource>,
    pub parent_id: Option<String>,
    pub children_ids: Vec<String>,
    pub friends_ids: Vec<String>,
    pub created_at: u64,
    pub metadata: ShardMetadata,
}
```

- **id**: Unique identifier (e.g. `shard_<nanos>`).
- **text**: Content of the message.
- **embedding**: Optional vector for semantic search; computed asynchronously.
- **sources**: References to PDFs, links, or friend shards.
- **parent_id** / **children_ids**: Conversation thread (parent message, follow-up replies).
- **friends_ids**: Linked shards for additional context (not parent-child).
- **metadata**: Role, contexts, citations.

### Methods

- `new(text, role)`: Create a new shard with generated id and timestamp.
- `set_embedding(embedding)`, `has_embedding()`, `embedding_dimension()` (384): Embedding support.
- `add_source(source)`, `set_parent(id)`, `add_child(id)`, `add_friend(id)`: Graph links.

## ShardMetadata

```rust
pub struct ShardMetadata {
    pub role: MessageRole,      // User | Assistant
    pub contexts: Vec<String>,
    pub citations: Vec<Citation>,
}
```

Used for rendering and API payloads.

## ShardSource

Sources referenced by a shard:

```rust
pub enum ShardSource {
    Pdf { paper_id: i32, page: Option<u32> },
    Link { url: String },
    FriendShard { shard_id: String },
}
```

Serialization supports legacy `sister_shard` / `sisters_ids` names.

## Relationship to chat and API

- **ChatState** holds conversations whose content is `Conversation.shards: Vec<Shard>`. Messages are added/updated/removed as shards; `get_messages()` converts shards to `ChatMessage` for the UI.
- **Pinned shards** (insights) are persisted on the backend via the shard HTTP API. The native client uses `ApiClient` methods `list_shards`, `create_or_update_shard`, `update_shard`, `delete_shard`, `search_shards`.

## Related Documentation

- [Chat State](chat.md) – Conversations and shard-based message handling
- [API Client](../api/client.md) – Shard CRUD and search
- [State Management](../../architecture/state.md)
