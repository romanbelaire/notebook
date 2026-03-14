# Chat State

The `state/chat.rs` module manages conversation state.

## ChatState

```rust
pub struct ChatState {
    pub conversations: Vec<Conversation>,
    pub current_conversation_id: Option<String>,
}
```

### Methods

- `new()`: Create new chat state
- `create_conversation()`: Create new conversation
- `get_current_conversation()`: Get current conversation
- `add_message()`: Add message to conversation
- `generate_title()`: Generate title from text

## Conversation

Conversations store content as **shards**, not raw messages. The chat UI converts shards to/from `ChatMessage` for display.

```rust
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub shards: Vec<Shard>,
    pub created_at: u64,
}
```

- `add_message()` / `set_messages()`: Convert to/from shards and maintain parent/child links.
- `get_messages()`: Returns messages converted from current shards.
- `get_shard()`, `get_shard_mut()`, `add_shard_to_current()`, `link_shards_as_friends()`: Direct shard access.

## Related Documentation

- [Shard](shard.md) – Data model for conversation content
- [State Management](../architecture/state.md)
- [Persistence](../persistence/conversation.md)

