# Persistence Module

The `persistence/` module provides data persistence to disk.

## Module Structure

```
persistence/
├── conversation.rs  # Conversation persistence
├── document.rs      # Document persistence
└── settings.rs      # Settings persistence
```

## Persistence Modules

### ConversationPersistence

Saves and loads conversations:

```rust
ConversationPersistence::save_conversation(&conversation)?;
let conversation = ConversationPersistence::load_conversation(id)?;
```

See [Conversation Documentation](conversation.md).

### DocumentPersistence

Saves and loads documents:

```rust
DocumentPersistence::save_document(&document)?;
let document = DocumentPersistence::load_document(id)?;
```

See [Document Documentation](document.md).

### SettingsPersistence

Saves and loads settings:

```rust
SettingsPersistence::save_settings(&settings)?;
let settings = SettingsPersistence::load_settings()?;
```

See [Settings Documentation](settings.md).

## Related Documentation

- [State Management](../architecture/state.md)

