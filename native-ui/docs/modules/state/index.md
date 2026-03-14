# State Module

The `state/` module provides application state management.

## Module Structure

```
state/
├── chat.rs      # Chat state
├── shard.rs     # Shard data model (conversation content)
├── ui.rs        # UI state
├── settings.rs  # Settings state
└── insights.rs  # Insights state
```

## State Modules

### ChatState

Manages conversation state:
- Conversations list
- Current conversation
- Messages (stored as shards; see Shard)

See [Chat State Documentation](chat.md).

### Shard

Core data model for conversation content (one blob per message). Conversations store `Vec<Shard>`; parent/child and friend links form a graph. Pinned shards are persisted via the backend shard API.

See [Shard Documentation](shard.md).

### UIState

Manages UI preferences:
- Sidebar open/closed
- Window positions
- View preferences

See [UI State Documentation](ui.md).

### SettingsState

Manages application settings:
- API base URL
- Model ID
- Theme preferences

See [Settings State Documentation](settings.md).

### InsightsState

Manages insights data:
- Insights list
- Selected insight
- Filters

See [Insights State Documentation](insights.md).

## Related Documentation

- [State Management](../architecture/state.md)
- [Persistence Module](../persistence/index.md)

